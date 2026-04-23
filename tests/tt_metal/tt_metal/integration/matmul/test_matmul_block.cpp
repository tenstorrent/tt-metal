// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar matmul_block demo test for tt-emule.
//
// This test exercises the matmul_block compute kernel on emulated Quasar hardware
// using Dataflow Buffers (DFBs) instead of Circular Buffers (CBs).
//
// Pipeline:
//   Reader DM (1 thread) --[DFB src0, src1]--> Compute (1 thread) --[DFB dst]--> Writer DM (1 thread)
//
// To add your own configurations, copy the QuasarMatmulBlock TEST_F and change
// M, K, N, math_fidelity, fp32_dest_acc_en, or dst_full_sync_en.
//
// Build:
//   cmake --build build_emule_clang --target test_matmul_block
//
// Run:
//   export ARCH_NAME=QUASAR
//   export TT_METAL_MOCK_CLUSTER_DESC_PATH=".../quasar_Q1.yaml"
//   export TT_METAL_EMULATED_MODE=1
//   export TT_METAL_SLOW_DISPATCH_MODE=1
//   export TT_METAL_RUNTIME_ROOT="/path/to/tt-metal"
//   export TT_METAL_SIMULATOR="/path/to/dir/with/soc_descriptor.yaml"
//   ./build_emule_clang/test/tt_emule/test_matmul_block

#include <tt_stl/reflection.hpp>
#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include "matmul_test_utils.hpp"
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;
namespace unit_tests_common::matmul::test_matmul_block {

struct MatmulTileStimuli {
    vector<bfloat16> t;  // Raw tensor values
    vector<uint32_t> a;  // Activations (tilized, transposed)
    vector<uint32_t> w;  // Weights (identity matrix, tilized)
};

struct MatmulTileConfig {
    uint32_t M, K, N;
    // Whether or not to use *_init_short LLK API calls:
    bool test_init_short = false;
    // Whether or not to use *_with_dt LLK API init calls:
    bool with_dt = false;
    // Whether or not we want the result to be stored in DST in FP32:
    bool fp32_dest_acc_en = false;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
    std::string reader_kernel;
    std::string compute_kernel;
    vector<uint32_t> compute_kernel_args;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
};

void create_test_stimuli(MatmulTileStimuli& stimuli, uint32_t M, uint32_t K, uint32_t N) {
    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape, tt::deprecated::Initialize::RANDOM, 0, 100, std::chrono::system_clock::now().time_since_epoch().count());
    stimuli.t = tensor.get_values();

    auto activations_tilized = tilize_swizzled(tensor.get_values(), M * 32, K * 32);
    auto activations_tile_layout =
        convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(activations_tilized));
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    auto activations_tile_transposed = tt::tt_metal::transpose_tiles(activations, M, K, 1);
    stimuli.a = activations_tile_transposed;

    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32);
    auto identity_tilized = tilize_swizzled(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity_tilized));
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
    stimuli.w = weights;
}

void set_math_fid_masks(uint16_t& math_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2:
        case MathFidelity::LoFi: {
            if (MetalContext::instance().get_cluster().arch() != ARCH::QUASAR) {
                math_fid_mask = 0xFFFE;
            }
            break;
        }
        default: {
            TT_THROW("Unsupported MathFidelity={}", math_fidelity);
            break;
        }
    }
}

// Run a Quasar matmul_block test.
//
// Data flow:
//   1. Host writes activations + weights to DRAM buffers
//   2. Reader DM kernel reads DRAM -> pushes tiles into src0/src1 DFBs
//   3. Compute kernel waits on src0/src1 DFBs -> runs matmul_block -> packs to dst DFB
//   4. Writer DM kernel reads dst DFB -> writes result to DRAM
//   5. Host reads DRAM result and compares against golden (activations * identity = activations)
void matmul_tile(
    tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const MatmulTileConfig& cfg,
    vector<uint32_t> activations,
    vector<uint32_t> weights,
    vector<bfloat16> tensor_vals) {
    if (MetalContext::instance().get_cluster().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "This test is Quasar-only";
    }

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    CoreCoord core = {0, 0};

    uint32_t M = cfg.M;
    uint32_t K = cfg.K;
    uint32_t N = cfg.N;
    uint32_t num_tiles = M * K;
    uint32_t single_tile_size_fp32 = 4 * 32 * 32;
    uint32_t single_tile_size_bfp16b = 2 * 32 * 32;
    uint32_t single_tile_size_out0 = cfg.fp32_dest_acc_en ? single_tile_size_fp32 : single_tile_size_bfp16b;
    const size_t dram_buffer_size_bfp16b = num_tiles * single_tile_size_bfp16b;
    const size_t dram_buffer_size_out0 = num_tiles * single_tile_size_out0;

    // --- DRAM buffers ---
    distributed::DeviceLocalBufferConfig input_buffer_config = {
        .page_size = dram_buffer_size_bfp16b, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig input_replicated_buffer_config = {.size = dram_buffer_size_bfp16b};

    distributed::DeviceLocalBufferConfig output_buffer_config = {
        .page_size = dram_buffer_size_out0, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig output_replicated_buffer_config = {.size = dram_buffer_size_out0};

    auto src0_dram_buffer =
        distributed::MeshBuffer::create(input_replicated_buffer_config, input_buffer_config, mesh_device.get());
    auto src1_dram_buffer =
        distributed::MeshBuffer::create(input_replicated_buffer_config, input_buffer_config, mesh_device.get());
    auto dst_dram_buffer =
        distributed::MeshBuffer::create(output_replicated_buffer_config, output_buffer_config, mesh_device.get());

    uint32_t num_input_tiles = 2 * M;

    // --- Dataflow Buffers (DFBs) ---
    // DFB replaces Circular Buffers on Quasar. Each DFB has:
    //   - producer_risc_mask: which processor(s) write to it (0x1 = DM0)
    //   - consumer_risc_mask: which processor(s) read from it (0x100 = compute engine 0)
    //   - AccessPattern::STRIDED: producer/consumer stride through entries
    tt_metal::experimental::dfb::DataflowBufferConfig dfb_src0_config = {
        .entry_size = single_tile_size_bfp16b,
        .num_entries = num_input_tiles,
        .producer_risc_mask = 0x1,       // DM0 produces
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x100,     // Compute engine 0 consumes
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};
    tt_metal::experimental::dfb::DataflowBufferConfig dfb_src1_config = {
        .entry_size = single_tile_size_bfp16b,
        .num_entries = num_input_tiles,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x100,
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};
    uint32_t src0_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, dfb_src0_config);
    uint32_t src1_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, dfb_src1_config);

    // Output DFB: compute engine produces, DM1 (writer) consumes
    tt_metal::experimental::dfb::DataflowBufferConfig dfb_output_config = {
        .entry_size = single_tile_size_out0,
        .num_entries = num_tiles,
        .producer_risc_mask = 0x100,     // Compute engine 0 produces
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,       // DM1 consumes
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false,
        .data_format = cfg.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b,
    };
    uint32_t dst_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program_, core, dfb_output_config);

    // Runtime args for reader kernel
    vector<uint32_t> reader_l1_args = {
        src0_dram_buffer->address(),
        0,
        src1_dram_buffer->address(),
        0,
        (std::uint32_t)K,
        (std::uint32_t)M,
        (std::uint32_t)N,
        (std::uint32_t)(M * single_tile_size_bfp16b),
        (std::uint32_t)(N * single_tile_size_bfp16b),
        0};  // with_bias = false

    // --- Compute defines ---
    std::map<std::string, std::string> compute_defines;
    compute_defines["WITH_DT"] = cfg.with_dt ? "1" : "0";
    compute_defines["TEST_INIT_SHORT"] = cfg.test_init_short ? "1" : "0";
    if (cfg.fp32_dest_acc_en) {
        compute_defines["DST_ACCUM_MODE"] = "1";
    }

    // --- Create Quasar kernels ---
    // Reader DM kernel: reads tiles from DRAM, pushes to src0/src1 DFBs
    std::vector<uint32_t> reader_cta = {src0_dfb, src1_dfb};
    KernelHandle mm_reader_kernel = tt_metal::experimental::quasar::CreateKernel(
        program_,
        cfg.reader_kernel,
        core,
        tt_metal::experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = reader_cta});

    // Writer DM kernel: reads from dst DFB, writes to DRAM
    std::vector<uint32_t> writer_cta = {dst_dfb};
    KernelHandle unary_writer_kernel = tt_metal::experimental::quasar::CreateKernel(
        program_,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = writer_cta});

    // Compute kernel: matmul_block with DFB support
    // compile_args = [block_tile_dim, dst_tile_rows, dst_tile_cols, block_cnt,
    //                 in0_block_tile_cnt, in1_block_tile_cnt, out_block_tile_cnt,
    //                 src0_dfb_id, src1_dfb_id, dst_dfb_id]
    std::vector<uint32_t> compute_cta = cfg.compute_kernel_args;
    compute_cta.insert(compute_cta.end(), {src0_dfb, src1_dfb, dst_dfb});
    KernelHandle compute_kernel = tt_metal::experimental::quasar::CreateKernel(
        program_,
        cfg.compute_kernel,
        core,
        tt_metal::experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
            .math_fidelity = cfg.math_fidelity,
            .fp32_dest_acc_en = cfg.fp32_dest_acc_en,
            .dst_full_sync_en = cfg.dst_full_sync_en,
            .compile_args = compute_cta,
            .defines = compute_defines});

    // --- Bind DFBs to producer/consumer kernels ---
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, src0_dfb, mm_reader_kernel, compute_kernel);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, src1_dfb, mm_reader_kernel, compute_kernel);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
        program_, dst_dfb, compute_kernel, unary_writer_kernel);

    // --- Write inputs and run ---
    fixture->WriteBuffer(mesh_device, src0_dram_buffer, activations);
    fixture->WriteBuffer(mesh_device, src1_dram_buffer, weights);

    tt_metal::SetRuntimeArgs(program_, mm_reader_kernel, core, reader_l1_args);
    tt_metal::SetRuntimeArgs(
        program_,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(), 0, num_tiles});

    fixture->RunProgram(mesh_device, workload);

    // --- Read result and verify ---
    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(mesh_device, dst_dram_buffer, result_vec);

    // Golden: activations * identity_matrix = activations (in tilized layout)
    std::vector<bfloat16> golden = std::move(tensor_vals);
    std::vector<bfloat16> golden_tilized = tilize_swizzled(golden, M * 32, N * 32);
    std::vector<bfloat16> golden_tilized_single =
        convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(golden_tilized));

    std::vector<uint32_t> golden_packed(golden_tilized_single.size());
    uint16_t math_fid_mask = 0xFFFF;
    set_math_fid_masks(math_fid_mask, cfg.math_fidelity);
    for (auto i = 0; i < golden_tilized.size(); i++) {
        golden_tilized_single[i] = std::bit_cast<bfloat16>(
            static_cast<uint16_t>(std::bit_cast<uint16_t>(golden_tilized_single[i]) & math_fid_mask));
        if (cfg.fp32_dest_acc_en) {
            golden_packed[i] = std::bit_cast<uint32_t>(static_cast<float>(golden_tilized_single[i]));
        }
    }
    if (!cfg.fp32_dest_acc_en) {
        golden_packed = pack_bfloat16_vec_into_uint32_vec(golden_tilized_single);
    }

    EXPECT_EQ(golden_packed.size(), result_vec.size());
    EXPECT_EQ(golden_packed, result_vec);

    src0_dram_buffer->deallocate();
    src1_dram_buffer->deallocate();
    dst_dram_buffer->deallocate();

    log_info(
        tt::LogTest,
        "Math Fidelity = {}, FP32_DestAcc = {}, DstSyncFull = {}",
        cfg.math_fidelity,
        cfg.fp32_dest_acc_en,
        cfg.dst_full_sync_en);
}
}  // namespace unit_tests_common::matmul::test_matmul_block

using namespace tt::test_utils;
using namespace unit_tests_common::matmul::test_matmul_block;

// Quasar matmul_block demo test.
//
// Configuration:
//   - M=4, K=4, N=4 (4x4 tile grid = 128x128 elements)
//   - matmul_block.cpp compute kernel (with ARCH_QUASAR DFB support)
//   - reader_matmul_with_bias_blocked.cpp reader (with ARCH_QUASAR DFB support)
//   - HiFi4 math fidelity
//   - BFloat16 accumulation
//   - 1 DM thread, 1 compute thread
//   - 3 DFBs: src0 (DM->Compute), src1 (DM->Compute), dst (Compute->DM)
//   - STRIDED access pattern, 1 producer / 1 consumer per DFB
//
// To experiment with different configurations, change M/K/N, math_fidelity,
// fp32_dest_acc_en, or dst_full_sync_en below.
TEST_F(MeshDispatchFixture, QuasarMatmulBlock) {
    uint32_t M = 4;
    uint32_t K = 4;
    uint32_t N = 4;
    MatmulTileConfig matmul_config = {
        .M = M,
        .K = K,
        .N = N,
        .test_init_short = false,
        .with_dt = false,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .reader_kernel =
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
        .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp",
        // compute_kernel_args: block_tile_dim, dst_tile_rows, dst_tile_cols, block_cnt,
        //                      in0_block_tile_cnt, in1_block_tile_cnt, out_block_tile_cnt
        .compute_kernel_args = {1, M, N, K, M, N, (M * N)},
        .math_fidelity = MathFidelity::HiFi4};
    MatmulTileStimuli stimuli;
    create_test_stimuli(stimuli, M, K, N);

    for (const auto& device : devices_) {
        matmul_tile(this, device, matmul_config, stimuli.a, stimuli.w, stimuli.t);
    }
}

}  // namespace tt::tt_metal
