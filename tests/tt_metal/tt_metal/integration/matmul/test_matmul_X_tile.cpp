// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
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

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;
namespace unit_tests_common::matmul::test_matmul_X_tile {

struct MatmulTileStimuli {
    vector<bfloat16> t;  // Raw tensor values
    vector<uint32_t> a;  // Activations
    vector<uint32_t> w;  // Weights
};

struct MatmulTileConfig {
    uint32_t M, K, N;
    // Whether or not to add matmul result with bias:
    bool with_bias = false;
    // Whether or not to use *_init_short LLK API calls:
    bool test_init_short = false;
    // Whether or not to use *_with_dt LLK API init calls:
    bool with_dt = true;
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

// This function creates bit masks to model math fidelity phases. This will mask the result only.
void set_math_fid_masks(uint16_t& math_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2:
        case MathFidelity::LoFi: {
            // Quasar's multiplier precision is higher,
            // so math fidelity masking of the golden is not needed.
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

// Shared state for the matmul_tile_{quasar,legacy} program builders. Captures the
// derived sizing and the DRAM buffers used by both arch paths so we can build them
// once in setup_matmul_tile_context() and verify them once in verify_matmul_tile_output().
struct MatmulTileContext {
    // Derived sizing
    uint32_t M, K, N;
    uint32_t num_tiles;
    uint32_t single_tile_size_bfp16b;
    uint32_t single_tile_size_out0;
    uint32_t num_input_tiles;
    size_t dram_buffer_size_bfp16b;
    size_t dram_buffer_size_out0;

    // DRAM buffers
    std::shared_ptr<distributed::MeshBuffer> src0_dram_buffer;
    std::shared_ptr<distributed::MeshBuffer> src1_dram_buffer;
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer;
};

static MatmulTileContext setup_matmul_tile_context(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const MatmulTileConfig& cfg) {
    MatmulTileContext ctx;
    ctx.M = cfg.M;
    ctx.K = cfg.K;
    ctx.N = cfg.N;
    // num_tile == M == N == K in the case of multi_tile, conveniently they were all the same!!
    // for single_tile case, num_tile = 1
    ctx.num_tiles = ctx.M * ctx.K;
    constexpr uint32_t single_tile_size_fp32 = 4 * 32 * 32;  // Single 32x32 tile size for Float32
    ctx.single_tile_size_bfp16b = 2 * 32 * 32;               // Single 32x32 tile size for Float16_b / Uint16
    ctx.single_tile_size_out0 = cfg.fp32_dest_acc_en ? single_tile_size_fp32 : ctx.single_tile_size_bfp16b;
    ctx.dram_buffer_size_bfp16b = ctx.num_tiles * ctx.single_tile_size_bfp16b;
    ctx.dram_buffer_size_out0 = ctx.num_tiles * ctx.single_tile_size_out0;
    ctx.num_input_tiles = 2 * ctx.M;

    distributed::DeviceLocalBufferConfig input_buffer_config = {
        .page_size = ctx.dram_buffer_size_bfp16b, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig input_replicated_buffer_config = {.size = ctx.dram_buffer_size_bfp16b};

    distributed::DeviceLocalBufferConfig output_buffer_config = {
        .page_size = ctx.dram_buffer_size_out0, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig output_replicated_buffer_config = {.size = ctx.dram_buffer_size_out0};

    ctx.src0_dram_buffer =
        distributed::MeshBuffer::create(input_replicated_buffer_config, input_buffer_config, mesh_device.get());
    ctx.src1_dram_buffer =
        distributed::MeshBuffer::create(input_replicated_buffer_config, input_buffer_config, mesh_device.get());
    ctx.dst_dram_buffer =
        distributed::MeshBuffer::create(output_replicated_buffer_config, output_buffer_config, mesh_device.get());

    return ctx;
}

static void verify_matmul_tile_output(
    tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const MatmulTileConfig& cfg,
    vector<bfloat16> tensor_vals,
    const MatmulTileContext& ctx) {
    // This is tilized result, will not be modified
    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(mesh_device, ctx.dst_dram_buffer, result_vec);

    std::vector<bfloat16> golden = std::move(tensor_vals);
    std::vector<bfloat16> golden_tilized = tilize_swizzled(golden, ctx.M * 32, ctx.N * 32);
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

    ctx.src0_dram_buffer->deallocate();
    ctx.src1_dram_buffer->deallocate();
    ctx.dst_dram_buffer->deallocate();

    log_info(
        tt::LogTest,
        "Math Fidelity = {}, FP32_DestAcc = {}, DstSyncFull = {}",
        cfg.math_fidelity,
        cfg.fp32_dest_acc_en,
        cfg.dst_full_sync_en);
}

// Metal 2.0 (Quasar) implementation. See matmul_tile() for the dispatcher.
static void matmul_tile_quasar(
    tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const MatmulTileConfig& cfg,
    vector<uint32_t>& activations,
    vector<uint32_t>& weights,
    vector<bfloat16> tensor_vals) {
    auto ctx = setup_matmul_tile_context(mesh_device, cfg);

    const experimental::metal2_host_api::NodeCoord node{0, 0};

    constexpr const char* SRC0_DFB = "src0_dfb";
    constexpr const char* SRC1_DFB = "src1_dfb";
    constexpr const char* DST_DFB = "dst_dfb";
    constexpr const char* READER = "reader";
    constexpr const char* WRITER = "writer";
    constexpr const char* COMPUTE = "compute";

    // Legacy DataflowBufferConfig used enable_implicit_sync = false on all DFBs.
    experimental::metal2_host_api::DataflowBufferSpec src0_dfb_spec{
        .unique_id = SRC0_DFB,
        .entry_size = ctx.single_tile_size_bfp16b,
        .num_entries = ctx.num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .disable_implicit_sync = true,
    };
    experimental::metal2_host_api::DataflowBufferSpec src1_dfb_spec{
        .unique_id = SRC1_DFB,
        .entry_size = ctx.single_tile_size_bfp16b,
        .num_entries = ctx.num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .disable_implicit_sync = true,
    };
    experimental::metal2_host_api::DataflowBufferSpec dst_dfb_spec{
        .unique_id = DST_DFB,
        .entry_size = ctx.single_tile_size_out0,
        .num_entries = ctx.num_tiles,
        .data_format_metadata = cfg.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b,
        .disable_implicit_sync = true,
    };

    experimental::metal2_host_api::KernelSpec reader_spec{
        .unique_id = READER,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{cfg.reader_kernel},
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = SRC0_DFB,
                 .local_accessor_name = "in0",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = SRC1_DFB,
                 .local_accessor_name = "in1",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .runtime_arguments_schema =
            {.named_runtime_args =
                 {"src0_addr",
                  "src0_dram_bank_id",
                  "src1_addr",
                  "src1_dram_bank_id",
                  "num_blocks",
                  "in0_block_tile_cnt",
                  "in1_block_tile_cnt",
                  "in0_block_size_bytes",
                  "in1_block_size_bytes",
                  "with_bias"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{"tt_metal/kernels/dataflow/writer_unary.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = DST_DFB,
            .local_accessor_name = "in",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"dst_addr", "bank_id", "num_tiles"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    // The Quasar matmul_block.cpp kernel uses named CTAs. Map cfg.compute_kernel_args (positional)
    // to named CTAs via the canonical {block_tile_dim, dst_tile_rows, dst_tile_cols, block_cnt,
    // in0_block_tile_cnt, in1_block_tile_cnt, out_block_tile_cnt} ordering.
    TT_FATAL(
        cfg.compute_kernel_args.size() == 7,
        "Quasar matmul_block expects 7 compile-time args but got {}",
        cfg.compute_kernel_args.size());
    experimental::metal2_host_api::KernelSpec::CompileTimeArgBindings compute_cta_bindings{
        {"block_tile_dim", cfg.compute_kernel_args[0]},
        {"dst_tile_rows", cfg.compute_kernel_args[1]},
        {"dst_tile_cols", cfg.compute_kernel_args[2]},
        {"block_cnt", cfg.compute_kernel_args[3]},
        {"in0_block_tile_cnt", cfg.compute_kernel_args[4]},
        {"in1_block_tile_cnt", cfg.compute_kernel_args[5]},
        {"out_block_tile_cnt", cfg.compute_kernel_args[6]},
    };

    experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines compute_defines = {
        {"WITH_DT", cfg.with_dt ? "1" : "0"},
        {"TEST_INIT_SHORT", cfg.test_init_short ? "1" : "0"},
    };
    if (cfg.fp32_dest_acc_en) {
        compute_defines.emplace_back("DST_ACCUM_MODE", "1");
    }

    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{cfg.compute_kernel},
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = SRC0_DFB,
                 .local_accessor_name = "in0",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = SRC1_DFB,
                 .local_accessor_name = "in1",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = DST_DFB,
                 .local_accessor_name = "out",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .compile_time_arg_bindings = compute_cta_bindings,
        .config_spec =
            experimental::metal2_host_api::ComputeConfiguration{
                .math_fidelity = cfg.math_fidelity,
                .fp32_dest_acc_en = cfg.fp32_dest_acc_en,
                .dst_full_sync_en = cfg.dst_full_sync_en,
            },
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "matmul_X_tile",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, dst_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    fixture->WriteBuffer(mesh_device, ctx.src0_dram_buffer, activations);
    fixture->WriteBuffer(mesh_device, ctx.src1_dram_buffer, weights);

    // Build reader runtime args (matches reader_matmul_with_bias_blocked.cpp Quasar named args).
    bool single_tile = !(ctx.M > 1 || ctx.N > 1 || ctx.K > 1);
    const uint32_t num_blocks = single_tile ? 1u : ctx.K;
    const uint32_t in0_block_tile_cnt = single_tile ? 1u : ctx.M;
    const uint32_t in1_block_tile_cnt = single_tile ? 1u : ctx.N;
    const uint32_t in0_block_size_bytes =
        static_cast<uint32_t>((single_tile ? 1u : ctx.M) * ctx.single_tile_size_bfp16b);
    const uint32_t in1_block_size_bytes =
        static_cast<uint32_t>((single_tile ? 1u : ctx.N) * ctx.single_tile_size_bfp16b);
    const uint32_t with_bias_arg = single_tile ? 0u : static_cast<uint32_t>(cfg.with_bias);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = READER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"src0_addr", ctx.src0_dram_buffer->address()},
                       {"src0_dram_bank_id", 0u},
                       {"src1_addr", ctx.src1_dram_buffer->address()},
                       {"src1_dram_bank_id", 0u},
                       {"num_blocks", num_blocks},
                       {"in0_block_tile_cnt", in0_block_tile_cnt},
                       {"in1_block_tile_cnt", in1_block_tile_cnt},
                       {"in0_block_size_bytes", in0_block_size_bytes},
                       {"in1_block_size_bytes", in1_block_size_bytes},
                       {"with_bias", with_bias_arg}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = WRITER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"dst_addr", ctx.dst_dram_buffer->address()}, {"bank_id", 0u}, {"num_tiles", ctx.num_tiles}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = COMPUTE,
        },
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    auto* device = mesh_device->get_devices()[0];
    tt::tt_metal::detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    verify_matmul_tile_output(fixture, mesh_device, cfg, std::move(tensor_vals), ctx);
}

// Legacy (WH/BH) implementation. See matmul_tile() for the dispatcher.
static void matmul_tile_legacy(
    tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const MatmulTileConfig& cfg,
    vector<uint32_t>& activations,
    vector<uint32_t>& weights,
    vector<bfloat16> tensor_vals) {
    auto ctx = setup_matmul_tile_context(mesh_device, cfg);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    CoreCoord core = {0, 0};

    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            ctx.num_input_tiles * ctx.single_tile_size_bfp16b, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, ctx.single_tile_size_bfp16b);
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(
            ctx.num_input_tiles * ctx.single_tile_size_bfp16b, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, ctx.single_tile_size_bfp16b);
    tt_metal::CreateCircularBuffer(program_, core, cb_src1_config);

    std::shared_ptr<distributed::MeshBuffer> src2_dram_buffer;
    std::shared_ptr<distributed::MeshBuffer> dst1_dram_buffer;
    if (cfg.with_bias) {  // with_bias only when M, N, or K > 1
        distributed::DeviceLocalBufferConfig bias_buffer_config = {
            .page_size = ctx.single_tile_size_bfp16b * ctx.N,
            .buffer_type = tt_metal::BufferType::DRAM,
            .bottom_up = false};
        distributed::ReplicatedBufferConfig bias_replicated_buffer_config = {
            .size = ctx.single_tile_size_bfp16b * ctx.N};
        src2_dram_buffer =
            distributed::MeshBuffer::create(bias_replicated_buffer_config, bias_buffer_config, mesh_device.get());

        uint32_t src2_cb_index = 2;
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(
                ctx.num_input_tiles * ctx.single_tile_size_bfp16b, {{src2_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src2_cb_index, ctx.single_tile_size_bfp16b);
        tt_metal::CreateCircularBuffer(program_, core, cb_src2_config);
    } else if (cfg.test_init_short) {  // This will be dummy input in uint16_t
        uint32_t in2_id = 2;
        uint32_t out1_id = 17;

        distributed::DeviceLocalBufferConfig dummy_buffer_config = {
            .page_size = ctx.single_tile_size_bfp16b * ctx.N,
            .buffer_type = tt_metal::BufferType::DRAM,
            .bottom_up = false};
        distributed::ReplicatedBufferConfig dummy_replicated_buffer_config = {
            .size = ctx.single_tile_size_bfp16b * ctx.N};
        // This will be srcB in uint16_t
        src2_dram_buffer =
            distributed::MeshBuffer::create(dummy_replicated_buffer_config, dummy_buffer_config, mesh_device.get());

        // This will be dummy output in uint16_t
        dst1_dram_buffer =
            distributed::MeshBuffer::create(dummy_replicated_buffer_config, dummy_buffer_config, mesh_device.get());

        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(
                ctx.num_input_tiles * ctx.single_tile_size_bfp16b, {{in2_id, tt::DataFormat::UInt16}})
                .set_page_size(in2_id, ctx.single_tile_size_bfp16b);
        tt_metal::CreateCircularBuffer(program_, core, cb_src2_config);

        tt_metal::CircularBufferConfig cb_dst1_config =
            tt_metal::CircularBufferConfig(
                ctx.num_input_tiles * ctx.single_tile_size_bfp16b, {{out1_id, tt::DataFormat::UInt16}})
                .set_page_size(out1_id, ctx.single_tile_size_bfp16b);
        tt_metal::CreateCircularBuffer(program_, core, cb_dst1_config);
    }

    uint32_t output_cb_index = 16;
    vector<uint32_t> reader_l1_args;
    if (ctx.M > 1 || ctx.N > 1 || ctx.K > 1) {
        uint32_t intermediate_cb_index = 24;
        std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
            {output_cb_index, (cfg.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)},
            {intermediate_cb_index, (cfg.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)}};

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(ctx.dram_buffer_size_out0, partials_and_out_data_format_spec)
                .set_page_size(output_cb_index, ctx.single_tile_size_out0)
                .set_page_size(intermediate_cb_index, ctx.single_tile_size_out0);
        tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

        reader_l1_args = {
            ctx.src0_dram_buffer->address(),
            0,
            ctx.src1_dram_buffer->address(),
            0,
            (std::uint32_t)ctx.K,
            (std::uint32_t)ctx.M,
            (std::uint32_t)ctx.N,
            (std::uint32_t)(ctx.M * ctx.single_tile_size_bfp16b),
            (std::uint32_t)(ctx.N * ctx.single_tile_size_bfp16b),
            cfg.with_bias};
    } else {
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * ctx.single_tile_size_out0,
                {{output_cb_index, (cfg.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)}})
                .set_page_size(output_cb_index, ctx.single_tile_size_out0);
        tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

        reader_l1_args = {
            ctx.src0_dram_buffer->address(),
            0,
            ctx.src1_dram_buffer->address(),
            0,
            1,
            1,
            1,
            1 * ctx.single_tile_size_bfp16b,
            1 * ctx.single_tile_size_bfp16b};
    }

    std::map<std::string, std::string> compute_defines;
    compute_defines["WITH_DT"] = cfg.with_dt ? "1" : "0";
    compute_defines["TEST_INIT_SHORT"] = cfg.test_init_short ? "1" : "0";
    if (cfg.fp32_dest_acc_en) {
        compute_defines["DST_ACCUM_MODE"] = "1";
    }

    auto mm_reader_kernel = tt_metal::CreateKernel(
        program_,
        cfg.reader_kernel,
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program_,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    tt_metal::CreateKernel(
        program_,
        cfg.compute_kernel,
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = cfg.math_fidelity,
            .fp32_dest_acc_en = cfg.fp32_dest_acc_en,
            .dst_full_sync_en = cfg.dst_full_sync_en,
            .compile_args = cfg.compute_kernel_args,
            .defines = compute_defines});

    fixture->WriteBuffer(mesh_device, ctx.src0_dram_buffer, activations);
    fixture->WriteBuffer(mesh_device, ctx.src1_dram_buffer, weights);

    if (cfg.with_bias || cfg.test_init_short) {
        vector<uint32_t> bias(ctx.N * 512, 0);
        fixture->WriteBuffer(mesh_device, src2_dram_buffer, bias);

        vector<uint32_t> bias_args = {
            src2_dram_buffer->address(), 0, (std::uint32_t)ctx.N, (std::uint32_t)(ctx.N * ctx.single_tile_size_bfp16b)};

        for (uint32_t arg : bias_args) {
            reader_l1_args.push_back(arg);
        }
    }

    tt_metal::SetRuntimeArgs(program_, mm_reader_kernel, core, reader_l1_args);

    tt_metal::SetRuntimeArgs(
        program_,
        unary_writer_kernel,
        core,
        {ctx.dst_dram_buffer->address(), 0, ctx.num_tiles});  // this is M * N in the multi_tile case !!

    fixture->RunProgram(mesh_device, workload);

    if ((cfg.with_bias || cfg.test_init_short)) {
        if (cfg.test_init_short) {
            dst1_dram_buffer->deallocate();
        }
        src2_dram_buffer->deallocate();
    }

    verify_matmul_tile_output(fixture, mesh_device, cfg, std::move(tensor_vals), ctx);
}

// Builds and runs a matmul tile program, then verifies the result against a
// CPU-computed golden. Dispatches to the Metal 2.0 (Quasar) or legacy (WH/BH)
// implementation based on the target arch.
void matmul_tile(
    tt_metal::MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const MatmulTileConfig& cfg,
    vector<uint32_t> activations,
    vector<uint32_t> weights,
    vector<bfloat16> tensor_vals) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        matmul_tile_quasar(fixture, mesh_device, cfg, activations, weights, std::move(tensor_vals));
    } else {
        matmul_tile_legacy(fixture, mesh_device, cfg, activations, weights, std::move(tensor_vals));
    }
}
}  // namespace unit_tests_common::matmul::test_matmul_X_tile

using namespace tt::test_utils;
using namespace unit_tests_common::matmul::test_matmul_X_tile;

/* matmul_config.compute_kernel_args = {
    // block_tile_dim, within block, how many tiles are on the K dim
    // dst_tile_rows
    // dst_tile_cols
    // block_cnt, across blocks, how many tiles are on the K dim
    // in0_block_tile_cnt, M * block_tile_dim
    // in1_block_tile_cnt, N * block_tile_dim
    // out_block_tile_cnt
}
*/

TEST_F(MeshDispatchFixture, TensixMatmulSingleTile) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        GTEST_SKIP() << "TensixMatmulSingleTile not implemented for Quasar yet";
    }
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        for (bool fp32_dest_acc_en : {true, false}) {
            for (bool dst_full_sync_en : {true, false}) {
                MatmulTileConfig matmul_config = {
                    .M = 1,
                    .K = 1,
                    .N = 1,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
                    .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
                    .compute_kernel_args = {1, 1, 1, 1, 1, 1, 1},
                    .math_fidelity = MathFidelity(i)};
                MatmulTileStimuli stimuli;
                create_test_stimuli(stimuli, 1, 1, 1);

                for (const auto& device : devices_) {
                    matmul_tile(this, device, matmul_config, stimuli.a, stimuli.w, stimuli.t);
                }
            }
        }
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulMultiTile) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        GTEST_SKIP() << "TensixMatmulMultiTile not implemented for Quasar yet";
    }
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        for (bool fp32_dest_acc_en : {true, false}) {
            for (bool dst_full_sync_en : {true, false}) {
                uint32_t M = fp32_dest_acc_en ? 2 : 4;
                uint32_t N = fp32_dest_acc_en ? 2 : 4;
                uint32_t K = fp32_dest_acc_en ? 2 : 4;
                MatmulTileConfig matmul_config = {
                    .M = M,
                    .K = K,
                    .N = N,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .reader_kernel =
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
                    .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_with_bias.cpp",
                    .compute_kernel_args = {1, M, N, K, M, N, (M * N), matmul_config.with_bias},
                    .math_fidelity = MathFidelity(i)};
                MatmulTileStimuli stimuli;
                create_test_stimuli(stimuli, M, K, N);

                for (const auto& device : devices_) {
                    matmul_tile(this, device, matmul_config, stimuli.a, stimuli.w, stimuli.t);
                    log_info(LogTest, "Multi tile with no bias passed");
                    matmul_config.with_bias = true;
                    matmul_tile(this, device, matmul_config, stimuli.a, stimuli.w, stimuli.t);
                    log_info(LogTest, "Multi tile with bias passed");
                }
            }
        }
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlock) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        for (bool fp32_dest_acc_en : {true, false}) {
            for (bool dst_full_sync_en : {true, false}) {
                uint32_t M = fp32_dest_acc_en ? 2 : 4;
                uint32_t N = fp32_dest_acc_en ? 2 : 4;
                uint32_t K = fp32_dest_acc_en ? 2 : 4;
                MatmulTileConfig matmul_config = {
                    .M = M,
                    .K = K,
                    .N = N,
                    .test_init_short = false,
                    .with_dt = false,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .reader_kernel =
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
                    .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp",
                    .compute_kernel_args = {1, M, N, K, M, N, (M * N)},
                    .math_fidelity = MathFidelity(i)};
                MatmulTileStimuli stimuli;
                create_test_stimuli(stimuli, M, K, N);

                for (const auto& device : devices_) {
                    matmul_tile(this, device, matmul_config, stimuli.a, stimuli.w, stimuli.t);
                }
            }
        }
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockInitShort) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        for (bool fp32_dest_acc_en : {true, false}) {
            for (bool dst_full_sync_en : {true, false}) {
                uint32_t M = fp32_dest_acc_en ? 2 : 4;
                uint32_t N = fp32_dest_acc_en ? 2 : 4;
                uint32_t K = fp32_dest_acc_en ? 2 : 4;
                MatmulTileConfig matmul_config = {
                    .M = M,
                    .K = K,
                    .N = N,
                    .test_init_short = true,
                    .with_dt = false,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .reader_kernel =
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
                    .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp",
                    .compute_kernel_args = {1, M, N, K, M, N, (M * N)},
                    .math_fidelity = MathFidelity(i)};
                MatmulTileStimuli stimuli;
                create_test_stimuli(stimuli, M, K, N);

                for (const auto& device : devices_) {
                    matmul_tile(this, device, matmul_config, stimuli.a, stimuli.w, stimuli.t);
                }
            }
        }
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockInitShortWithDt) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        GTEST_SKIP() << "TensixMatmulBlockInitShortWithDt not implemented for Quasar yet";
    }
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        for (bool fp32_dest_acc_en : {true, false}) {
            for (bool dst_full_sync_en : {true, false}) {
                uint32_t M = fp32_dest_acc_en ? 2 : 4;
                uint32_t N = fp32_dest_acc_en ? 2 : 4;
                uint32_t K = fp32_dest_acc_en ? 2 : 4;
                MatmulTileConfig matmul_config = {
                    .M = M,
                    .K = K,
                    .N = N,
                    .test_init_short = true,
                    .with_dt = true,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .reader_kernel =
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
                    .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp",
                    .compute_kernel_args = {1, M, N, K, M, N, (M * N)},
                    .math_fidelity = MathFidelity(i)};
                MatmulTileStimuli stimuli;
                create_test_stimuli(stimuli, M, K, N);

                for (const auto& device : devices_) {
                    matmul_tile(this, device, matmul_config, stimuli.a, stimuli.w, stimuli.t);
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
