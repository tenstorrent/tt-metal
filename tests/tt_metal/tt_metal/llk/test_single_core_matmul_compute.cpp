// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <chrono>
#include <cmath>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <unistd.h>
#include <array>
#include <map>
#include <memory>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/mxfp4.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include "single_core_compute_runners.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include "impl/data_format/bfloat16_utils.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/float8_utils.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::matmul {

// Per-block matmul: out[M x N] = in0[M x K] * in1[K x N]
// Repeated num_blocks times
// If K > 1 -> dest accumulation within each block
// If num_blocks > 1 -> partials accumulation (either l1 accumulation or spill and reload)
struct BlockedMatmulConfig {
    std::uint32_t M = 1;           // per-block rows (tiles)
    std::uint32_t K = 1;           // per-block inner dim (tiles)
    std::uint32_t N = 1;           // per-block cols (tiles)
    std::uint32_t num_blocks = 1;  // number of K-blocks
    bool packer_l1_acc = false;
    // Format / DEST-mode parameters. Defaults preserve the original BF16 + fp32_dest=false
    // behaviour exercised by `TensixTestSingleCoreMultiBlock*ComputeMatmul` (and the Quasar
    // multi-block matmul tests). On Blackhole, any in/out FP8 path requires
    // fp32_dest_acc_en=true (asserted at JIT time in ComputeKernel::set_build_options).
    tt::DataFormat in0_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat in1_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat out_fmt = tt::DataFormat::Float16_b;
    bool fp32_dest_acc_en = false;
};

void create_CBs_for_fused_matmul(
    distributed::MeshWorkload& workload,
    const std::shared_ptr<distributed::MeshDevice>& /*mesh_device*/,
    CoreCoord core,
    bool activations_rm,
    bool output_rm,
    std::uint32_t M,
    std::uint32_t N,
    std::uint32_t in0_block_w,
    std::uint32_t /*out_subblock_h*/) {
    std::uint32_t num_bytes_for_df = 2;
    std::uint32_t in0_cb = 0;
    std::uint32_t in1_cb = 1;
    std::uint32_t tilize_mode_tilized_in0_cb = 24;
    std::uint32_t matmul_partials_cb = 25;
    std::uint32_t untilize_mode_final_matmul_partials_cb = 26;
    std::uint32_t untilize_mode_reblock_cb = 27;
    std::uint32_t out0_cb = 16;

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program = workload.get_programs().at(device_range);

    std::uint32_t single_tile_size = num_bytes_for_df * 1024;

    std::uint32_t num_output_tiles = M * N;

    // Invariants
    std::uint32_t cb0_tiles = M * in0_block_w * 2;
    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{in0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(in0_cb, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    std::uint32_t cb1_tiles = N * in0_block_w * 2;
    tt_metal::CircularBufferConfig cb_in1_config =
        tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{in1_cb, tt::DataFormat::Float16_b}})
            .set_page_size(in1_cb, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_in1_config);

    if (not activations_rm and not output_rm) {  // no tilize, no untilize
        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Partials share same L1 address space as output
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else if (not activations_rm and output_rm) {  // no tilize, just untilize

        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Need a new CB to push output block to since other
        // intermediate read pointer changes in enable reload
        // block
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        std::uint32_t reblock_cb_tiles = N;  // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config =
            tt_metal::CircularBufferConfig(
                reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else if (activations_rm and not output_rm) {  // just tilize, no untilize

        tt_metal::CircularBufferConfig cb_src0_tilized_config =
            tt_metal::CircularBufferConfig(
                cb0_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else {  // tilize activations and untilize output

        // Used for placing tilized activations
        tt_metal::CircularBufferConfig cb_src0_tilized_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Shares same address space as matmul partials
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size,
                {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        std::uint32_t reblock_cb_tiles = N;  // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config =
            tt_metal::CircularBufferConfig(
                reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    }
}

// U(-1, +1) packed inputs (per in_fmt) and float views after quantization.
// Trivial / constant stimulus is avoided so structural K-stride bugs are not
// masked. M=K=N=1 is the single-tile case; larger (M, K, N) for single_block_matmul.
struct MatmulStimulus {
    std::vector<std::uint32_t> packed_input0;
    std::vector<std::uint32_t> packed_input1;
    std::vector<float> in0_floats;
    std::vector<float> in1_floats;
};

// Per-operand stimulus generation. Returns the packed L1 representation and a
// float reference vector in face-major-within-tile order (matching the matmul
// golden's byte_tile_face_major_index addressing).  For Bfp8_b, the float
// reference is the unpack-after-pack roundtrip so the golden reflects the
// values the hardware actually sees, not the raw RNG samples.
struct OperandStimulus {
    std::vector<std::uint32_t> packed;
    std::vector<float> floats;
};

OperandStimulus make_operand_stimulus(tt::DataFormat fmt, std::uint32_t tile_count, std::uint32_t seed) {
    constexpr float rng = 1.0f;
    const size_t num_elements = tt::constants::TILE_HW * tile_count;
    OperandStimulus out;
    if (fmt == tt::DataFormat::Fp8_e4m3) {
        out.packed = generate_packed_uniform_random_vector<std::uint32_t, float8_e4m3>(
            float8_e4m3(-rng), float8_e4m3(+rng), num_elements, seed);
        out.floats = fp8_to_floats(out.packed);
    } else if (fmt == tt::DataFormat::Float16_b) {
        out.packed = generate_packed_uniform_random_vector<std::uint32_t, bfloat16>(
            bfloat16(-rng), bfloat16(+rng), num_elements, seed);
        out.floats = bf16_to_floats(out.packed);
    } else if (fmt == tt::DataFormat::Bfp8_b) {
        // Generate random floats in face-major tile order (no spatial reshape),
        // pack to Bfp8_b L1 layout, then unpack to get the float values
        // post-quantization for the golden reference.
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-rng, +rng);
        std::vector<float> raw(num_elements);
        for (float& v : raw) {
            v = dist(gen);
        }
        out.packed =
            pack_as_bfp8_tiles<float>(ttsl::make_const_span(raw), /*row_major_input=*/false, /*is_exp_a=*/false);
        out.floats = unpack_bfp8_tiles_into_float_vec(
            ttsl::make_const_span(out.packed), /*row_major_output=*/false, /*is_exp_a=*/false);
    } else if (fmt == tt::DataFormat::MxFp4) {
        // MXFP4 (S1E2M1, OCP microscaling) input. Same staging as Bfp8_b: generate random
        // floats in face-major tile order, pack to the MXFP4 L1 layout, then unpack to recover
        // the post-quantization float values used as the golden reference. NOTE: this L1 layout
        // is identical for plain MxFp4 and the register-only MxFp4_2x variants -- the 2x packing
        // happens in the unpacker (L1->SrcReg), not here.
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-rng, +rng);
        std::vector<float> raw(num_elements);
        for (float& v : raw) {
            v = dist(gen);
        }
        out.packed = pack_as_mxfp4_tiles<float>(ttsl::make_const_span(raw), /*row_major_input=*/false);
        out.floats = unpack_mxfp4_tiles_into_float_vec(ttsl::make_const_span(out.packed), /*row_major_output=*/false);
    } else {
        TT_FATAL(false, "make_operand_stimulus: unsupported fmt {}", static_cast<int>(fmt));
    }
    return out;
}

MatmulStimulus make_matmul_stimulus(
    tt::DataFormat in0_fmt, tt::DataFormat in1_fmt, std::uint32_t M, std::uint32_t K, std::uint32_t N) {
    OperandStimulus a = make_operand_stimulus(in0_fmt, M * K, /*seed=*/0);
    OperandStimulus b = make_operand_stimulus(in1_fmt, K * N, /*seed=*/1);
    MatmulStimulus out;
    out.packed_input0 = std::move(a.packed);
    out.packed_input1 = std::move(b.packed);
    out.in0_floats = std::move(a.floats);
    out.in1_floats = std::move(b.floats);
    return out;
}

MatmulStimulus make_matmul_stimulus(tt::DataFormat in_fmt, std::uint32_t M, std::uint32_t K, std::uint32_t N) {
    return make_matmul_stimulus(in_fmt, in_fmt, M, K, N);
}

// Host reference matmul over face-major tiles: output layout is M×N tiles × TILE_HW
// elements per tile.
std::vector<float> make_matmul_golden(
    const std::vector<float>& in0_floats,
    const std::vector<float>& in1_floats,
    std::uint32_t M,
    std::uint32_t K,
    std::uint32_t N) {
    std::vector<float> golden_floats(M * N * tt::constants::TILE_HW, 0.0f);
    for (std::uint32_t mt = 0; mt < M; mt++) {
        for (std::uint32_t nt = 0; nt < N; nt++) {
            const size_t out_tile_off = (mt * N + nt) * tt::constants::TILE_HW;
            for (std::uint32_t y = 0; y < tt::constants::TILE_HEIGHT; y++) {
                for (std::uint32_t x = 0; x < tt::constants::TILE_WIDTH; x++) {
                    float acc = 0.0f;
                    for (std::uint32_t kt = 0; kt < K; kt++) {
                        const size_t in0_tile_off = (mt * K + kt) * tt::constants::TILE_HW;
                        const size_t in1_tile_off = (kt * N + nt) * tt::constants::TILE_HW;
                        for (std::uint32_t z = 0; z < tt::constants::TILE_WIDTH; z++) {
                            acc += in0_floats[in0_tile_off + byte_tile_face_major_index(z, y)] *
                                   in1_floats[in1_tile_off + byte_tile_face_major_index(x, z)];
                        }
                    }
                    golden_floats[out_tile_off + byte_tile_face_major_index(x, y)] = acc;
                }
            }
        }
    }
    return golden_floats;
}

inline void dump_matmul_debug(
    const std::vector<float>& in0_floats,
    const std::vector<float>& in1_floats,
    const std::vector<float>& golden_floats,
    const std::vector<float>& dest_floats) {
    log_info(tt::LogTest, "Matmul mismatch; dumping in0/in1/golden/device (face-major, TILE_WIDTH per row):");
    log_info(tt::LogTest, "in0_floats:");
    tt::test_utils::print_vector_fixed_numel_per_row(in0_floats, tt::constants::TILE_WIDTH);
    log_info(tt::LogTest, "in1_floats:");
    tt::test_utils::print_vector_fixed_numel_per_row(in1_floats, tt::constants::TILE_WIDTH);
    log_info(tt::LogTest, "golden_floats:");
    tt::test_utils::print_vector_fixed_numel_per_row(golden_floats, tt::constants::TILE_WIDTH);
    log_info(tt::LogTest, "device_floats:");
    tt::test_utils::print_vector_fixed_numel_per_row(dest_floats, tt::constants::TILE_WIDTH);
}

// Single-tile matmul. Inputs and output formats are programmable; default
// (Float16_b in, Float16_b out, fp32_dest_acc_en=false) preserves the legacy
// BF16 test semantics. Pass Fp8_e4m3 for the FP8 enablement variants on
// Blackhole (see PR #40287/#41142 for the LLK family fix-up).  in0_fmt and
// in1_fmt may differ for mixed-family verification (e.g. Float16_b A x Fp8 B).
bool single_tile_matmul(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    tt::DataFormat in0_fmt = tt::DataFormat::Float16_b,
    tt::DataFormat in1_fmt = tt::DataFormat::Float16_b,
    tt::DataFormat out_fmt = tt::DataFormat::Float16_b,
    bool fp32_dest_acc_en = false) {
    bool pass = true;
    CoreCoord core(0, 0);
    const std::uint32_t in0_cb_index = 0;
    const std::uint32_t in1_cb_index = 1;
    const std::uint32_t out_cb_index = 16;
    const std::uint32_t in0_tile_size = tt::tile_size(in0_fmt);
    const std::uint32_t in1_tile_size = tt::tile_size(in1_fmt);
    const std::uint32_t out_tile_size = tt::tile_size(out_fmt);

    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram(*mesh_device);
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = in0_tile_size},
        {.page_size = in0_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device.get());
    auto input1_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = in1_tile_size},
        {.page_size = in1_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device.get());
    auto output_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = out_tile_size},
        {.page_size = out_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device.get());

    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(in0_tile_size, {{in0_cb_index, in0_fmt}})
            .set_page_size(in0_cb_index, in0_tile_size));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(in1_tile_size, {{in1_cb_index, in1_fmt}})
            .set_page_size(in1_cb_index, in1_tile_size));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(out_tile_size, {{out_cb_index, out_fmt}})
            .set_page_size(out_cb_index, out_tile_size));

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/single_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {in0_cb_index, in1_cb_index, out_cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus & Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    const MatmulStimulus stimulus = make_matmul_stimulus(in0_fmt, in1_fmt, /*M=*/1, /*K=*/1, /*N=*/1);
    const std::vector<float> golden_floats =
        make_matmul_golden(stimulus.in0_floats, stimulus.in1_floats, /*M=*/1, /*K=*/1, /*N=*/1);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    distributed::EnqueueWriteMeshBuffer(cq, input0_dram_buffer, stimulus.packed_input0, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, input1_dram_buffer, stimulus.packed_input1, /*blocking=*/true);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {(std::uint32_t)input0_dram_buffer->address(),
         (std::uint32_t)0,
         (std::uint32_t)input1_dram_buffer->address(),
         (std::uint32_t)0,
         (std::uint32_t)1});
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {(std::uint32_t)output_dram_buffer->address(), (std::uint32_t)0, (std::uint32_t)1});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::uint32_t> dest_buffer_data;
    distributed::EnqueueReadMeshBuffer(cq, dest_buffer_data, output_dram_buffer, /*blocking=*/true);
    std::vector<float> dest_floats;
    if (out_fmt == tt::DataFormat::Fp8_e4m3) {
        dest_floats = fp8_to_floats(dest_buffer_data);
    } else if (out_fmt == tt::DataFormat::Float16_b) {
        dest_floats = bf16_to_floats(dest_buffer_data);
    } else {
        TT_FATAL(false, "single_tile_matmul: unsupported out_fmt {}", static_cast<int>(out_fmt));
    }

    // Tolerances under random U(-1, +1) stimulus: FP8 output tracks ~1/8
    // quantization error; BF16 output (with default fp32_dest_acc=false uses
    // BF16 dest accumulation, so per-element error grows with the K=32 inner
    // dim) needs a few-percent slack.
    float rtol = 0.05f;
    float atol = 0.2f;
    if (out_fmt == tt::DataFormat::Fp8_e4m3) {
        rtol = 0.125f;
        atol = 0.125f;
    }
    pass &= tt::test_utils::is_close_vectors<float>(
        dest_floats, golden_floats, [&](float a, float b) { return tt::test_utils::is_close(a, b, rtol, atol); });
    if (not pass) {
        dump_matmul_debug(stimulus.in0_floats, stimulus.in1_floats, golden_floats, dest_floats);
    }
    return pass;
}
// Single-block matmul: blocking that still fits within Dst (no spill/reload).
// Inputs and output formats are programmable; defaults preserve the legacy
// BF16 test semantics.
bool single_block_matmul(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    std::uint32_t M,
    std::uint32_t K,
    std::uint32_t N,
    tt::DataFormat in_fmt = tt::DataFormat::Float16_b,
    tt::DataFormat out_fmt = tt::DataFormat::Float16_b,
    bool fp32_dest_acc_en = false) {
    bool pass = true;
    CoreCoord core(0, 0);
    const std::uint32_t in0_cb_index = 0;
    const std::uint32_t in1_cb_index = 1;
    const std::uint32_t out_cb_index = 16;
    const size_t in_tile_size = tt::tile_size(in_fmt);
    const size_t out_tile_size = tt::tile_size(out_fmt);
    const size_t in0_byte_size = M * K * in_tile_size;
    const size_t in1_byte_size = K * N * in_tile_size;
    const size_t out_byte_size = M * N * out_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    tt_metal::Program program = tt_metal::CreateProgram(*mesh_device);
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto input0_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = in0_byte_size},
        {.page_size = in0_byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device.get());
    auto input1_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = in1_byte_size},
        {.page_size = in1_byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device.get());
    auto output_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = out_byte_size},
        {.page_size = out_byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device.get());

    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(in0_byte_size, {{in0_cb_index, in_fmt}})
            .set_page_size(in0_cb_index, in_tile_size));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(in1_byte_size, {{in1_cb_index, in_fmt}})
            .set_page_size(in1_cb_index, in_tile_size));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(out_byte_size, {{out_cb_index, out_fmt}})
            .set_page_size(out_cb_index, out_tile_size));

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = {in0_cb_index, in1_cb_index, out_cb_index, M * K, K * N, M * N, M, N, K}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus & Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    const MatmulStimulus stimulus = make_matmul_stimulus(in_fmt, M, K, N);
    const std::vector<float> golden_floats = make_matmul_golden(stimulus.in0_floats, stimulus.in1_floats, M, K, N);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    distributed::EnqueueWriteMeshBuffer(cq, input0_dram_buffer, stimulus.packed_input0, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, input1_dram_buffer, stimulus.packed_input1, /*blocking=*/true);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {(std::uint32_t)input0_dram_buffer->address(),
         (std::uint32_t)0,
         (std::uint32_t)input1_dram_buffer->address(),
         (std::uint32_t)0,
         (std::uint32_t)1,              // num_blocks
         (std::uint32_t)M * K,          // in0_block_tile_cnt
         (std::uint32_t)K * N,          // in1_block_tile_cnt
         (std::uint32_t)in0_byte_size,  // in0_block_size_bytes
         (std::uint32_t)in1_byte_size});
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {(std::uint32_t)output_dram_buffer->address(), (std::uint32_t)0, (std::uint32_t)M * N});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::uint32_t> dest_buffer_data;
    distributed::EnqueueReadMeshBuffer(cq, dest_buffer_data, output_dram_buffer, /*blocking=*/true);
    std::vector<float> dest_floats;
    // Tolerances under random U(-1, +1) stimulus. FP8 output: ~1/8
    // quantization plus deeper accumulation rounding for K>1. BF16 output
    // (default fp32_dest_acc=false uses BF16 dest accumulation, so per-element
    // error grows with K * TILE_WIDTH inner-dim length) needs a few-percent
    // slack. PCC backstop catches structural mis-permutations that pointwise
    // tolerances would let slip.
    float rtol = 0.05f;
    float atol = (K > 1) ? 0.4f : 0.2f;
    double min_pcc = 0.99;
    if (out_fmt == tt::DataFormat::Fp8_e4m3) {
        dest_floats = fp8_to_floats(dest_buffer_data);
        rtol = 0.125f;
        atol = (K > 1) ? 0.25f : 0.125f;
        min_pcc = (K > 1) ? 0.98 : 0.99;
    } else if (out_fmt == tt::DataFormat::Float16_b) {
        dest_floats = bf16_to_floats(dest_buffer_data);
    } else {
        TT_FATAL(false, "single_block_matmul: unsupported out_fmt {}", static_cast<int>(out_fmt));
    }

    pass &= tt::test_utils::is_close_vectors<float>(
        dest_floats, golden_floats, [&](float a, float b) { return tt::test_utils::is_close(a, b, rtol, atol); });
    pass &= check_pcc(dest_floats, golden_floats, min_pcc);
    if (not pass) {
        dump_matmul_debug(stimulus.in0_floats, stimulus.in1_floats, golden_floats, dest_floats);
    }
    return pass;
}
// blocked matmul has blocking on output, spill/reloads or does l1 accumulation using intermediate
bool blocked_matmul(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const BlockedMatmulConfig& cfg) {
    const std::uint32_t M = cfg.M;
    const std::uint32_t K = cfg.K;
    const std::uint32_t N = cfg.N;
    const std::uint32_t num_blocks = cfg.num_blocks;
    const tt::DataFormat in0_fmt = cfg.in0_fmt;
    const tt::DataFormat in1_fmt = cfg.in1_fmt;
    const tt::DataFormat out_fmt = cfg.out_fmt;

    const bool is_quasar = MetalContext::instance().get_cluster().arch() == ARCH::QUASAR;

    bool pass = true;
    CoreCoord core(0, 0);
    const size_t in0_tile_size = tt::tile_size(in0_fmt);
    const size_t in1_tile_size = tt::tile_size(in1_fmt);
    const size_t out_tile_size = tt::tile_size(out_fmt);
    // Partials CB carries the in-flight DEST tiles. Sizing it as the output format is a
    // conservative bound — it matches existing BF16 behaviour and works for FP8 since the
    // packer gasket converts Float32 DEST → out_fmt at L1 write time.
    const size_t partials_tile_size = out_tile_size;
    const size_t in0_block_size_bytes = M * K * in0_tile_size;
    const size_t in1_block_size_bytes = K * N * in1_tile_size;
    const size_t in0_total_size_bytes = num_blocks * in0_block_size_bytes;
    const size_t in1_total_size_bytes = num_blocks * in1_block_size_bytes;
    const size_t out_byte_size = M * N * out_tile_size;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    auto input0_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = in0_total_size_bytes},
        {.page_size = in0_total_size_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device.get());
    const std::uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input1_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = in1_total_size_bytes},
        {.page_size = in1_total_size_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device.get());
    const std::uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto output_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = out_byte_size},
        {.page_size = out_byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device.get());
    const std::uint32_t out_dram_addr = output_dram_buffer->address();

    // Metal 2.0: one dataflow-buffer + kernel-spec path for every architecture. Only the hardware
    // config differs (Gen1 = Wormhole/Blackhole, Gen2 = Quasar)
    const experimental::DFBSpecName IN0_DFB{"in0_dfb"};
    const experimental::DFBSpecName IN1_DFB{"in1_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::DFBSpecName PARTIALS_DFB{"partials_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};
    const experimental::NodeCoord node{core.x, core.y};

    const experimental::DataflowBufferSpec in0_dfb_spec{
        .unique_id = IN0_DFB,
        .entry_size = static_cast<std::uint32_t>(in0_tile_size),
        .num_entries = M * K,
        .data_format_metadata = in0_fmt,
    };
    const experimental::DataflowBufferSpec in1_dfb_spec{
        .unique_id = IN1_DFB,
        .entry_size = static_cast<std::uint32_t>(in1_tile_size),
        .num_entries = K * N,
        .data_format_metadata = in1_fmt,
    };
    const experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = static_cast<std::uint32_t>(out_tile_size),
        .num_entries = M * N,
        .data_format_metadata = out_fmt,
    };
    const experimental::DataflowBufferSpec partials_dfb_spec{
        .unique_id = PARTIALS_DFB,
        .entry_size = static_cast<std::uint32_t>(partials_tile_size),
        .num_entries = M * N,
        .data_format_metadata = out_fmt,
    };

    experimental::KernelSpec::CompilerOptions::Defines compute_defines;
    if (cfg.packer_l1_acc) {
        compute_defines.emplace("PACKER_L1_ACC", "1");
    }

    experimental::DataMovementHardwareConfig reader_hw_config;
    experimental::DataMovementHardwareConfig writer_hw_config;
    experimental::ComputeHardwareConfig compute_hw_config;
    if (is_quasar) {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
        // Quasar's first fidelity phase covers the Float16_b and MxFp4_2x operands here in full
        compute_hw_config = experimental::ComputeHardwareConfig{
            .fpu_math_fidelity = MathFidelity::LoFi, .enable_32_bit_dest = cfg.fp32_dest_acc_en};
    } else {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                },
        };
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                },
        };
        compute_hw_config = experimental::ComputeHardwareConfig{.enable_32_bit_dest = cfg.fp32_dest_acc_en};
    }

    const experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(IN0_DFB, "in0"), experimental::ProducerOf(IN1_DFB, "in1")},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src0_addr",
                  "src0_dram_bank_id",
                  "src1_addr",
                  "src1_dram_bank_id",
                  "num_blocks",
                  "in0_block_tile_cnt",
                  "in1_block_tile_cnt",
                  "in0_block_size_bytes",
                  "in1_block_size_bytes"}},
        .hw_config = reader_hw_config,
    };

    const experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = writer_hw_config,
    };

    const experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_block_compute.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             // Partials is compute-only scratch (spill/reload, or the L1-acc accumulator), so
             // compute is both its producer and its consumer.
             {
                 .dfb_spec_name = PARTIALS_DFB,
                 .accessor_name = "partials",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = PARTIALS_DFB,
                 .accessor_name = "partials",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args =
            {{"in0_block_num_tiles", M * K},
             {"in1_block_num_tiles", K * N},
             {"out_block_num_tiles", M * N},
             {"out_r", M},
             {"out_c", N},
             {"in0_k", K},
             {"num_blocks", num_blocks}},
        .hw_config = compute_hw_config,
    };

    const experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    const experimental::ProgramSpec spec{
        .name = "blocked_matmul",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in0_dfb_spec, in1_dfb_spec, out_dfb_spec, partials_dfb_spec},
        .work_units = {wu},
    };

    workload.add_program(device_range, experimental::MakeProgramFromSpec(*mesh_device, spec));
    auto& program_ = workload.get_programs().at(device_range);

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    // Stimulus + reference floats are emitted in face-major-tile order across the full
    // num_blocks × (M×K) / (K×N) tile counts; we slice into per-block windows in the golden loop.
    const OperandStimulus in0_stim = make_operand_stimulus(in0_fmt, num_blocks * M * K, /*seed=*/0);
    const OperandStimulus in1_stim = make_operand_stimulus(in1_fmt, num_blocks * K * N, /*seed=*/1);

    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    // Output is M×N tiles in face-major-tile order (TILED_NFACES layout). For each block,
    // accumulate the per-block matmul into the shared golden, using byte_tile_face_major_index
    // for both inputs and output to match the layout the device produces.
    std::vector<float> golden_floats(M * N * tt::constants::TILE_HW, 0.0f);
    for (std::uint32_t b = 0; b < num_blocks; b++) {
        const size_t in0_block_off = b * M * K * tt::constants::TILE_HW;
        const size_t in1_block_off = b * K * N * tt::constants::TILE_HW;
        for (std::uint32_t mt = 0; mt < M; mt++) {
            for (std::uint32_t nt = 0; nt < N; nt++) {
                const size_t out_tile_off = (mt * N + nt) * tt::constants::TILE_HW;
                for (std::uint32_t y = 0; y < tt::constants::TILE_HEIGHT; y++) {
                    for (std::uint32_t x = 0; x < tt::constants::TILE_WIDTH; x++) {
                        float acc = 0.0f;
                        for (std::uint32_t kt = 0; kt < K; kt++) {
                            const size_t in0_tile_off = in0_block_off + (mt * K + kt) * tt::constants::TILE_HW;
                            const size_t in1_tile_off = in1_block_off + (kt * N + nt) * tt::constants::TILE_HW;
                            for (std::uint32_t z = 0; z < tt::constants::TILE_WIDTH; z++) {
                                acc += in0_stim.floats[in0_tile_off + byte_tile_face_major_index(z, y)] *
                                       in1_stim.floats[in1_tile_off + byte_tile_face_major_index(x, z)];
                            }
                        }
                        golden_floats[out_tile_off + byte_tile_face_major_index(x, y)] += acc;
                    }
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    distributed::EnqueueWriteMeshBuffer(cq, input0_dram_buffer, in0_stim.packed, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, input1_dram_buffer, in1_stim.packed, /*blocking=*/true);

    experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", in0_dram_addr},
                 {"src0_dram_bank_id", 0u},
                 {"src1_addr", in1_dram_addr},
                 {"src1_dram_bank_id", 0u},
                 {"num_blocks", num_blocks},
                 {"in0_block_tile_cnt", M * K},
                 {"in1_block_tile_cnt", K * N},
                 {"in0_block_size_bytes", static_cast<std::uint32_t>(in0_block_size_bytes)},
                 {"in1_block_size_bytes", static_cast<std::uint32_t>(in1_block_size_bytes)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"dst_addr", out_dram_addr}, {"bank_id", 0u}, {"num_tiles", M * N}}),
        },
    };
    experimental::SetProgramRunArgs(program_, run_args);

    auto blocking = is_quasar;
    distributed::EnqueueMeshWorkload(cq, workload, blocking);
    if (not blocking) {
        distributed::Finish(cq);
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::uint32_t> dest_buffer_data;
    distributed::EnqueueReadMeshBuffer(cq, dest_buffer_data, output_dram_buffer, /*blocking=*/true);
    std::vector<float> dest_floats;
    if (out_fmt == tt::DataFormat::Fp8_e4m3) {
        dest_floats = fp8_to_floats(dest_buffer_data);
    } else if (out_fmt == tt::DataFormat::Float16_b) {
        dest_floats = bf16_to_floats(dest_buffer_data);
    } else {
        TT_FATAL(false, "blocked_matmul: unsupported out_fmt {}", static_cast<int>(out_fmt));
    }

    // For BF16 output: per-element close (tight tolerances are realistic).
    // For FP8 output: per-element checks are not meaningful — FP8 quantization compounds
    // across blocks, and PACKER_L1_ACC re-quantizes every block-output through Fp8 storage.
    // PCC is the structural-correctness backstop; thresholds reflect realistic FP8 fidelity loss
    // (lower for L1Acc, which round-trips through Fp8 L1 every block instead of through Float32 DEST).
    if (out_fmt == tt::DataFormat::Fp8_e4m3) {
        const double min_pcc = cfg.packer_l1_acc ? 0.85 : 0.95;
        pass &= check_pcc(dest_floats, golden_floats, min_pcc);
    } else {
        const float rtol = 0.05f;
        const float atol = 0.05f + 0.05f * static_cast<float>(K * num_blocks);
        pass &= tt::test_utils::is_close_vectors<float>(
            dest_floats, golden_floats, [&](float a, float b) { return tt::test_utils::is_close(a, b, rtol, atol); });
        pass &= check_pcc(dest_floats, golden_floats, /*min_pcc=*/0.99);
    }
    return pass;
}

// MOP-less matmul (issue #52329): out[rt x ct] = in0[rt x kt] * in1[kt x ct] in one dest acquire.
struct MatmulConfig {
    std::uint32_t rt_dim = 1;
    std::uint32_t ct_dim = 1;
    std::uint32_t kt_dim = 1;
    MathFidelity math_fidelity = MathFidelity::LoFi;
};

void run_matmul_no_mop(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const MatmulConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    const experimental::NodeCoord node{0, 0};
    const bool is_quasar = mesh_device->arch() == tt::ARCH::QUASAR;

    const std::uint32_t rt_dim = test_config.rt_dim;
    const std::uint32_t ct_dim = test_config.ct_dim;
    const std::uint32_t kt_dim = test_config.kt_dim;
    const std::uint32_t in0_num_tiles = rt_dim * kt_dim;
    const std::uint32_t in1_num_tiles = kt_dim * ct_dim;
    const std::uint32_t out_num_tiles = rt_dim * ct_dim;
    const std::uint32_t single_tile_size = tt::constants::TILE_HW * sizeof(bfloat16);

    log_info(
        tt::LogTest,
        "Testing matmul_block_no_mop rt_dim={} ct_dim={} kt_dim={} fidelity={}",
        rt_dim,
        ct_dim,
        kt_dim,
        static_cast<int>(test_config.math_fidelity));

    // One page per buffer, not one per tile: the reader and writer kernels walk a single DRAM bank
    // linearly (one bank_id, the address advancing by a tile per transfer), and a page-per-tile
    // buffer is interleaved across the channels, so tile k would be fetched from page
    // k * num_banks and the walk would leave that bank's share of the buffer part way through.
    auto make_dram = [&](std::uint32_t num_tiles) {
        const std::uint32_t buffer_size = single_tile_size * num_tiles;
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = buffer_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
        distributed::ReplicatedBufferConfig cfg{.size = buffer_size};
        return distributed::MeshBuffer::create(cfg, dram_config, mesh_device.get());
    };
    auto in0_dram = make_dram(in0_num_tiles);
    auto in1_dram = make_dram(in1_num_tiles);
    auto out_dram = make_dram(out_num_tiles);

    experimental::KernelSpec::CompilerOptions::Defines defines_vec;
    defines_vec.emplace("RT_DIM", std::to_string(rt_dim));
    defines_vec.emplace("CT_DIM", std::to_string(ct_dim));
    defines_vec.emplace("KT_DIM", std::to_string(kt_dim));

    const experimental::DFBSpecName INP0_DFB{"inp0_dfb"};
    const experimental::DFBSpecName INP1_DFB{"inp1_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_dfb = [&](const experimental::DFBSpecName& name, std::uint32_t num_entries) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = single_tile_size,
            .num_entries = num_entries,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
    };
    // One matmul_block_no_mop call indexes a rt_dim x 1 slice of in0 and a 1 x ct_dim slice of in1 off
    // the read pointer within a single dest acquire, so both operands must be fully resident.
    experimental::DataflowBufferSpec inp0_dfb_spec = make_dfb(INP0_DFB, in0_num_tiles);
    experimental::DataflowBufferSpec inp1_dfb_spec = make_dfb(INP1_DFB, in1_num_tiles);
    experimental::DataflowBufferSpec out_dfb_spec = make_dfb(OUT_DFB, out_num_tiles);

    // Quasar drives dataflow through the Gen2 config and takes its DFB sync explicitly
    experimental::DataMovementHardwareConfig reader_hw_config;
    experimental::DataMovementHardwareConfig writer_hw_config;
    experimental::ComputeHardwareConfig compute_hw_config;
    if (is_quasar) {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
        compute_hw_config = experimental::ComputeHardwareConfig{.fpu_math_fidelity = test_config.math_fidelity};
    } else {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                },
        };
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                },
        };
        compute_hw_config = experimental::ComputeHardwareConfig{.fpu_math_fidelity = test_config.math_fidelity};
    }

    // Reads num_tiles into in0 and num_bcast_tiles into in1 from two DRAM buffers, which is exactly
    // the two matmul operands; nothing about it is broadcast-specific.
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_bcast_col_reuse.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles", "num_bcast_tiles"}},
        .hw_config = reader_hw_config,
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = writer_hw_config,
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_no_mop.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = defines_vec},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };
    experimental::ProgramSpec spec{
        .name = "single_core_matmul_no_mop",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {inp0_dfb_spec, inp1_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program built_program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    workload.add_program(device_range, std::move(built_program));
    auto& program_run = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", static_cast<std::uint32_t>(in0_dram->address())},
                 {"src0_bank_id", 0u},
                 {"src1_addr", static_cast<std::uint32_t>(in1_dram->address())},
                 {"src1_bank_id", 0u},
                 {"num_tiles", in0_num_tiles},
                 {"num_bcast_tiles", in1_num_tiles}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<std::uint32_t>(out_dram->address())},
                 {"bank_id", 0u},
                 {"num_tiles", out_num_tiles}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    // make_operand_stimulus emits face-major tile order, which is the DRAM page order the reader
    // streams and the order make_matmul_golden indexes, so no tilize/untilize round trip is needed.
    MatmulStimulus stimulus = make_matmul_stimulus(tt::DataFormat::Float16_b, rt_dim, kt_dim, ct_dim);
    std::vector<float> golden_floats =
        make_matmul_golden(stimulus.in0_floats, stimulus.in1_floats, rt_dim, kt_dim, ct_dim);

    distributed::WriteShard(cq, in0_dram, stimulus.packed_input0, zero_coord);
    distributed::WriteShard(cq, in1_dram, stimulus.packed_input1, zero_coord);

    distributed::EnqueueMeshWorkload(cq, workload, is_quasar);
    distributed::Finish(cq);

    std::vector<std::uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, out_dram, zero_coord);
    std::vector<float> dest_floats = bf16_to_floats(dest_buffer_data);

    // BF16 dest accumulation, so per-element error grows with the kt_dim * TILE_WIDTH inner length;
    // the PCC backstop catches structural mis-permutations pointwise tolerances would let slip.
    const float rtol = 0.05f;
    const float atol = (kt_dim > 1) ? 0.4f : 0.2f;
    bool pass = tt::test_utils::is_close_vectors<float>(
        dest_floats, golden_floats, [&](float a, float b) { return tt::test_utils::is_close(a, b, rtol, atol); });
    pass &= check_pcc(dest_floats, golden_floats, /*min_pcc=*/0.99);
    if (not pass) {
        dump_matmul_debug(stimulus.in0_floats, stimulus.in1_floats, golden_floats, dest_floats);
    }
    ASSERT_TRUE(pass);
}

}  // namespace unit_tests::compute::matmul

// Compute-layer coverage: experimental MOP-less matmul (issue #52329). Ordered so the first
// failure localises: 1x1x1 is a single tile through init + execute, kt_dim>1 adds dest
// accumulation across inner steps, rt/ct > 1 adds the block reuse walk.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixTestSingleCoreComputeMatmulNoMop) {
    using unit_tests::compute::matmul::MatmulConfig;
    const std::vector<MatmulConfig> cases = {
        {.rt_dim = 1, .ct_dim = 1, .kt_dim = 1},
        {.rt_dim = 1, .ct_dim = 1, .kt_dim = 2},
        {.rt_dim = 2, .ct_dim = 2, .kt_dim = 1},
        {.rt_dim = 2, .ct_dim = 2, .kt_dim = 2},
    };
    for (const auto& cfg : cases) {
        unit_tests::compute::matmul::run_matmul_no_mop(this->devices_.at(0), cfg);
    }
}

TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleTileComputeMatmul) {
    if (arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "single_tile_matmul test is not supported on Quasar";
    }
    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul(device));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileComputeMatmul) {
    if (arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "single_block_matmul test is not supported on Quasar";
    }
    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(device, 1, 1, 1));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileAccumulationComputeMatmul) {
    if (arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "single_block_matmul test is not supported on Quasar";
    }
    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(device, 1, 2, 1));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileNoAccumulationComputeMatmul) {
    if (arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "single_block_matmul test is not supported on Quasar";
    }
    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(device, 2, 1, 2));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreMultiBlockSpillReloadComputeMatmul) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2, .K = 2, .N = 2, .num_blocks = 4, .packer_l1_acc = false};
    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(device, config));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreMultiBlockL1AccComputeMatmul) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2, .K = 2, .N = 2, .num_blocks = 4, .packer_l1_acc = true};
    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(device, config));
    }
}

// Quasar-only test for matmul variant that enables the 2x source format optimization for MxFp4_2x format,
// This optimization allows src register datums to store two elements in one.
// Since MxFp4_2x only supports GAPOOL and MVMUL/MVMULDI instructions, we cannot test it with multiple blocks
// since multi-block matmul kernel uses datacopy from SRC to DST which is not supported by MxFp4_2x format.
// We can still verify the correctness of the optimization with single block matmul.
// L1 acc doesn't work in this case since it also relies on datacopy from SRC to DST for the accumulation.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixTestSingleCoreSingleBlockComputeMatmulMxFp4X2) {
    if (MetalContext::instance().get_cluster().get_quasar_ip_variant() == tt::IpVariant::TRINITY) {
        GTEST_SKIP() << "MxFp4_2x register formats do not exist on Trinity";
    }
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2,
        .K = 2,
        .N = 2,
        .num_blocks = 1,
        .packer_l1_acc = false,
        // MxFp4 in0/in1 auto-selects the 2x-packed src-register format for matmul on Quasar.
        .in0_fmt = tt::DataFormat::MxFp4,
        .in1_fmt = tt::DataFormat::MxFp4,
        .out_fmt = tt::DataFormat::Float16_b};
    ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(this->devices_.at(0), config));
}

// FP8 variants of the multi-block matmul. Blackhole-gated because Fp8_e4m3 only exists on BH
// and the JIT-time assert in ComputeKernel::set_build_options requires fp32_dest_acc_en=true
// for any FP8 path. Mirrors the BF16 SpillReload / L1Acc tests at M=K=N=2, num_blocks=4.
TEST_F(LLKBlackholeSingleCardFixture, TensixTestSingleCoreMultiBlockSpillReloadComputeMatmulFp8e4m3) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2,
        .K = 2,
        .N = 2,
        .num_blocks = 4,
        .packer_l1_acc = false,
        .in0_fmt = tt::DataFormat::Fp8_e4m3,
        .in1_fmt = tt::DataFormat::Fp8_e4m3,
        .out_fmt = tt::DataFormat::Fp8_e4m3,
        .fp32_dest_acc_en = true};
    ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(this->devices_.at(0), config));
}
TEST_F(LLKBlackholeSingleCardFixture, TensixTestSingleCoreMultiBlockL1AccComputeMatmulFp8e4m3) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2,
        .K = 2,
        .N = 2,
        .num_blocks = 4,
        .packer_l1_acc = true,
        .in0_fmt = tt::DataFormat::Fp8_e4m3,
        .in1_fmt = tt::DataFormat::Fp8_e4m3,
        .out_fmt = tt::DataFormat::Fp8_e4m3,
        .fp32_dest_acc_en = true};
    ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(this->devices_.at(0), config));
}

// Sweeps in × out data format (4 cells). fp32_dest_acc_en is fixed to true: BH requires it
// whenever any CB is Fp8 (see ComputeKernel::set_build_options assert), and for the non-Fp8
// cells the fp32-dest=false variant is already covered by other tests.
TEST_F(LLKBlackholeSingleCardFixture, TensixTestSingleCoreSingleTileComputeMatmulFormatSweep) {
    static constexpr std::array<tt::DataFormat, 2> kInFormats = {
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Float16_b,
    };
    static constexpr std::array<tt::DataFormat, 2> kOutFormats = {
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Float16_b,
    };
    static constexpr bool kFp32DestAccEn = true;

    for (tt::DataFormat in_fmt : kInFormats) {
        for (tt::DataFormat out_fmt : kOutFormats) {
            log_info(
                tt::LogTest,
                "TensixTestSingleCoreSingleTileComputeMatmulFormatSweep: in_fmt={} out_fmt={} "
                "fp32_dest_acc_en={}",
                in_fmt,
                out_fmt,
                kFp32DestAccEn);
            ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul(
                this->devices_.at(0), in_fmt, in_fmt, out_fmt, kFp32DestAccEn));
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixTestSingleCoreSingleBlockSingleTileComputeMatmulFp8e4m3) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(
        this->devices_.at(0),
        1,
        1,
        1,
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/true));
}

TEST_F(LLKBlackholeSingleCardFixture, TensixTestSingleCoreSingleBlockSingleTileAccumulationComputeMatmulFp8e4m3) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(
        this->devices_.at(0),
        1,
        2,
        1,
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/true));
}

// ============================================================================
// Id-free (2.0) matmul, validated against the make_matmul_golden host golden (float reference), not a legacy
// kernel. Covers single-tile (C = A*B, 32x32x32) and a 2x2x2 block matmul. All Float16_b. Runs on Blackhole
// (BH-only API). in0 (A) -> c_0 -> SrcB, in1 (B) -> c_1 -> SrcA (the run_matmul_* convention).
// ============================================================================
TEST_F(LLKBlackholeSingleCardFixture, TensixMatmulSingleTileIdFreeGolden) {
    auto src0 = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b), /*rand_max_float=*/2, /*seed=*/42, /*offset=*/-1.0f);
    auto src1 = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b), /*rand_max_float=*/2, /*seed=*/7, /*offset=*/-1.0f);

    auto golden = unit_tests::compute::matmul::make_matmul_golden(
        bf16_to_floats(src0), bf16_to_floats(src1), /*M=*/1, /*K=*/1, /*N=*/1);
    auto result = unit_tests::llk::single_core::run_matmul_single(
        *this->devices_.at(0), src0, src1, "tests/tt_metal/tt_metal/test_kernels/compute/matmul_idfree.cpp");
    auto result_floats = bf16_to_floats(result);

    EXPECT_TRUE(is_close_vectors<float>(
        result_floats, golden, [](float a, float b) { return is_close(a, b, /*rtol=*/0.05f, /*atol=*/0.2f); }));
    EXPECT_TRUE(check_pcc(result_floats, golden, /*min_pcc=*/0.99));
}

// Block matmul: one matmul_block call produces the whole rt_dim x ct_dim output block (A[rt x kt] * B[kt x ct]).
// kt_dim == 1 here so the block is exercised on the output-grid axes (rt x ct) against a real host golden;
// multi-tile K accumulation is driven by the compute kernel's own K loop (out of this op-level golden's scope).
TEST_F(LLKBlackholeSingleCardFixture, TensixMatmulBlockIdFreeGolden) {
    // rt == ct required by run_matmul_block (reader_binary feeds c_0/c_1 the same count). 2x2 block, kt = 1.
    constexpr std::uint32_t ct_dim = 2, rt_dim = 2, kt_dim = 1;
    auto src0 = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * (rt_dim * kt_dim),
        /*rand_max_float=*/2,
        /*seed=*/42,
        /*offset=*/-1.0f);
    auto src1 = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * (kt_dim * ct_dim),
        /*rand_max_float=*/2,
        /*seed=*/7,
        /*offset=*/-1.0f);

    auto golden = unit_tests::compute::matmul::make_matmul_golden(
        bf16_to_floats(src0), bf16_to_floats(src1), /*M=*/rt_dim, /*K=*/kt_dim, /*N=*/ct_dim);
    auto result = unit_tests::llk::single_core::run_matmul_block(
        *this->devices_.at(0),
        src0,
        src1,
        ct_dim,
        rt_dim,
        kt_dim,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul_block_2_0.cpp");
    auto result_floats = bf16_to_floats(result);

    EXPECT_TRUE(is_close_vectors<float>(
        result_floats, golden, [](float a, float b) { return is_close(a, b, /*rtol=*/0.05f, /*atol=*/0.2f); }));
    EXPECT_TRUE(check_pcc(result_floats, golden, /*min_pcc=*/0.99));
}

// Compile-time contract of the id-free matmul operand geometry; these programs are never launched. The LLK has
// unpack + math programs only for the pairs of the Blackhole matmul support matrix: in0 columns == in1 rows,
// in0 not 16x16 (any 1..32-row height with 32 columns, or 32x16), in1 one of 32x32 / 32x16 / 16x32. The
// programs are selected from the operands' total dimensions and assume the face grid behind them: an operand
// shorter than 32 rows is unpacked two faces per tile and the math takes a 16-row operand as two 16-row faces, so
// an in0 with short faces must be a single face row and in1 faces must be 16 rows high. On the unthrottled pairs
// the LLK's only guard is a runtime LLK_ASSERT on the 16x16-by-16x16 pair that production builds compile out
// (its other two runtime asserts are transpose with a 32x16 in1, a runtime argument, and the throttled program's
// 32x32-only rule, which the throttle rows below exercise); every other unsupported pair (a single-face operand
// next to a multi-face one, a mismatched inner dimension, four short faces whose total matches a two-face tile)
// silently runs a program over faces the operand does not have, or leaves faces unread. The API must reject all
// of them at compile time, in both the single-tile and the block init and again in the execute (matmul_tiles /
// matmul_block deduce their own operands and configure the unpacker from them, so an execute with a pair the
// init was not given must not compile either), and keep accepting every supported pair.
TEST_F(LLKBlackholeSingleCardFixture, TensixMatmulIdFreeCompileLimits) {
    struct Case {
        std::uint32_t in0_faces_r, in0_faces_c, in1_faces_r, in1_faces_c;
        std::uint32_t in0_face_r_dim, in1_face_r_dim;
        std::uint32_t mm_throttle;  // MM_THROTTLE compile define; 0 = unthrottled
        MathFidelity fidelity;
        bool accepted;
    };
    // Face dims are 16x16 unless in0_face_r_dim / in1_face_r_dim say otherwise; tile dims = faces x face dims.
    const std::array<Case, 24> cases{{
        // Supported pairs (in0 x in1).
        {2, 2, 2, 2, 16, 16, 0, MathFidelity::LoFi, true},  // 32x32 x 32x32
        {2, 2, 2, 1, 16, 16, 0, MathFidelity::LoFi, true},  // 32x32 x 32x16
        {2, 1, 1, 2, 16, 16, 0, MathFidelity::LoFi, true},  // 32x16 x 16x32
        {1, 2, 2, 2, 16, 16, 0, MathFidelity::LoFi, true},  // 16x32 x 32x32
        {1, 2, 2, 1, 16, 16, 0, MathFidelity::LoFi, true},  // 16x32 x 32x16
        {1, 2, 2, 2, 8, 16, 0, MathFidelity::LoFi, true},   // 8x32 x 32x32 (tiny-tile in0: one row of two 8-row faces)
        // 16x16 in0 (no math program), inner dimension matched or not.
        {1, 1, 1, 1, 16, 16, 0, MathFidelity::LoFi, false},  // 16x16 x 16x16
        {1, 1, 1, 2, 16, 16, 0, MathFidelity::LoFi, false},  // 16x16 x 16x32
        {1, 1, 2, 2, 16, 16, 0, MathFidelity::LoFi, false},  // 16x16 x 32x32
        {1, 1, 2, 1, 16, 16, 0, MathFidelity::LoFi, false},  // 16x16 x 32x16
        {1, 1, 1, 2, 8, 16, 0, MathFidelity::LoFi, false},   // 8x16 x 16x32 (sub-16-row single face in0)
        // Inner dimension mismatch with a multi-face in0.
        {2, 2, 1, 1, 16, 16, 0, MathFidelity::LoFi, false},  // 32x32 x 16x16
        {1, 2, 1, 1, 16, 16, 0, MathFidelity::LoFi, false},  // 16x32 x 16x16
        {2, 1, 2, 2, 16, 16, 0, MathFidelity::LoFi, false},  // 32x16 x 32x32
        // Inner dimension matched but in1 is 16x16 (not a supported in1 tile).
        {2, 1, 1, 1, 16, 16, 0, MathFidelity::LoFi, false},  // 32x16 x 16x16
        // Totals of a supported pair, face grid of an unsupported one: four short faces where the programs expect
        // two 16-row faces (in0 16x32 / 8x32 as 2x2 faces of 8 / 4 rows; in1 16x32 as 2x2 faces of 8 rows).
        {2, 2, 2, 2, 8, 16, 0, MathFidelity::LoFi, false},  // {8,16,2,2} x 32x32
        {2, 2, 2, 2, 4, 16, 0, MathFidelity::LoFi, false},  // {4,16,2,2} x 32x32
        {2, 1, 2, 2, 16, 8, 0, MathFidelity::LoFi, false},  // 32x16 x {8,16,2,2}
        // Throttled matmul. Levels 4 / 5 at HiFi2 or higher keep the throttled program, which exists for the
        // 32x32 by 32x32 pair only; levels 1-3, and any level at LoFi, fall back to the unthrottled program.
        {2, 2, 2, 2, 16, 16, 4, MathFidelity::HiFi2, true},   // 32x32 x 32x32, throttle 4 HiFi2
        {2, 2, 2, 1, 16, 16, 4, MathFidelity::HiFi2, false},  // 32x32 x 32x16, throttle 4 HiFi2
        {1, 2, 2, 2, 16, 16, 5, MathFidelity::HiFi4, false},  // 16x32 x 32x32, throttle 5 HiFi4
        {1, 2, 2, 2, 8, 16, 4, MathFidelity::HiFi2, false},   // 8x32 x 32x32 (tiny in0), throttle 4 HiFi2
        {2, 2, 2, 1, 16, 16, 3, MathFidelity::HiFi2, true},   // 32x32 x 32x16, throttle 3 (unthrottled fallback)
        {2, 2, 2, 1, 16, 16, 4, MathFidelity::LoFi, true},    // 32x32 x 32x16, throttle 4 LoFi (fallback)
    }};
    const CoreCoord core{0, 0};
    for (const auto& test : cases) {
        for (const bool block : {false, true}) {
            // execute_only: the kernel inits with the 32x32 by 32x32 pair and hands the pair under test to the
            // execute alone, so the execute's own assert is what rejects it. The throttle rule lives in the math
            // init, and the 32x32 pair satisfies it at every level, so the throttled rows probe the init only.
            for (const bool execute_only : {false, true}) {
                if (execute_only && test.mm_throttle != 0) {
                    continue;
                }
                SCOPED_TRACE(
                    ::testing::Message() << "in0_faces=" << test.in0_faces_r << "x" << test.in0_faces_c
                                         << ", in0_face_r_dim=" << test.in0_face_r_dim
                                         << ", in1_faces=" << test.in1_faces_r << "x" << test.in1_faces_c
                                         << ", in1_face_r_dim=" << test.in1_face_r_dim << ", mm_throttle="
                                         << test.mm_throttle << ", fidelity=" << static_cast<int>(test.fidelity)
                                         << ", block=" << block << ", execute_only=" << execute_only);
                std::map<std::string, std::string> defines;
                if (test.mm_throttle != 0) {
                    defines.emplace("MM_THROTTLE", std::to_string(test.mm_throttle));
                }
                Program program = CreateProgram();
                CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/compute/matmul_idfree_compile_limits.cpp",
                    core,
                    ComputeConfig{
                        .math_fidelity = test.fidelity,
                        .fp32_dest_acc_en = false,
                        .compile_args =
                            {test.in0_faces_r,
                             test.in0_faces_c,
                             test.in1_faces_r,
                             test.in1_faces_c,
                             static_cast<std::uint32_t>(block),
                             test.in0_face_r_dim,
                             test.in1_face_r_dim,
                             static_cast<std::uint32_t>(execute_only)},
                        .defines = defines});
                if (test.accepted) {
                    EXPECT_NO_THROW(program.impl().compile(&this->device()));
                    continue;
                }
                // The build failure carries the compiler log; the rejection must come from the API's static_assert,
                // not from some unrelated build error. The throttled rows use supported pairs, so their rejection is
                // the math init's throttle assert; the execute-only rows must be rejected by the execute itself.
                const std::string entry_point = block ? (execute_only ? "matmul_block" : "matmul_block_init")
                                                      : (execute_only ? "matmul_tiles" : "matmul_init");
                const std::string expected_assert = test.mm_throttle != 0 ? "llk_math_matmul_init: MM_THROTTLE"
                                                                          : entry_point + ": unsupported operand pair";
                try {
                    program.impl().compile(&this->device());
                    ADD_FAILURE() << "unsupported operand pair compiled";
                } catch (const std::runtime_error& e) {
                    EXPECT_NE(std::string(e.what()).find(expected_assert), std::string::npos) << e.what();
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
