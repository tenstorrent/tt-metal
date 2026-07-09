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
#include "impl/dataflow_buffer/dataflow_buffer.hpp"
#include "impl/host_api/temp_quasar_api.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
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
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::matmul {

// Per-block matmul: out[M x N] = in0[M x K] * in1[K x N]
// Repeated num_blocks times
// If K > 1 -> dest accumulation within each block
// If num_blocks > 1 -> partials accumulation (either l1 accumulation or spill and reload)
struct BlockedMatmulConfig {
    uint32_t M = 1;           // per-block rows (tiles)
    uint32_t K = 1;           // per-block inner dim (tiles)
    uint32_t N = 1;           // per-block cols (tiles)
    uint32_t num_blocks = 1;  // number of K-blocks
    bool packer_l1_acc = false;
    // Format / DEST-mode parameters. Defaults preserve the original BF16 + fp32_dest=false
    // behaviour exercised by `TensixTestSingleCoreMultiBlock*ComputeMatmul` (and the Quasar
    // multi-block matmul tests). On Blackhole, any in/out FP8 path requires
    // fp32_dest_acc_en=true (asserted at JIT time in ComputeKernel::set_build_options).
    tt::DataFormat in0_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat in1_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat out_fmt = tt::DataFormat::Float16_b;
    bool fp32_dest_acc_en = false;
    bool enable_2x_src_format = false;
};

void create_CBs_for_fused_matmul(
    distributed::MeshWorkload& workload,
    const std::shared_ptr<distributed::MeshDevice>& /*mesh_device*/,
    CoreCoord core,
    bool activations_rm,
    bool output_rm,
    uint32_t M,
    uint32_t N,
    uint32_t in0_block_w,
    uint32_t /*out_subblock_h*/) {
    uint32_t num_bytes_for_df = 2;
    uint32_t in0_cb = 0;
    uint32_t in1_cb = 1;
    uint32_t tilize_mode_tilized_in0_cb = 24;
    uint32_t matmul_partials_cb = 25;
    uint32_t untilize_mode_final_matmul_partials_cb = 26;
    uint32_t untilize_mode_reblock_cb = 27;
    uint32_t out0_cb = 16;

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program = workload.get_programs().at(device_range);

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;

    // Invariants
    uint32_t cb0_tiles = M * in0_block_w * 2;
    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{in0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(in0_cb, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    uint32_t cb1_tiles = N * in0_block_w * 2;
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
        uint32_t reblock_cb_tiles = N;  // Only space for one row
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
        uint32_t reblock_cb_tiles = N;  // Only space for one row
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
    std::vector<uint32_t> packed_input0;
    std::vector<uint32_t> packed_input1;
    std::vector<float> in0_floats;
    std::vector<float> in1_floats;
};

// Per-operand stimulus generation. Returns the packed L1 representation and a
// float reference vector in face-major-within-tile order (matching the matmul
// golden's byte_tile_face_major_index addressing).  For Bfp8_b, the float
// reference is the unpack-after-pack roundtrip so the golden reflects the
// values the hardware actually sees, not the raw RNG samples.
struct OperandStimulus {
    std::vector<uint32_t> packed;
    std::vector<float> floats;
};

OperandStimulus make_operand_stimulus(tt::DataFormat fmt, uint32_t tile_count, uint32_t seed) {
    constexpr float rng = 1.0f;
    const size_t num_elements = tt::constants::TILE_HW * tile_count;
    OperandStimulus out;
    if (fmt == tt::DataFormat::Fp8_e4m3) {
        out.packed = generate_packed_uniform_random_vector<uint32_t, float8_e4m3>(
            float8_e4m3(-rng), float8_e4m3(+rng), num_elements, seed);
        out.floats = fp8_to_floats(out.packed);
    } else if (fmt == tt::DataFormat::Float16_b) {
        out.packed = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
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
    tt::DataFormat in0_fmt, tt::DataFormat in1_fmt, uint32_t M, uint32_t K, uint32_t N) {
    OperandStimulus a = make_operand_stimulus(in0_fmt, M * K, /*seed=*/0);
    OperandStimulus b = make_operand_stimulus(in1_fmt, K * N, /*seed=*/1);
    MatmulStimulus out;
    out.packed_input0 = std::move(a.packed);
    out.packed_input1 = std::move(b.packed);
    out.in0_floats = std::move(a.floats);
    out.in1_floats = std::move(b.floats);
    return out;
}

MatmulStimulus make_matmul_stimulus(tt::DataFormat in_fmt, uint32_t M, uint32_t K, uint32_t N) {
    return make_matmul_stimulus(in_fmt, in_fmt, M, K, N);
}

// Host reference matmul over face-major tiles: output layout is M×N tiles × TILE_HW
// elements per tile.
std::vector<float> make_matmul_golden(
    const std::vector<float>& in0_floats, const std::vector<float>& in1_floats, uint32_t M, uint32_t K, uint32_t N) {
    std::vector<float> golden_floats(M * N * tt::constants::TILE_HW, 0.0f);
    for (uint32_t mt = 0; mt < M; mt++) {
        for (uint32_t nt = 0; nt < N; nt++) {
            const size_t out_tile_off = (mt * N + nt) * tt::constants::TILE_HW;
            for (uint32_t y = 0; y < tt::constants::TILE_HEIGHT; y++) {
                for (uint32_t x = 0; x < tt::constants::TILE_WIDTH; x++) {
                    float acc = 0.0f;
                    for (uint32_t kt = 0; kt < K; kt++) {
                        const size_t in0_tile_off = (mt * K + kt) * tt::constants::TILE_HW;
                        const size_t in1_tile_off = (kt * N + nt) * tt::constants::TILE_HW;
                        for (uint32_t z = 0; z < tt::constants::TILE_WIDTH; z++) {
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
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const uint32_t in0_tile_size = tt::tile_size(in0_fmt);
    const uint32_t in1_tile_size = tt::tile_size(in1_fmt);
    const uint32_t out_tile_size = tt::tile_size(out_fmt);

    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::InterleavedBufferConfig dram_in0_config{
        .device = device,
        .size = in0_tile_size,
        .page_size = in0_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig dram_in1_config{
        .device = device,
        .size = in1_tile_size,
        .page_size = in1_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig dram_out_config{
        .device = device,
        .size = out_tile_size,
        .page_size = out_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = CreateBuffer(dram_in0_config);
    auto input1_dram_buffer = CreateBuffer(dram_in1_config);
    auto output_dram_buffer = CreateBuffer(dram_out_config);

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
    tt_metal::detail::WriteToBuffer(input0_dram_buffer, stimulus.packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, stimulus.packed_input1);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {(uint32_t)input0_dram_buffer->address(),
         (uint32_t)0,
         (uint32_t)input1_dram_buffer->address(),
         (uint32_t)0,
         (uint32_t)1});
    tt_metal::SetRuntimeArgs(
        program_, writer_kernel, core, {(uint32_t)output_dram_buffer->address(), (uint32_t)0, (uint32_t)1});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
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
    uint32_t M,
    uint32_t K,
    uint32_t N,
    tt::DataFormat in_fmt = tt::DataFormat::Float16_b,
    tt::DataFormat out_fmt = tt::DataFormat::Float16_b,
    bool fp32_dest_acc_en = false) {
    bool pass = true;
    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
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
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config_0{
        .device = device,
        .size = in0_byte_size,
        .page_size = in0_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig dram_config_1{
        .device = device,
        .size = in1_byte_size,
        .page_size = in1_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig dram_config_out{
        .device = device,
        .size = out_byte_size,
        .page_size = out_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = CreateBuffer(dram_config_0);
    auto input1_dram_buffer = CreateBuffer(dram_config_1);
    auto output_dram_buffer = CreateBuffer(dram_config_out);

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
    tt_metal::detail::WriteToBuffer(input0_dram_buffer, stimulus.packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, stimulus.packed_input1);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {(uint32_t)input0_dram_buffer->address(),
         (uint32_t)0,
         (uint32_t)input1_dram_buffer->address(),
         (uint32_t)0,
         (uint32_t)1,              // num_blocks
         (uint32_t)M * K,          // in0_block_tile_cnt
         (uint32_t)K * N,          // in1_block_tile_cnt
         (uint32_t)in0_byte_size,  // in0_block_size_bytes
         (uint32_t)in1_byte_size});
    tt_metal::SetRuntimeArgs(
        program_, writer_kernel, core, {(uint32_t)output_dram_buffer->address(), (uint32_t)0, (uint32_t)M * N});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
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
    const uint32_t M = cfg.M;
    const uint32_t K = cfg.K;
    const uint32_t N = cfg.N;
    const uint32_t num_blocks = cfg.num_blocks;
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
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config_0{
        .device = device,
        .size = in0_total_size_bytes,
        .page_size = in0_total_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_1{
        .device = device,
        .size = in1_total_size_bytes,
        .page_size = in1_total_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_out{
        .device = device,
        .size = out_byte_size,
        .page_size = out_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = CreateBuffer(dram_config_0);
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input1_dram_buffer = CreateBuffer(dram_config_1);
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto output_dram_buffer = CreateBuffer(dram_config_out);
    const uint32_t out_dram_addr = output_dram_buffer->address();

    uint32_t in0_id = 0;
    uint32_t in1_id = 0;
    uint32_t out_id = 0;
    uint32_t partials_id = 0;

    if (is_quasar) {
        auto make_dfb = [&](uint32_t entry_size,
                            uint32_t num_entries,
                            uint16_t producer_mask,
                            uint16_t consumer_mask,
                            tt::DataFormat fmt) {
            return tt_metal::experimental::dfb::CreateDataflowBuffer(
                program_,
                core,
                tt_metal::experimental::dfb::DataflowBufferConfig{
                    .entry_size = entry_size,
                    .num_entries = num_entries,
                    .producer_risc_mask = producer_mask,
                    .num_producers = 1,
                    .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                    .consumer_risc_mask = consumer_mask,
                    .num_consumers = 1,
                    .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                    .enable_producer_implicit_sync = false,
                    .enable_consumer_implicit_sync = false,
                    .data_format = fmt});
        };
        in0_id = make_dfb(in0_tile_size, M * K, 0x1, 0x100, in0_fmt);
        in1_id = make_dfb(in1_tile_size, K * N, 0x1, 0x100, in1_fmt);
        out_id = make_dfb(out_tile_size, M * N, 0x100, 0x2, out_fmt);
        partials_id = tt_metal::experimental::dfb::CreateDataflowBuffer(
            program_,
            core,
            tt_metal::experimental::dfb::DataflowBufferConfig{
                .entry_size = partials_tile_size,
                .num_entries = M * N,
                .num_producers = 1,
                .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 1,
                .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                .enable_producer_implicit_sync = false,
                .enable_consumer_implicit_sync = false,
                .data_format = out_fmt,
                .tensix_scope = tt_metal::experimental::dfb::TensixScope::INTRA});
    } else {
        const uint32_t in0_cb_index = 0;
        const uint32_t in1_cb_index = 1;
        const uint32_t out_cb_index = 16;
        const uint32_t partials_cb_index = 24;

        tt_metal::CircularBufferConfig l1_input0_cb_config =
            tt_metal::CircularBufferConfig(in0_block_size_bytes, {{in0_cb_index, in0_fmt}})
                .set_page_size(in0_cb_index, in0_tile_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_input0_cb_config);

        tt_metal::CircularBufferConfig l1_input1_cb_config =
            tt_metal::CircularBufferConfig(in1_block_size_bytes, {{in1_cb_index, in1_fmt}})
                .set_page_size(in1_cb_index, in1_tile_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_input1_cb_config);

        tt_metal::CircularBufferConfig l1_output_cb_config =
            tt_metal::CircularBufferConfig(out_byte_size, {{out_cb_index, out_fmt}})
                .set_page_size(out_cb_index, out_tile_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_output_cb_config);

        tt_metal::CircularBufferConfig l1_partials_cb_config =
            tt_metal::CircularBufferConfig(out_byte_size, {{partials_cb_index, out_fmt}})
                .set_page_size(partials_cb_index, partials_tile_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_partials_cb_config);

        in0_id = in0_cb_index;
        in1_id = in1_cb_index;
        out_id = out_cb_index;
        partials_id = partials_cb_index;
    }

    std::map<std::string, std::string> compute_defines;
    if (cfg.packer_l1_acc) {
        compute_defines["PACKER_L1_ACC"] = "1";
    }

    std::vector<uint32_t> compute_compile_args = {
        in0_id, in1_id, out_id, partials_id, M * K, K * N, M * N, M, N, K, num_blocks};

    KernelHandle reader_kernel;
    KernelHandle writer_kernel;
    KernelHandle compute_kernel;

    if (is_quasar) {
        reader_kernel = tt_metal::experimental::quasar::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
            core,
            tt_metal::experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .compile_args = {in0_id, in1_id}});

        writer_kernel = tt_metal::experimental::quasar::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
            core,
            tt_metal::experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .compile_args = {out_id}});

        compute_kernel = tt_metal::experimental::quasar::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_block_compute.cpp",
            core,
            tt_metal::experimental::quasar::QuasarComputeConfig{
                .num_threads_per_cluster = 1,
                .enable_2x_src_format = cfg.enable_2x_src_format,
                .compile_args = compute_compile_args,
                .defines = compute_defines});

        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, in0_id, reader_kernel, compute_kernel);
        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, in1_id, reader_kernel, compute_kernel);
        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, out_id, compute_kernel, writer_kernel);
        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, partials_id, compute_kernel, compute_kernel);
    } else {
        reader_kernel = tt_metal::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {in0_id, in1_id}});

        writer_kernel = tt_metal::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = {out_id}});

        compute_kernel = tt_metal::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_block_compute.cpp",
            core,
            tt_metal::ComputeConfig{
                .fp32_dest_acc_en = cfg.fp32_dest_acc_en,
                .compile_args = compute_compile_args,
                .defines = compute_defines});
    }

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
    for (uint32_t b = 0; b < num_blocks; b++) {
        const size_t in0_block_off = b * M * K * tt::constants::TILE_HW;
        const size_t in1_block_off = b * K * N * tt::constants::TILE_HW;
        for (uint32_t mt = 0; mt < M; mt++) {
            for (uint32_t nt = 0; nt < N; nt++) {
                const size_t out_tile_off = (mt * N + nt) * tt::constants::TILE_HW;
                for (uint32_t y = 0; y < tt::constants::TILE_HEIGHT; y++) {
                    for (uint32_t x = 0; x < tt::constants::TILE_WIDTH; x++) {
                        float acc = 0.0f;
                        for (uint32_t kt = 0; kt < K; kt++) {
                            const size_t in0_tile_off = in0_block_off + (mt * K + kt) * tt::constants::TILE_HW;
                            const size_t in1_tile_off = in1_block_off + (kt * N + nt) * tt::constants::TILE_HW;
                            for (uint32_t z = 0; z < tt::constants::TILE_WIDTH; z++) {
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

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, in0_stim.packed);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, in1_stim.packed);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)0,
            (uint32_t)in1_dram_addr,
            (uint32_t)0,
            (uint32_t)num_blocks,
            (uint32_t)(M * K),  // in0_block_tile_cnt
            (uint32_t)(K * N),  // in1_block_tile_cnt
            (uint32_t)in0_block_size_bytes,
            (uint32_t)in1_block_size_bytes,
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)0,
            (uint32_t)(M * N),
        });

    auto blocking = is_quasar;
    distributed::EnqueueMeshWorkload(cq, workload, blocking);
    if (not blocking) {
        distributed::Finish(cq);
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
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

}  // namespace unit_tests::compute::matmul

TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleTileComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul(this->devices_.at(id)));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 1, 1, 1));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileAccumulationComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 1, 2, 1));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileNoAccumulationComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 2, 1, 2));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreMultiBlockSpillReloadComputeMatmul) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2, .K = 2, .N = 2, .num_blocks = 4, .packer_l1_acc = false};
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(this->devices_.at(id), config));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreMultiBlockL1AccComputeMatmul) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2, .K = 2, .N = 2, .num_blocks = 4, .packer_l1_acc = true};
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(this->devices_.at(id), config));
    }
}

// Quasar-only test for matmul variant that enables the 2x source format optimization for MxFp4_2x format,
// This optimization allows src register datums to store two elements in one.
// Since MxFp4_2x only supports GAPOOL and MVMUL/MVMULDI instructions, we cannot test it with multiple blocks
// since multi-block matmul kernel uses datacopy from SRC to DST which is not supported by MxFp4_2x format.
// We can still verify the correctness of the optimization with single block matmul.
// L1 acc doesn't work in this case since it also relies on datacopy from SRC to DST for the accumulation.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixTestSingleCoreSingleBlockComputeMatmulMxFp4X2) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2,
        .K = 2,
        .N = 2,
        .num_blocks = 1,
        .packer_l1_acc = false,
        .in0_fmt = tt::DataFormat::MxFp4,
        .in1_fmt = tt::DataFormat::MxFp4,
        .out_fmt = tt::DataFormat::Float16_b,
        .enable_2x_src_format = true};
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

}  // namespace tt::tt_metal
