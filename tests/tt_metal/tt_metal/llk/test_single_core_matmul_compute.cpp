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
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include <tt-metalium/experimental/host_api.hpp>
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
        auto rand_float = std::bind(std::uniform_real_distribution<float>(-rng, +rng), std::mt19937(seed));
        std::vector<float> raw(num_elements);
        for (float& v : raw) {
            v = rand_float();
        }
        out.packed =
            pack_as_bfp8_tiles<float>(ttsl::make_const_span(raw), /*row_major_input=*/false, /*is_exp_a=*/false);
        out.floats = unpack_bfp8_tiles_into_float_vec(
            ttsl::make_const_span(out.packed), /*row_major_output=*/false, /*is_exp_a=*/false);
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
    const bool fp8_out = (out_fmt == tt::DataFormat::Fp8_e4m3);
    auto dest_floats = fp8_out ? fp8_to_floats(dest_buffer_data) : bf16_to_floats(dest_buffer_data);

    // Tolerances under random U(-1, +1) stimulus. FP8 output: ~1/8
    // quantization plus deeper accumulation rounding for K>1. BF16 output
    // (default fp32_dest_acc=false uses BF16 dest accumulation, so per-element
    // error grows with K * TILE_WIDTH inner-dim length) needs a few-percent
    // slack. PCC backstop catches structural mis-permutations that pointwise
    // tolerances would let slip.
    const float rtol = fp8_out ? 0.125f : 0.05f;
    const float atol = fp8_out ? ((K > 1) ? 0.25f : 0.125f) : ((K > 1) ? 0.4f : 0.2f);
    pass &= tt::test_utils::is_close_vectors<float>(
        dest_floats, golden_floats, [&](float a, float b) { return tt::test_utils::is_close(a, b, rtol, atol); });
    const double min_pcc = fp8_out ? ((K > 1) ? 0.98 : 0.99) : 0.99;
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

    const bool is_quasar = MetalContext::instance().get_cluster().arch() == ARCH::QUASAR;

    bool pass = true;
    CoreCoord core(0, 0);
    const size_t cb_page_size = 2 * tt::constants::TILE_HW;  // 2 bytes per bfloat16 element
    const size_t in0_block_size_bytes = M * K * cb_page_size;
    const size_t in1_block_size_bytes = K * N * cb_page_size;
    const size_t in0_total_size_bytes = num_blocks * in0_block_size_bytes;
    const size_t in1_total_size_bytes = num_blocks * in1_block_size_bytes;
    const size_t out_byte_size = M * N * cb_page_size;
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
        auto make_dfb = [&](uint32_t entry_size, uint32_t num_entries, uint16_t producer_mask, uint16_t consumer_mask) {
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
                    .data_format = tt::DataFormat::Float16_b});
        };
        in0_id = make_dfb(cb_page_size, M * K, 0x1, 0x100);
        in1_id = make_dfb(cb_page_size, K * N, 0x1, 0x100);
        out_id = make_dfb(cb_page_size, M * N, 0x100, 0x2);
        partials_id = tt_metal::experimental::dfb::CreateDataflowBuffer(
            program_,
            core,
            tt_metal::experimental::dfb::DataflowBufferConfig{
                .entry_size = cb_page_size,
                .num_entries = M * N,
                .num_producers = 1,
                .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 1,
                .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                .enable_producer_implicit_sync = false,
                .enable_consumer_implicit_sync = false,
                .data_format = tt::DataFormat::Float16_b,
                .tensix_scope = tt_metal::experimental::dfb::TensixScope::INTRA});
    } else {
        const uint32_t in0_cb_index = 0;
        const uint32_t in1_cb_index = 1;
        const uint32_t out_cb_index = 16;
        const uint32_t partials_cb_index = 24;

        tt_metal::CircularBufferConfig l1_input0_cb_config =
            tt_metal::CircularBufferConfig(in0_block_size_bytes, {{in0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(in0_cb_index, cb_page_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_input0_cb_config);

        tt_metal::CircularBufferConfig l1_input1_cb_config =
            tt_metal::CircularBufferConfig(in1_block_size_bytes, {{in1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(in1_cb_index, cb_page_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_input1_cb_config);

        tt_metal::CircularBufferConfig l1_output_cb_config =
            tt_metal::CircularBufferConfig(out_byte_size, {{out_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(out_cb_index, cb_page_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_output_cb_config);

        tt_metal::CircularBufferConfig l1_partials_cb_config =
            tt_metal::CircularBufferConfig(out_byte_size, {{partials_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(partials_cb_index, cb_page_size);
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
                .num_threads_per_cluster = 1, .compile_args = compute_compile_args, .defines = compute_defines});

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
            tt_metal::ComputeConfig{.compile_args = compute_compile_args, .defines = compute_defines});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0.0f,
        1.0f,
        in0_total_size_bytes / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -0.45f,
        1.0f,
        in1_total_size_bytes / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    // Data in DRAM is in TILED_NFACES layout, stored as num_blocks consecutive
    // blocks of [M*32, K*32] and [K*32, N*32] respectively.
    const uint32_t M_elem = M * 32;
    const uint32_t K_elem = K * 32;
    const uint32_t N_elem = N * 32;
    const uint32_t block_in0_elems = M_elem * K_elem;
    const uint32_t block_in1_elems = K_elem * N_elem;
    const uint32_t out_elems = M_elem * N_elem;

    auto u16_in0 = u16_from_u32_vector(packed_input0);
    auto u16_in1 = u16_from_u32_vector(packed_input1);

    std::vector<float> golden_float(out_elems, 0.0f);
    std::vector<uint32_t> block_shape_in0 = {1, 1, M_elem, K_elem};
    std::vector<uint32_t> block_shape_in1 = {1, 1, K_elem, N_elem};

    for (uint32_t b = 0; b < num_blocks; b++) {
        // Extract this block's tile data and untilize to row-major
        std::vector<uint16_t> block_in0(
            u16_in0.begin() + b * block_in0_elems, u16_in0.begin() + (b + 1) * block_in0_elems);
        std::vector<uint16_t> block_in1(
            u16_in1.begin() + b * block_in1_elems, u16_in1.begin() + (b + 1) * block_in1_elems);

        auto in0_rm = convert_layout<uint16_t>(
            block_in0, block_shape_in0, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
        auto in1_rm = convert_layout<uint16_t>(
            block_in1, block_shape_in1, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);

        for (uint32_t i = 0; i < M_elem; i++) {
            for (uint32_t j = 0; j < N_elem; j++) {
                for (uint32_t k = 0; k < K_elem; k++) {
                    float a = static_cast<float>(std::bit_cast<bfloat16>(in0_rm[i * K_elem + k]));
                    float bb = static_cast<float>(std::bit_cast<bfloat16>(in1_rm[k * N_elem + j]));
                    golden_float[i * N_elem + j] += a * bb;
                }
            }
        }
    }

    // Convert golden from float to bfloat16, tilize, and pack
    std::vector<uint16_t> golden_u16(out_elems);
    for (uint32_t i = 0; i < out_elems; i++) {
        golden_u16[i] = std::bit_cast<uint16_t>(bfloat16(golden_float[i]));
    }
    std::vector<uint32_t> out_shape = {1, 1, M_elem, N_elem};
    auto golden_tiled = convert_layout<uint16_t>(
        golden_u16, out_shape, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
    auto packed_golden = u32_from_u16_vector(golden_tiled);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);

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

    int failed_index;
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.05f, 0.05f); },
        &failed_index);
    if (not pass) {
        log_info(tt::LogTest, "Failed Index={}", failed_index);
        print_vector_fixed_numel_per_row(
            unpack_vector<bfloat16, uint32_t>(dest_buffer_data), tt::constants::TILE_WIDTH);
    }
    return pass;
}

}  // namespace unit_tests::compute::matmul

namespace {
const char* matmul_data_format_name(tt::DataFormat f) {
    switch (f) {
        case tt::DataFormat::Fp8_e4m3: return "Fp8_e4m3";
        case tt::DataFormat::Float16_b: return "Float16_b";
        default: return "DataFormat(unknown)";
    }
}
}  // namespace

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

// sweeps in×out data format × fp32 dest acc (8 cases). Logs each case at start.
// Skips mixed Fp8/non-Fp8 cells at fp32_dest=false — those require per-CB role
// metadata in JIT to pick the correct DEST family.
TEST_F(LLKBlackholeSingleCardFixture, TensixTestSingleCoreSingleTileComputeMatmulFormatSweep) {
    static constexpr std::array<tt::DataFormat, 2> kInFormats = {
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Float16_b,
    };
    static constexpr std::array<tt::DataFormat, 2> kOutFormats = {
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Float16_b,
    };
    static constexpr std::array<bool, 2> kFp32DestAccEn = {true, false};

    for (tt::DataFormat in_fmt : kInFormats) {
        for (tt::DataFormat out_fmt : kOutFormats) {
            for (bool fp32_dest_acc_en : kFp32DestAccEn) {
                const bool mixed_fp8 = (in_fmt == tt::DataFormat::Fp8_e4m3) != (out_fmt == tt::DataFormat::Fp8_e4m3);
                if (mixed_fp8 && !fp32_dest_acc_en) {
                    log_info(
                        tt::LogTest,
                        "TensixTestSingleCoreSingleTileComputeMatmulFormatSweep: SKIP in_fmt={} out_fmt={} "
                        "fp32_dest_acc_en=false (unsupported on this branch)",
                        matmul_data_format_name(in_fmt),
                        matmul_data_format_name(out_fmt));
                    continue;
                }
                log_info(
                    tt::LogTest,
                    "TensixTestSingleCoreSingleTileComputeMatmulFormatSweep: in_fmt={} out_fmt={} "
                    "fp32_dest_acc_en={}",
                    matmul_data_format_name(in_fmt),
                    matmul_data_format_name(out_fmt),
                    fp32_dest_acc_en);
                ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul(
                    this->devices_.at(0), in_fmt, in_fmt, out_fmt, fp32_dest_acc_en));
            }
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

TEST_F(
    LLKBlackholeSingleCardFixture, TensixTestSingleCoreSingleBlockSingleTileAccumulationComputeMatmulFp8e4m3Fp16Dest) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(
        this->devices_.at(0),
        1,
        2,
        1,
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/false));
}

}  // namespace tt::tt_metal
