// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace {

constexpr std::uint32_t FACE = 16;
constexpr std::uint32_t TILE_DIM = 32;

// custom_mm through the CB API with N = 16: [32, 16] weight tiles, or [16, 32] ones holding B^T when transposed, and
// [rows, 16] output tiles. api picks the compute overloads: CB ids, LLKOperand in0 and in1, or CB in0 with
// LLKOperand in1 and output. Small integer operands keep every partial sum exact in a bf16 DEST, so the result must
// match the golden bit for bit.
void run_custom_mm_n16(
    distributed::MeshDevice& mesh,
    std::uint32_t rows,
    std::uint32_t kt,
    std::uint32_t ct,
    bool transpose,
    bool split_acc,
    std::uint32_t api) {
    const std::uint32_t input_tiles = kt * ct;  // in0 is padded to the reader's shared tile count
    const std::uint32_t K = kt * TILE_DIM;
    const std::uint32_t N = ct * FACE;
    const Tile activation_tile({rows, TILE_DIM});
    const Tile weight_tile = transpose ? Tile({FACE, TILE_DIM}) : Tile({TILE_DIM, FACE});
    const Tile output_tile({rows, FACE});
    const auto format = tt::DataFormat::Float16_b;
    const auto a_bytes = activation_tile.get_tile_size(format);
    const auto b_bytes = weight_tile.get_tile_size(format);
    const auto out_bytes = output_tile.get_tile_size(format);
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
    auto buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    auto a_buffer = buffer(input_tiles * a_bytes);
    auto b_buffer = buffer(input_tiles * b_bytes);
    auto result_buffer = buffer(ct * out_bytes);
    auto make_cb = [&](tt::CBIndex index, const Tile& tile, std::uint32_t depth) {
        const auto bytes = tile.get_tile_size(format);
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(depth * bytes, {{index, format}})
                .set_page_size(index, bytes)
                .set_tile_dims(index, tile));
    };
    make_cb(tt::CBIndex::c_0, activation_tile, input_tiles);
    make_cb(tt::CBIndex::c_1, weight_tile, input_tiles);
    make_cb(tt::CBIndex::c_16, output_tile, ct);
    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/custom_mm_n16.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {
                kt, ct, static_cast<std::uint32_t>(transpose), static_cast<std::uint32_t>(split_acc), rows, api}});

    // A in {0, 1, 2} and B in {-1, 0, 1}: |sums| stay within 2 * K <= 256, exact in bf16. Both vary along K with a
    // period of 3, which divides neither a face (16) nor a tile (32), so reading the wrong in0 face or tile shows.
    auto a_value = [](std::uint32_t m, std::uint32_t k) { return static_cast<float>((m + k) % 3); };
    auto b_value = [](std::uint32_t k, std::uint32_t n) { return static_cast<float>((k * 5 + n * 7) % 3) - 1.0f; };

    // in0: kt [rows, 32] tiles, each two dense [rows, 16] faces along K.
    std::vector<bfloat16> activations(input_tiles * rows * TILE_DIM, bfloat16(0.0f));
    for (std::uint32_t t = 0; t < kt; ++t) {
        for (std::uint32_t m = 0; m < rows; ++m) {
            for (std::uint32_t col = 0; col < TILE_DIM; ++col) {
                const auto index = t * rows * TILE_DIM + (col / FACE) * rows * FACE + m * FACE + col % FACE;
                activations[index] = bfloat16(a_value(m, t * TILE_DIM + col));
            }
        }
    }
    // in1: kt * ct tiles, k-major / c-minor. Each holds two 16x16 faces along K; a face is B[k][n] row-major, or
    // B^T[n][k] when transposed (the unpacker transposes each face back).
    std::vector<bfloat16> weights(input_tiles * TILE_DIM * FACE);
    for (std::uint32_t r = 0; r < kt; ++r) {
        for (std::uint32_t c = 0; c < ct; ++c) {
            for (std::uint32_t face = 0; face < 2; ++face) {
                for (std::uint32_t i = 0; i < FACE; ++i) {
                    for (std::uint32_t j = 0; j < FACE; ++j) {
                        const auto k = r * TILE_DIM + face * FACE + (transpose ? j : i);
                        const auto n = c * FACE + (transpose ? i : j);
                        weights[(r * ct + c) * TILE_DIM * FACE + face * FACE * FACE + i * FACE + j] =
                            bfloat16(b_value(k, n));
                    }
                }
            }
        }
    }
    // Output: ct [rows, 16] tiles.
    std::vector<bfloat16> golden(ct * rows * FACE);
    for (std::uint32_t m = 0; m < rows; ++m) {
        for (std::uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (std::uint32_t k = 0; k < K; ++k) {
                sum += a_value(m, k) * b_value(k, n);
            }
            golden[(n / FACE) * rows * FACE + m * FACE + n % FACE] = bfloat16(sum);
        }
    }

    auto& cq = mesh.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, a_buffer, pack_bfloat16_vec_into_uint32_vec(activations), true);
    distributed::EnqueueWriteMeshBuffer(cq, b_buffer, pack_bfloat16_vec_into_uint32_vec(weights), true);
    SetRuntimeArgs(program, reader, core, {a_buffer->address(), 0, b_buffer->address(), 0, input_tiles});
    SetRuntimeArgs(program, writer, core, {result_buffer->address(), 0, ct});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, result_buffer, true);
    EXPECT_EQ(result, pack_bfloat16_vec_into_uint32_vec(golden));
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, CustomMatmulN16) {
    // rows 1 would give 32-byte output tiles, below the DRAM write alignment; the tt-llk test covers it.
    // The CB API takes the full sweep; the LLKOperand overloads share its LLK calls, so one shape per mode suffices.
    for (const bool transpose : {false, true}) {
        for (const bool split_acc : {false, true}) {
            for (const auto api : {0u, 1u, 2u}) {
                for (const auto rows : {2u, 8u}) {
                    for (const auto kt : {1u, 2u, 4u}) {
                        for (const auto ct : {1u, 3u, 8u}) {
                            if (api != 0 && (kt != 2 || ct != 3)) {
                                continue;
                            }
                            SCOPED_TRACE(
                                ::testing::Message()
                                << "rows=" << rows << ", kt=" << kt << ", ct=" << ct << ", transpose=" << transpose
                                << ", split_acc=" << split_acc << ", api=" << api);
                            run_custom_mm_n16(*devices_.at(0), rows, kt, ct, transpose, split_acc, api);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
