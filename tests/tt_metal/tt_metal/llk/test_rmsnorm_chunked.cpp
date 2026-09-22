// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace {

constexpr std::uint32_t rows = 3;
constexpr std::uint32_t tile_elements = 1024;

std::vector<float> run_rmsnorm_chunked(
    distributed::MeshDevice& mesh,
    std::uint32_t num_tiles,
    std::uint32_t capacity,
    bool fp32_dest,
    bool full_sync,
    MathFidelity fidelity,
    bool clear_only,
    const std::vector<bfloat16>& source = {}) {
    const CoreCoord core{0, 0};
    constexpr std::uint32_t input_tile_bytes = tile_elements * sizeof(bfloat16);
    const auto output_format = fp32_dest ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t output_tile_bytes = tile_elements * (fp32_dest ? sizeof(float) : sizeof(bfloat16));
    const std::uint32_t output_tiles = rows * (clear_only ? capacity * (capacity - 1) : 1);
    Program program = CreateProgram();
    auto& cq = mesh.mesh_command_queue();
    auto make_buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    auto output = make_buffer(output_tiles * output_tile_bytes);
    for (const auto index : {tt::CBIndex::c_0, tt::CBIndex::c_1}) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(num_tiles * input_tile_bytes, {{index, tt::DataFormat::Float16_b}})
                .set_page_size(index, input_tile_bytes));
    }
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig((clear_only ? capacity : 1) * output_tile_bytes, {{tt::CBIndex::c_16, output_format}})
            .set_page_size(tt::CBIndex::c_16, output_tile_bytes));

    // clear_only needs operand metadata for initialization but consumes no CB.
    // Keep the input buffer alive until the workload completes in either case.
    auto input = make_buffer(rows * num_tiles * input_tile_bytes);
    if (!clear_only) {
        distributed::EnqueueWriteMeshBuffer(cq, input, pack_bfloat16_vec_into_uint32_vec(source), true);
        const auto reader = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, reader, core, {input->address(), 0, input->address(), 0, rows * num_tiles});
    }
    const auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {output->address(), 0, output_tiles});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/rmsnorm_chunked.cpp",
        core,
        ComputeConfig{
            .math_fidelity = fidelity,
            .fp32_dest_acc_en = fp32_dest,
            .dst_full_sync_en = full_sync,
            .math_approx_mode = false,
            .compile_args = {num_tiles, capacity, rows, static_cast<std::uint32_t>(clear_only)}});

    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> packed;
    distributed::EnqueueReadMeshBuffer(cq, packed, output, true);
    std::vector<float> result;
    result.reserve(output_tiles * tile_elements);
    for (const auto word : packed) {
        if (fp32_dest) {
            result.push_back(std::bit_cast<float>(word));
        } else {
            result.push_back(static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(word))));
            result.push_back(static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(word >> 16))));
        }
    }
    return result;
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, RmsnormProductClearPreservesOtherDestTiles) {
    for (const bool full_sync : {false, true}) {
        for (const bool fp32_dest : {false, true}) {
            // FP32 full sync deliberately clears slots 4..6 in the second bank.
            const std::uint32_t capacity = fp32_dest && !full_sync ? 4 : 8;
            SCOPED_TRACE(::testing::Message() << "full_sync=" << full_sync << ", fp32_dest=" << fp32_dest);
            const auto result = run_rmsnorm_chunked(
                *devices_.at(0), capacity + 1, capacity, fp32_dest, full_sync, MathFidelity::HiFi4, true);
            ASSERT_EQ(result.size(), rows * (capacity - 1) * capacity * tile_elements);
            for (std::uint32_t row = 0; row < rows; ++row) {
                for (std::uint32_t target = 0; target < capacity - 1; ++target) {
                    for (std::uint32_t tile = 0; tile < capacity; ++tile) {
                        const float expected = tile == target ? 0.0f : static_cast<float>(row * 16 + tile + 1);
                        const auto start = ((row * (capacity - 1) + target) * capacity + tile) * tile_elements;
                        for (std::uint32_t lane = 0; lane < tile_elements; ++lane) {
                            ASSERT_EQ(result[start + lane], expected)
                                << "row=" << row << ", target=" << target << ", tile=" << tile << ", lane=" << lane;
                        }
                    }
                }
            }
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, RmsnormChunkedReductionPreservesAccumulator) {
    for (const bool full_sync : {false, true}) {
        for (const bool fp32_dest : {false, true}) {
            const std::uint32_t capacity = fp32_dest && !full_sync ? 4 : 8;
            const std::uint32_t batch_size = capacity - 1;
            for (const bool partial_tail : {false, true}) {
                const std::uint32_t num_tiles = 3 * batch_size - (partial_tail ? 1 : 0);
                std::vector<bfloat16> source(rows * num_tiles * tile_elements);
                std::vector<double> golden(rows, 0.0);
                for (std::uint32_t row = 0; row < rows; ++row) {
                    const float magnitude = std::ldexp(1.0f, static_cast<int>(row) - 2);
                    for (std::uint32_t i = 0; i < num_tiles * tile_elements; ++i) {
                        // Signed powers of two keep products and the three-/seven-
                        // tile chunk sums exact even in LoFi. This isolates DEST
                        // reuse from partial-mantissa multiplication error.
                        const bool negative = (i + i / tile_elements + row) % 2 != 0;
                        const bfloat16 value(negative ? -magnitude : magnitude);
                        source[row * num_tiles * tile_elements + i] = value;
                        golden[row] += static_cast<double>(value) * static_cast<double>(value) / tile_elements;
                    }
                }
                for (const auto fidelity : {MathFidelity::LoFi, MathFidelity::HiFi4}) {
                    SCOPED_TRACE(
                        ::testing::Message()
                        << "full_sync=" << full_sync << ", fp32_dest=" << fp32_dest << ", num_tiles=" << num_tiles
                        << ", fidelity=" << static_cast<int>(fidelity));
                    const auto result = run_rmsnorm_chunked(
                        *devices_.at(0), num_tiles, capacity, fp32_dest, full_sync, fidelity, false, source);
                    ASSERT_EQ(result.size(), rows * tile_elements);
                    double squared_error = 0.0;
                    double squared_golden = 0.0;
                    for (std::uint32_t row = 0; row < rows; ++row) {
                        const float actual = result[row * tile_elements];
                        ASSERT_TRUE(std::isfinite(actual)) << "row=" << row;
                        EXPECT_NEAR(actual, golden[row], 0.01 * golden[row]) << "row=" << row;
                        squared_error += std::pow(actual - golden[row], 2);
                        squared_golden += golden[row] * golden[row];
                    }
                    // Correlation alone would accept the original uniform scaling error.
                    EXPECT_LT(std::sqrt(squared_error / squared_golden), 0.01);
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
