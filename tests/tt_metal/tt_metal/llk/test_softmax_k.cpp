// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/data_formats/bfloat16.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/execution/slow_dispatch.hpp>
#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

namespace {

constexpr uint32_t kRowWidth = tt::constants::FACE_WIDTH;  // softmax_k works on the 16 columns of face 0, row 0

// One input row per tile. Column 0 is the row maximum: the public API broadcasts it as the max. Columns >= k are
// padding and get the tile's padding fill, which the API must ignore.
struct SoftmaxKCase {
    std::array<float, kRowWidth> logits;
    float padding;
};

std::vector<SoftmaxKCase> softmax_k_cases() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<SoftmaxKCase> cases;
    SoftmaxKCase c{};
    // Descending, evenly spaced.
    for (uint32_t i = 0; i < kRowWidth; ++i) {
        c.logits[i] = 2.0f - 0.5f * i;
    }
    c.padding = nan;
    cases.push_back(c);
    // All zero: 0.0 is an ordinary logit, so the result is uniform over the k columns.
    c.logits.fill(0.0f);
    c.padding = inf;
    cases.push_back(c);
    // Unordered below the maximum.
    for (uint32_t i = 0; i < kRowWidth; ++i) {
        c.logits[i] = i == 0 ? 3.0f : 3.0f - static_cast<float>((i * 7) % 17) / 2.0f;
    }
    c.padding = -inf;
    cases.push_back(c);
    // Negative maximum, with zeros among the valid columns' differences.
    for (uint32_t i = 0; i < kRowWidth; ++i) {
        c.logits[i] = -5.0f - static_cast<float>(i % 3);
    }
    c.padding = 1e30f;
    cases.push_back(c);
    // Wide spread: the tail columns underflow towards 0.
    for (uint32_t i = 0; i < kRowWidth; ++i) {
        c.logits[i] = 10.0f - 1.5f * i;
    }
    c.padding = 0.0f;
    cases.push_back(c);
    return cases;
}

/**
 * @brief Run the public softmax_k compute API (unpack + row copy + max broadcast + SFPU) on one tile per case.
 *
 * Only row 0 of face 0 is defined: columns < k must match softmax(x[c] - x[0]) over the first k columns, and
 * columns >= k must be exactly 0.0. Every other element of the input tile is NaN and is never read.
 * The program is dispatched twice to exercise reuse.
 */
void run_softmax_k(distributed::MeshDevice& mesh, uint32_t k, bool fp32_dest_acc_en) {
    constexpr uint32_t tile_size = tt::constants::TILE_HW;
    const auto cases = softmax_k_cases();
    const auto tiles = static_cast<uint32_t>(cases.size());

    std::vector<bfloat16> input(tiles * tile_size, bfloat16(std::numeric_limits<float>::quiet_NaN()));
    // Row 0 of face 0 is the first kRowWidth elements of a tile in tile order.
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        for (uint32_t col = 0; col < kRowWidth; ++col) {
            input[tile * tile_size + col] = bfloat16(col < k ? cases[tile].logits[col] : cases[tile].padding);
        }
    }

    auto& cq = mesh.mesh_command_queue();
    const CoreCoord core{0, 0};
    Program program = CreateProgram(mesh);
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t buffer_bytes = tiles * tile_bytes;
    auto src_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buffer_bytes},
        {.page_size = buffer_bytes, .buffer_type = BufferType::DRAM},
        &mesh);
    auto dst_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buffer_bytes},
        {.page_size = buffer_bytes, .buffer_type = BufferType::DRAM},
        &mesh);
    for (const auto index : {tt::CBIndex::c_0, tt::CBIndex::c_16}) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(2 * tile_bytes, {{index, tt::DataFormat::Float16_b}}).set_page_size(index, tile_bytes));
    }
    const auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    const auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/softmax_k.cpp",
        core,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {tiles, k}});
    SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(src_buffer->address()), 0, tiles});
    SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(dst_buffer->address()), 0, tiles});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));

    // bf16 output: one rounding of the result plus the SFPU exp / reciprocal error.
    constexpr double rtol = 1.0 / 64;
    constexpr double atol = 1e-3;
    const auto packed = pack_bfloat16_vec_into_uint32_vec(input);
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        distributed::EnqueueWriteMeshBuffer(cq, src_buffer, packed, true);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::vector<uint32_t> raw;
        distributed::EnqueueReadMeshBuffer(cq, raw, dst_buffer, true);
        const auto actual = unpack_uint32_vec_into_bfloat16_vec(raw);
        ASSERT_EQ(actual.size(), input.size());
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            // Golden from the bf16-rounded inputs, with column 0 as the max, as the API uses it.
            std::array<double, kRowWidth> golden{};
            double sum = 0.0;
            const double max = static_cast<float>(input[tile * tile_size]);
            for (uint32_t col = 0; col < k; ++col) {
                golden[col] = std::exp(static_cast<float>(input[tile * tile_size + col]) - max);
                sum += golden[col];
            }
            double actual_sum = 0.0;
            for (uint32_t col = 0; col < kRowWidth; ++col) {
                const bfloat16 out = actual[tile * tile_size + col];
                if (col < k) {
                    const double expected = golden[col] / sum;
                    const double got = static_cast<float>(out);
                    actual_sum += got;
                    EXPECT_NEAR(got, expected, rtol * expected + atol)
                        << "repeat=" << repeat << " k=" << k << " fp32_dest=" << fp32_dest_acc_en << " tile=" << tile
                        << " col=" << col;
                } else {
                    EXPECT_EQ(std::bit_cast<uint16_t>(out), 0u)
                        << "padding not +0.0: repeat=" << repeat << " k=" << k << " fp32_dest=" << fp32_dest_acc_en
                        << " tile=" << tile << " col=" << col << " value=" << static_cast<float>(out);
                }
            }
            EXPECT_NEAR(actual_sum, 1.0, 2 * rtol)
                << "row sum: repeat=" << repeat << " k=" << k << " fp32_dest=" << fp32_dest_acc_en << " tile=" << tile;
        }
    }
}

}  // namespace

// k = 1 (single column), even/odd k, 15 (no even padding column) and 16 (no padding) cover every masking branch.
constexpr std::array<uint32_t, 7> kSoftmaxKValues{1, 2, 7, 8, 9, 15, 16};

TEST_F(LLKBlackholeSingleCardFixture, SoftmaxKPublicApi16BitDest) {
    for (const uint32_t k : kSoftmaxKValues) {
        run_softmax_k(*devices_.at(0), k, false /*fp32_dest_acc_en*/);
    }
}

TEST_F(LLKBlackholeSingleCardFixture, SoftmaxKPublicApi32BitDest) {
    for (const uint32_t k : kSoftmaxKValues) {
        run_softmax_k(*devices_.at(0), k, true /*fp32_dest_acc_en*/);
    }
}

}  // namespace tt::tt_metal
