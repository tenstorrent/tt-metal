// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

/**
 * @brief Validate the public block-max API, row packing, and neighboring DST slots.
 *
 * Keep the golden and device cases in one host-owned table. Compile arguments
 * describe the tile/packing geometry; runtime arguments contain a DST index and
 * valid-score count per acquisition. Dispatch twice to exercise program reuse.
 */
TEST_F(LLKBlackholeSingleCardFixture, BlockMax8CompactRowsAndNeighborPreservation) {
    constexpr uint32_t slots = 8;
    constexpr uint32_t block_size = 8;
    constexpr uint32_t result_rows = 8;
    constexpr std::array<uint32_t, 13> valid_counts{0, 1, 7, 8, 9, 15, 16, 17, 511, 512, 513, 1023, 1024};
    constexpr std::array<uint32_t, 3> dst_indices{0, 3, 7};
    constexpr uint32_t batches = valid_counts.size();
    constexpr uint32_t dst_index_count = dst_indices.size();
    constexpr uint32_t bf16_per_word = sizeof(uint32_t) / sizeof(bfloat16);
    constexpr uint32_t tile_size = tt::constants::TILE_HW;
    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t face_width = tt::constants::FACE_WIDTH;
    constexpr uint32_t tiles = batches * slots;
    std::vector<bfloat16> input(tiles * tile_size);
    auto expected = input;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        const uint32_t batch = tile / slots;
        const uint32_t dst = dst_indices[batch % dst_index_count];
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < tile_size; ++i) {
            const uint32_t row = i / tile_width;
            const uint32_t col = i % tile_width;
            const uint32_t physical = ((row / face_width) * 2 + col / face_width) * face_width * face_width +
                                      row % face_width * face_width + col % face_width;
            const float value = tile % slots == dst && i >= valid_counts[batch]
                                    ? 2048.0f
                                    : static_cast<float>(static_cast<int>((i * 17 + tile * 11) % 127) - 127) / 8.0f;
            input[tile * tile_size + physical] = bfloat16(value);
            if (tile % slots != dst) {
                expected[tile * tile_size + physical] = bfloat16(value);
            } else {
                if (i < valid_counts[batch]) {
                    maximum = std::max(maximum, value);
                }
                if (i % block_size == block_size - 1) {
                    expected[tile * tile_size + i / block_size] = bfloat16(maximum);
                    maximum = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }
    const auto packed = pack_bfloat16_vec_into_uint32_vec(input);
    const auto golden = pack_bfloat16_vec_into_uint32_vec(expected);
    auto& mesh = *devices_.at(0);
    auto& cq = mesh.mesh_command_queue();
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
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
            CircularBufferConfig(slots * tile_bytes, {{index, tt::DataFormat::Float16_b}})
                .set_page_size(index, tile_bytes));
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
    const auto compute = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/block_max8.cpp",
        core,
        ComputeConfig{.fp32_dest_acc_en = false, .compile_args = {tiles, slots, result_rows}});
    std::vector<uint32_t> cases;
    for (uint32_t batch = 0; batch < batches; ++batch) {
        cases.push_back(dst_indices[batch % dst_index_count]);
        cases.push_back(valid_counts[batch]);
    }
    SetRuntimeArgs(program, compute, core, cases);
    SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(src_buffer->address()), 0, tiles});
    SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(dst_buffer->address()), 0, tiles});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        distributed::EnqueueWriteMeshBuffer(cq, src_buffer, packed, true);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::vector<uint32_t> actual;
        distributed::EnqueueReadMeshBuffer(cq, actual, dst_buffer, true);
        ASSERT_EQ(actual.size(), golden.size());
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            const auto dst = dst_indices[(tile / slots) % dst_index_count];
            const uint32_t words =
                tile % slots == dst ? tile_size / block_size / bf16_per_word : tile_size / bf16_per_word;
            for (uint32_t word = 0; word < words; ++word) {
                const auto offset = tile * tile_size / bf16_per_word + word;
                EXPECT_EQ(actual[offset], golden[offset])
                    << "repeat=" << repeat << " tile=" << tile << " word=" << word;
            }
        }
    }
}
}  // namespace tt::tt_metal
