// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

namespace {

/**
 * @brief Run copy_tile (unpack_A) then add_tiles (unpack_AB) in one DST section, with math delayed between copies.
 *
 * Dispatch twice; the first dispatch after tt-smi -r is the cold run.
 */
void run_unpack_a_ab_race(distributed::MeshDevice& mesh, uint32_t delay_cycles, bool unpack_stall) {
    constexpr uint32_t batches = 3;
    constexpr uint32_t slots = 8;
    constexpr uint32_t copies_after = 6;
    constexpr uint32_t tile_size = tt::constants::TILE_HW;
    constexpr uint32_t in_tiles = batches * 2 * slots;
    constexpr uint32_t out_tiles = batches * slots;
    // Integer values keep A+B exact in bfloat16: A in [-256, 0), B in [0, 64].
    std::vector<bfloat16> input(in_tiles * tile_size);
    std::vector<bfloat16> expected(out_tiles * tile_size);
    for (uint32_t batch = 0; batch < batches; ++batch) {
        for (uint32_t slot = 0; slot < slots; ++slot) {
            for (uint32_t i = 0; i < tile_size; ++i) {
                const float a = -static_cast<float>((i * 7 + slot * 13 + batch * 29) % 256 + 1);
                const float b = static_cast<float>((i * 11 + slot * 5 + batch * 3) % 65);
                input[((batch * 2 * slots) + slot) * tile_size + i] = bfloat16(a);
                input[((batch * 2 * slots) + slots + slot) * tile_size + i] = bfloat16(b);
                expected[(batch * slots + slot) * tile_size + i] = bfloat16(slot < copies_after ? a : a + b);
            }
        }
    }
    const auto packed = pack_bfloat16_vec_into_uint32_vec(input);
    auto& cq = mesh.mesh_command_queue();
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    auto src_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = in_tiles * tile_bytes},
        {.page_size = in_tiles * tile_bytes, .buffer_type = BufferType::DRAM},
        &mesh);
    auto dst_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = out_tiles * tile_bytes},
        {.page_size = out_tiles * tile_bytes, .buffer_type = BufferType::DRAM},
        &mesh);
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(2 * slots * tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, tile_bytes));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(slots * tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, tile_bytes));
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
        "tests/tt_metal/tt_metal/test_kernels/compute/unpack_a_ab_race.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = false, .compile_args = {batches, delay_cycles, static_cast<uint32_t>(unpack_stall)}});
    SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(src_buffer->address()), 0, in_tiles});
    SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(dst_buffer->address()), 0, out_tiles});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    for (uint32_t run = 0; run < 2; ++run) {
        distributed::EnqueueWriteMeshBuffer(cq, src_buffer, packed, true);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::vector<uint32_t> raw;
        distributed::EnqueueReadMeshBuffer(cq, raw, dst_buffer, true);
        const auto actual = unpack_uint32_vec_into_bfloat16_vec(raw);
        ASSERT_EQ(actual.size(), expected.size());
        for (uint32_t tile = 0; tile < out_tiles; ++tile) {
            uint32_t wrong = 0;
            for (uint32_t i = 0; i < tile_size; ++i) {
                wrong += static_cast<float>(actual[tile * tile_size + i]) !=
                         static_cast<float>(expected[tile * tile_size + i]);
            }
            EXPECT_EQ(wrong, 0u) << "run=" << run << " batch=" << tile / slots << " slot=" << tile % slots;
        }
    }
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, UnpackAToABRaceDelay300) {
    run_unpack_a_ab_race(*devices_.at(0), 300 /*delay_cycles*/, false /*unpack_stall*/);
}

TEST_F(LLKBlackholeSingleCardFixture, UnpackAToABRaceNoDelay) {
    run_unpack_a_ab_race(*devices_.at(0), 0 /*delay_cycles*/, false /*unpack_stall*/);
}

TEST_F(LLKBlackholeSingleCardFixture, UnpackAToABRaceDelay300UnpackStall) {
    run_unpack_a_ab_race(*devices_.at(0), 300 /*delay_cycles*/, true /*unpack_stall*/);
}

}  // namespace tt::tt_metal
