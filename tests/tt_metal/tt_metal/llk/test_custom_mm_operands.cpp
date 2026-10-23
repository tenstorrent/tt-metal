// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace {

void run_custom_mm_operands(
    distributed::MeshDevice& mesh, std::uint32_t rows, bool mixed, std::uint32_t mode, bool full_sync) {
    constexpr std::uint32_t blocks = 3;
    const std::uint32_t input_block_tiles = mode == 2 ? 5 : 3;
    const std::uint32_t input_tiles = blocks * input_block_tiles;
    const Tile tiny({rows, 32});
    const Tile full;
    const auto activation_format = tt::DataFormat::Float16_b;
    const auto weight_format = mode == 1 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    const auto a_bytes = tiny.get_tile_size(activation_format);
    const auto b_bytes = full.get_tile_size(weight_format);
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
    auto result_buffer = buffer(blocks * a_bytes);
    auto make_cb = [&](tt::CBIndex index, tt::DataFormat format, const Tile& tile, std::uint32_t depth) {
        const auto bytes = tile.get_tile_size(format);
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(depth * bytes, {{index, format}})
                .set_page_size(index, bytes)
                .set_tile_dims(index, tile));
    };
    // Two blocks of capacity force successive executions to use different L1
    // bases, then wrap the CB. Static input metadata stays unchanged.
    make_cb(tt::CBIndex::c_0, activation_format, tiny, 2 * input_block_tiles);
    make_cb(tt::CBIndex::c_1, weight_format, full, 2 * input_block_tiles);
    make_cb(tt::CBIndex::c_16, activation_format, tiny, 2);
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
        "tests/tt_metal/tt_metal/test_kernels/compute/custom_mm_operands.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = full_sync,
            .math_approx_mode = false,
            .compile_args = {blocks, rows, static_cast<std::uint32_t>(mixed), mode}});

    std::vector<bfloat16> activations(input_tiles * rows * 32);
    std::vector<bfloat16> weights(input_tiles * 1024, bfloat16(0.0f));
    std::vector<bfloat16> golden(blocks * rows * 32);
    for (std::uint32_t block = 0; block < blocks; ++block) {
        for (std::uint32_t tile = 0; tile < input_block_tiles; ++tile) {
            for (std::uint32_t row = 0; row < rows; ++row) {
                for (std::uint32_t col = 0; col < 32; ++col) {
                    const auto index =
                        (block * input_block_tiles + tile) * rows * 32 + (col / 16) * rows * 16 + row * 16 + col % 16;
                    const float value =
                        tile == 1 || tile == 2 ? static_cast<float>(block + row + col % 7 + tile) / 8.0f : 64.0f;
                    activations[index] = bfloat16(value);
                }
            }
            for (std::uint32_t d = 0; d < 32; ++d) {
                const auto index = (block * input_block_tiles + tile) * 1024 + (d / 16) * 3 * 256 + (d % 16) * 17;
                // The two-column producer uses [I, 2I; 2I, I]. The reuse
                // consumer reads weight tiles 2 and 4, exercising a K stride of 2.
                float weight = 2.0f;
                if (tile == 0) {
                    weight = 32.0f;
                } else if (tile == 1 || tile == 4) {
                    weight = 1.0f;
                }
                weights[index] = bfloat16(weight);
            }
        }
        for (std::uint32_t i = 0; i < rows * 32; ++i) {
            const auto a0 = static_cast<float>(activations[(block * input_block_tiles + 1) * rows * 32 + i]);
            const auto a1 = static_cast<float>(activations[(block * input_block_tiles + 2) * rows * 32 + i]);
            // Reuse consumes both populated DEST tiles:
            // 2 * (a0 + 2*a1) + (2*a0 + a1) = 4*a0 + 5*a1.
            golden[block * rows * 32 + i] = bfloat16(mode == 2 ? 4.0f * a0 + 5.0f * a1 : a0 + 2.0f * a1);
        }
    }
    auto& cq = mesh.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, a_buffer, pack_bfloat16_vec_into_uint32_vec(activations), true);
    auto packed_weights =
        mode == 1 ? pack_as_bfp8_tiles<bfloat16>(weights, false, false) : pack_bfloat16_vec_into_uint32_vec(weights);
    distributed::EnqueueWriteMeshBuffer(cq, b_buffer, packed_weights, true);
    SetRuntimeArgs(program, reader, core, {a_buffer->address(), 0, b_buffer->address(), 0, input_tiles});
    SetRuntimeArgs(program, writer, core, {result_buffer->address(), 0, blocks});
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

TEST_F(LLKBlackholeSingleCardFixture, CustomMatmulOperandOverloads) {
    for (const auto rows : {1u, 2u, 4u, 8u}) {
        for (const bool mixed : {false, true}) {
            for (const auto mode : {0u, 1u, 2u}) {
                for (const bool full_sync : {false, true}) {
                    SCOPED_TRACE(
                        ::testing::Message()
                        << "rows=" << rows << ", mixed=" << mixed << ", mode=" << mode << ", full_sync=" << full_sync);
                    run_custom_mm_operands(*devices_.at(0), rows, mixed, mode, full_sync);
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
