// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
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

constexpr std::uint32_t iterations = 3;
constexpr std::uint32_t dest_tiles = 3;
constexpr std::uint32_t target = 1;
constexpr std::uint32_t full_tile_elements = 1024;

void run_deepseek_binary_dest_reuse(
    distributed::MeshDevice& mesh, std::uint32_t tile_height, MathFidelity fidelity, bool fp32_dest, bool reuse_srcb) {
    const Tile input_tile({tile_height, 32});
    const Tile output_tile;
    constexpr auto format = tt::DataFormat::Float16_b;
    const auto input_tile_bytes = input_tile.get_tile_size(format);
    const auto output_tile_bytes = output_tile.get_tile_size(format);
    const std::uint32_t input_tile_elements = tile_height * 32;
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
    auto buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            {.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    auto input = buffer(iterations * input_tile_bytes);
    auto output = buffer(iterations * dest_tiles * output_tile_bytes);
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(2 * input_tile_bytes, {{tt::CBIndex::c_0, format}})
            .set_page_size(tt::CBIndex::c_0, input_tile_bytes)
            .set_tile_dims(tt::CBIndex::c_0, input_tile));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(dest_tiles * output_tile_bytes, {{tt::CBIndex::c_16, format}})
            .set_page_size(tt::CBIndex::c_16, output_tile_bytes)
            .set_tile_dims(tt::CBIndex::c_16, output_tile));
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
        "tests/tt_metal/tt_metal/test_kernels/compute/deepseek_binary_dest_reuse.cpp",
        core,
        ComputeConfig{
            .math_fidelity = fidelity,
            .fp32_dest_acc_en = fp32_dest,
            .dst_full_sync_en = false,
            .math_approx_mode = false,
            .compile_args = {iterations, static_cast<std::uint32_t>(reuse_srcb)}});

    std::vector<bfloat16> source(iterations * input_tile_elements);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        for (std::uint32_t lane = 0; lane < input_tile_elements; ++lane) {
            // Signed powers of two make products exact at every fidelity. Both
            // faces contain distinct values, so losing face 1 cannot pass.
            const auto exponent = static_cast<int>((lane + 3 * iteration) % 6) - 3;
            const float magnitude = std::ldexp(1.0f, exponent);
            source[iteration * input_tile_elements + lane] = bfloat16(lane % 2 == 0 ? magnitude : -magnitude);
        }
    }
    auto& cq = mesh.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, input, pack_bfloat16_vec_into_uint32_vec(source), true);
    SetRuntimeArgs(program, reader, core, {input->address(), 0, iterations});
    SetRuntimeArgs(program, writer, core, {output->address(), 0, iterations * dest_tiles});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> packed;
    distributed::EnqueueReadMeshBuffer(cq, packed, output, true);
    const auto result = unpack_uint32_vec_into_bfloat16_vec(packed);
    ASSERT_EQ(result.size(), iterations * dest_tiles * full_tile_elements);

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        for (std::uint32_t tile = 0; tile < dest_tiles; ++tile) {
            const auto start = (iteration * dest_tiles + tile) * full_tile_elements;
            const float initial = static_cast<float>(1u << (iteration + tile));
            if (tile != target) {
                for (std::uint32_t lane = 0; lane < full_tile_elements; ++lane) {
                    ASSERT_EQ(static_cast<float>(result[start + lane]), initial)
                        << "iteration=" << iteration << ", untouched tile=" << tile << ", lane=" << lane;
                }
                continue;
            }
            const auto [face_height, face_width] = input_tile.get_face_shape();
            const std::uint32_t face_elements = face_height * face_width;
            for (std::uint32_t lane = 0; lane < input_tile_elements; ++lane) {
                // Input packs partial faces tightly; full-tile output preserves
                // the 16-row spacing of faces in DEST. Unused rows are scratch.
                const auto output_lane = lane / face_elements * 256 + lane % face_elements;
                const float expected = initial * static_cast<float>(source[iteration * input_tile_elements + lane]);
                ASSERT_EQ(static_cast<float>(result[start + output_lane]), expected)
                    << "iteration=" << iteration << ", input lane=" << lane << ", output lane=" << output_lane;
            }
        }
    }
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, DeepseekBinaryDestReusePreservesTinyTileGeometry) {
    for (const std::uint32_t height : {1u, 2u, 4u, 8u, 32u}) {
        for (const auto fidelity : {MathFidelity::LoFi, MathFidelity::HiFi2, MathFidelity::HiFi4}) {
            for (const bool fp32_dest : {false, true}) {
                for (const bool reuse_srcb : {false, true}) {
                    SCOPED_TRACE(
                        ::testing::Message() << "height=" << height << ", fidelity=" << static_cast<int>(fidelity)
                                             << ", fp32_dest=" << fp32_dest << ", reuse_srcb=" << reuse_srcb);
                    run_deepseek_binary_dest_reuse(*devices_.at(0), height, fidelity, fp32_dest, reuse_srcb);
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
