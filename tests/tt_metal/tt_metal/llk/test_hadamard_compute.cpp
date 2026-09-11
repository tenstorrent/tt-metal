// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <bit>
#include <cmath>
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

void run_hadamard_reconfiguration(
    distributed::MeshDevice& mesh, MathFidelity fidelity, bool normalize, bool fp32_dest, bool full_sync) {
    constexpr std::uint32_t count = 3;
    constexpr std::uint32_t face_elements = 256;
    constexpr std::uint32_t vector_length = 128;
    const Tile face({16, 16});
    const Tile full_tile;
    const auto input_bytes = face.get_tile_size(tt::DataFormat::Float16_b);
    const auto output_bytes = face.get_tile_size(tt::DataFormat::Bfp8_b);
    const auto copy_bytes = full_tile.get_tile_size(tt::DataFormat::Float16_b);
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
    auto& cq = mesh.mesh_command_queue();
    auto buffer = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = count * bytes},
            {.page_size = count * bytes, .buffer_type = BufferType::DRAM},
            &mesh);
    };
    auto input = buffer(input_bytes);
    auto weights = buffer(input_bytes);
    auto copy_input = buffer(copy_bytes);
    auto output = buffer(output_bytes);
    auto copy_output = buffer(copy_bytes);
    auto cb = [&](tt::CBIndex index, tt::DataFormat format, const Tile& shape, std::uint32_t depth) {
        const auto bytes = shape.get_tile_size(format);
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(depth * bytes, {{index, format}})
                .set_page_size(index, bytes)
                .set_tile_dims(index, shape));
    };
    // reader_binary loads c_0/c_1 first, then c_2. They must fit in their
    // entirety, and H16 must remain resident while its address is in unpacker context 1.
    cb(tt::CBIndex::c_0, tt::DataFormat::Float16_b, face, count);
    cb(tt::CBIndex::c_1, tt::DataFormat::Float16_b, face, count);
    cb(tt::CBIndex::c_2, tt::DataFormat::Float16_b, full_tile, 1);
    cb(tt::CBIndex::c_16, tt::DataFormat::Bfp8_b, face, 1);
    cb(tt::CBIndex::c_17, tt::DataFormat::Float16_b, full_tile, 1);

    const auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .defines = {{"LOAD_BUF2_DATA", "1"}}});
    const auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/hadamard_reconfiguration.cpp",
        core,
        ComputeConfig{
            .math_fidelity = fidelity,
            .fp32_dest_acc_en = fp32_dest,
            .dst_full_sync_en = full_sync,
            .math_approx_mode = false,
            .compile_args = {count, static_cast<std::uint32_t>(normalize)}});

    std::vector<bfloat16> inputs(count * face_elements);
    std::vector<bfloat16> h16(count * face_elements);
    std::vector<bfloat16> copies(count * 1024);
    std::vector<float> golden(count * face_elements, 0.0f);
    constexpr std::array<std::uint32_t, count> rows{13, 41, 70};
    for (std::uint32_t tile = 0; tile < count; ++tile) {
        // Distinct Walsh basis vectors have a single H128 output at the
        // corresponding row. This golden does not duplicate the two-pass LLK.
        const float sign = tile % 2 == 0 ? 1.0f : -1.0f;
        for (std::uint32_t i = 0; i < face_elements; ++i) {
            inputs[tile * face_elements + i] =
                bfloat16(i < vector_length ? sign * (std::popcount(i & rows[tile]) % 2 == 0 ? 1.0f : -1.0f) : 7.0f);
            h16[tile * face_elements + i] = bfloat16(std::popcount((i / 16) & (i % 16)) % 2 == 0 ? 1.0f : -1.0f);
        }
        const auto expected = sign * (normalize ? std::sqrt(128.0f) : 128.0f);
        // The normalized SFPU path stores BF16 before the BFP8 packer runs.
        golden[tile * face_elements + rows[tile]] = static_cast<float>(bfloat16(expected));
        for (std::uint32_t i = 0; i < 1024; ++i) {
            copies[tile * 1024 + i] = bfloat16(static_cast<float>(static_cast<int>((i + tile * 19) % 61) - 30) / 8.0f);
        }
    }
    const auto packed_copies = pack_bfloat16_vec_into_uint32_vec(copies);
    distributed::EnqueueWriteMeshBuffer(cq, input, pack_bfloat16_vec_into_uint32_vec(inputs), true);
    distributed::EnqueueWriteMeshBuffer(cq, weights, pack_bfloat16_vec_into_uint32_vec(h16), true);
    distributed::EnqueueWriteMeshBuffer(cq, copy_input, packed_copies, true);
    SetRuntimeArgs(
        program, reader, core, {input->address(), 0, weights->address(), 0, count, copy_input->address(), 0});
    SetRuntimeArgs(
        program,
        writer,
        core,
        {output->address(), 0, tt::CBIndex::c_16, copy_output->address(), 0, tt::CBIndex::c_17, count, 1});

    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> result;
    std::vector<std::uint32_t> copied;
    distributed::EnqueueReadMeshBuffer(cq, result, output, true);
    distributed::EnqueueReadMeshBuffer(cq, copied, copy_output, true);
    const auto packed_golden = pack_as_bfp8_tiles<float>(golden, true, false, face);
    EXPECT_EQ(
        unpack_bfp8_tiles_into_float_vec(result, true, false, face),
        unpack_bfp8_tiles_into_float_vec(packed_golden, true, false, face));
    EXPECT_EQ(copied, packed_copies);
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, HadamardShortInitPreservesDestState) {
    for (const auto fidelity : {MathFidelity::LoFi, MathFidelity::HiFi4}) {
        for (const bool full_sync : {false, true}) {
            for (const bool fp32_dest : {false, true}) {
                for (const bool normalize : {false, true}) {
                    if (normalize && fp32_dest) {
                        continue;  // The API explicitly rejects this combination.
                    }
                    SCOPED_TRACE(
                        ::testing::Message() << "fidelity=" << static_cast<int>(fidelity) << ", full_sync=" << full_sync
                                             << ", fp32_dest=" << fp32_dest << ", normalize=" << normalize);
                    run_hadamard_reconfiguration(*devices_.at(0), fidelity, normalize, fp32_dest, full_sync);
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
