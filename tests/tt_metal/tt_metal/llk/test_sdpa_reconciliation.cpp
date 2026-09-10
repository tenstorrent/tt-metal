// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

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

void run_sdpa_recip(distributed::MeshDevice& mesh, std::uint32_t fidelity, std::uint32_t granularity) {
    auto* device = mesh.get_devices()[0];
    const CoreCoord core{0, 0};
    constexpr std::uint32_t tiles = 3;
    constexpr std::uint32_t tile_bytes = 2048;
    Program program = CreateProgram();
    auto make_buffer = [&]() {
        return CreateBuffer(InterleavedBufferConfig{
            .device = device,
            .size = tiles * tile_bytes,
            .page_size = tiles * tile_bytes,
            .buffer_type = BufferType::DRAM});
    };
    auto input = make_buffer();
    auto output = make_buffer();
    for (const auto cb : {tt::CBIndex::c_0, tt::CBIndex::c_16}) {
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tile_bytes, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, tile_bytes));
    }
    // The normalizer's logical tile is two 8x16 faces. Data is already in
    // DEST; this CB supplies the compute API's operand format/shape metadata.
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(512, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, 512)
            .set_unpack_face_geometry(tt::CBIndex::c_1, 8, 2));
    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/sdpa_recip_reconciliation.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {tiles, fidelity, granularity}});

    std::vector<bfloat16> source(tiles * 1024);
    std::vector<bfloat16> golden(source.size());
    for (std::uint32_t i = 0; i < source.size(); ++i) {
        source[i] = bfloat16(static_cast<float>(static_cast<int>(i * 17 % 63) - 31) / 8.0f);
        // Two 8x32 SDPA tiles occupy the first 512 elements of each full tile.
        // The other half is a guard for writes beyond the normalized region.
        const float divisor = i % 1024 < 512 ? 2.0f : 1.0f;
        golden[i] = bfloat16(static_cast<float>(source[i]) / divisor);
    }
    detail::WriteToBuffer(input, pack_bfloat16_vec_into_uint32_vec(source));
    SetRuntimeArgs(program, reader, core, {input->address(), 0, tiles});
    SetRuntimeArgs(program, writer, core, {output->address(), 0, tiles});
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    auto& cq = mesh.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    std::vector<std::uint32_t> result;
    detail::ReadFromBuffer(output, result);
    EXPECT_EQ(result, pack_bfloat16_vec_into_uint32_vec(golden));
}

}  // namespace

TEST_F(LLKBlackholeSingleCardFixture, SdpaRecipFidelityAndSignalling) {
    for (const auto fidelity : {0u, 2u, 3u, 4u, 255u}) {
        for (const auto granularity : {1u, 2u}) {
            SCOPED_TRACE(::testing::Message() << "fidelity=" << fidelity << ", granularity=" << granularity);
            run_sdpa_recip(*devices_.at(0), fidelity, granularity);
        }
    }
}

}  // namespace tt::tt_metal
