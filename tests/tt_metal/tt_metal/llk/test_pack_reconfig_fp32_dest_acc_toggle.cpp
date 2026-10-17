// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Packer data-format reconfig inside a mid-kernel dest-acc override.
//
// A kernel built with fp32_dest_acc_en = true switches Dest to 16-bit mode with
// set_fp32_dest_acc<false>(), copies a tile into Dest, reconfigures the packer from the Float16_b
// output configured at startup to the Bfp8_b output, and packs. The reconfig must leave the packer's
// Dest read width at what the override programmed; re-deriving it from the kernel-wide flag makes
// the packer read 32-bit rows out of a 16-bit Dest and the tile comes out scrambled.
//
// Stimuli are whole numbers in [0, 127], which Float16_b and Bfp8_b both represent exactly, so the
// decoded output is compared with the input datum for datum.

#include <gtest/gtest.h>
#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/program/program_impl.hpp"
#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

using std::vector;

namespace unit_tests::llk::pack_reconfig_fp32_dest_acc_toggle {

constexpr std::uint32_t kNumTiles = 1;
constexpr std::uint32_t kDatumsPerTile = tt::constants::TILE_HW;
// Bfp8_b keeps whole numbers up to this limit exact whatever the block's shared exponent is.
constexpr std::uint32_t kWholeNumberLimit = 128;

static constexpr const char* kComputeKernel =
    "tests/tt_metal/tt_metal/test_kernels/compute/pack_reconfig_fp32_dest_acc_toggle.cpp";

// c_0 (Float16_b) -> compute -> c_16 (Bfp8_b). c_17 (Float16_b) only configures the packer at startup.
static vector<std::uint32_t> run_pack_reconfig_inside_override(
    distributed::MeshDevice& mesh_device, const vector<std::uint32_t>& src_vec) {
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    const std::uint32_t in_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const std::uint32_t out_tile_size = tt::tile_size(tt::DataFormat::Bfp8_b);

    auto src_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kNumTiles * in_tile_size},
        {.page_size = kNumTiles * in_tile_size, .buffer_type = BufferType::DRAM},
        &mesh_device);
    auto dst_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kNumTiles * out_tile_size},
        {.page_size = kNumTiles * out_tile_size, .buffer_type = BufferType::DRAM},
        &mesh_device);

    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(in_tile_size, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, in_tile_size));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(out_tile_size, {{tt::CBIndex::c_16, tt::DataFormat::Bfp8_b}})
            .set_page_size(tt::CBIndex::c_16, out_tile_size));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(in_tile_size, {{tt::CBIndex::c_17, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_17, in_tile_size));

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
    CreateKernel(program, kComputeKernel, core, ComputeConfig{.fp32_dest_acc_en = true, .compile_args = {kNumTiles}});

    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src_buffer, src_vec, /*blocking=*/true);
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, kNumTiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, kNumTiles});

    LaunchProgram(mesh_device, std::move(program));

    vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_buffer, /*blocking=*/true);
    return result_vec;
}

}  // namespace unit_tests::llk::pack_reconfig_fp32_dest_acc_toggle

TEST_F(LLKMeshDeviceFixture, TensixPackReconfigInsideFp32DestAccOverride) {
    using namespace unit_tests::llk::pack_reconfig_fp32_dest_acc_toggle;
    if (this->arch_ == ARCH::QUASAR) {
        GTEST_SKIP() << "set_fp32_dest_acc is not supported on Quasar";
    }

    std::mt19937 rng(0x5EED);
    vector<bfloat16> input(kNumTiles * kDatumsPerTile);
    for (auto& datum : input) {
        datum = bfloat16(static_cast<float>(rng() % kWholeNumberLimit));
    }
    const vector<std::uint32_t> src_vec = pack_bfloat16_vec_into_uint32_vec(input);

    for (auto& device : this->devices_) {
        const vector<std::uint32_t> result = run_pack_reconfig_inside_override(*device, src_vec);
        const vector<float> output = unpack_bfp8_tiles_into_float_vec(
            ttsl::make_const_span(result), /*row_major_output=*/false, /*is_exp_a=*/false);
        ASSERT_EQ(output.size(), input.size());

        std::uint32_t mismatches = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            const float expected = static_cast<float>(input[i]);
            if (output[i] != expected) {
                if (mismatches < 8) {
                    log_error(tt::LogTest, "datum {}: expected {} got {}", i, expected, output[i]);
                }
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0u) << "Bfp8_b pack after pack_reconfig_data_format inside set_fp32_dest_acc<false>() "
                                     "differs from the input in "
                                  << mismatches << " of " << input.size() << " datums";
    }
}

}  // namespace tt::tt_metal
