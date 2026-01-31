// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "print_tile_helpers.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test checking that DPRINTS print multiple tiles correctly.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {

namespace {

constexpr uint32_t elements_in_tile = 32 * 32;

/* Number of tiles in a circular buffer. The kernels will print num_tiles tiles. */
constexpr uint32_t num_tiles = 4;

void RunTest(
    DPrintMeshFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    tt::DataFormat data_format) {
    // Set up program + CQ, run on just one core
    constexpr CoreCoord core = {0, 0};
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto& cq = mesh_device->mesh_command_queue();

    /* Creating input CB. Page size of CB = tile size. Each tile is a separate page for CB. */
    uint32_t tile_size = tt::tile_size(data_format);
    CircularBufferConfig cb_src0_config = CircularBufferConfig(tile_size * num_tiles, {{CBIndex::c_0, data_format}})
                                              .set_page_size(CBIndex::c_0, tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

    distributed::DeviceLocalBufferConfig dram_config{.page_size = tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    dram_config.sharding_args = BufferShardingArgs(std::nullopt, TensorMemoryLayout::INTERLEAVED);

    distributed::ReplicatedBufferConfig buffer_config{.size = tile_size * num_tiles};
    auto src_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();

    KernelHandle brisc_print_kernel_id = CreateKernel(
        program_,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/print_tiles_multiple_brisc.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    bool is_tilized = (data_format == tt::DataFormat::Bfp8_b) || (data_format == tt::DataFormat::Bfp4_b);
    tt_metal::SetRuntimeArgs(
        program_, brisc_print_kernel_id, core, {dram_buffer_src_addr, (std::uint32_t)0, is_tilized, num_tiles});

    using tt::tt_metal::test::dprint::GenerateInputTileWithOffset;

    std::vector<uint32_t> u32_vec{};
    std::string expected_output_write;
    std::string expected_output_read;

    /* Generating input tiles so that all numbers are consecutive in the uint32_t representation.
       Tile with i=0 starts at 0, tile with i=1 starts at tile_size, etc.
    */
    for (int i = 0; i < num_tiles; i++) {
        std::vector<uint32_t> tile = GenerateInputTileWithOffset(data_format, i * elements_in_tile);

        u32_vec.insert(u32_vec.end(), tile.begin(), tile.end());

        using tt::tt_metal::test::dprint::GenerateExpectedData;
        std::string golden_output = GenerateExpectedData(data_format, tile);
        expected_output_write += fmt::format("Write tile {}:{}\n", i, golden_output);
        expected_output_read += fmt::format("Read tile {}:{}\n", i, golden_output);
    }

    distributed::WriteShard(cq, src_dram_buffer, u32_vec, zero_coord, true);
    fixture->RunProgram(mesh_device, workload);

    const auto* filename = "generated/dprint/device-0_worker-core-0-0_BRISC.txt";
    auto expected_output = expected_output_write + expected_output_read;

    EXPECT_TRUE(FilesMatchesString(filename, expected_output));
}

struct TestParams {
    tt::DataFormat data_format;
};

class PrintTilesMultipleFixture : public DPrintSeparateFilesFixture,
                                  public ::testing::WithParamInterface<TestParams> {};

INSTANTIATE_TEST_SUITE_P(
    PrintTilesMultipleTests,
    PrintTilesMultipleFixture,
    ::testing::Values(
        TestParams{tt::DataFormat::Float32},
        TestParams{tt::DataFormat::Float16_b},
        TestParams{tt::DataFormat::Bfp4_b},
        TestParams{tt::DataFormat::Bfp8_b},
        TestParams{tt::DataFormat::Int8},
        TestParams{tt::DataFormat::Int32},
        TestParams{tt::DataFormat::UInt8},
        TestParams{tt::DataFormat::UInt16},
        TestParams{tt::DataFormat::UInt32}),
    [](const ::testing::TestParamInfo<PrintTilesMultipleFixture::ParamType>& info) {
        return std::string(enchantum::to_string(info.param.data_format));
    });

TEST_P(PrintTilesMultipleFixture, TestPrintTilesMultiple) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, GetParam().data_format);
            },
            mesh_device);
    }
}

}  // namespace

}  // namespace CMAKE_UNIQUE_NAMESPACE
