// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
// A simple test for checking DPRINTs from all harts.
// Note that in this test the kernels print only 1 tile.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {

namespace {

std::vector<std::string> GenerateGoldenOutput(tt::DataFormat data_format, std::vector<uint32_t>& input_tile) {
    using tt::tt_metal::test::dprint::GenerateExpectedData;
    std::string data = GenerateExpectedData(data_format, input_tile);
    std::vector<std::string> expected;
    expected.push_back(fmt::format("Print tile from Data0:{}", data));
    expected.push_back(fmt::format("Print tile from Data1:{}", data));
    expected.push_back(fmt::format("Print tile from Unpack:{}", data));
    expected.push_back(fmt::format(
        "Print tile from Math:\nWarning: MATH core does not support TileSlice printing, omitting print..."));
    expected.push_back(fmt::format("Print tile from Pack:{}", data));
    return expected;
}

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

    // Create an input CB with the right data format
    uint32_t tile_size = tt::tile_size(data_format);
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(tile_size, {{CBIndex::c_0, data_format}}).set_page_size(CBIndex::c_0, tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);
    CircularBufferConfig cb_intermed_config =
        CircularBufferConfig(tile_size, {{CBIndex::c_1, data_format}}).set_page_size(CBIndex::c_1, tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_intermed_config);

    // Dram buffer to send data to, device will read it out of here to print
    distributed::DeviceLocalBufferConfig dram_config{.page_size = tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = tile_size};
    auto src_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();

    // Create kernels on device
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program_,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/print_tile_brisc.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    KernelHandle ncrisc_print_kernel_id = CreateKernel(
        program_,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/print_tile_ncrisc.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    KernelHandle trisc_print_kernel_id = CreateKernel(
        program_,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/print_tile_trisc.cpp",
        core,
        ComputeConfig{});

    // BRISC kernel needs dram info via rtargs, every risc needs to know if data is tilized
    bool is_tilized = (data_format == tt::DataFormat::Bfp8_b) || (data_format == tt::DataFormat::Bfp4_b);
    tt_metal::SetRuntimeArgs(
        program_, brisc_print_kernel_id, core, {dram_buffer_src_addr, (std::uint32_t)0, is_tilized});
    tt_metal::SetRuntimeArgs(program_, ncrisc_print_kernel_id, core, {is_tilized});
    tt_metal::SetRuntimeArgs(program_, trisc_print_kernel_id, core, {is_tilized});

    using tt::tt_metal::test::dprint::GenerateInputTile;
    // Create input tile
    std::vector<uint32_t> u32_vec = GenerateInputTile(data_format);
    /*for (int idx = 0; idx < u32_vec.size(); idx+= 16) {
        string tmp = fmt::format("data[{:#03}:{:#03}]:", idx - 1, idx - 16);
        for (int i = 0; i < 16; i++)
            tmp += fmt::format(" 0x{:08x}", u32_vec[idx + 15 - i]);
        log_info(tt::LogTest, "{}", tmp);
    }*/

    // Send input tile to dram
    distributed::WriteShard(cq, src_dram_buffer, u32_vec, zero_coord);

    // Run the program
    fixture->RunProgram(mesh_device, workload);

    // Check against expected prints
    auto expected = GenerateGoldenOutput(data_format, u32_vec);
    DPrintSeparateFilesFixture::check_output(expected);
}

struct TestParams {
    tt::DataFormat data_format;
};

class PrintTileFixture : public DPrintSeparateFilesFixture, public ::testing::WithParamInterface<TestParams> {};

INSTANTIATE_TEST_SUITE_P(
    PrintTileTests,
    PrintTileFixture,
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
    [](const ::testing::TestParamInfo<PrintTileFixture::ParamType>& info) {
        return std::string(enchantum::to_string(info.param.data_format));
    });

TEST_P(PrintTileFixture, TestPrintTile) {
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
