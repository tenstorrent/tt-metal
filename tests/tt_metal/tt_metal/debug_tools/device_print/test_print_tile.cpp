// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
#include <tt-metalium/kernel_types.hpp>
#include "debug_tools_fixture.hpp"
#include "print_tile_helpers.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
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

// What BRISC prints, given the expected text of the hw0_32_8 slice and the tile.
using BriscGolden = std::function<std::string(const std::string& data, const std::vector<uint32_t>& input_tile)>;

// The text TileSlice prints for rows [h0, h1) step hs and columns [w0, w1) step ws of a Float16_b tile that isn't
// tilized, each value formatted with `spec`.
std::string Float16bSliceText(
    const std::vector<uint32_t>& input_tile,
    uint32_t h0,
    uint32_t h1,
    uint32_t hs,
    uint32_t w0,
    uint32_t w1,
    uint32_t ws,
    const std::string& spec = "{}") {
    const std::vector<bfloat16> values = unpack_uint32_vec_into_bfloat16_vec(input_tile);
    std::string text;
    for (uint32_t h = h0; h < h1; h += hs) {
        text += '\n';
        for (uint32_t w = w0; w < w1; w += ws) {
            if (w != w0) {
                text += ' ';
            }
            text += fmt::format(fmt::runtime(spec), static_cast<float>(values[(h * 32) + w]));
        }
    }
    return text;
}

// The line the host prints before a print whose {:x} doesn't suit its Float16_b values, which then print as {} would.
constexpr const char* kSpecNotSuitingFloat16bWarning =
    "Warning: invalid format specifier {:x} for Float16_b data, printed with {}\n";

std::vector<std::string> GenerateGoldenOutput(
    tt::DataFormat data_format, std::vector<uint32_t>& input_tile, const BriscGolden& brisc_golden = {}) {
    using tt::tt_metal::test::dprint::GenerateExpectedData;
    std::string data = GenerateExpectedData(data_format, input_tile);
    // The kernels print "...:\n{}\n". The slice (or the warning that replaces it) ends with a newline, so the format's
    // final \n leaves an empty line.
    std::vector<std::string> expected;
    expected.push_back(
        brisc_golden ? brisc_golden(data, input_tile) : fmt::format("Print tile from Data0:{}\n\n", data));
    expected.push_back(fmt::format("Print tile from Data1:{}\n\n", data));
    expected.push_back(fmt::format("Print tile from Unpack:{}\n\n", data));
    expected.push_back(fmt::format(
        "Print tile from Math:\nWarning: MATH core does not support TileSlice printing, omitting print...\n\n"));
    expected.push_back(fmt::format("Print tile from Pack:{}\n\n", data));
    return expected;
}

// Runs the BRISC, NCRISC and TRISC tile kernels on one tile of `data_format`. `brisc_kernel` and `brisc_golden` replace
// print_tile_brisc.cpp and what it prints; such a kernel reads the tile into c_in0 the same way.
void RunTest(
    DevicePrintSeparateFilesFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    tt::DataFormat data_format,
    const std::string& brisc_kernel = "print_tile_brisc.cpp",
    const BriscGolden& brisc_golden = {}) {
    // Set up program + CQ, run on just one core
    constexpr CoreCoord core = {0, 0};
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram(*mesh_device);
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
            "tests/tt_metal/tt_metal/test_kernels/device_print/" + brisc_kernel,
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    KernelHandle ncrisc_print_kernel_id = CreateKernel(
        program_,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/device_print/print_tile_ncrisc.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    KernelHandle trisc_print_kernel_id = CreateKernel(
        program_,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/device_print/print_tile_trisc.cpp",
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

    // Send input tile to dram
    distributed::WriteShard(cq, src_dram_buffer, u32_vec, zero_coord);

    // Run the program
    const ChipId device_id = mesh_device->get_device_ids()[0];
    DevicePrintSeparateFilesFixture::remove_output_files(device_id);
    fixture->RunProgram(mesh_device, workload);

    // Check against expected prints
    auto expected = GenerateGoldenOutput(data_format, u32_vec, brisc_golden);
    DevicePrintSeparateFilesFixture::check_output(device_id, expected);
}

struct TestParams {
    tt::DataFormat data_format;
};

class DevicePrintTileFixture : public DevicePrintSeparateFilesFixture,
                               public ::testing::WithParamInterface<TestParams> {};

INSTANTIATE_TEST_SUITE_P(
    DevicePrintTileTests,
    DevicePrintTileFixture,
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
    [](const ::testing::TestParamInfo<DevicePrintTileFixture::ParamType>& info) {
        return std::string(enchantum::to_string(info.param.data_format));
    });

TEST_P(DevicePrintTileFixture, TestPrintTile) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [&](DevicePrintSeparateFilesFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, GetParam().data_format);
            },
            mesh_device);
    }
}

class DevicePrintTileKnownIssueFixture : public DevicePrintSeparateFilesFixture {
public:
    void RunTestOnEveryDevice(
        tt::DataFormat data_format, const std::string& brisc_kernel, const BriscGolden& brisc_golden = {}) {
        for (auto& mesh_device : this->devices_) {
            this->RunTestOnDevice(
                [&](DevicePrintSeparateFilesFixture* fixture,
                    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                    RunTest(fixture, mesh_device, data_format, brisc_kernel, brisc_golden);
                },
                mesh_device);
        }
    }
};

// Known issue: TileSlice's size is an int but the generic serialization's is a uint32_t, so only TileSlice<64> and
// TileSlice<128>, which have their own, can be printed; today this kernel fails to compile with "No serialization
// defined for this type". Disabled until that is fixed.
TEST_F(DevicePrintTileKnownIssueFixture, DISABLED_PrintTileSliceSizes) {
    RunTestOnEveryDevice(
        tt::DataFormat::Float16_b,
        "print_tile_slice_sizes.cpp",
        [](const std::string& data, const std::vector<uint32_t>& input_tile) {
            return fmt::format(
                "TileSlice<100>:{}\n\nTileSlice<252>:{}\n\n", data, Float16bSliceText(input_tile, 0, 28, 4, 0, 32, 2));
        });
}

// Known issue: TileSlice has no default constructor outside tt-llk's test environment, so today tile_slice_from_l1
// fails to compile. Disabled until that is fixed.
TEST_F(DevicePrintTileKnownIssueFixture, DISABLED_PrintTileSliceFromL1) {
    RunTestOnEveryDevice(tt::DataFormat::Bfp8_b, "print_tile_slice_from_l1.cpp");
}

// Known issue: today the host hands the spec to fmt unchecked, fmt throws, and none of the run's output is written,
// from any RISC. Disabled until that is fixed.
TEST_F(DevicePrintTileKnownIssueFixture, DISABLED_PrintTileWithSpecNotSuitingData) {
    RunTestOnEveryDevice(
        tt::DataFormat::Float16_b,
        "print_tile_spec_not_suiting_data.cpp",
        [](const std::string& data, const std::vector<uint32_t>& input_tile) {
            return "Fixed:" + Float16bSliceText(input_tile, 0, 32, 8, 0, 32, 8, "{:.3f}") + "\n\n" +
                   kSpecNotSuitingFloat16bWarning + "Hex:" + data + "\n\n";
        });
}

// What print_tile_after_spec_not_suiting_data.cpp prints.
std::string GoesOnAfterSpecNotSuitingDataGolden(const std::string& data, const std::vector<uint32_t>&) {
    return "Before\n" + std::string(kSpecNotSuitingFloat16bWarning) + "Hex:" + data + "\n\nAfter\n";
}

// Known issue: today a print the host can't format stops the print server. That RISC's later prints, the other RISCs'
// prints and every later program's prints are lost, in fast dispatch the earlier ones too, and nothing is reported. The
// second run checks a later program. Disabled until that is fixed.
TEST_F(DevicePrintTileKnownIssueFixture, DISABLED_PrintGoesOnAfterSpecNotSuitingData) {
    for (int run = 0; run < 2; run++) {
        RunTestOnEveryDevice(
            tt::DataFormat::Float16_b,
            "print_tile_after_spec_not_suiting_data.cpp",
            GoesOnAfterSpecNotSuitingDataGolden);
    }
}

// Known issue: the same as users run it, outside test mode. Today the print server's thread rethrows fmt's exception,
// which ends the whole process, so run this test on its own. Disabled until that is fixed.
TEST_F(DevicePrintTileKnownIssueFixture, DISABLED_PrintGoesOnAfterSpecNotSuitingDataOutsideTestMode) {
    MetalContext::instance().rtoptions().set_test_mode_enabled(false);
    for (int run = 0; run < 2; run++) {
        RunTestOnEveryDevice(
            tt::DataFormat::Float16_b,
            "print_tile_after_spec_not_suiting_data.cpp",
            GoesOnAfterSpecNotSuitingDataGolden);
    }
}

}  // namespace

}  // namespace CMAKE_UNIQUE_NAMESPACE
