// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <variant>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/utils.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

//////////////////////////////////////////////////////////////////////////////////////////
// A simple test for checking that disabling dprints on a device won't cause a hang.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
void RunTest(DPrintFixture* fixture, IDevice* device) {
    // Set up program and command queue
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    Program program = Program();

    // Create a CB for testing TSLICE, dimensions are 32x32 bfloat16s
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t buffer_size = 32*32*sizeof(bfloat16);
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
        buffer_size,
        {{src0_cb_index, tt::DataFormat::Float16_b}}
    ).set_page_size(src0_cb_index, buffer_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // This kernel is enough to fill up the print buffer, even though the device is not being
    // printed from, we still need to drain the print buffer to prevent hanging the core.
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/brisc_print.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    // Run the program
    fixture->RunProgram(device, program);

    // Check that the log file is empty.
    std::fstream log_file;
    std::string file_name = fixture->dprint_file_name;
    EXPECT_TRUE(OpenFile(file_name, log_file, std::fstream::in));
    EXPECT_TRUE(log_file.peek() == std::ifstream::traits_type::eof());
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}

TEST_F(DPrintDisableDevicesFixture, TensixTestPrintMuteDevice) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunTest, device);
    }
}
