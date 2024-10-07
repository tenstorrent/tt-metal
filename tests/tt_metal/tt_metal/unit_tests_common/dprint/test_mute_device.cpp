// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint_fixture.hpp"
#include "common/bfloat16.hpp"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A simple test for checking that disabling dprints on a device won't cause a hang.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

const std::string golden_output =
R"(Test Debug Print: Data0
Basic Types:
101-1.618@0.122559
e5551234569123456789
-17-343-44444-5123456789
SETPRECISION/FIXED/DEFAULTFLOAT:
3.1416
3.14159012
3.14159
3.141590118
SETW:
    123456123456  ab
HEX/OCT/DEC:
1e240361100123456
SLICE:
0.122558594 0.127929688 0.490234375 0.51171875
0.245117188 0.255859375 0.98046875 1.0234375
1.9609375 2.046875 7.84375 8.1875
3.921875 4.09375 15.6875 16.375
0.122558594 0.124511719 0.127929688 0.131835938 0.490234375 0.498046875 0.51171875 0.52734375
0.182617188 0.186523438 0.190429688 0.194335938 0.73046875 0.74609375 0.76171875 0.77734375
0.245117188 0.249023438 0.255859375 0.263671875 0.98046875 0.99609375 1.0234375 1.0546875
0.365234375 0.373046875 0.380859375 0.388671875 1.4609375 1.4921875 1.5234375 1.5546875
<TileSlice data truncated due to exceeding max count (32)>)";

static void RunTest(DPrintFixture* fixture, Device* device) {
    // Set up program and command queue
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    auto program = CreateScopedProgram();

    // Create a CB for testing TSLICE, dimensions are 32x32 bfloat16s
    constexpr uint32_t src0_cb_index = CB::c_in0;
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
    string file_name = fixture->dprint_file_name;
    EXPECT_TRUE(OpenFile(file_name, log_file, std::fstream::in));
    EXPECT_TRUE(log_file.peek() == std::ifstream::traits_type::eof());
}

TEST_F(DPrintFixtureDisableDevices, TestPrintMuteDevice) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(RunTest, device);
    }
}
