// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "gtest/gtest.h"
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

//////////////////////////////////////////////////////////////////////////////////////////
// A simple test for checking DPRINTs from all harts.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
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
<TileSlice data truncated due to exceeding max count (32)>
Tried printing CBIndex::c_1: Unsupported data format (Bfp2_b)
Test Debug Print: Unpack
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
<TileSlice data truncated due to exceeding max count (32)>
Tried printing CBIndex::c_1: Unsupported data format (Bfp2_b)
Test Debug Print: Math
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
Warning: MATH core does not support TileSlice printing, omitting print...
Warning: MATH core does not support TileSlice printing, omitting print...
Warning: MATH core does not support TileSlice printing, omitting print...
Test Debug Print: Pack
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
<TileSlice data truncated due to exceeding max count (32)>
Tried printing CBIndex::c_1: Unsupported data format (Bfp2_b)
Test Debug Print: Data1
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
<TileSlice data truncated due to exceeding max count (32)>
Tried printing CBIndex::c_1: Unsupported data format (Bfp2_b))";

void RunTest(DPrintFixture* fixture, IDevice* device) {
    // Set up program and command queue
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    Program program = Program();

    // Create a CB for testing TSLICE, dimensions are 32x32 bfloat16s
    constexpr uint32_t buffer_size = 32*32*sizeof(bfloat16);
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
        buffer_size,
        {{CBIndex::c_0, tt::DataFormat::Float16_b}}
    ).set_page_size(CBIndex::c_0, buffer_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // A CB with an unsupported data format
    CircularBufferConfig cb_src1_config = CircularBufferConfig(
        buffer_size,
        {{CBIndex::c_1, tt::DataFormat::Bfp2_b}}
    ).set_page_size(CBIndex::c_1, buffer_size);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    // Three different kernels to mirror typical usage and some previously
    // failing test cases, although all three kernels simply print.
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/brisc_print.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    KernelHandle ncrisc_print_kernel_id = CreateKernel(
        program,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/ncrisc_print.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    KernelHandle trisc_print_kernel_id = CreateKernel(
        program,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/trisc_print.cpp",
        core,
        ComputeConfig{});

    // Run the program
    fixture->RunProgram(device, program);

    // Check that the expected print messages are in the log file
    EXPECT_TRUE(
        FilesMatchesString(
            DPrintFixture::dprint_file_name,
            golden_output
        )
    );
}
}
}

TEST_F(DPrintFixture, TensixTestPrintFromAllHarts) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunTest, device);
    }
}
