// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <memory>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <variant>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "gtest/gtest.h"
#include "hal_types.hpp"
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

namespace CMAKE_UNIQUE_NAMESPACE {
const std::string golden_output_data0 =
    R"(Test Debug Print: Data0
Basic Types:
101-1.618@0.122559
e5551234569123456789
-17-343-44444-5123456789
Pointer:
123
456
789
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
)";
const std::string golden_output_compute =
    R"(Test Debug Print: Unpack
Basic Types:
101-1.618@0.122559
e5551234569123456789
-17-343-44444-5123456789
Pointer:
123
456
789
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
Pointer:
123
456
789
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
Pointer:
123
456
789
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
)";
const std::string golden_output_data1 =
    R"(Test Debug Print: Data1
Basic Types:
101-1.618@0.122559
e5551234569123456789
-17-343-44444-5123456789
Pointer:
123
456
789
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
)";

void RunTest(
    DPrintMeshFixture* fixture,
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    const std::string& golden_output) {
    // Set up program and command queue
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    auto& program_ = workload.get_programs().at(device_range);

    // Create a CB for testing TSLICE, dimensions are 32x32 bfloat16s
    constexpr uint32_t buffer_size = 32*32*sizeof(bfloat16);
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
        buffer_size,
        {{CBIndex::c_0, tt::DataFormat::Float16_b}}
    ).set_page_size(CBIndex::c_0, buffer_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

    // A CB with an unsupported data format
    CircularBufferConfig cb_src1_config = CircularBufferConfig(
        buffer_size,
        {{CBIndex::c_1, tt::DataFormat::Bfp2_b}}
    ).set_page_size(CBIndex::c_1, buffer_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_src1_config);

    // Three different kernels to mirror typical usage and some previously
    // failing test cases, although all three kernels simply print.
    CreateKernel(
        program_,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/brisc_print.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program_,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/ncrisc_print.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernel(
        program_,
        tt_metal::MetalContext::instance().rtoptions().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/trisc_print.cpp",
        core,
        ComputeConfig{});

    // Run the program
    fixture->RunProgram(mesh_device, workload);

    // Check that the expected print messages are in the log file
    EXPECT_TRUE(FilesMatchesString(DPrintMeshFixture::dprint_file_name, golden_output));
}

struct TestParams {
    std::string test_name;
    std::vector<HalProcessorIdentifier> enabled_processors;
    std::string golden_output;
};

using enum HalProgrammableCoreType;
using enum HalProcessorClassType;

class PrintAllHartsFixture : public DPrintMeshFixture, public ::testing::WithParamInterface<TestParams> {
private:
    HalProcessorSet original_enabled_processors_;

protected:
    void ExtraSetUp() override {
        original_enabled_processors_ = tt::tt_metal::MetalContext::instance().rtoptions().get_feature_processors(
            tt::llrt::RunTimeDebugFeatureDprint);
        HalProcessorSet processor_set;
        for (const auto& proc : GetParam().enabled_processors) {
            processor_set.add(
                proc.core_type,
                tt::tt_metal::MetalContext::instance().hal().get_processor_index(
                    proc.core_type, proc.processor_class, proc.processor_type));
        }
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_processors(
            tt::llrt::RunTimeDebugFeatureDprint, processor_set);
    }
    void ExtraTearDown() override {
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_processors(
            tt::llrt::RunTimeDebugFeatureDprint, original_enabled_processors_);
    }
};

INSTANTIATE_TEST_SUITE_P(
    PrintAllHartsTests,
    PrintAllHartsFixture,
    ::testing::Values(
        TestParams{
            "All",
            {
                {TENSIX, DM, 0},
                {TENSIX, DM, 1},
                {TENSIX, COMPUTE, 0},
                {TENSIX, COMPUTE, 1},
                {TENSIX, COMPUTE, 2},
            },
            golden_output_data0 + golden_output_compute + golden_output_data1,
        },
        TestParams{
            "Brisc",
            {
                {TENSIX, DM, 0},
            },
            golden_output_data0,
        },
        TestParams{
            "BriscCompute",
            {
                // DPRINT server timeout if BRISC is disabled.
                {TENSIX, DM, 0},
                {TENSIX, COMPUTE, 0},
                {TENSIX, COMPUTE, 1},
                {TENSIX, COMPUTE, 2},
            },
            golden_output_data0 + golden_output_compute,
        }),
    [](const ::testing::TestParamInfo<TestParams>& info) { return info.param.test_name; });

TEST_P(PrintAllHartsFixture, TensixTestPrint) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](DPrintMeshFixture* fixture, std::shared_ptr<distributed::MeshDevice> mesh_device) {
                RunTest(fixture, mesh_device, GetParam().golden_output);
            },
            mesh_device);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
