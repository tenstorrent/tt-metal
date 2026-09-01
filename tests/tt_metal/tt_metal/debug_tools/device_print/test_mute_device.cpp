// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <variant>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

//////////////////////////////////////////////////////////////////////////////////////////
// A simple test for checking that disabling DPRINTs on a device won't cause a hang.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

// For usage by tests that need the DPRINT server devices disabled.
class DevicePrintDisableMeshDevicesFixture : public DevicePrintFixture {
protected:
    void ExtraSetUp() override {
        // For this test, mute each devices using the environment variable
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_all_chips(
            tt::llrt::RunTimeDebugFeatureDprint, false);
        tt::tt_metal::MetalContext::instance().rtoptions().set_feature_chip_ids(
            tt::llrt::RunTimeDebugFeatureDprint, {});
    }
    void ExtraTearDown() override {
        MetalContext::instance()
            .teardown();  // Teardown dprint server so we can re-init later with all devices enabled again
    }
};

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
void RunTest(DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // This kernel is enough to fill up the print buffer, even though the device is not being
    // printed from, we still need to drain the print buffer to prevent hanging the core.
    //
    // (There used to be a circular buffer here "for testing TSLICE", but print_all_argument_sizes.cpp
    // contains nothing but DEVICE_PRINT calls and never touched it, so it was dead weight.)
    fixture->RunProgram(mesh_device, "tests/tt_metal/tt_metal/test_kernels/device_print/print_all_argument_sizes.cpp");

    // Check that the log file is empty.
    std::fstream log_file;
    std::string file_name = fixture->dprint_file_name;
    EXPECT_TRUE(OpenFile(file_name, log_file, std::fstream::in));
    EXPECT_TRUE(log_file.peek() == std::ifstream::traits_type::eof());
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(DevicePrintDisableMeshDevicesFixture, TensixTestPrintMuteDevice) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunTest, mesh_device);
    }
}
