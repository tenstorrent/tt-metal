// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//////////////////////////////////////////////////////////////////////////////////////////
// Watcher Enablement Mode Tests
//
// This test file validates the watcher's selective enablement feature, which allows
// enabling watcher monitoring for specific kernel types (USER, DISPATCH, FABRIC).
//
// The file contains TWO complementary test suites:
//
// 1. WatcherEnablementModeTests (Non-USER modes):
//    - Tests: DISPATCH only, DISPATCH+FABRIC
//    - Verifies: User kernels do NOT trigger asserts when USER mode is disabled
//    - Test count: 4 tests (2 modes × 2 core types: TENSIX, ACTIVE_ETH)
//
// 2. WatcherUserEnabledTests (USER mode enabled):
//    - Tests: USER only, ALL (USER+DISPATCH+FABRIC)
//    - Verifies: User kernels DO trigger asserts when USER mode is enabled
//    - Test count: 4 tests (2 modes × 2 core types: TENSIX, ACTIVE_ETH)
//
// Together, these tests provide comprehensive coverage of the watcher enablement
// mode feature, ensuring it correctly enables/disables monitoring based on kernel type.
// Total: 8 focused test cases covering the key scenarios.
//////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>
#include <stdint.h>
#include <functional>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt_stl/assert.hpp>
#include "debug_tools_fixture.hpp"
#include "hal_types.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {

class MeshWatcherNonUserFixture : public MeshWatcherFixture {
public:
    llrt::WatcherEnablementMode previous_enablement_mode_{};
    llrt::WatcherEnablementMode test_enablement_mode_{};

    void SetEnablementMode(llrt::WatcherEnablementMode mode) { test_enablement_mode_ = mode; }

    void SetUp() override {
        previous_enablement_mode_ = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enablement_mode();

        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enablement_mode(test_enablement_mode_);

        log_info(
            LogTest,
            "Watcher enablement mode set to: 0x{:02x} (DISPATCH={}, FABRIC={}, USER={})",
            static_cast<uint8_t>(test_enablement_mode_),
            (test_enablement_mode_ & llrt::WatcherEnablementMode::DISPATCH),
            (test_enablement_mode_ & llrt::WatcherEnablementMode::FABRIC),
            (test_enablement_mode_ & llrt::WatcherEnablementMode::USER));

        MeshWatcherFixture::SetUp();
    }

    void TearDown() override {
        MeshWatcherFixture::TearDown();
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enablement_mode(previous_enablement_mode_);
    }
};

static void RunTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    HalProgrammableCoreType programmable_core_type,
    llrt::WatcherEnablementMode enablement_mode) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto device = mesh_device->get_devices()[0];

    // Choose a core to run the test on
    CoreCoord logical_core, virtual_core;
    switch (programmable_core_type) {
        case HalProgrammableCoreType::TENSIX:
            logical_core = {0, 0};
            virtual_core = device->worker_core_from_logical_core(logical_core);
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
            if (device->get_active_ethernet_cores(true).empty()) {
                log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_active_ethernet_cores(true).begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            break;
        case HalProgrammableCoreType::IDLE_ETH:
            if (device->get_inactive_ethernet_cores().empty()) {
                log_info(LogTest, "Skipping this test since device has no inactive ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_inactive_ethernet_cores().begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            break;
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported programmable core type");
    }

    log_info(LogTest, "Running test on device {} core {}...", device->id(), virtual_core.str());

    // Create a kernel that would normally trigger an assert
    KernelHandle assert_kernel;
    switch (programmable_core_type) {
        case HalProgrammableCoreType::TENSIX:
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{.noc = tt_metal::NOC::NOC_0});
            break;
        case HalProgrammableCoreType::IDLE_ETH:
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0});
            break;
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported programmable core type");
    }

    // Write runtime args that would normally trip an assert (a == b)
    // Since watcher is disabled for USER kernels, this should NOT trigger an assert
    const std::vector<uint32_t> assert_args = {3, 3, static_cast<uint32_t>(dev_msgs::DebugAssertTripped)};
    SetRuntimeArgs(program_, assert_kernel, logical_core, assert_args);

    // Run the kernel - it should complete successfully without triggering watcher assert
    // because watcher is only enabled for DISPATCH/FABRIC, not USER kernels
    log_info(LogTest, "Running user kernel with assert condition (a==b, would normally trip ASSERT)...");
    log_info(
        LogTest,
        "Since watcher is disabled for USER kernels (mode=0x{:02x}), no assert should trigger.",
        static_cast<uint8_t>(enablement_mode));

    fixture->RunProgram(mesh_device, workload, true);

    log_info(LogTest, "User kernel completed without triggering watcher assert.");

    // Verify that no watcher exception was raised
    std::string exception = MetalContext::instance().watcher_server()->exception_message();
    EXPECT_TRUE(exception.empty()) << "Expected no watcher exception, but got: " << exception;
}

// Test function for when USER mode IS enabled - expects assert to trigger
static void RunTestWithUserEnabled(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    HalProgrammableCoreType programmable_core_type,
    llrt::WatcherEnablementMode enablement_mode) {
    // Verify that USER mode IS enabled
    ASSERT_TRUE(enablement_mode & llrt::WatcherEnablementMode::USER)
        << "Test configuration error: USER mode should be enabled for this test";

    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto device = mesh_device->get_devices()[0];

    // Choose a core to run the test on
    CoreCoord logical_core, virtual_core;
    std::string risc;
    switch (programmable_core_type) {
        case HalProgrammableCoreType::TENSIX:
            logical_core = {0, 0};
            virtual_core = device->worker_core_from_logical_core(logical_core);
            risc = " brisc";
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
            if (device->get_active_ethernet_cores(true).empty()) {
                log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_active_ethernet_cores(true).begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            risc = "erisc";
            break;
        case HalProgrammableCoreType::IDLE_ETH:
            if (device->get_inactive_ethernet_cores().empty()) {
                log_info(LogTest, "Skipping this test since device has no inactive ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_inactive_ethernet_cores().begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            risc = "erisc";
            break;
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported programmable core type");
    }

    log_info(LogTest, "Running test on device {} core {}...", device->id(), virtual_core.str());

    // Create a kernel that will trigger an assert
    KernelHandle assert_kernel;
    switch (programmable_core_type) {
        case HalProgrammableCoreType::TENSIX:
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{.noc = tt_metal::NOC::NOC_0});
            break;
        case HalProgrammableCoreType::IDLE_ETH:
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0});
            break;
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported programmable core type");
    }

    // First, run with safe args to verify kernel works
    const std::vector<uint32_t> safe_args = {3, 4, static_cast<uint32_t>(dev_msgs::DebugAssertTripped)};
    SetRuntimeArgs(program_, assert_kernel, logical_core, safe_args);

    log_info(LogTest, "Running kernel with safe args (should complete normally)...");
    fixture->RunProgram(mesh_device, workload, true);
    log_info(LogTest, "Safe run completed successfully.");

    const std::vector<uint32_t> assert_args = {3, 3, static_cast<uint32_t>(dev_msgs::DebugAssertTripped)};
    SetRuntimeArgs(program_, assert_kernel, logical_core, assert_args);

    log_info(LogTest, "Running user kernel with assert condition (a==b, SHOULD trip ASSERT)...");
    log_info(
        LogTest,
        "Since watcher is enabled for USER kernels (mode=0x{:02x}), assert should trigger.",
        static_cast<uint8_t>(enablement_mode));

    fixture->RunProgram(mesh_device, workload);

    // We should be able to find the expected watcher error in the log
    const std::string kernel = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp";
    const uint32_t line_num = 67;
    std::string expected = fmt::format(
        "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} tripped an assert on line {}. "
        "Note that file name reporting is not yet implemented, and the reported line number for the assert may be "
        "from a different file. Current kernel: {}.",
        device->id(),
        (programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH) ? "acteth" : "worker",
        logical_core.x,
        logical_core.y,
        virtual_core.x,
        virtual_core.y,
        risc,
        line_num,
        kernel);

    std::string exception = "";
    do {
        exception = MetalContext::instance().watcher_server()->exception_message();
    } while (exception.empty());

    log_info(LogTest, "Watcher correctly caught the assert in user kernel.");
    EXPECT_EQ(expected, exception);
}

// Fixture for testing USER mode enabled
class MeshWatcherUserEnabledFixture : public MeshWatcherFixture {
public:
    llrt::WatcherEnablementMode previous_enablement_mode_{};
    llrt::WatcherEnablementMode test_enablement_mode_{};

    void SetEnablementMode(llrt::WatcherEnablementMode mode) { test_enablement_mode_ = mode; }

    void SetUp() override {
        previous_enablement_mode_ = tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enablement_mode();

        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enablement_mode(test_enablement_mode_);

        log_info(
            LogTest,
            "Watcher enablement mode set to: 0x{:02x} (DISPATCH={}, FABRIC={}, USER={})",
            static_cast<uint8_t>(test_enablement_mode_),
            (test_enablement_mode_ & llrt::WatcherEnablementMode::DISPATCH),
            (test_enablement_mode_ & llrt::WatcherEnablementMode::FABRIC),
            (test_enablement_mode_ & llrt::WatcherEnablementMode::USER));

        MeshWatcherFixture::SetUp();
    }

    void TearDown() override {
        MeshWatcherFixture::TearDown();
        tt::tt_metal::MetalContext::instance().rtoptions().set_watcher_enablement_mode(previous_enablement_mode_);
    }
};

}  // namespace CMAKE_UNIQUE_NAMESPACE

// Test parameters
struct WatcherEnablementModeTestParams {
    std::string test_name;
    HalProgrammableCoreType core_type;
    llrt::WatcherEnablementMode enablement_mode;
};

class WatcherEnablementModeTest : public CMAKE_UNIQUE_NAMESPACE::MeshWatcherNonUserFixture,
                                  public ::testing::WithParamInterface<WatcherEnablementModeTestParams> {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        this->SetEnablementMode(params.enablement_mode);
        MeshWatcherNonUserFixture::SetUp();
    }
};

TEST_P(WatcherEnablementModeTest, TestWatcherNonUserModeDoesNotAssertUserKernels) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& params = GetParam();

    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }

    this->RunTestOnDevice(
        [&params](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTest(fixture, mesh_device, params.core_type, params.enablement_mode);
        },
        this->devices_[0]);
}

namespace {

using enum HalProgrammableCoreType;
using llrt::WatcherEnablementMode;

// Test combinations: DISPATCH only and DISPATCH+FABRIC
// In all cases, USER mode should NOT be enabled, so user kernels should not trigger asserts
// Only test TENSIX and ACTIVE_ETH cores
INSTANTIATE_TEST_SUITE_P(
    WatcherEnablementModeTests,
    WatcherEnablementModeTest,
    ::testing::Values(
        // Test with DISPATCH only - user kernels should not assert
        WatcherEnablementModeTestParams{"Tensix_DispatchOnly", TENSIX, WatcherEnablementMode::DISPATCH},
        WatcherEnablementModeTestParams{"ActiveEth_DispatchOnly", ACTIVE_ETH, WatcherEnablementMode::DISPATCH},

        // Test with DISPATCH + FABRIC - user kernels should not assert
        WatcherEnablementModeTestParams{"Tensix_DispatchFabric", TENSIX, WatcherEnablementMode::DISPATCH_FABRIC},
        WatcherEnablementModeTestParams{
            "ActiveEth_DispatchFabric", ACTIVE_ETH, WatcherEnablementMode::DISPATCH_FABRIC}),
    [](const ::testing::TestParamInfo<WatcherEnablementModeTestParams>& info) { return info.param.test_name; });

}  // namespace

//////////////////////////////////////////////////////////////////////////////////////////
// Test suite for USER mode ENABLED - verifies asserts DO trigger
//////////////////////////////////////////////////////////////////////////////////////////

// Test parameters for USER-enabled tests
struct WatcherUserEnabledTestParams {
    std::string test_name;
    HalProgrammableCoreType core_type;
    llrt::WatcherEnablementMode enablement_mode;
};

class WatcherUserEnabledTest : public CMAKE_UNIQUE_NAMESPACE::MeshWatcherUserEnabledFixture,
                               public ::testing::WithParamInterface<WatcherUserEnabledTestParams> {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        this->SetEnablementMode(params.enablement_mode);
        MeshWatcherUserEnabledFixture::SetUp();
    }
};

TEST_P(WatcherUserEnabledTest, TestWatcherUserModeDoesAssertUserKernels) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& params = GetParam();

    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }

    this->RunTestOnDevice(
        [&params](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTestWithUserEnabled(fixture, mesh_device, params.core_type, params.enablement_mode);
        },
        this->devices_[0]);
}

namespace {

using enum HalProgrammableCoreType;
using llrt::WatcherEnablementMode;

// Test combinations with USER mode enabled - user kernels SHOULD trigger asserts
// Only test TENSIX and ACTIVE_ETH cores
INSTANTIATE_TEST_SUITE_P(
    WatcherUserEnabledTests,
    WatcherUserEnabledTest,
    ::testing::Values(
        // Test with USER only - user kernels should assert
        WatcherUserEnabledTestParams{"Tensix_UserOnly", TENSIX, WatcherEnablementMode::USER},
        WatcherUserEnabledTestParams{"ActiveEth_UserOnly", ACTIVE_ETH, WatcherEnablementMode::USER},

        // Test with ALL (USER + DISPATCH + FABRIC) - user kernels should assert
        WatcherUserEnabledTestParams{"Tensix_All", TENSIX, WatcherEnablementMode::ALL},
        WatcherUserEnabledTestParams{"ActiveEth_All", ACTIVE_ETH, WatcherEnablementMode::ALL}),
    [](const ::testing::TestParamInfo<WatcherUserEnabledTestParams>& info) { return info.param.test_name; });

}  // namespace
