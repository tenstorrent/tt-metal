// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/cluster_descriptor_types.h>
#include "gtest/gtest.h"
#include <map>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include "llrt.hpp"

namespace tt::tt_metal {

// A dispatch-agnostic test fixture using CRTP for per-class static instances
// If using Fast Dispatch, number of HW CQs is determined from the RTOptions
//
// Concrete classes must provide
// static bool WillSkip() : Returns true if the test should be skipped
// static std::string_view GetSkipMessage() : Returns a string describing the test requirements. This is used for the
// skip message
//
// For GTEST to properly instantiate a SetUpSuite and TearDownSuite for each suite, the concrete classes must
// each implement their own SetUpSuite and TearDownSuite. Calling DispatchFixture::SetUpTestSuite() and
// DispatchFixture::TearDownTestSuite() will throw an exception.
//
//
//
//
template <typename IMPL>
class DispatchFixture : public ::testing::Test {
private:
    // These get copied to the instance member variables in SetUp() for backward compatibility with existing tests
    inline static ARCH arch;
    inline static std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices;
    inline static std::vector<tt::tt_metal::IDevice*> devices;
    inline static bool slow_dispatch = false;

    //
    // Configuration
    //

    // Trace region size
    inline static uint32_t trace_region_size_ = DEFAULT_TRACE_REGION_SIZE;
    // L1 Small size
    inline static uint32_t l1_small_size_ = DEFAULT_L1_SMALL_SIZE;
    // Limit the number of devices to create
    inline static size_t num_devices_to_create_ = std::numeric_limits<size_t>::max();

public:
    //
    // Instance member variables for backward compatibility with existing tests
    // They are setup and reset in SetUp() and TearDown()
    // TODO: Remove these
    //

    // Devices
    std::vector<tt::tt_metal::IDevice*> devices_;
    // Device ID to Device Map
    std::map<chip_id_t, tt::tt_metal::IDevice*> reserved_devices_;
    // Convenience pointer to a single MMIO device
    tt::tt_metal::IDevice* device_ = nullptr;
    // Architecture of the devices
    ARCH arch_ = tt::ARCH::Invalid;
    // Number of devices
    size_t num_devices_ = 0;
    // Slow dispatch
    bool slow_dispatch_ = false;

    static void SetUpTestSuite() { TT_THROW("SetUpTestSuite not implemented in DispatchFixture"); }

    static void TearDownTestSuite() { TT_THROW("TearDownTestSuite not implemented in DispatchFixture"); }

    // Initialize the test suite with the given parameters
    static void DoSetUpTestSuite(
        uint32_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        uint32_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t num_devices_to_create = std::numeric_limits<size_t>::max()) {
        l1_small_size_ = l1_small_size;
        trace_region_size_ = trace_region_size;
        num_devices_to_create_ = num_devices_to_create;

        slow_dispatch = IsSlowDispatch();
        arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        std::vector<chip_id_t> chip_ids;
        for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
            chip_ids.push_back(id);
            if (chip_ids.size() >= num_devices_to_create) {
                break;
            }
        }

        auto dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();

        auto num_hw_cqs = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();

        if (arch == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() != 1 && num_hw_cqs > 1) {
            if (!tt::tt_metal::IsGalaxyCluster()) {
                log_warning(
                    tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in Setup()");
                dispatch_core_config = DispatchCoreType::ETH;
            }
        }

        reserved_devices = tt::tt_metal::detail::CreateDevices(
            chip_ids, num_hw_cqs, l1_small_size_, trace_region_size_, dispatch_core_config);
        for (const auto& [id, device] : reserved_devices) {
            devices.push_back(device);
        }
    }

    // Helper function to setup with a specific number of devices
    static void DoSetUpTestSuiteWithNumberOfDevices(size_t num_devices_to_create) {
        DoSetUpTestSuite(DEFAULT_TRACE_REGION_SIZE, DEFAULT_L1_SMALL_SIZE, num_devices_to_create);
    }

    // Helper function to setup with a specific trace region size
    static void DoSetUpTestSuiteWithTrace(uint32_t trace_region_size) {
        DoSetUpTestSuite(trace_region_size, DEFAULT_L1_SMALL_SIZE, std::numeric_limits<size_t>::max());
    }

    // Teardown the test suite
    static void DoTearDownTestSuite() {
        // Checking if devices are empty because DPrintFixture.TensixTestPrintFinish already
        // closed all devices
        if (!reserved_devices.empty()) {
            tt::tt_metal::detail::CloseDevices(reserved_devices);
            reserved_devices.clear();
            devices.clear();
        }
        arch = tt::ARCH::Invalid;
        num_devices_to_create_ = std::numeric_limits<size_t>::max();
        l1_small_size_ = DEFAULT_L1_SMALL_SIZE;
        trace_region_size_ = DEFAULT_TRACE_REGION_SIZE;
    }

    // A function to run a program, according to which dispatch mode is set.
    void RunProgram(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, const bool skip_finish = false) {
        const uint64_t program_id = program.get_id();
        if (slow_dispatch) {
            tt::tt_metal::detail::LaunchProgram(device, program);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueProgram(cq, program, false);
            if (!skip_finish) {
                tt::tt_metal::Finish(cq);
            }
        }
    }

    // Reset the devices to the original configuration for this test suite
    static void ResetTestSuite() {
        uint32_t previous_l1_small_size = l1_small_size_;
        uint32_t previous_trace_region_size = trace_region_size_;
        size_t previous_num_devices_to_create = num_devices_to_create_;
        DoTearDownTestSuite();
        DoSetUpTestSuite(previous_l1_small_size, previous_trace_region_size, previous_num_devices_to_create);
    }

    void FinishCommands(tt::tt_metal::IDevice* device) {
        if (!slow_dispatch) {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::Finish(cq);
        }
    }

    void WriteBuffer(
        tt::tt_metal::IDevice* device, std::shared_ptr<tt::tt_metal::Buffer> in_buffer, std::vector<uint32_t>& src_vec) {
        if (slow_dispatch) {
            tt::tt_metal::detail::WriteToBuffer(in_buffer, src_vec);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueWriteBuffer(cq, in_buffer, src_vec, false);
        }
    }

    void ReadBuffer(
        tt::tt_metal::IDevice* device,
        const std::shared_ptr<tt::tt_metal::Buffer>& out_buffer,
        std::vector<uint32_t>& dst_vec) {
        if (slow_dispatch) {
            tt::tt_metal::detail::ReadFromBuffer(out_buffer, dst_vec);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueReadBuffer(cq, out_buffer, dst_vec, true);
        }
    }

    int NumDevices() { return this->devices_.size(); }

    static bool IsSlowDispatch() { return getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr; }

    static std::vector<tt::tt_metal::IDevice*> GetDevices() { return devices; }

    static std::map<chip_id_t, tt::tt_metal::IDevice*> GetDevicesMap() { return reserved_devices; }

    void SetUp() override {
        if (IMPL::WillSkip()) {
            GTEST_SKIP() << IMPL::GetSkipMessage();
        }
        if (devices.size() > 0) {
            device_ = devices[0];
        }

        reserved_devices_ = reserved_devices;
        arch_ = arch;
        num_devices_ = devices.size();
        slow_dispatch_ = IsSlowDispatch();
        devices_ = devices;
    }

    void TearDown() override {
        device_ = nullptr;
        devices_.clear();
        reserved_devices_.clear();
        arch_ = tt::ARCH::Invalid;
        num_devices_ = 0;
    }

    void RunTestOnDevice(const std::function<void()>& run_function, tt::tt_metal::IDevice* device) {
        log_info(tt::LogTest, "Running test on device {}.", device->id());
        run_function();
        log_info(tt::LogTest, "Finished running test on device {}.", device->id());
    }

    void DetectDispatchMode() {
        if (slow_dispatch) {
            log_info(tt::LogTest, "Running test using Slow Dispatch");
        } else {
            log_info(tt::LogTest, "Running test using Fast Dispatch");
        }
    }
};

// Dispatch Agnostic Fixture that runs on all devices and all dispatch modes
class AnyDeviceDispatchFixture : public DispatchFixture<AnyDeviceDispatchFixture> {
public:
    static bool WillSkip() { return false; }

    static std::string_view GetSkipMessage() { return "This test does not skip"; }

    static void SetUpTestSuite() { DispatchFixture<AnyDeviceDispatchFixture>::DoSetUpTestSuite(); }

    static void TearDownTestSuite() { DispatchFixture<AnyDeviceDispatchFixture>::DoTearDownTestSuite(); }
};

}  // namespace tt::tt_metal
