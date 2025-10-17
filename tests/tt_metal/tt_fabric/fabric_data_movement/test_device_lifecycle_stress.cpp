// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include <chrono>
#include <thread>
#include <algorithm>

namespace tt::tt_fabric {
namespace fabric_router_tests {

// Helper class to manage device open/close without fabric
class DeviceLifecycleHelper {
public:
    static void OpenDevices(
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        log_debug(tt::LogTest, "Opening devices (no fabric config)...");
        
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<chip_id_t> ids;
        ids.reserve(num_devices);
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        
        devices_map = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
            ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);
        
        for (auto& [id, device] : devices_map) {
            devices.push_back(device);
        }
        
        log_debug(tt::LogTest, "Devices opened successfully ({} devices)", devices.size());
    }
    
    static void CloseDevices(
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        log_debug(tt::LogTest, "Closing devices...");
        for (auto& [id, device] : devices_map) {
            device->close();
        }
        devices_map.clear();
        devices.clear();
        log_debug(tt::LogTest, "Devices closed");
    }
    
    static void OpenCloseCycle(
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        OpenDevices(devices_map, devices);
        CloseDevices(devices_map, devices);
    }
};

// Fixture for device lifecycle stress testing (no fabric)
class DeviceLifecycleStressFixture : public ::testing::Test {
protected:
    std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_map_;
    std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    
    void SetUp() override {
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices < 1) {
            GTEST_SKIP() << "Test requires at least 1 device";
        }
        
        bool slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "Running device lifecycle stress tests with slow dispatch");
        } else {
            log_info(tt::LogTest, "Running device lifecycle stress tests with fast dispatch");
        }
    }
    
    void TearDown() override {
        // Clean up any remaining devices
        if (!devices_map_.empty()) {
            DeviceLifecycleHelper::CloseDevices(devices_map_, devices_);
        }
    }
};

// Test 1: Basic open/close cycle - 10 times
TEST_F(DeviceLifecycleStressFixture, BasicOpenClose_10Cycles) {
    for (int i = 0; i < 10; i++) {
        log_info(tt::LogTest, "Cycle {}/10: Opening and closing devices", i + 1);
        DeviceLifecycleHelper::OpenCloseCycle(devices_map_, devices_);
    }
}

// Test 2: Rapid open/close - 20 cycles
TEST_F(DeviceLifecycleStressFixture, RapidOpenClose_20Cycles) {
    for (int i = 0; i < 20; i++) {
        log_info(tt::LogTest, "Cycle {}/20: Opening and closing devices", i + 1);
        DeviceLifecycleHelper::OpenCloseCycle(devices_map_, devices_);
    }
}

// Test 3: Extended stress - 50 cycles
TEST_F(DeviceLifecycleStressFixture, ExtendedStress_50Cycles) {
    for (int i = 0; i < 50; i++) {
        log_info(tt::LogTest, "Cycle {}/50: Opening and closing devices", i + 1);
        DeviceLifecycleHelper::OpenCloseCycle(devices_map_, devices_);
    }
}

// Test 4: Open once, close, open again - verify repeatability
TEST_F(DeviceLifecycleStressFixture, RepeatableOpenClose_5Times) {
    for (int i = 0; i < 5; i++) {
        log_info(tt::LogTest, "Iteration {}/5: Open -> Close -> Open -> Close", i + 1);
        
        // First open
        DeviceLifecycleHelper::OpenDevices(devices_map_, devices_);
        EXPECT_FALSE(devices_.empty()) << "Devices should be opened on iteration " << i + 1;
        
        // First close
        DeviceLifecycleHelper::CloseDevices(devices_map_, devices_);
        EXPECT_TRUE(devices_.empty()) << "Devices should be closed on iteration " << i + 1;
        
        // Second open
        DeviceLifecycleHelper::OpenDevices(devices_map_, devices_);
        EXPECT_FALSE(devices_.empty()) << "Devices should reopen on iteration " << i + 1;
        
        // Second close
        DeviceLifecycleHelper::CloseDevices(devices_map_, devices_);
        EXPECT_TRUE(devices_.empty()) << "Devices should close again on iteration " << i + 1;
    }
}

// Test 5: Keep devices open, verify state, then close - 10 times
TEST_F(DeviceLifecycleStressFixture, OpenVerifyClose_10Times) {
    for (int i = 0; i < 10; i++) {
        log_info(tt::LogTest, "Cycle {}/10: Open, verify, close", i + 1);
        
        DeviceLifecycleHelper::OpenDevices(devices_map_, devices_);
        
        // Verify devices are actually open and accessible
        EXPECT_FALSE(devices_.empty());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        EXPECT_EQ(devices_.size(), num_devices) << "Should have all available devices";
        
        // Verify we can access device properties
        for (const auto& device : devices_) {
            EXPECT_NE(device, nullptr);
            EXPECT_FALSE(device->get_devices().empty());
        }
        
        DeviceLifecycleHelper::CloseDevices(devices_map_, devices_);
        EXPECT_TRUE(devices_.empty());
    }
}

// Test 6: Alternating long and short lived devices
TEST_F(DeviceLifecycleStressFixture, AlternateLongShortLifetime_15Cycles) {
    for (int i = 0; i < 15; i++) {
        if (i % 2 == 0) {
            // Quick open/close
            log_info(tt::LogTest, "Cycle {}/15: Quick open/close", i + 1);
            DeviceLifecycleHelper::OpenCloseCycle(devices_map_, devices_);
        } else {
            // Open, do verification, then close
            log_info(tt::LogTest, "Cycle {}/15: Open, verify state, close", i + 1);
            DeviceLifecycleHelper::OpenDevices(devices_map_, devices_);
            EXPECT_FALSE(devices_.empty());
            // Small delay to keep devices open longer
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            DeviceLifecycleHelper::CloseDevices(devices_map_, devices_);
        }
    }
}

// Test 7: Verify device IDs are consistent across open/close cycles
TEST_F(DeviceLifecycleStressFixture, ConsistentDeviceIDs_10Cycles) {
    std::vector<chip_id_t> expected_ids;
    
    // First cycle - capture device IDs
    DeviceLifecycleHelper::OpenDevices(devices_map_, devices_);
    for (const auto& [id, device] : devices_map_) {
        expected_ids.push_back(id);
    }
    std::sort(expected_ids.begin(), expected_ids.end());
    DeviceLifecycleHelper::CloseDevices(devices_map_, devices_);
    
    // Subsequent cycles - verify same IDs
    for (int i = 0; i < 10; i++) {
        log_info(tt::LogTest, "Cycle {}/10: Verifying consistent device IDs", i + 1);
        
        DeviceLifecycleHelper::OpenDevices(devices_map_, devices_);
        
        std::vector<chip_id_t> current_ids;
        for (const auto& [id, device] : devices_map_) {
            current_ids.push_back(id);
        }
        std::sort(current_ids.begin(), current_ids.end());
        
        EXPECT_EQ(current_ids, expected_ids) 
            << "Device IDs should be consistent across cycles";
        
        DeviceLifecycleHelper::CloseDevices(devices_map_, devices_);
    }
}

// Test 8: Maximum stress - 100 rapid cycles
TEST_F(DeviceLifecycleStressFixture, MaximumStress_100Cycles) {
    for (int i = 0; i < 100; i++) {
        if (i % 10 == 0) {
            log_info(tt::LogTest, "Cycle {}/100: Opening and closing devices", i + 1);
        }
        DeviceLifecycleHelper::OpenCloseCycle(devices_map_, devices_);
    }
    log_info(tt::LogTest, "Completed 100 device open/close cycles successfully");
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric

