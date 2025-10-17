// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/control_plane.hpp>

namespace tt::tt_fabric {
namespace fabric_router_tests {

// Helper class to manage topology switching with full device teardown/setup
class TopologySwapHelper {
public:
    static void CloseDevices(
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        log_info(tt::LogTest, "=== START: Closing devices (total: {}) ===", devices_map.size());
        
        int device_count = 0;
        for (auto& [id, device] : devices_map) {
            log_info(tt::LogTest, "  Closing device {}/{} (chip_id: {})", ++device_count, devices_map.size(), id);
            device->close();
            log_info(tt::LogTest, "  Device {} closed successfully", id);
        }
        
        log_info(tt::LogTest, "  Clearing devices_map...");
        devices_map.clear();
        log_info(tt::LogTest, "  devices_map cleared");
        
        log_info(tt::LogTest, "  Clearing devices vector...");
        devices.clear();
        log_info(tt::LogTest, "  devices vector cleared");
        
        log_info(tt::LogTest, "=== COMPLETE: All devices closed ===");
    }
    
    static void OpenDevicesWithFabricConfig(
        tt::tt_fabric::FabricConfig fabric_config,
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        log_info(tt::LogTest, "=== START: Opening devices with FabricConfig: {} ===", static_cast<int>(fabric_config));
        
        // Set fabric config before opening devices
        log_info(tt::LogTest, "  Setting fabric config...");
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
        
        tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode, std::nullopt);
        log_info(tt::LogTest, "  Fabric config set successfully");
        
        log_info(tt::LogTest, "  Getting available devices...");
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        log_info(tt::LogTest, "  Found {} available devices", num_devices);
        
        std::vector<chip_id_t> ids;
        ids.reserve(num_devices);
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        log_info(tt::LogTest, "  Device IDs prepared: {}", ids.size());
        
        log_info(tt::LogTest, "  Getting dispatch core config...");
        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
        log_info(tt::LogTest, "  Dispatch core config obtained");
        
        log_info(tt::LogTest, "  Creating unit meshes (this may take a while)...");
        devices_map = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
            ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);
        log_info(tt::LogTest, "  Unit meshes created successfully, {} devices in map", devices_map.size());
        
        log_info(tt::LogTest, "  Populating devices vector...");
        int device_count = 0;
        for (auto& [id, device] : devices_map) {
            devices.push_back(device);
            log_info(tt::LogTest, "    Added device {}/{} (chip_id: {}) to vector", ++device_count, devices_map.size(), id);
        }
        log_info(tt::LogTest, "  Devices vector populated with {} devices", devices.size());
        
        log_info(tt::LogTest, "=== COMPLETE: Devices opened successfully ===");
    }
    
    static void FullTopologySwap(
        tt::tt_fabric::FabricConfig fabric_config,
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        log_info(tt::LogTest, "======================================");
        log_info(tt::LogTest, "BEGIN TOPOLOGY SWAP TO: {}", GetFabricConfigName(fabric_config));
        log_info(tt::LogTest, "======================================");
        
        // Close existing devices
        log_info(tt::LogTest, "STEP 1/2: Closing existing devices");
        CloseDevices(devices_map, devices);
        
        // Open with new fabric config
        log_info(tt::LogTest, "STEP 2/2: Opening devices with new topology");
        OpenDevicesWithFabricConfig(fabric_config, devices_map, devices);
        
        log_info(tt::LogTest, "======================================");
        log_info(tt::LogTest, "TOPOLOGY SWAP COMPLETE");
        log_info(tt::LogTest, "======================================");
    }
    
    static void DisableFabric(
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        log_info(tt::LogTest, "=== DISABLING FABRIC ===");
        CloseDevices(devices_map, devices);
        log_info(tt::LogTest, "Setting fabric config to DISABLED...");
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        log_info(tt::LogTest, "Fabric disabled successfully");
    }
    
    static std::string GetFabricConfigName(tt::tt_fabric::FabricConfig config) {
        switch (config) {
            case tt::tt_fabric::FabricConfig::FABRIC_1D: return "FABRIC_1D (Linear)";
            case tt::tt_fabric::FabricConfig::FABRIC_2D: return "FABRIC_2D (Mesh)";
            case tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC: return "FABRIC_2D_DYNAMIC (Mesh Dynamic)";
            default: return "UNKNOWN";
        }
    }
};

// Simple test that validates topology is set correctly
void ValidateTopology(
    const std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices,
    tt::tt_fabric::FabricConfig expected_config) {
    
    if (devices.empty()) {
        return;
    }
    
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    
    // Map FabricConfig to expected Topology
    tt::tt_fabric::Topology expected_topology;
    switch (expected_config) {
        case tt::tt_fabric::FabricConfig::FABRIC_1D:
            expected_topology = tt::tt_fabric::Topology::Linear;
            break;
        case tt::tt_fabric::FabricConfig::FABRIC_2D:
        case tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC:
            expected_topology = tt::tt_fabric::Topology::Mesh;
            break;
        default:
            GTEST_FAIL() << "Unexpected fabric config";
    }
    
    EXPECT_EQ(topology, expected_topology) 
        << "Expected topology: " << static_cast<int>(expected_topology) 
        << ", got: " << static_cast<int>(topology);
}

// Stress test fixture that doesn't use static setup
class TopologySwapStressFixture : public ::testing::Test {
protected:
    std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_map_;
    std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    
    void SetUp() override {
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices < 2) {
            log_info(tt::LogTest, "Skipping topology swap stress tests as there are less than 2 devices available");
            GTEST_SKIP();
        }
        
        bool slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "Running topology swap stress tests with slow dispatch");
        } else {
            log_info(tt::LogTest, "Running topology swap stress tests with fast dispatch");
        }
    }
    
    void TearDown() override {
        TopologySwapHelper::DisableFabric(devices_map_, devices_);
    }
};

// Test 1: Alternating between 1D and 2D
TEST_F(TopologySwapStressFixture, Alternate1DAnd2D_10Swaps) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
    };
    
    for (size_t i = 0; i < configs.size(); i++) {
        log_info(tt::LogTest, "Swap {}/{}: {}", i + 1, configs.size(), 
            TopologySwapHelper::GetFabricConfigName(configs[i]));
        TopologySwapHelper::FullTopologySwap(configs[i], devices_map_, devices_);
        ValidateTopology(devices_, configs[i]);
    }
}

// Test 2: Alternating between 2D and 2D Dynamic
TEST_F(TopologySwapStressFixture, Alternate2DAnd2DDynamic_10Swaps) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
    };
    
    for (size_t i = 0; i < configs.size(); i++) {
        log_info(tt::LogTest, "Swap {}/{}: {}", i + 1, configs.size(), 
            TopologySwapHelper::GetFabricConfigName(configs[i]));
        TopologySwapHelper::FullTopologySwap(configs[i], devices_map_, devices_);
        ValidateTopology(devices_, configs[i]);
    }
}

// Test 3: Alternating between 1D and 2D Dynamic
TEST_F(TopologySwapStressFixture, Alternate1DAnd2DDynamic_10Swaps) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
    };
    
    for (size_t i = 0; i < configs.size(); i++) {
        log_info(tt::LogTest, "Swap {}/{}: {}", i + 1, configs.size(), 
            TopologySwapHelper::GetFabricConfigName(configs[i]));
        TopologySwapHelper::FullTopologySwap(configs[i], devices_map_, devices_);
        ValidateTopology(devices_, configs[i]);
    }
}

// Test 4: Cycling through all three topologies
TEST_F(TopologySwapStressFixture, CycleAll3Topologies_15Swaps) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
    };
    
    for (size_t i = 0; i < configs.size(); i++) {
        log_info(tt::LogTest, "Swap {}/{}: {}", i + 1, configs.size(), 
            TopologySwapHelper::GetFabricConfigName(configs[i]));
        TopologySwapHelper::FullTopologySwap(configs[i], devices_map_, devices_);
        ValidateTopology(devices_, configs[i]);
    }
}

// Test 5: Reverse cycling through all three topologies
TEST_F(TopologySwapStressFixture, ReverseCycleAll3Topologies_15Swaps) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
    };
    
    for (size_t i = 0; i < configs.size(); i++) {
        log_info(tt::LogTest, "Swap {}/{}: {}", i + 1, configs.size(), 
            TopologySwapHelper::GetFabricConfigName(configs[i]));
        TopologySwapHelper::FullTopologySwap(configs[i], devices_map_, devices_);
        ValidateTopology(devices_, configs[i]);
    }
}

// Test 6: Random pattern maximizing swaps - 20 swaps
TEST_F(TopologySwapStressFixture, RandomPattern_20Swaps) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
    };
    
    for (size_t i = 0; i < configs.size(); i++) {
        log_info(tt::LogTest, "Swap {}/{}: {}", i + 1, configs.size(), 
            TopologySwapHelper::GetFabricConfigName(configs[i]));
        TopologySwapHelper::FullTopologySwap(configs[i], devices_map_, devices_);
        ValidateTopology(devices_, configs[i]);
    }
}

// Test 7: Rapid swaps between 1D and 2D - 30 swaps
TEST_F(TopologySwapStressFixture, RapidAlternate1DAnd2D_30Swaps) {
    for (int i = 0; i < 30; i++) {
        auto config = (i % 2 == 0) ? tt::tt_fabric::FabricConfig::FABRIC_1D 
                                   : tt::tt_fabric::FabricConfig::FABRIC_2D;
        log_info(tt::LogTest, "Swap {}/30: {}", i + 1, TopologySwapHelper::GetFabricConfigName(config));
        TopologySwapHelper::FullTopologySwap(config, devices_map_, devices_);
        ValidateTopology(devices_, config);
    }
}

// Test 8: Rapid swaps between 2D and 2D Dynamic - 30 swaps
TEST_F(TopologySwapStressFixture, RapidAlternate2DAnd2DDynamic_30Swaps) {
    for (int i = 0; i < 30; i++) {
        auto config = (i % 2 == 0) ? tt::tt_fabric::FabricConfig::FABRIC_2D 
                                   : tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC;
        log_info(tt::LogTest, "Swap {}/30: {}", i + 1, TopologySwapHelper::GetFabricConfigName(config));
        TopologySwapHelper::FullTopologySwap(config, devices_map_, devices_);
        ValidateTopology(devices_, config);
    }
}

// Test 9: Rapid cycling through all three - 30 swaps
TEST_F(TopologySwapStressFixture, RapidCycleAll3Topologies_30Swaps) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
    };
    
    for (int i = 0; i < 30; i++) {
        auto config = configs[i % 3];
        log_info(tt::LogTest, "Swap {}/30: {}", i + 1, TopologySwapHelper::GetFabricConfigName(config));
        TopologySwapHelper::FullTopologySwap(config, devices_map_, devices_);
        ValidateTopology(devices_, config);
    }
}

// Test 10: Extended stress test - 50 swaps with all patterns
TEST_F(TopologySwapStressFixture, ExtendedStressTest_50Swaps) {
    std::vector<tt::tt_fabric::FabricConfig> pattern = {
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
    };
    
    for (int i = 0; i < 50; i++) {
        auto config = pattern[i % pattern.size()];
        log_info(tt::LogTest, "Swap {}/50: {}", i + 1, TopologySwapHelper::GetFabricConfigName(config));
        TopologySwapHelper::FullTopologySwap(config, devices_map_, devices_);
        ValidateTopology(devices_, config);
    }
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric

