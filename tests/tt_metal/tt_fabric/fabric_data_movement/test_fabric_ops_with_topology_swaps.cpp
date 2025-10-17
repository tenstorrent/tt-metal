// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/buffer.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/control_plane.hpp>
#include "fabric_fixture.hpp"
#include "utils.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

// Helper to setup and teardown topologies dynamically with full device lifecycle
class DynamicTopologyManager {
public:
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
    
    static void OpenDevicesWithFabricConfig(
        tt::tt_fabric::FabricConfig fabric_config,
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        log_debug(tt::LogTest, "Opening devices with FabricConfig: {}", static_cast<int>(fabric_config));
        
        // Set fabric config before opening devices
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
        
        tt::tt_fabric::SetFabricConfig(fabric_config, reliability_mode, std::nullopt);
        
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
        
        log_debug(tt::LogTest, "Devices opened successfully");
    }
    
    static void FullTopologySwap(
        tt::tt_fabric::FabricConfig fabric_config,
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        // Close existing devices
        CloseDevices(devices_map, devices);
        
        // Open with new fabric config
        OpenDevicesWithFabricConfig(fabric_config, devices_map, devices);
    }
    
    static void DisableFabric(
        std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices_map,
        std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
        
        CloseDevices(devices_map, devices);
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
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

// Fixture for testing fabric operations with topology swaps
class FabricOpsWithTopologySwapsFixture : public ::testing::Test {
protected:
    std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_map_;
    std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    bool slow_dispatch_;
    
    void SetUp() override {
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices < 2) {
            GTEST_SKIP() << "Test requires at least 2 devices, found " << num_devices;
        }
        
        slow_dispatch_ = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        log_info(tt::LogTest, "Running fabric ops with topology swaps - slow_dispatch: {}", slow_dispatch_);
    }
    
    void TearDown() override {
        DynamicTopologyManager::DisableFabric(devices_map_, devices_);
    }
    
    // Simple sanity check operation
    void RunSimpleSanityCheck() {
        if (devices_.size() < 2) {
            return;
        }
        
        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        const auto& device0 = devices_[0];
        const auto& device1 = devices_[1];
        
        auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(
            device0->get_devices()[0]->id());
        auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(
            device1->get_devices()[0]->id());
        
        // Just verify that routing direction exists
        auto direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
        EXPECT_TRUE(direction.has_value()) 
            << "No routing direction found from device 0 to device 1";
    }
};

// Test 1: Alternate 1D <-> 2D with sanity checks
TEST_F(FabricOpsWithTopologySwapsFixture, Alternate1DAnd2DWithSanityChecks) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
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
            DynamicTopologyManager::GetFabricConfigName(configs[i]));
        DynamicTopologyManager::FullTopologySwap(configs[i], devices_map_, devices_);
        RunSimpleSanityCheck();
    }
}

// Test 2: Alternate 2D <-> 2D Dynamic with sanity checks
TEST_F(FabricOpsWithTopologySwapsFixture, Alternate2DAnd2DDynamicWithSanityChecks) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
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
            DynamicTopologyManager::GetFabricConfigName(configs[i]));
        DynamicTopologyManager::FullTopologySwap(configs[i], devices_map_, devices_);
        RunSimpleSanityCheck();
    }
}

// Test 3: Cycle through all three topologies with sanity checks
TEST_F(FabricOpsWithTopologySwapsFixture, CycleAll3TopologiesWithSanityChecks) {
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
    };
    
    for (size_t i = 0; i < configs.size(); i++) {
        log_info(tt::LogTest, "Swap {}/{}: {}", i + 1, configs.size(), 
            DynamicTopologyManager::GetFabricConfigName(configs[i]));
        DynamicTopologyManager::FullTopologySwap(configs[i], devices_map_, devices_);
        RunSimpleSanityCheck();
    }
}

// Test 4: Rapid swaps between 1D and 2D - 20 iterations
TEST_F(FabricOpsWithTopologySwapsFixture, RapidSwaps1DAnd2D_20Iterations) {
    for (int i = 0; i < 20; i++) {
        auto config = (i % 2 == 0) ? tt::tt_fabric::FabricConfig::FABRIC_1D 
                                   : tt::tt_fabric::FabricConfig::FABRIC_2D;
        log_info(tt::LogTest, "Swap {}/20: {}", i + 1, DynamicTopologyManager::GetFabricConfigName(config));
        DynamicTopologyManager::FullTopologySwap(config, devices_map_, devices_);
        RunSimpleSanityCheck();
    }
}

// Test 5: Rapid swaps between 2D and 2D Dynamic - 20 iterations
TEST_F(FabricOpsWithTopologySwapsFixture, RapidSwaps2DAnd2DDynamic_20Iterations) {
    for (int i = 0; i < 20; i++) {
        auto config = (i % 2 == 0) ? tt::tt_fabric::FabricConfig::FABRIC_2D 
                                   : tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC;
        log_info(tt::LogTest, "Swap {}/20: {}", i + 1, DynamicTopologyManager::GetFabricConfigName(config));
        DynamicTopologyManager::FullTopologySwap(config, devices_map_, devices_);
        RunSimpleSanityCheck();
    }
}

// Test 6: Rapid swaps between 1D and 2D Dynamic - 20 iterations
TEST_F(FabricOpsWithTopologySwapsFixture, RapidSwaps1DAnd2DDynamic_20Iterations) {
    for (int i = 0; i < 20; i++) {
        auto config = (i % 2 == 0) ? tt::tt_fabric::FabricConfig::FABRIC_1D 
                                   : tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC;
        log_info(tt::LogTest, "Swap {}/20: {}", i + 1, DynamicTopologyManager::GetFabricConfigName(config));
        DynamicTopologyManager::FullTopologySwap(config, devices_map_, devices_);
        RunSimpleSanityCheck();
    }
}

// Test 7: Maximum chaos - random pattern with maximum swaps (25 iterations)
TEST_F(FabricOpsWithTopologySwapsFixture, MaximumChaos_25Iterations) {
    std::vector<tt::tt_fabric::FabricConfig> configs = {
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
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
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
    };
    
    for (size_t i = 0; i < configs.size(); i++) {
        log_info(tt::LogTest, "Swap {}/{}: {}", i + 1, configs.size(), 
            DynamicTopologyManager::GetFabricConfigName(configs[i]));
        DynamicTopologyManager::FullTopologySwap(configs[i], devices_map_, devices_);
        RunSimpleSanityCheck();
    }
}

// Test 8: Extended stress test - 40 iterations
TEST_F(FabricOpsWithTopologySwapsFixture, ExtendedStress_40Iterations) {
    std::vector<tt::tt_fabric::FabricConfig> pattern = {
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
        tt::tt_fabric::FabricConfig::FABRIC_1D,
        tt::tt_fabric::FabricConfig::FABRIC_2D,
    };
    
    for (int i = 0; i < 40; i++) {
        auto config = pattern[i % pattern.size()];
        log_info(tt::LogTest, "Swap {}/40: {}", i + 1, DynamicTopologyManager::GetFabricConfigName(config));
        DynamicTopologyManager::FullTopologySwap(config, devices_map_, devices_);
        RunSimpleSanityCheck();
    }
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric

