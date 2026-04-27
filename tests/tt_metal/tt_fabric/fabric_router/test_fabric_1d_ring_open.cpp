// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <llrt/tt_cluster.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "common/tt_backend_api_types.hpp"

namespace tt::tt_fabric::test_1d_ring {

// Test fixture for FABRIC_1D_RING configuration
class Fabric1DRingFixture : public ::testing::Test {
public:
    inline static tt::ARCH arch_;
    inline static std::map<ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_map_;
    inline static std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    inline static bool should_skip_ = false;

protected:
    static void SetUpTestSuite() {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            log_info(tt::LogTest, "This test requires TT_METAL_SLOW_DISPATCH_MODE to be set");
        }

        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices < 2) {
            log_info(tt::LogTest, "Skipping: requires at least 2 devices, found {}", num_devices);
            should_skip_ = true;
            return;
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        // Set up device IDs
        std::vector<ChipId> ids;
        ids.reserve(num_devices);
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }

        // Use FABRIC_1D_RING config - this is the key config being tested
        tt::tt_fabric::FabricReliabilityMode reliability_mode =
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;

        log_info(tt::LogTest, "Setting FabricConfig to FABRIC_1D_RING");
        tt::tt_fabric::SetFabricConfig(
            tt::tt_fabric::FabricConfig::FABRIC_1D_RING,
            reliability_mode,
            std::nullopt,  // num_routing_planes
            tt::tt_fabric::FabricTensixConfig::DISABLED,
            tt::tt_fabric::FabricUDMMode::DISABLED);

        const auto& dispatch_core_config =
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();

        log_info(tt::LogTest, "Creating unit meshes with FABRIC_1D_RING...");
        devices_map_ = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
            ids,
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            1,
            dispatch_core_config,
            {},
            DEFAULT_WORKER_L1_SIZE);

        for (auto& [id, device] : devices_map_) {
            devices_.push_back(device);
        }
        log_info(tt::LogTest, "Successfully created {} mesh devices", devices_.size());
    }

    static void TearDownTestSuite() {
        if (should_skip_) {
            return;
        }
        for (auto& [id, device] : devices_map_) {
            device->close();
        }
        devices_map_.clear();
        devices_.clear();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }

    void SetUp() override {
        if (should_skip_) {
            GTEST_SKIP() << "Skipping test - insufficient devices or setup failed";
        }
    }

    const std::vector<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& get_devices() const {
        return devices_;
    }
};

// Simple test that just verifies devices were opened successfully
TEST_F(Fabric1DRingFixture, OpenDevicesWithFabric1DRing) {
    ASSERT_FALSE(devices_.empty()) << "No devices were created";
    log_info(tt::LogTest, "Test passed: {} devices opened with FABRIC_1D_RING", devices_.size());

    // Log some basic device info
    for (const auto& device : devices_) {
        auto physical_devices = device->get_devices();
        for (const auto* phys_dev : physical_devices) {
            log_info(tt::LogTest, "  Physical device ID: {}", phys_dev->id());
        }
    }
}

}  // namespace tt::tt_fabric::test_1d_ring
