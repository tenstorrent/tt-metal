// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <tt_stl/indestructible.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

/**
 * @brief Singleton class for managing fabric switch mesh device setup and teardown.
 *
 * This class provides setup() and teardown() methods to create and close devices
 * specifically for switch meshes. It ensures proper fabric handshake between tests
 * by managing device lifecycle.
 */
class FabricSwitchManager {
public:
    /**
     * @brief Get the singleton instance of FabricSwitchManager.
     * @return Reference to the singleton instance.
     */
    static FabricSwitchManager& instance();

    /**
     * @brief Create devices for switch meshes.
     *
     * This method creates devices for all switch meshes using the switch mesh device IDs
     * from the control plane. It initializes the device manager with appropriate configuration
     * for switch mesh devices.
     *
     * @param fabric_config The fabric configuration to use for switch mesh devices.
     *                      The workload calling this method knows which fabric config it needs.
     * @param fabric_reliability_mode The fabric reliability mode to use. Defaults to strict mode.
     */
    void setup(
        FabricConfig fabric_config,
        FabricReliabilityMode fabric_reliability_mode = FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

    /**
     * @brief Close devices for switch meshes.
     *
     * This method closes devices that belong to switch meshes to ensure proper fabric
     * handshake between tests. This is critical because fabric routers wait for peer handshake,
     * and if devices remain open from a previous test, the handshake won't be re-initiated,
     * causing subsequent tests to hang.
     *
     * @note See Issue #34040 for future optimization to keep routers in standby mode
     *       instead of fully terminating them between workloads.
     */
    void teardown();

    // Delete copy and move constructors/assignments
    FabricSwitchManager(const FabricSwitchManager&) = delete;
    FabricSwitchManager& operator=(const FabricSwitchManager&) = delete;
    FabricSwitchManager(FabricSwitchManager&&) = delete;
    FabricSwitchManager& operator=(FabricSwitchManager&&) = delete;

private:
    friend class tt::stl::Indestructible<FabricSwitchManager>;
    FabricSwitchManager() = default;
    ~FabricSwitchManager() = default;

    // Cache the device map returned by CreateDevices to use directly in CloseDevices
    std::map<tt::ChipId, tt::tt_metal::IDevice*> switch_devices_;
};

}  // namespace tt::tt_fabric
