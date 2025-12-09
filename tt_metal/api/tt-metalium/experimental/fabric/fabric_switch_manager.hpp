// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <tt_stl/indestructible.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

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
     */
    void setup();

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
    // Track if we set the fabric config so we can clean it up properly in teardown
    bool fabric_config_was_set_by_manager_ = false;
};

}  // namespace tt::tt_fabric
