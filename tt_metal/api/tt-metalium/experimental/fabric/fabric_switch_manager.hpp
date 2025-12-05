// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/indestructible.hpp>

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
     * from the control plane. It initializes the device pool with appropriate configuration
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
};

}  // namespace tt::tt_fabric
