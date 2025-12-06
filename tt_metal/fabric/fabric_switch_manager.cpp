// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric_switch_manager.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <unordered_set>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/device/device_pool.hpp"
#include "hostdevcommon/common_values.hpp"

namespace tt::tt_fabric {

FabricSwitchManager& FabricSwitchManager::instance() {
    static tt::stl::Indestructible<FabricSwitchManager> inst;
    return inst.get();
}

void FabricSwitchManager::setup() {
    // Create devices for the switch mesh
    const auto& switch_device_ids =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_switch_mesh_device_ids();

    // Only create devices if there are switch devices
    if (switch_device_ids.empty()) {
        return;
    }

    tt::tt_metal::detail::CreateDevices(
        switch_device_ids,
        1,  // num_command_queues
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER},
        {},                      // l1_bank_remap
        DEFAULT_WORKER_L1_SIZE,  // worker_l1_size
        false,                   // init_profiler
        true,                    // use_max_eth_core_count_on_all_devices
        true);                   // initialize_fabric_and_dispatch_fw
}

void FabricSwitchManager::teardown() {
    // Close devices for switch meshes to ensure proper fabric handshake between tests.
    // This is critical because fabric routers wait for peer handshake, and if
    // devices remain open from a previous test, the handshake won't be re-initiated,
    // causing subsequent tests to hang.
    if (tt::DevicePool::is_initialized()) {
        // Copy switch device IDs first to avoid accessing control plane after device closure
        const std::vector<tt::ChipId> switch_device_ids =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_switch_mesh_device_ids();

        // Get all active devices and filter to only switch mesh devices
        auto active_devices = tt::DevicePool::instance().get_all_active_devices();
        std::vector<tt::tt_metal::IDevice*> switch_devices_to_close;

        std::unordered_set<tt::ChipId> switch_device_ids_set(switch_device_ids.begin(), switch_device_ids.end());

        for (auto* device : active_devices) {
            if (switch_device_ids_set.find(device->id()) != switch_device_ids_set.end()) {
                switch_devices_to_close.push_back(device);
            }
        }

        if (!switch_devices_to_close.empty()) {
            tt::DevicePool::instance().close_devices(switch_devices_to_close);
        }
    }
}

}  // namespace tt::tt_fabric
