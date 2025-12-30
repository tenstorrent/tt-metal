// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric_switch_manager.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <map>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "hostdevcommon/common_values.hpp"

namespace tt::tt_fabric {

FabricSwitchManager& FabricSwitchManager::instance() {
    static tt::stl::Indestructible<FabricSwitchManager> inst;
    return inst.get();
}

void FabricSwitchManager::setup(FabricConfig fabric_config, FabricReliabilityMode fabric_reliability_mode) {
    // Create devices for the switch mesh
    const auto& switch_device_ids =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_switch_mesh_device_ids();

    // Only create devices if there are switch devices
    if (switch_device_ids.empty()) {
        switch_devices_.clear();
        return;
    }

    // Set fabric config explicitly before creating devices
    // This is required for workloads that need fabric (not just minimal fabric for dispatch)
    // The workload calling setup() knows which fabric config it needs, so we use the provided config
    tt::tt_fabric::SetFabricConfig(fabric_config, fabric_reliability_mode);

    // Cache the device map returned by CreateDevices to use directly in CloseDevices
    // TODO: Issue #34040 - If routers are in standby mode, we could skip full reinitialization
    // and just reactivate them instead of calling CreateDevices.
    switch_devices_ = tt::tt_metal::detail::CreateDevices(
        switch_device_ids,
        1,  // num_command_queues
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER},
        {},                      // l1_bank_remap
        DEFAULT_WORKER_L1_SIZE,  // worker_l1_size
        false,                   // init_profiler
        true,                    // use_max_eth_core_count_on_all_devices
        // TOD: for future optimzation, switch meshes don't need dispatch fw.
        true);  // initialize_fabric_and_dispatch_fw
}

void FabricSwitchManager::teardown() {
    // Close devices for switch meshes to ensure proper fabric handshake between tests.
    // This is critical because fabric routers wait for peer handshake, and if
    // devices remain open from a previous test, the handshake won't be re-initiated,
    // causing subsequent tests to hang.
    //
    // TODO: Issue #34040 - Router Standby/Reactivation Optimization
    // In the future, we could keep routers in standby mode instead of fully terminating
    // them, allowing faster reactivation without recompilation and re-handshake overhead.
    if (!switch_devices_.empty()) {
        // Use the cached device map returned by CreateDevices
        tt::tt_metal::detail::CloseDevices(switch_devices_);
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);

        switch_devices_.clear();
    }
}

}  // namespace tt::tt_fabric
