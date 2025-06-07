// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>

#include "dev_msgs.h"
#include <tt-metalium/core_descriptor.hpp>
#include "hostdevcommon/dprint_common.h"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// Helper function for comparing CoreDescriptors for using in sets.
struct CoreDescriptorComparator {
    bool operator()(const CoreDescriptor& x, const CoreDescriptor& y) const {
        if (x.coord == y.coord) {
            return x.type < y.type;
        } else {
            return x.coord < y.coord;
        }
    }
};
using CoreDescriptorSet = std::set<CoreDescriptor, CoreDescriptorComparator>;

// Helper function to get CoreDescriptors for all debug-relevant cores on device.
static CoreDescriptorSet GetAllCores(chip_id_t device_id) {
    CoreDescriptorSet all_cores;
    // The set of all printable cores is Tensix + Eth cores
    CoreCoord logical_grid_size =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            all_cores.insert({{x, y}, CoreType::WORKER});
        }
    }
    for (const auto& logical_core :
         tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(device_id)) {
        all_cores.insert({logical_core, CoreType::ETH});
    }
    for (const auto& logical_core :
         tt::tt_metal::MetalContext::instance().get_cluster().get_inactive_ethernet_cores(device_id)) {
        all_cores.insert({logical_core, CoreType::ETH});
    }

    return all_cores;
}

// Helper function to get CoreDescriptors for all cores that are used for dispatch. Should be a subset of
// GetAllCores().
static CoreDescriptorSet GetDispatchCores(chip_id_t device_id) {
    CoreDescriptorSet dispatch_cores;
    unsigned num_cqs = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_num_hw_cqs();
    const auto& dispatch_core_config =
        tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();
    log_warning(tt::LogAlways, "Dispatch Core Type = {}", dispatch_core_type);
    for (auto logical_core : tt::get_logical_dispatch_cores(device_id, num_cqs, dispatch_core_config)) {
        dispatch_cores.insert({logical_core, dispatch_core_type});
    }
    return dispatch_cores;
}

// Helper function to convert virtual core -> HalProgrammableCoreType. TODO: Remove when we fix core types.
static tt::tt_metal::HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core, chip_id_t device_id) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_core, device_id)) {
        return tt::tt_metal::HalProgrammableCoreType::TENSIX;
    }

    // Eth pcores have a different address, but only active ones.
    CoreCoord logical_core =
        tt::tt_metal::MetalContext::instance().get_cluster().get_logical_ethernet_core_from_virtual(
            device_id, virtual_core);
    auto active_ethernet_cores =
        tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(device_id);
    if (active_ethernet_cores.find(logical_core) != active_ethernet_cores.end()) {
        return tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
    }

    return tt::tt_metal::HalProgrammableCoreType::IDLE_ETH;
}

inline uint64_t GetDprintBufAddr(chip_id_t device_id, const CoreCoord& virtual_core, int risc_id) {
    dprint_buf_msg_t* buf = tt::tt_metal::MetalContext::instance().hal().get_dev_addr<dprint_buf_msg_t*>(
        get_programmable_core_type(virtual_core, device_id), tt::tt_metal::HalL1MemAddrType::DPRINT);
    return reinterpret_cast<uint64_t>(&(buf->data[risc_id]));
}

// TODO(#17275): Move this and others to the HAL
#define DPRINT_NRISCVS 5
#define DPRINT_NRISCVS_ETH 1

inline int GetNumRiscs(chip_id_t device_id, const CoreDescriptor& core) {
    if (core.type == CoreType::ETH) {
        if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE) {
            // TODO: Update this to be `DPRINT_NRISCVS_ETH + 1` when active erisc0 is running Metal FW
            auto logical_active_eths =
                tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(device_id);
            CoreCoord logical_eth(core.coord.x, core.coord.y);
            return (logical_active_eths.find(logical_eth) != logical_active_eths.end()) ? DPRINT_NRISCVS_ETH
                                                                                        : DPRINT_NRISCVS_ETH + 1;
        }
        return DPRINT_NRISCVS_ETH;
    } else {
        return DPRINT_NRISCVS;
    }
}

inline std::string_view get_core_type_name(CoreType ct) {
    switch (ct) {
        case CoreType::ARC: return "ARC";
        case CoreType::DRAM: return "DRAM";
        case CoreType::ETH: return "ethernet";
        case CoreType::PCIE: return "PCIE";
        case CoreType::WORKER: return "worker";
        case CoreType::HARVESTED: return "harvested";
        case CoreType::ROUTER_ONLY: return "router_only";
        case CoreType::ACTIVE_ETH: return "active_eth";
        case CoreType::IDLE_ETH: return "idle_eth";
        case CoreType::TENSIX: return "tensix";
        default: return "UNKNOWN";
    }
}

}  // namespace tt::tt_metal
