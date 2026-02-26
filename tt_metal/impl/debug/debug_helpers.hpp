// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>

#include <fmt/format.h>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "llrt/core_descriptor.hpp"
#include "hostdevcommon/dprint_common.h"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "llrt.hpp"
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal {

// Helper function for comparing CoreDescriptors for using in sets.
struct CoreDescriptorComparator {
    bool operator()(const umd::CoreDescriptor& x, const umd::CoreDescriptor& y) const {
        if (x.coord == y.coord) {
            return x.type < y.type;
        }
        return x.coord < y.coord;
    }
};
using CoreDescriptorSet = std::set<umd::CoreDescriptor, CoreDescriptorComparator>;

// Helper function to get CoreDescriptors for all debug-relevant cores on device.
inline static CoreDescriptorSet GetAllCores(ChipId device_id) {
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
         tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id)) {
        all_cores.insert({logical_core, CoreType::ETH});
    }
    for (const auto& logical_core :
         tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(device_id)) {
        all_cores.insert({logical_core, CoreType::ETH});
    }

    return all_cores;
}

// Helper function to get CoreDescriptors for all cores that are used for dispatch. Should be a subset of
// GetAllCores().
[[maybe_unused]] static CoreDescriptorSet GetDispatchCores(ChipId device_id) {
    CoreDescriptorSet dispatch_cores;
    unsigned num_cqs = tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_num_hw_cqs();
    const auto& dispatch_core_config =
        tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);
    log_debug(tt::LogAlways, "Dispatch Core Type = {}", dispatch_core_type);
    for (auto logical_core : tt::get_logical_dispatch_cores(device_id, num_cqs, dispatch_core_config)) {
        dispatch_cores.insert({logical_core, dispatch_core_type});
    }
    return dispatch_cores;
}

inline uint64_t GetDprintBufAddr(ChipId device_id, const CoreCoord& virtual_core, int risc_id) {
    uint64_t addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        llrt::get_core_type(device_id, virtual_core), tt::tt_metal::HalL1MemAddrType::DPRINT_BUFFERS);
    return addr + (sizeof(DebugPrintMemLayout) * risc_id);
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

// Returns the assert message portion for a given assert type
// Returns empty string for unknown types (callers must handle this)
// For DebugAssertTripped, line_num is used in the message
inline std::string get_debug_assert_message(dev_msgs::debug_assert_type_t type, uint16_t line_num = 0) {
    switch (type) {
        case dev_msgs::DebugAssertTripped:
            return fmt::format(
                "tripped an assert on line {}. Note that file name reporting is not yet "
                "implemented, and the reported line number for the assert may be from a different file.",
                line_num);
        case dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped:
            return "detected an inter-kernel data race due to kernel completing with pending NOC "
                   "transactions (missing NOC reads flushed barrier).";
        case dev_msgs::DebugAssertNCriscNOCNonpostedWritesSentTripped:
            return "detected an inter-kernel data race due to kernel completing with pending NOC "
                   "transactions (missing NOC non-posted writes sent barrier).";
        case dev_msgs::DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped:
            return "detected an inter-kernel data race due to kernel completing with pending NOC "
                   "transactions (missing NOC non-posted atomics flushed barrier).";
        case dev_msgs::DebugAssertNCriscNOCPostedWritesSentTripped:
            return "detected an inter-kernel data race due to kernel completing with pending NOC "
                   "transactions (missing NOC posted writes sent barrier).";
        case dev_msgs::DebugAssertRtaOutOfBounds: return "accessed unique runtime arg index out of bounds.";
        case dev_msgs::DebugAssertCrtaOutOfBounds: return "accessed common runtime arg index out of bounds.";
        default: return "";
    }
}

}  // namespace tt::tt_metal
