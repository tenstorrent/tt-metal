// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <set>
#include <vector>
#include <cctype>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <fmt/format.h>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "context/metal_env_accessor.hpp"
#include "llrt/core_descriptor.hpp"
#include "hostdevcommon/dprint_common.h"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "llrt.hpp"
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>
#include "llrt/hal.hpp"

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
inline static CoreDescriptorSet GetAllCores(
    tt::Cluster& cluster, tt::tt_fabric::ControlPlane& control_plane, ChipId device_id) {
    CoreDescriptorSet all_cores;
    // The set of all printable cores is Tensix + Eth cores
    CoreCoord logical_grid_size = cluster.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            all_cores.insert({{x, y}, CoreType::WORKER});
        }
    }
    for (const auto& logical_core : control_plane.get_active_ethernet_cores(device_id)) {
        all_cores.insert({logical_core, CoreType::ETH});
    }
    for (const auto& logical_core : control_plane.get_inactive_ethernet_cores(device_id)) {
        all_cores.insert({logical_core, CoreType::ETH});
    }

    return all_cores;
}

// Helper function to get CoreDescriptors for all cores that are used for dispatch. Should be a subset of
// GetAllCores().
[[maybe_unused]] static CoreDescriptorSet GetDispatchCores(
    MetalEnvImpl& env, ChipId device_id, uint8_t num_hw_cqs, const DispatchCoreConfig& dispatch_core_config) {
    CoreDescriptorSet dispatch_cores;
    CoreType dispatch_core_type = get_core_type_from_config(dispatch_core_config);
    log_debug(tt::LogAlways, "Dispatch Core Type = {}", dispatch_core_type);
    for (auto logical_core : tt::get_logical_dispatch_cores(env, device_id, num_hw_cqs, dispatch_core_config)) {
        dispatch_cores.insert({logical_core, dispatch_core_type});
    }
    return dispatch_cores;
}

inline uint64_t GetDprintBufAddr(ChipId device_id, const CoreCoord& virtual_core, int risc_id) {
    uint64_t addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        llrt::get_core_type(device_id, virtual_core), tt::tt_metal::HalL1MemAddrType::DPRINT_BUFFERS);
    return addr + (sizeof(DebugPrintMemLayout) * risc_id);
}

inline uint64_t GetDevicePrintBufAddr(ChipId device_id, const CoreCoord& virtual_core) {
    return tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        llrt::get_core_type(device_id, virtual_core), tt::tt_metal::HalL1MemAddrType::DPRINT_BUFFERS);
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
inline std::string get_debug_assert_message(
    dev_msgs::debug_assert_type_t type, uint16_t line_num = 0, uint64_t hw_fault_info = 0) {
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
        case dev_msgs::DebugAssertHwFault:
            return fmt::format(
                "hardware fault occurred at PC 0x{:x}. Cause: 0x{:x}, faulting address or instruction: 0x{:08x}",
                line_num,
                hw_fault_info & 0xffffffff,
                (hw_fault_info >> 32) & 0xffffffff);
        default: return "";
    }
}

// Metadata for identifying and logging processor info in the watcher (Tensix and Ethernet)
struct EnableSymbolsInfo {
    std::string main_processor;
    std::vector<std::string> processor_names;  // All RISC processors
    std::vector<std::string> symbols;  // Labels per log line. Quasar: (DM:, NEO:) or (E:), BH/WH: (B, N, T) or (E)
    std::string enable_legend;         // Legend in the watcher log header for enable/disable flags
};

// This function gets enable/disable flags for watcher header/legend in the log file
inline EnableSymbolsInfo get_enable_symbols_info(HalProgrammableCoreType core_type) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;
    EnableSymbolsInfo info;
    info.main_processor = hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, 0, false);

    std::vector<std::string> legend_parts;

    // Create the enable flags for BH/WH (e.g. B/b=BRISC)
    auto add_legacy_entry = [&](const std::string& sym, const std::string& name) {
        std::string lo = sym;
        std::transform(lo.begin(), lo.end(), lo.begin(), [](unsigned char c) { return std::tolower(c); });
        info.symbols.push_back(sym);
        legend_parts.push_back(fmt::format("{}/{}={}", sym, lo, name));
    };

    if (core_type == HalProgrammableCoreType::TENSIX) {
        for (uint32_t cls = 0; cls < hal.get_processor_classes_count(core_type); cls++) {
            auto type = static_cast<HalProcessorClassType>(cls);
            uint32_t base = hal.get_processor_index(core_type, type, 0);
            uint32_t count = hal.get_processor_types_count(core_type, cls);

            // Log all processor names
            for (uint32_t i = 0; i < count; ++i) {
                auto name = hal.get_processor_class_name(core_type, base + i, false);
                info.processor_names.push_back(name);
                // On WH/BH: For BRISC and NCRISC, create enable flags
                if (!is_quasar && type != HalProcessorClassType::COMPUTE) {
                    add_legacy_entry(std::string{name[0]}, name);
                }
            }

            // On Quasar, enable flags are displayed using a bitmask in hex
            if (is_quasar) {
                // DM: one symbol per processor
                if (type == HalProcessorClassType::DM) {
                    info.symbols.push_back("DM:");
                    legend_parts.push_back(fmt::format("DM:[hex]=DataMovement(0-{})", count - 1));
                } else {
                    uint32_t ct_idx = hal.get_programmable_core_type_index(core_type);
                    // Neo0, Neo1, Neo2, Neo3
                    uint32_t num_clusters = count / hal.get_processor_class_num_fw_binaries(ct_idx, cls);
                    info.symbols.push_back("NEO:");
                    legend_parts.push_back(fmt::format("NEO:[hex]=ComputeClusters(NeoCluster0-{})", num_clusters - 1));
                }
            }  // On WH/BH Compute: collapse all TRISCs to one symbol, strip trailing digit (TRISC0 -> TRISC)
            else if (type == HalProcessorClassType::COMPUTE) {
                auto name = hal.get_processor_class_name(core_type, base, false);
                if (!name.empty() && std::isdigit(static_cast<unsigned char>(name.back()))) {
                    name.pop_back();
                }
                add_legacy_entry(std::string{name[0]}, name);
            }
        }
    } else {
        // ACTIVE_ETH/IDLE_ETH: collect names (arch-independent), then symbols (arch-specific)
        uint32_t num = hal.get_num_risc_processors(core_type);
        for (uint32_t i = 0; i < num; ++i) {
            info.processor_names.push_back(hal.get_processor_class_name(core_type, i, false));
        }
        if (is_quasar) {
            info.symbols.push_back("E:");
            legend_parts.push_back(fmt::format("E:[hex]=Ethernet(0-{})", num - 1));
        } else {
            for (uint32_t i = 0; i < num; ++i) {
                std::string abbrev = hal.get_processor_class_name(core_type, i, true);
                add_legacy_entry(std::string{abbrev[0]}, info.processor_names[i]);
            }
        }
    }
    info.enable_legend = fmt::format("{}", fmt::join(legend_parts, " "));
    if (!is_quasar) {
        info.enable_legend = "UPPER=enabled, lower=disabled: " + info.enable_legend;
    }
    return info;
}
}  // namespace tt::tt_metal
