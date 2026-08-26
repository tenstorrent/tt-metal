// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>
#include <set>
#include <span>
#include <vector>
#include <cctype>
#include <cstdio>
#include <enchantum/enchantum.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "context/metal_env_accessor.hpp"
#include "llrt/core_descriptor.hpp"
#include "hostdevcommon/dprint_common.h"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "llrt.hpp"
#include <impl/dispatch/dispatch_core_manager.hpp>
#include "impl/dispatch/dispatch_engine_cores.hpp"
#include <llrt/tt_cluster.hpp>
#include "llrt/hal.hpp"
#include "internal/tt-2xx/quasar/error_handling.h"
#include "internal/tt-2xx/quasar/overlay/remapper_common.hpp"
#include "hostdev/debug_ring_buffer_common.h"

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
    // The set of all printable cores is Tensix + Eth + DRAM (when supported)
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
    const auto& hal = MetalContext::instance().hal();
    if (hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        const auto& soc_desc = cluster.get_soc_desc(device_id);
        for (const auto& dram_core : soc_desc.get_cores(CoreType::DRAM, CoordSystem::LOGICAL)) {
            all_cores.insert({{dram_core.x, dram_core.y}, CoreType::DRAM});
        }
    }
    // Quasar dispatch-engine tiles use synthetic logical coords CoreCoord(index, 0) over the soc
    // `dispatch:` list (see dispatch_engine_cores). They are printable just like worker/eth/dram cores.
    if (hal.has_programmable_core_type(HalProgrammableCoreType::DISPATCH)) {
        const auto& soc_desc = cluster.get_soc_desc(device_id);
        for (const auto& logical_core : detail::get_quasar_soc_dispatch_engine_logical_cores(soc_desc)) {
            all_cores.insert({{logical_core.x, logical_core.y}, CoreType::DISPATCH});
        }
    }

    return all_cores;
}

// Helper function to get CoreDescriptors for all cores that are used for dispatch. Should be a subset of
// GetAllCores().
[[maybe_unused]] static CoreDescriptorSet GetDispatchCores(
    MetalEnvImpl& env, ChipId device_id, uint8_t num_hw_cqs, const DispatchCoreConfig& dispatch_core_config) {
    CoreDescriptorSet dispatch_cores;
    CoreType dispatch_core_type = resolve_dispatch_core_type(env, device_id, dispatch_core_config);
    log_debug(tt::LogAlways, "Dispatch Core Type = {}", dispatch_core_type);
    for (auto logical_core : tt::get_logical_dispatch_cores(env, device_id, num_hw_cqs, dispatch_core_config)) {
        dispatch_cores.insert({logical_core, dispatch_core_type});
    }
    return dispatch_cores;
}

inline uint64_t GetDevicePrintBufAddr(ChipId device_id, const CoreCoord& virtual_core) {
    return tt::tt_metal::MetalContext::instance().hal().get_dev_noc_addr(
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
        case CoreType::DISPATCH: return "dispatch";
        default: return "UNKNOWN";
    }
}

// The kQuasarErr* shifts and masks for the error code layout live in error_handling.h, shared
// with the device side so both decode the same bits.

// enchantum::to_string gives back an empty view for unnamed values, which ends up as a blank
// field in the watcher log.
template <typename E>
inline std::string quasar_enum_name_or_hex(uint32_t raw) {
    const auto name = enchantum::to_string(static_cast<E>(raw));
    return name.empty() ? fmt::format("unknown code 0x{:02x}", raw) : std::string{name};
}

// Only the per-TRISC blocks put a PC in ERR_DATA, and those print it up front like the DM
// faults. Everything else gets it as a trailing field, see get_quasar_error_data_name().
inline bool quasar_error_data_is_pc(TriscErrors block) {
    switch (block) {
        case TriscErrors::ERROR_TRISC0:
        case TriscErrors::ERROR_TRISC1:
        case TriscErrors::ERROR_TRISC2:
        case TriscErrors::ERROR_TRISC3: return true;
        default: return false;
    }
}

// What's in ERR_DATA, which depends on the block. For the per-TRISC errors it's a PC, the last
// instruction to commit, so for a timeout it's roughly where the thread gave up rather than the
// exact culprit. Disassemble around it: on a MEM_READ_NO_RESPONSE that points at the load whose
// response never arrived.
inline std::string_view get_quasar_error_data_name(TriscErrors block) {
    switch (block) {
        case TriscErrors::ERROR_TRISC0:
        case TriscErrors::ERROR_TRISC1:
        case TriscErrors::ERROR_TRISC2:
        case TriscErrors::ERROR_TRISC3: return "PC";
        case TriscErrors::NEO_SEMAPHORES:
        case TriscErrors::GLOBAL_SEMAPHORES: return "semaphore index";
        case TriscErrors::TILE_COUNTERS: return "tile counter index";
        case TriscErrors::ILLEGAL_INSTRUCTION_TRISC0:
        case TriscErrors::ILLEGAL_INSTRUCTION_TRISC1:
        case TriscErrors::ILLEGAL_INSTRUCTION_TRISC2:
        case TriscErrors::ILLEGAL_INSTRUCTION_TRISC3: return "offending instruction";
        default: return "faulting address or instruction";
    }
}

// Which TRISC reported, or nullopt for the Neo-level blocks (TDMA, EDC, semaphores, SFPU, tile
// counters) since those aren't tied to a single thread.
//
// Careful: the illegal-instruction blocks count backwards, 32 is TRISC3 and 35 is TRISC0, the
// opposite way round to ERROR_TRISC0..3. Keep that in here rather than open-coding it.
inline std::optional<uint32_t> get_quasar_error_trisc_id(TriscErrors block) {
    const auto raw = static_cast<uint32_t>(block);
    if (raw <= static_cast<uint32_t>(TriscErrors::ERROR_TRISC3)) {
        return raw;
    }
    if (raw >= static_cast<uint32_t>(TriscErrors::ILLEGAL_INSTRUCTION_TRISC3) &&
        raw <= static_cast<uint32_t>(TriscErrors::ILLEGAL_INSTRUCTION_TRISC0)) {
        return static_cast<uint32_t>(TriscErrors::ILLEGAL_INSTRUCTION_TRISC0) - raw;
    }
    return std::nullopt;
}

// Decodes error_code[7:0]. Takes the block as well because the index means nothing on its own.
// Empty for the blocks that don't carry an index (EDC, unallocated IDs).
inline std::string get_quasar_error_index_description(TriscErrors block, uint32_t index) {
    switch (block) {
        case TriscErrors::ERROR_TRISC0:
        case TriscErrors::ERROR_TRISC1:
        case TriscErrors::ERROR_TRISC2:
        case TriscErrors::ERROR_TRISC3: return quasar_enum_name_or_hex<TriscRiscErrors>(index);

        case TriscErrors::UNPACKER_0:
        case TriscErrors::UNPACKER_1:
        case TriscErrors::UNPACKER_2:
        case TriscErrors::PACKER_0:
        case TriscErrors::PACKER_1: return quasar_enum_name_or_hex<TdmaErrors>(index);

        case TriscErrors::NEO_SEMAPHORES:
        case TriscErrors::GLOBAL_SEMAPHORES: return quasar_enum_name_or_hex<SemaphoreErrors>(index & 0x7);

        // Sticky bitmask rather than a single value, so both bits can be up at once.
        case TriscErrors::SFPU: {
            std::vector<std::string_view> flags;
            if (index & static_cast<uint32_t>(SfpuErrors::CC_STACK_OVERFLOW)) {
                flags.emplace_back("CC_STACK_OVERFLOW");
            }
            if (index & static_cast<uint32_t>(SfpuErrors::CC_STACK_UNDERFLOW)) {
                flags.emplace_back("CC_STACK_UNDERFLOW");
            }
            return flags.empty() ? fmt::format("unknown code 0x{:02x}", index)
                                 : fmt::format("{}", fmt::join(flags, " + "));
        }

        // Which counter went bad, not a reason code.
        case TriscErrors::TILE_COUNTERS: return fmt::format("counter {}", index);

        case TriscErrors::EDC_FATAL_ERROR:
        case TriscErrors::EDC_CORRECTABLE_ERROR: return {};

        // Only the opcode here, the caller prints the whole instruction out of ERR_DATA.
        case TriscErrors::ILLEGAL_INSTRUCTION_TRISC0:
        case TriscErrors::ILLEGAL_INSTRUCTION_TRISC1:
        case TriscErrors::ILLEGAL_INSTRUCTION_TRISC2:
        case TriscErrors::ILLEGAL_INSTRUCTION_TRISC3: return fmt::format("opcode 0x{:02x}", index);

        default: return {};
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
        case dev_msgs::DebugAssertNCriscNOCPacketTagClearedTripped:
            return "detected invalid NOC command buffer state before starting the next kernel "
                   "(write-capable NOC packet tags must be zero so implicit transaction ID users start with "
                   "transaction ID 0).";
        case dev_msgs::DebugAssertRtaOutOfBounds: return "accessed unique runtime arg index out of bounds.";
        case dev_msgs::DebugAssertCrtaOutOfBounds: return "accessed common runtime arg index out of bounds.";
        case dev_msgs::DebugAssertHwFault:
            if ((hw_fault_info & 0xffffffff) <= 7) {
                return fmt::format(
                    "hardware fault occurred at PC 0x{:x}. Cause: {}, faulting address or instruction: 0x{:08x}",
                    line_num,
                    enchantum::to_string(static_cast<DmErrors>(hw_fault_info & 0xffffffff)),
                    (hw_fault_info >> 32) & 0xffffffff);
            } else {
                const uint32_t error_code = hw_fault_info & 0xffffffff;
                const uint32_t neo = (error_code >> kQuasarErrNeoShift) & kQuasarErrNeoMask;
                const auto block =
                    static_cast<TriscErrors>((error_code >> kQuasarErrBlockShift) & kQuasarErrBlockMask);
                const uint32_t index = error_code & kQuasarErrIndexMask;

                const std::string block_name =
                    quasar_enum_name_or_hex<TriscErrors>((error_code >> kQuasarErrBlockShift) & kQuasarErrBlockMask);
                const std::string detail = get_quasar_error_index_description(block, index);

                // Left off entirely for Neo-level blocks instead of defaulting to 0, otherwise
                // "no thread" and "thread 0" look the same.
                const auto trisc = get_quasar_error_trisc_id(block);
                const std::string where = fmt::format(
                    "Neo {}{}", neo, trisc.has_value() ? fmt::format(" TRISC{}", *trisc) : std::string{});
                const std::string cause = fmt::format(
                    "{}{}", block_name, detail.empty() ? std::string{} : fmt::format(" ({})", detail));
                const uint32_t err_data = (hw_fault_info >> 32) & 0xffffffff;

                if (quasar_error_data_is_pc(block)) {
                    return fmt::format(
                        "hardware fault occurred at PC 0x{:08x} on {} with cause: {}, error_code 0x{:04x}",
                        err_data,
                        where,
                        cause,
                        error_code & 0xffff);
                }
                return fmt::format(
                    "hardware fault occurred on {} with cause: {}, error_code 0x{:04x}, {}: 0x{:08x}",
                    where,
                    cause,
                    error_code & 0xffff,
                    get_quasar_error_data_name(block),
                    err_data);
            }
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
    } else if (core_type == HalProgrammableCoreType::DRAM) {
        // DRAM cores (Blackhole DRISC): single processor
        uint32_t num = hal.get_num_risc_processors(core_type);
        for (uint32_t i = 0; i < num; ++i) {
            info.processor_names.push_back(hal.get_processor_class_name(core_type, i, false));
        }
        for (uint32_t i = 0; i < num; ++i) {
            std::string abbrev = hal.get_processor_class_name(core_type, i, true);
            add_legacy_entry(abbrev, info.processor_names[i]);
        }
    } else if (core_type == HalProgrammableCoreType::DISPATCH) {
        // Quasar dispatch-engine cores run DM-only firmware/kernels; display the DM enable flags using a
        // hex bitmask, mirroring the Quasar TENSIX DM / ETH style.
        uint32_t num = hal.get_num_risc_processors(core_type);
        for (uint32_t i = 0; i < num; ++i) {
            info.processor_names.push_back(hal.get_processor_class_name(core_type, i, false));
        }
        if (is_quasar) {
            info.symbols.push_back("DM:");
            legend_parts.push_back(fmt::format("DM:[hex]=DataMovement(0-{})", num - 1));
        } else {
            for (uint32_t i = 0; i < num; ++i) {
                std::string abbrev = hal.get_processor_class_name(core_type, i, true);
                add_legacy_entry(std::string{abbrev[0]}, info.processor_names[i]);
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

// Format client name for tile counter output.
// DM clients show paired DMs (e.g., DM0/DM4) because DM0-3 and DM4-7 share tile counter groups.
inline void fprintClientName(FILE* f, uint32_t client_id) {
    if (client_id < overlay::NEO_0) {
        fprintf(f, "DM%u/DM%u", client_id, client_id + overlay::NEO_0);
    } else {
        fprintf(f, "NEO_%u", client_id - overlay::NEO_0);
    }
}

// DPRINT queries can take tens of seconds per poll on RTL sim.
inline int debug_server_wait_timeout_sec(const llrt::RunTimeOptions& rtoptions) {
    return rtoptions.get_simulator_enabled() ? 30 : 5;
}

inline int debug_server_finish_timeout_sec(const llrt::RunTimeOptions& rtoptions) {
    return rtoptions.get_simulator_enabled() ? 30 : 2;
}

// Format ring buffer output - auto-detects SPSC (WH) vs MPSC (Quasar/BH) based on arch
// For MPSC, thread_indices and core_type are used to prefix entries with processor name
// Returns vector of lines like ["[0x00270028,...,", " 0x001f0020,...,", "]"]
// or for MPSC: ["[[DM0]0x00270028,...,", " [DM0]0x001f0020,...,", "]"]
inline std::vector<std::string> FormatRingBuffer(
    std::span<const uint32_t> data,
    std::span<const uint32_t> thread_indices = {},
    HalProgrammableCoreType core_type = HalProgrammableCoreType::TENSIX) {
    if (data.empty()) {
        return {};
    }
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const bool is_mpsc = hal.has_mpsc_ring_buffer();
    TT_ASSERT(
        !is_mpsc || thread_indices.size() == data.size(),
        "FormatRingBuffer: MPSC data requires one thread index per entry");

    constexpr size_t entries_per_line = 8;

    std::vector<std::string> lines;
    std::string line = "[";
    for (size_t i = 0; i < data.size(); i++) {
        if (is_mpsc) {
            auto name = hal.get_processor_class_name(core_type, thread_indices[i], false);
            line += fmt::format("[{}]0x{:08x},", name, data[i]);
        } else {
            line += fmt::format("0x{:08x},", data[i]);
        }
        if ((i + 1) % entries_per_line == 0 && i + 1 < data.size()) {
            lines.push_back(line);
            line = " ";  // Continuation lines start with space
        }
    }
    line.pop_back();  // Remove trailing comma
    line += "]";
    lines.push_back(line);
    return lines;
}

// SPSC overload - extracts data in newest-first order and formats
inline std::vector<std::string> FormatRingBuffer(
    const debug_spsc_ring_buf_msg_t& buf, HalProgrammableCoreType core_type = HalProgrammableCoreType::TENSIX) {
    if (buf.current_ptr == DEBUG_RING_BUFFER_STARTING_INDEX) {
        return {};
    }
    // Extract newest-first: walk backwards from the last written entry, wrapping at 0
    std::vector<uint32_t> data;
    const int16_t last_written_idx = buf.current_ptr;
    const int16_t count = buf.wrapped ? DEBUG_RING_BUFFER_SPSC_ELEMENTS : (last_written_idx + 1);
    int16_t idx = last_written_idx;
    for (int16_t i = 0; i < count; i++) {
        data.push_back(buf.data[idx]);
        if (--idx < 0) {
            idx = DEBUG_RING_BUFFER_SPSC_ELEMENTS - 1;
        }
    }
    return FormatRingBuffer(data, {}, core_type);
}

}  // namespace tt::tt_metal
