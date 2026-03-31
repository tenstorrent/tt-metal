// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <cstring>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
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

// Host-side FNV-1a hash — mirrors device-side constexpr debug_file_hash in dev_msgs.h.
inline uint16_t host_debug_file_hash(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= static_cast<uint32_t>(*str++);
        hash *= 16777619u;
    }
    return static_cast<uint16_t>((hash >> 16) ^ (hash & 0xFFFF));
}

// Read all source file paths from a .o.dephash file.
// These are always written during kernel compilation (no debug info required).
// Format per line: "<absolute-path>"\t<uint64-content-hash>
inline std::unordered_set<std::string> collect_dephash_filenames(const std::filesystem::path& dephash_path) {
    std::unordered_set<std::string> result;
    std::ifstream f(dephash_path);
    if (!f.is_open()) {
        return result;
    }
    std::filesystem::path dep;
    while (f >> dep) {
        uint64_t content_hash;
        f >> content_hash;  // skip the file content hash
        if (f.bad()) {
            break;
        }
        result.insert(dep.string());
    }
    return result;
}

// Try to match file_id against a path and all its stripped-prefix suffixes.
// Returns the shortest matching path form (the one __FILE__ expanded to), or empty string.
inline std::string match_hash_in_path(uint16_t file_id, const std::string& path) {
    std::string best;
    auto check = [&](const char* s) {
        if (host_debug_file_hash(s) == file_id) {
            if (best.empty()) {
                best = s;  // keep first match; loop goes full→stripped so this is the longest match
            }
        }
    };
    // Try full path, then strip one leading directory component at a time.
    // This recovers the relative path form __FILE__ expanded to at compile time.
    check(path.c_str());
    size_t start = 0;
    while ((start = path.find('/', start)) != std::string::npos) {
        ++start;
        if (start < path.size()) {
            check(path.c_str() + start);
        }
    }
    return best;
}

// Resolve a file_id hash back to a source filename using the .o.dephash files that sit
// alongside each kernel ELF.  These are always written during compilation (no debug info
// required) and record the absolute paths of every header compiled into the kernel.
inline std::string resolve_file_from_hash(uint16_t file_id, const std::vector<std::string>& elf_paths = {}) {
    std::unordered_set<std::string> candidates;
    for (const auto& elf_path : elf_paths) {
        auto elf_dir = std::filesystem::path(elf_path).parent_path();
        try {
            for (const auto& entry : std::filesystem::directory_iterator(elf_dir)) {
                const auto& p = entry.path();
                // Only *.o.dephash — *.elf.dephash contains only linker scripts.
                if (p.extension() == ".dephash" && std::filesystem::path(p.stem()).extension() == ".o") {
                    auto filenames = collect_dephash_filenames(p);
                    candidates.insert(filenames.begin(), filenames.end());
                }
            }
        } catch (const std::filesystem::filesystem_error&) {
        }
    }

    for (const auto& path : candidates) {
        auto m = match_hash_in_path(file_id, path);
        if (!m.empty()) {
            return m;
        }
    }

    return fmt::format("unknown file (hash=0x{:04x})", file_id);
}

// Read the assert message for (file_id, line_num) from the .debug_assert_msgs section
// in the kernel ELF.  Section entries are variable-length and packed:
//   [uint16_t file_id][uint16_t line_num][char msg... '\0']
// The string is stored inline — no pointer resolution, no device L1 cost.
// Returns empty string if the section is absent or no matching entry is found.
inline std::string resolve_assert_message(
    uint16_t file_id, uint16_t line_num, const std::vector<std::string>& elf_paths) {
    for (const auto& elf_path : elf_paths) {
        std::ifstream f(elf_path, std::ios::binary);
        if (!f) {
            continue;
        }

        unsigned char ident[EI_NIDENT];
        f.read(reinterpret_cast<char*>(ident), EI_NIDENT);
        if (f.gcount() != EI_NIDENT || ident[EI_MAG0] != ELFMAG0 || ident[EI_MAG1] != ELFMAG1 ||
            ident[EI_MAG2] != ELFMAG2 || ident[EI_MAG3] != ELFMAG3 || ident[EI_CLASS] != ELFCLASS32) {
            continue;
        }

        f.seekg(0);
        Elf32_Ehdr ehdr;
        f.read(reinterpret_cast<char*>(&ehdr), sizeof(ehdr));
        if (f.fail() || ehdr.e_shnum == 0 || ehdr.e_shstrndx >= ehdr.e_shnum) {
            continue;
        }

        std::vector<Elf32_Shdr> shdrs(ehdr.e_shnum);
        f.seekg(ehdr.e_shoff);
        f.read(reinterpret_cast<char*>(shdrs.data()), ehdr.e_shnum * sizeof(Elf32_Shdr));
        if (f.fail()) {
            continue;
        }

        const auto& shstrtab_hdr = shdrs[ehdr.e_shstrndx];
        std::vector<char> shstrtab(shstrtab_hdr.sh_size);
        f.seekg(shstrtab_hdr.sh_offset);
        f.read(shstrtab.data(), shstrtab_hdr.sh_size);
        if (f.fail()) {
            continue;
        }

        auto section_name = [&](const Elf32_Shdr& sh) -> const char* {
            if (sh.sh_name >= shstrtab.size()) {
                return "";
            }
            return shstrtab.data() + sh.sh_name;
        };

        const Elf32_Shdr* msgs_shdr = nullptr;
        for (const auto& sh : shdrs) {
            if (std::strcmp(section_name(sh), ".debug_assert_msgs") == 0) {
                msgs_shdr = &sh;
                break;
            }
        }
        if (!msgs_shdr || msgs_shdr->sh_size == 0) {
            continue;
        }

        std::vector<char> section(msgs_shdr->sh_size);
        f.seekg(msgs_shdr->sh_offset);
        f.read(section.data(), msgs_shdr->sh_size);
        if (f.fail()) {
            continue;
        }

        // Scan variable-length entries: [fid:2][lnum:2][msg...\0]
        const char* p = section.data();
        const char* end = p + section.size();
        while (p + 4 < end) {
            uint16_t fid, lnum;
            std::memcpy(&fid, p, 2);
            std::memcpy(&lnum, p + 2, 2);
            p += 4;
            const char* msg_start = p;
            while (p < end && *p != '\0') {
                ++p;
            }
            if (fid == file_id && lnum == line_num && p > msg_start) {
                return std::string(msg_start, p);
            }
            if (p < end) {
                ++p;  // skip null terminator
            }
        }
    }
    return {};
}

// Returns the assert message portion for a given assert type.
// Returns empty string for unknown types (callers must handle this).
inline std::string get_debug_assert_message(
    dev_msgs::debug_assert_type_t type,
    uint16_t line_num = 0,
    uint16_t file_id = 0,
    uint64_t hw_fault_info = 0,
    const std::vector<std::string>& elf_paths = {}) {
    switch (type) {
        case dev_msgs::DebugAssertTripped: {
            std::string file_str = (file_id != 0) ? resolve_file_from_hash(file_id, elf_paths) : "unknown file";
            std::string msg = resolve_assert_message(file_id, line_num, elf_paths);
            if (!msg.empty()) {
                return fmt::format("tripped an assert in {} on line {}: \"{}\".", file_str, line_num, msg);
            }
            return fmt::format("tripped an assert in {} on line {}.", file_str, line_num);
        }
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
            // line_num holds mepc (faulting instruction address) for hardware faults
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
