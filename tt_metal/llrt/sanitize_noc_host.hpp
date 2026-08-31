// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "impl/context/metal_context.hpp"
#include "llrt/tlb_config.hpp"  // kL2cpuLimBase / kL2cpuLimTlbEnd

namespace tt {

// Host MMIO reads/writes don't have alignment restrictions, so no need to check alignment here.
inline bool debug_valid_reg_addr(const tt::tt_metal::Hal& hal, uint64_t addr, uint32_t len) {
    return hal.valid_reg_addr(static_cast<uint32_t>(addr)) && len == 4;
}

inline bool debug_valid_l1_addr(const tt::tt_metal::Hal& hal, uint64_t addr, uint32_t len) {
    const uint64_t base =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE);
    const uint64_t size =
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE);
    return addr >= base && addr + len <= base + size;
}

inline bool debug_valid_worker_addr(const tt::tt_metal::Hal& hal, uint64_t addr, uint32_t len) {
    return debug_valid_l1_addr(hal, addr, len) || debug_valid_reg_addr(hal, addr, len);
}

inline bool debug_valid_dram_addr(uint64_t addr, uint32_t len, uint64_t base, uint64_t end) {
    return addr >= base && addr + len <= end;
}

inline bool debug_valid_eth_addr(const tt::tt_metal::Hal& hal, uint64_t addr, uint32_t len) {
    const uint64_t base =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::BASE);
    const uint64_t size =
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::BASE);
    return (addr >= base && addr + len <= base + size) || debug_valid_reg_addr(hal, addr, len);
}

inline bool debug_valid_dram_l1_addr(const tt::tt_metal::Hal& hal, uint64_t addr, uint32_t len) {
    if (!hal.has_programmable_core_type(tt::tt_metal::HalProgrammableCoreType::DRAM)) {
        return debug_valid_reg_addr(hal, addr, len);
    }
    const uint64_t noc_offset = hal.get_l1_noc_offset(tt::tt_metal::HalProgrammableCoreType::DRAM);
    const uint64_t size =
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::DRAM, tt::tt_metal::HalL1MemAddrType::BASE);
    return (addr >= noc_offset && addr + len <= noc_offset + size) || debug_valid_reg_addr(hal, addr, len);
}

static bool coord_found_p(const std::vector<tt::umd::CoreCoord>& coords, tt::tt_metal::xy_pair core) {
    for (const tt::umd::CoreCoord& core_coord : coords) {
        tt::tt_metal::xy_pair item = {core_coord.x, core_coord.y};
        if (item == core) {
            return true;
        }
    }
    return false;
}

static bool coord_found_p(const std::unordered_set<tt::tt_metal::xy_pair>& coords, tt::tt_metal::xy_pair core) {
    return coords.contains(core);
}

static std::string noc_address(tt::tt_metal::xy_pair core, uint64_t a, uint32_t l) {
    std::stringstream ss;
    ss << "noc{" << core.str() << ", 0x" << std::setfill('0') << std::setw(8) << std::hex << a << ", " << std::dec << l
       << "}";
    return ss.str();
}

// NOLINTBEGIN(cppcoreguidelines-no-malloc)
static void print_stack_trace() {
    void* array[15];

    int size = backtrace(array, 15);
    char** strings = backtrace_symbols(array, size);
    if (strings != nullptr) {
        fprintf(stderr, "Obtained %d stack frames.\n", size);
        for (int i = 0; i < size; i++) {
            fprintf(stderr, "%s\n", strings[i]);
        }
    }

    free(strings);  // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
}
// NOLINTEND(cppcoreguidelines-no-malloc)

static void watcher_sanitize_host_noc(
    const char* what,
    const metal_SocDescriptor& soc_d,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_worker_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_eth_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_pcie_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_dram_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_dram_hw_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_dispatch_cores,
    const tt::tt_metal::xy_pair& core,
    uint64_t addr,
    uint32_t lbytes) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const bool dram_l1_available = hal.has_programmable_core_type(tt::tt_metal::HalProgrammableCoreType::DRAM);
    const uint64_t dram_l1_noc_offset =
        dram_l1_available ? hal.get_l1_noc_offset(tt::tt_metal::HalProgrammableCoreType::DRAM) : 0;

    if (coord_found_p(soc_d.get_cores(CoreType::PCIE, CoordSystem::NOC0), core) ||
        coord_found_p(virtual_pcie_cores, core)) {
        TT_THROW("Host watcher: bad {} NOC coord {}", what, core.str());
    } else if (
        coord_found_p(soc_d.get_cores(CoreType::DRAM, CoordSystem::NOC0), core) ||
        coord_found_p(virtual_dram_cores, core)) {
        if (dram_l1_available && coord_found_p(virtual_dram_hw_cores, core) && addr >= dram_l1_noc_offset) {
            if (!debug_valid_dram_l1_addr(hal, addr, lbytes)) {
                print_stack_trace();
                TT_THROW("Host watcher: bad {} dram L1 address {}", what, noc_address(core, addr, lbytes));
            }
        } else {
            uint64_t dram_addr_base = 0;
            uint64_t dram_addr_size = soc_d.dram_core_size;
            uint64_t dram_addr_end = dram_addr_size - dram_addr_base;
            if (!debug_valid_dram_addr(addr, lbytes, dram_addr_base, dram_addr_end)) {
                print_stack_trace();
                TT_THROW("Host watcher: bad {} dram address {}", what, noc_address(core, addr, lbytes));
            }
        }
    } else if (dram_l1_available && coord_found_p(virtual_dram_hw_cores, core)) {
        if (!debug_valid_dram_l1_addr(hal, addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} dram hw core address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(virtual_eth_cores, core)) {
        if (!debug_valid_eth_addr(hal, addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} eth address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (
        coord_found_p(soc_d.get_cores(CoreType::DISPATCH, CoordSystem::NOC0), core) ||
        coord_found_p(virtual_dispatch_cores, core)) {
        if (!debug_valid_worker_addr(hal, addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} dispatch address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(virtual_worker_cores, core)) {
        if (!debug_valid_worker_addr(hal, addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} worker address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(soc_d.get_cores(CoreType::L2CPU, CoordSystem::NOC0), core)) {
        // L2CPU tiles address LIM, which does not start at 0, so the worker/eth
        // address predicates do not apply. Validate against the LIM aperture the
        // per-tile static TLB covers. get_cores() is empty on architectures
        // without L2CPU tiles, making this branch unreachable there.
        if (addr < ll_api::kL2cpuLimBase || addr + lbytes > ll_api::kL2cpuLimTlbEnd) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} L2CPU LIM address {}", what, noc_address(core, addr, lbytes));
        }
    } else {
        // Bad COORD
        print_stack_trace();
        TT_THROW("Host watcher: bad {} NOC coord {}", what, core.str());
    }
}

inline void watcher_sanitize_host_noc_read(
    const metal_SocDescriptor& soc_d,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_worker_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_eth_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_pcie_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_dram_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_dram_hw_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_dispatch_cores,
    const tt::tt_metal::xy_pair& core,
    uint64_t addr,
    uint32_t lbytes) {
    watcher_sanitize_host_noc(
        "read",
        soc_d,
        virtual_worker_cores,
        virtual_eth_cores,
        virtual_pcie_cores,
        virtual_dram_cores,
        virtual_dram_hw_cores,
        virtual_dispatch_cores,
        core,
        addr,
        lbytes);
}

inline void watcher_sanitize_host_noc_write(
    const metal_SocDescriptor& soc_d,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_worker_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_eth_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_pcie_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_dram_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_dram_hw_cores,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_dispatch_cores,
    const tt::tt_metal::xy_pair& core,
    uint64_t addr,
    uint32_t lbytes) {
    watcher_sanitize_host_noc(
        "write",
        soc_d,
        virtual_worker_cores,
        virtual_eth_cores,
        virtual_pcie_cores,
        virtual_dram_cores,
        virtual_dram_hw_cores,
        virtual_dispatch_cores,
        core,
        addr,
        lbytes);
}

inline void watcher_sanitize_host_noc_multicast_write(
    const metal_SocDescriptor& soc_d,
    const std::unordered_set<tt::tt_metal::xy_pair>& virtual_worker_cores,
    const tt::tt_metal::xy_pair& core_start,
    const tt::tt_metal::xy_pair& core_end,
    uint64_t addr,
    uint32_t lbytes) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    // NoC torus architectures (WH/BH) support wrap-around multicasts where end < start,
    // but only for Tensix cores. DRAM/PCIe/Eth cores don't support wrap-around.
    bool has_noc_torus = (hal.get_noc_topology() == tt::tt_metal::NoCTopologyType::TORUS);
    bool is_tensix_multicast = (coord_found_p(soc_d.get_cores(CoreType::TENSIX, CoordSystem::NOC0), core_start) ||
                                coord_found_p(virtual_worker_cores, core_start)) &&
                               (coord_found_p(soc_d.get_cores(CoreType::TENSIX, CoordSystem::NOC0), core_end) ||
                                coord_found_p(virtual_worker_cores, core_end));

    // Allow wrap-around only for Tensix cores on torus architectures
    bool allow_wrap_around = has_noc_torus && is_tensix_multicast;

    if (!allow_wrap_around && (core_start.x > core_end.x || core_start.y > core_end.y)) {
        TT_THROW(
            "Host watcher: bad multicast write coordinates - start {} must be <= end {} in both x and y (multicast "
            "invalid range)",
            core_start.str(),
            core_end.str());
    }

    if (not coord_found_p(soc_d.get_cores(CoreType::TENSIX, CoordSystem::NOC0), core_start) and
        not coord_found_p(virtual_worker_cores, core_start)) {
        TT_THROW("Host watcher: bad multicast write NOC coord {} - start core is not tensix", core_start.str());
    } else if (coord_found_p(virtual_worker_cores, core_start)) {
        if (!debug_valid_worker_addr(hal, addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad multicast write worker address {}", noc_address(core_start, addr, lbytes));
        }
    }

    if (not coord_found_p(soc_d.get_cores(CoreType::TENSIX, CoordSystem::NOC0), core_end) and
        not coord_found_p(virtual_worker_cores, core_end)) {
        TT_THROW("Host watcher: bad multicast write NOC coord {} - end core is not tensix", core_end.str());
    } else if (coord_found_p(virtual_worker_cores, core_end)) {
        if (!debug_valid_worker_addr(hal, addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad multicast write worker address {}", noc_address(core_end, addr, lbytes));
        }
    }
}

}  // namespace tt
