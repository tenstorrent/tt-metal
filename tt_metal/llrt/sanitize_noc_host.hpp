// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "impl/context/metal_context.hpp"

namespace tt {

// Host MMIO reads/writes don't have alignment restrictions, so no need to check alignment here.
#define DEBUG_VALID_L1_ADDR(a, l) (((a) >= HAL_MEM_L1_BASE) && ((a) + (l) <= HAL_MEM_L1_BASE + HAL_MEM_L1_SIZE))

#define DEBUG_VALID_REG_ADDR(a) tt::tt_metal::MetalContext::instance().hal().valid_reg_addr(a)
#define DEBUG_VALID_WORKER_ADDR(a, l) (DEBUG_VALID_L1_ADDR(a, l) || (DEBUG_VALID_REG_ADDR(a) && (l) == 4))
#define DEBUG_VALID_DRAM_ADDR(a, l, b, e) (((a) >= (b)) && ((a) + (l) <= (e)))

#define DEBUG_VALID_ETH_ADDR(a, l)                                                        \
    ((((a) >= HAL_MEM_ETH_BASE) && ((a) + (l) <= HAL_MEM_ETH_BASE + HAL_MEM_ETH_SIZE)) || \
     (DEBUG_VALID_REG_ADDR(a) && (l) == 4))

static bool coord_found_p(const std::vector<tt::umd::CoreCoord>& coords, CoreCoord core) {
    for (const tt::umd::CoreCoord& core_coord : coords) {
        CoreCoord item = {core_coord.x, core_coord.y};
        if (item == core) {
            return true;
        }
    }
    return false;
}

static bool coord_found_p(const std::unordered_set<CoreCoord>& coords, CoreCoord core) { return coords.contains(core); }

static std::string noc_address(CoreCoord core, uint64_t a, uint32_t l) {
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
    const std::unordered_set<CoreCoord>& virtual_worker_cores,
    const std::unordered_set<CoreCoord>& virtual_eth_cores,
    const std::unordered_set<CoreCoord>& virtual_pcie_cores,
    const std::unordered_set<CoreCoord>& virtual_dram_cores,
    const CoreCoord& core,
    uint64_t addr,
    uint32_t lbytes) {
    if (coord_found_p(soc_d.get_cores(CoreType::PCIE, CoordSystem::NOC0), core) ||
        coord_found_p(virtual_pcie_cores, core)) {
        TT_THROW("Host watcher: bad {} NOC coord {}", what, core.str());
    } else if (
        coord_found_p(soc_d.get_cores(CoreType::DRAM, CoordSystem::NOC0), core) ||
        coord_found_p(virtual_dram_cores, core)) {
        uint64_t dram_addr_base = 0;
        uint64_t dram_addr_size = soc_d.dram_core_size;
        uint64_t dram_addr_end = dram_addr_size - dram_addr_base;
        if (!DEBUG_VALID_DRAM_ADDR(addr, lbytes, dram_addr_base, dram_addr_end)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} dram address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(virtual_eth_cores, core)) {
        if (!DEBUG_VALID_ETH_ADDR(addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} eth address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(virtual_worker_cores, core)) {
        if (!DEBUG_VALID_WORKER_ADDR(addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} worker address {}", what, noc_address(core, addr, lbytes));
        }
    } else {
        // Bad COORD
        print_stack_trace();
        TT_THROW("Host watcher: bad {} NOC coord {}", what, core.str());
    }
}

inline void watcher_sanitize_host_noc_read(
    const metal_SocDescriptor& soc_d,
    const std::unordered_set<CoreCoord>& virtual_worker_cores,
    const std::unordered_set<CoreCoord>& virtual_eth_cores,
    const std::unordered_set<CoreCoord>& virtual_pcie_cores,
    const std::unordered_set<CoreCoord>& virtual_dram_cores,
    const CoreCoord& core,
    uint64_t addr,
    uint32_t lbytes) {
    watcher_sanitize_host_noc(
        "read",
        soc_d,
        virtual_worker_cores,
        virtual_eth_cores,
        virtual_pcie_cores,
        virtual_dram_cores,
        core,
        addr,
        lbytes);
}

inline void watcher_sanitize_host_noc_write(
    const metal_SocDescriptor& soc_d,
    const std::unordered_set<CoreCoord>& virtual_worker_cores,
    const std::unordered_set<CoreCoord>& virtual_eth_cores,
    const std::unordered_set<CoreCoord>& virtual_pcie_cores,
    const std::unordered_set<CoreCoord>& virtual_dram_cores,
    const CoreCoord& core,
    uint64_t addr,
    uint32_t lbytes) {
    watcher_sanitize_host_noc(
        "write",
        soc_d,
        virtual_worker_cores,
        virtual_eth_cores,
        virtual_pcie_cores,
        virtual_dram_cores,
        core,
        addr,
        lbytes);
}

inline void watcher_sanitize_host_noc_multicast_write(
    const metal_SocDescriptor& soc_d,
    const std::unordered_set<CoreCoord>& virtual_worker_cores,
    const CoreCoord& core_start,
    const CoreCoord& core_end,
    uint64_t addr,
    uint32_t lbytes) {
    if (core_start.x > core_end.x || core_start.y > core_end.y) {
        TT_THROW(
            "Host watcher: bad multicast write coordinates - start {} must be <= end {} in both x and y",
            core_start.str(),
            core_end.str());
    }

    if (not coord_found_p(soc_d.get_cores(CoreType::TENSIX, CoordSystem::NOC0), core_start) and
        not coord_found_p(virtual_worker_cores, core_start)) {
        TT_THROW("Host watcher: bad multicast write NOC coord {} - start core is not tensix", core_start.str());
    } else if (coord_found_p(virtual_worker_cores, core_start)) {
        if (!DEBUG_VALID_WORKER_ADDR(addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad multicast write worker address {}", noc_address(core_start, addr, lbytes));
        }
    }

    if (not coord_found_p(soc_d.get_cores(CoreType::TENSIX, CoordSystem::NOC0), core_end) and
        not coord_found_p(virtual_worker_cores, core_end)) {
        TT_THROW("Host watcher: bad multicast write NOC coord {} - end core is not tensix", core_end.str());
    } else if (coord_found_p(virtual_worker_cores, core_end)) {
        if (!DEBUG_VALID_WORKER_ADDR(addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad multicast write worker address {}", noc_address(core_end, addr, lbytes));
        }
    }
}

}  // namespace tt
