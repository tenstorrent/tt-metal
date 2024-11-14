// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "noc/noc_parameters.h"
#include "noc/noc_overlay_parameters.h"

// FIXME: ARCH_NAME specific include
#include "dev_mem_map.h" // MEM_[L1/ETH]_BASE

#pragma once

#undef MEM_L1_BASE
#undef MEM_ETH_BASE
constexpr std::uint32_t MEM_L1_BASE = 0x0;
constexpr std::uint32_t MEM_ETH_BASE = 0x0;

namespace {
  inline std::uint32_t compute_size(const tt::tt_metal::Hal& hal, tt::tt_metal::HalProgrammableCoreType core_type) {
    return hal.get_dev_size(core_type, tt::tt_metal::HalL1MemAddrType::UNRESERVED) +
           hal.get_dev_addr(core_type, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
  }
  inline std::uint32_t mem_l1_size(const tt::tt_metal::Hal& hal) {
    return compute_size(hal, tt::tt_metal::HalProgrammableCoreType::TENSIX);
  }
  inline std::uint32_t mem_eth_size(const tt::tt_metal::Hal& hal) {
    if (hal.get_arch() == tt::ARCH::GRAYSKULL) {
      return 0;
    }
    return compute_size(hal, tt::tt_metal::HalProgrammableCoreType::IDLE_ETH);
  }
}

namespace tt {

// Host MMIO reads/writes don't have alignment restrictions, so no need to check alignment here.
#define DEBUG_VALID_L1_ADDR(a, l) (((a) >= MEM_L1_BASE) && ((a) + (l) <= MEM_L1_BASE + mem_l1_size(tt_metal::hal)))

// what's the size of the NOC<n> address space?  using 0x1000 for now
#define DEBUG_VALID_REG_ADDR(a)                                                        \
    (                                                                                  \
     (((a) >= NOC_OVERLAY_START_ADDR) && ((a) < NOC_OVERLAY_START_ADDR + NOC_STREAM_REG_SPACE_SIZE * NOC_NUM_STREAMS)) || \
     (((a) >= NOC0_REGS_START_ADDR) && ((a) < NOC0_REGS_START_ADDR + 0x1000)) || \
     (((a) >= NOC1_REGS_START_ADDR) && ((a) < NOC1_REGS_START_ADDR + 0x1000)) || \
     ((a) == RISCV_DEBUG_REG_SOFT_RESET_0))
#define DEBUG_VALID_WORKER_ADDR(a, l) (DEBUG_VALID_L1_ADDR(a, l) || (DEBUG_VALID_REG_ADDR(a) && (l) == 4))
#define DEBUG_VALID_DRAM_ADDR(a, l, b, e) (((a) >= b) && ((a) + (l) <= e))

#define DEBUG_VALID_ETH_ADDR(a, l) (((a) >= MEM_ETH_BASE) && ((a) + (l) <= MEM_ETH_BASE + mem_eth_size(tt_metal::hal)))

static bool coord_found_p(std::vector<CoreCoord>coords, CoreCoord core) {
    for (CoreCoord item : coords) {
        if (item == core) return true;
    }
    return false;
}

static bool coord_found_p(CoreCoord range, CoreCoord core) {
    return
        core.x >= 1 && core.x <= range.x &&
        core.y >= 1 && core.y <= range.y;
}

static string noc_address(CoreCoord core, uint64_t a, uint32_t l) {
    std::stringstream ss;
    ss << "noc{" << core.str() << ", 0x" << std::setfill('0') << std::setw(8) << std::hex << a << ", " << std::dec << l << "}";
    return ss.str();
}

static void print_stack_trace (void) {
    void *array[15];

    int size = backtrace (array, 15);
    char **strings = backtrace_symbols(array, size);
    if (strings != NULL) {
        fprintf(stderr, "Obtained %d stack frames.\n", size);
        for (int i = 0; i < size; i++) fprintf(stderr, "%s\n", strings[i]);
    }

    free (strings);
}

static void watcher_sanitize_host_noc(const char* what,
                                      const metal_SocDescriptor& soc_d,
                                      const CoreCoord &core,
                                      uint64_t addr,
                                      uint32_t lbytes) {

    if (coord_found_p(soc_d.get_pcie_cores(), core)) {
        TT_THROW("Host watcher: bad {} NOC coord {}", what, core.str());
    } else if (coord_found_p(soc_d.get_dram_cores(), core)) {
        uint64_t dram_addr_base = 0;
        uint64_t dram_addr_size = soc_d.dram_core_size;
        uint64_t dram_addr_end = dram_addr_size - dram_addr_base;
        if (!DEBUG_VALID_DRAM_ADDR(addr, lbytes, dram_addr_base, dram_addr_end)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} dram address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(soc_d.get_physical_ethernet_cores(), core)) {
        if (!DEBUG_VALID_ETH_ADDR(addr, lbytes)) {
            print_stack_trace();
            TT_THROW("Host watcher: bad {} eth address {}", what, noc_address(core, addr, lbytes));
        }
    } else if (coord_found_p(soc_d.grid_size, core)) {
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

void watcher_sanitize_host_noc_read(const metal_SocDescriptor& soc_d, const CoreCoord& core, uint64_t addr, uint32_t lbytes) {
    watcher_sanitize_host_noc("read", soc_d, core, addr, lbytes);
}

void watcher_sanitize_host_noc_write(const metal_SocDescriptor& soc_d, const CoreCoord& core, uint64_t addr, uint32_t lbytes) {
    watcher_sanitize_host_noc("write", soc_d, core, addr, lbytes);
}

} // namespace tt
