// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/regions/metal_l1_region.hpp -- checked L1 window query.

#ifndef X280_HARNESS_REGIONS_METAL_L1_REGION_HPP
#define X280_HARNESS_REGIONS_METAL_L1_REGION_HPP

#include "harness/regions/l1_window.hpp"

#include "llrt/hal.hpp"

#include <cstdint>
#include <stdexcept>

namespace x280::harness::regions {

using tt::tt_metal::Hal;
using tt::tt_metal::HalL1MemAddrType;
using tt::tt_metal::HalProgrammableCoreType;

// Queries unreserved L1 window for DRAM, IDLE_ETH or ACTIVE_ETH.
// local.base is a device-local byte address and local.size a byte count passed
// to kernels. host_noc_base is the full tagged NoC address for DRAM (used by
// cluster host APIs) or local.base for either Ethernet type. Throws std::invalid_argument for any
// other core_type. Propagates HAL exceptions on failure.
// An `alignment` other than 1 moves the start up to it with align_window_start,
// which throws as documented there (an empty window among them).
inline L1Window query_unreserved_window(const Hal& hal, HalProgrammableCoreType core_type, uint64_t alignment = 1u) {
    if (core_type != HalProgrammableCoreType::DRAM && core_type != HalProgrammableCoreType::IDLE_ETH &&
        core_type != HalProgrammableCoreType::ACTIVE_ETH) {
        throw std::invalid_argument(
            "query_unreserved_window: unsupported core type (expected DRAM, IDLE_ETH or ACTIVE_ETH)");
    }
    const uint64_t base = hal.get_dev_addr(core_type, HalL1MemAddrType::UNRESERVED);
    const uint64_t size = hal.get_dev_size(core_type, HalL1MemAddrType::UNRESERVED);
    const uint64_t host_noc_base = (core_type == HalProgrammableCoreType::DRAM)
                                       ? hal.get_dev_noc_addr(core_type, HalL1MemAddrType::UNRESERVED)
                                       : base;
    const L1Window window{
        .local = {.base = base, .size = size},
        .host_noc_base = host_noc_base,
    };
    return alignment == 1u ? window : align_window_start(window, alignment);
}

}  // namespace x280::harness::regions

#endif  // X280_HARNESS_REGIONS_METAL_L1_REGION_HPP
