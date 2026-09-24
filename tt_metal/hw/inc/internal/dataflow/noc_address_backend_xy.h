// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Legacy XY NoC address backend: addresses carry raw (x, y) coordinates in the
// bits above NOC_ADDR_COORD_SHIFT. Each function expands to exactly the
// expression it replaced at its call sites (verified by object-code
// comparison). Include through the arch-selected "noc_address_backend.h"
// wrapper (internal/tt-1xx/ or internal/tt-2xx/quasar/), which sets the
// backend-neutral noc_address_backend alias call sites use; never include or
// name this namespace directly.
//
// No defaulted noc parameter here on purpose: callers always pass it, and a
// noc_index default would break translation units where the global is not
// declared at this point (e.g. DRISC firmware).

#include <cstdint>

namespace noc_address_backend_xy {

FORCE_INLINE uint64_t worker_address(uint32_t x, uint32_t y, uint32_t local_address, uint8_t noc) {
    return NOC_XY_ADDR(DYNAMIC_NOC_X(noc, x), DYNAMIC_NOC_Y(noc, y), local_address);
}

FORCE_INLINE uint64_t self_address(uint32_t local_address, uint8_t noc) {
    return NOC_XY_ADDR(my_x[noc], my_y[noc], local_address);
}

FORCE_INLINE uint64_t packed_worker_address(uint32_t packed_xy, uint32_t local_address) {
    return ((uint64_t)(packed_xy) << NOC_ADDR_COORD_SHIFT) | local_address;
}

FORCE_INLINE uint64_t multicast_descriptor(
    uint32_t start_x,
    uint32_t start_y,
    uint32_t end_x,
    uint32_t end_y,
    uint32_t local_address,
    uint8_t noc) {
    return NOC_MULTICAST_ADDR(
        DYNAMIC_NOC_X(noc, start_x),
        DYNAMIC_NOC_Y(noc, start_y),
        DYNAMIC_NOC_X(noc, end_x),
        DYNAMIC_NOC_Y(noc, end_y),
        local_address);
}

template <bool DRAM>
FORCE_INLINE uint64_t bank_address(uint32_t bank_index, uint32_t local_address, uint8_t noc) {
    uint32_t packed_xy;
    if constexpr (DRAM) {
        packed_xy = dram_bank_to_noc_xy[noc][bank_index];
    } else {
        packed_xy = l1_bank_to_noc_xy[noc][bank_index];
    }
    return packed_worker_address(packed_xy, local_address);
}

FORCE_INLINE uint32_t extract_local_address(uint64_t address) { return static_cast<uint32_t>(address); }

FORCE_INLINE bool is_local(uint64_t address, uint8_t noc) {
    uint32_t x = NOC_UNICAST_ADDR_X(address);
    uint32_t y = NOC_UNICAST_ADDR_Y(address);
    return x == my_x[noc] && y == my_y[noc];
}

// Dispatch go-message coordinates arrive as the raw uint8_t fields of go_msg_t.
FORCE_INLINE uint64_t dispatch_address(uint8_t x, uint8_t y, uint32_t local_address) {
    return NOC_XY_ADDR(NOC_X(x), NOC_Y(y), local_address);
}

FORCE_INLINE uint64_t system_memory_address(uint32_t local_address, uint8_t noc) {
    uint64_t pcie_core_noc_encoding =
        uint64_t(NOC_XY_PCIE_ENCODING(DYNAMIC_NOC_X(noc, PCIE_NOC_X), DYNAMIC_NOC_Y(noc, PCIE_NOC_Y)));
    return pcie_core_noc_encoding | local_address;
}

}  // namespace noc_address_backend_xy
