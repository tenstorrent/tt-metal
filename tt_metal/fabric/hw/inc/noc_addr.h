// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp"

#include "dataflow_api.h"
#include "noc_nonblocking_api.h"
#include <cstdint>

// NOTE: This will eventually be updated with an official API
static constexpr size_t VIRTUAL_COORDS_START_X = 16;
static constexpr size_t VIRTUAL_COORDS_START_Y = 16;
FORCE_INLINE bool is_using_noc_coords(uint16_t noc_x, uint16_t noc_y) {
#ifdef ARCH_WORMHOLE
    return noc_x < VIRTUAL_COORDS_START_X && noc_y < VIRTUAL_COORDS_START_Y;
#elif defined(COORDINATE_VIRTUALIZATION_ENABLED) && COORDINATE_VIRTUALIZATION_ENABLED == 1
    return false;
#else
    return true;
#endif
}

FORCE_INLINE uint64_t
safe_get_noc_addr(uint8_t dest_noc_x, uint8_t dest_noc_y, uint32_t dest_bank_addr, uint8_t noc_id = noc_index) {
    bool using_noc_coords = is_using_noc_coords(dest_noc_x, dest_noc_y);
    uint8_t noc_x = dest_noc_x;
    uint8_t noc_y = dest_noc_y;
    if (using_noc_coords) {
        noc_x = NOC_0_X_PHYS_COORD(noc_id, noc_size_x, dest_noc_x);
        noc_y = NOC_0_Y_PHYS_COORD(noc_id, noc_size_y, dest_noc_y);
    }
    return get_noc_addr(noc_x, noc_y, dest_bank_addr, noc_id);
}
// TODO: COMMONIZE WITH THE ONE IN `ccl_send_writer.cpp`
FORCE_INLINE std::pair<tt::tt_fabric::WorkerXY, uint32_t> get_noc_address_components(uint64_t noc_addr) {
    const size_t bank_addr = noc_addr & 0xFFFFFFFF;
    const size_t noc_x = NOC_UNICAST_ADDR_X(noc_addr);
    const size_t noc_y = NOC_UNICAST_ADDR_Y(noc_addr);
    return {tt::tt_fabric::WorkerXY(noc_x, noc_y), bank_addr};
}
