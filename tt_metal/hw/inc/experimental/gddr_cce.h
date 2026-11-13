// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "dev_mem_map.h"
#include "hostdev/mimir_cce_spa_routes.h"
#include "internal/risc_attribs.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DRISC)

namespace experimental {

inline __attribute__((always_inline)) volatile tt_l1_ptr MimirCceSpaRoutes* mimir_cce_spa_routes() {
    return reinterpret_cast<volatile tt_l1_ptr MimirCceSpaRoutes*>(MEM_MIMIR_CCE_SPA_ROUTES_BASE);
}

// The host writes each core's route table, so the destination instance alone selects the D2D port.
inline __attribute__((always_inline)) uint64_t gddr_cce_address(uint32_t mimir_index, uint64_t offset) {
    if (mimir_index >= MIMIR_CCE_SPA_ROUTES_MAX_INSTANCES) {
        __builtin_trap();
    }
    volatile tt_l1_ptr MimirCceSpaRoutes* routes = mimir_cce_spa_routes();
    const uint32_t port = routes->gddr_d2d[mimir_index];
    return routes->gddr_d2d_base[port] + static_cast<uint64_t>(mimir_index) * routes->gddr_instance_stride + offset;
}

inline __attribute__((always_inline)) void gddr_cce_read(
    uint32_t mimir_index, uint64_t offset, uint32_t l1_address, uint32_t num_words) {
    const volatile uint32_t* src = reinterpret_cast<volatile const uint32_t*>(gddr_cce_address(mimir_index, offset));
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
}

inline __attribute__((always_inline)) void gddr_cce_write(
    uint32_t mimir_index, uint64_t offset, uint32_t l1_address, uint32_t num_words) {
    const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile const tt_l1_ptr uint32_t*>(l1_address);
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(gddr_cce_address(mimir_index, offset));
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
}

inline __attribute__((always_inline)) uint64_t mimir_cce_sram_address(uint32_t mimir_cce_index, uint64_t offset) {
    volatile tt_l1_ptr MimirCceSpaRoutes* routes = mimir_cce_spa_routes();
    return routes->sram_spa_base + static_cast<uint64_t>(mimir_cce_index) * routes->sram_tile_stride + offset;
}

inline __attribute__((always_inline)) void mimir_cce_sram_write_uint32(uint32_t mimir_cce_index, uint64_t offset, uint32_t value) {
    *reinterpret_cast<volatile uint32_t*>(mimir_cce_sram_address(mimir_cce_index, offset)) = value;
}

inline __attribute__((always_inline)) void mimir_cce_sram_write(
    uint32_t mimir_cce_index, uint64_t offset, uint32_t l1_address, uint32_t num_words) {
    const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile const tt_l1_ptr uint32_t*>(l1_address);
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(mimir_cce_sram_address(mimir_cce_index, offset));
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
}

}  // namespace experimental

#endif
