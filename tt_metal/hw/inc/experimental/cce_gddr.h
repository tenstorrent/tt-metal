// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "dev_mem_map.h"
#include "hostdev/cce_spa_routes.h"
#include "internal/risc_attribs.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DRISC)

namespace experimental {

inline __attribute__((always_inline)) volatile tt_l1_ptr CceSpaRoutes* cce_spa_routes() {
    return reinterpret_cast<volatile tt_l1_ptr CceSpaRoutes*>(MEM_CCE_SPA_ROUTES_BASE);
}

// local_mimir stays in the signature so existing kernels compile. The host wrote this core's
// route table, so the destination instance selects the port.
inline __attribute__((always_inline)) uint64_t
cce_gddr_address(uint32_t local_mimir, uint32_t mimir_index, uint64_t offset) {
    (void)local_mimir;
    if (mimir_index >= CCE_SPA_ROUTES_MAX_INSTANCES) {
        __builtin_trap();
    }
    volatile tt_l1_ptr CceSpaRoutes* routes = cce_spa_routes();
    const uint32_t port = routes->gddr_d2d[mimir_index];
    return routes->gddr_d2d_base[port] + static_cast<uint64_t>(mimir_index) * routes->gddr_instance_stride + offset;
}

inline __attribute__((always_inline)) void cce_gddr_read(
    uint32_t local_mimir, uint32_t mimir_index, uint64_t offset, uint32_t l1_address, uint32_t num_words) {
    const volatile uint32_t* src =
        reinterpret_cast<volatile const uint32_t*>(cce_gddr_address(local_mimir, mimir_index, offset));
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
}

inline __attribute__((always_inline)) void cce_gddr_write(
    uint32_t local_mimir, uint32_t mimir_index, uint64_t offset, uint32_t l1_address, uint32_t num_words) {
    const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile const tt_l1_ptr uint32_t*>(l1_address);
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(cce_gddr_address(local_mimir, mimir_index, offset));
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
}

inline __attribute__((always_inline)) uint64_t cce_sram_address(uint32_t cce_index, uint64_t offset) {
    volatile tt_l1_ptr CceSpaRoutes* routes = cce_spa_routes();
    return routes->sram_spa_base + static_cast<uint64_t>(cce_index) * routes->sram_tile_stride + offset;
}

inline __attribute__((always_inline)) void cce_sram_write_uint32(uint32_t cce_index, uint64_t offset, uint32_t value) {
    *reinterpret_cast<volatile uint32_t*>(cce_sram_address(cce_index, offset)) = value;
}

inline __attribute__((always_inline)) void cce_sram_write(
    uint32_t cce_index, uint64_t offset, uint32_t l1_address, uint32_t num_words) {
    const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile const tt_l1_ptr uint32_t*>(l1_address);
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(cce_sram_address(cce_index, offset));
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
}

}  // namespace experimental

#endif
