// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "internal/risc_attribs.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DRISC)

namespace experimental {

// Package GDDR is one 8 GiB block per Mimir on the D2D0 route. The CCE remapper
// claims only this die's block, so a local index hits the remapper.
inline constexpr uint64_t CCE_GDDR_SPA_BASE = 0x1000000000000ULL;
inline constexpr uint64_t CCE_GDDR_MIMIR_STRIDE = 0x200000000ULL;
// Same cells on the D2D1 route: 16 instance strides above the D2D0 base.
// On MMK the Mimir-to-Mimir wire is M0 D2D1 <-> M1 D2D0. A sender on Mimir 0
// reaches another die through this window. A sender on Mimir 1 uses the D2D0
// window. M0 D2D0 is not connected to Mimir 1.
inline constexpr uint64_t CCE_GDDR_D2D1_SPA_BASE = 0x1002000000000ULL;

inline __attribute__((always_inline)) uint64_t
cce_gddr_address(uint32_t local_mimir, uint32_t mimir_index, uint64_t offset) {
    const uint64_t route_base =
        (local_mimir == 0 && mimir_index != local_mimir) ? CCE_GDDR_D2D1_SPA_BASE : CCE_GDDR_SPA_BASE;
    return route_base + static_cast<uint64_t>(mimir_index) * CCE_GDDR_MIMIR_STRIDE + offset;
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

// Package CCE SRAM is 4 MiB per CCE, grouped as 8 MiB per Mimir.
inline constexpr uint64_t CCE_SRAM_SPA_BASE = 0x1280000000ULL;
inline constexpr uint64_t CCE_SRAM_TILE_STRIDE = 0x400000ULL;

inline __attribute__((always_inline)) uint64_t cce_sram_address(uint32_t cce_index, uint64_t offset) {
    return CCE_SRAM_SPA_BASE + static_cast<uint64_t>(cce_index) * CCE_SRAM_TILE_STRIDE + offset;
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
