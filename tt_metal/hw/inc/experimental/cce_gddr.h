// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "internal/risc_attribs.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DRISC)

namespace experimental {

// Package GDDR is one 8 GiB block per Mimir. The CCE remapper claims only this
// die's window, so a local index hits the remapper and a remote index falls
// through to the NOC.
inline constexpr uint64_t CCE_GDDR_SPA_BASE = 0x1000000000000ULL;
inline constexpr uint64_t CCE_GDDR_MIMIR_STRIDE = 0x200000000ULL;

inline __attribute__((always_inline)) uint64_t cce_gddr_address(uint32_t mimir_index, uint64_t offset) {
    return CCE_GDDR_SPA_BASE + static_cast<uint64_t>(mimir_index) * CCE_GDDR_MIMIR_STRIDE + offset;
}

inline __attribute__((always_inline)) void cce_gddr_read(
    uint32_t mimir_index, uint64_t offset, uint32_t l1_address, uint32_t num_words) {
    const volatile uint32_t* src = reinterpret_cast<volatile const uint32_t*>(cce_gddr_address(mimir_index, offset));
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
}

inline __attribute__((always_inline)) void cce_gddr_write(
    uint32_t mimir_index, uint64_t offset, uint32_t l1_address, uint32_t num_words) {
    const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile const tt_l1_ptr uint32_t*>(l1_address);
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(cce_gddr_address(mimir_index, offset));
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
