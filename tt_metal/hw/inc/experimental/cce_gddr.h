// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "internal/risc_attribs.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DRISC)

namespace experimental {

// CCE load/store accesses reach GDDR through the local address remapper. Metal
// buffer addresses are offsets within a partition; convert one to the CCE SPA
// window before dereferencing it.
inline constexpr uint64_t CCE_GDDR_SPA_BASE = 0x1000000000000ULL;
inline constexpr uint64_t CCE_GDDR_PARTITION_STRIDE = 0x200000000ULL;

inline __attribute__((always_inline)) uint64_t cce_gddr_address(uint32_t partition, uint32_t buffer_address) {
    return CCE_GDDR_SPA_BASE + static_cast<uint64_t>(partition) * CCE_GDDR_PARTITION_STRIDE + buffer_address;
}

inline __attribute__((always_inline)) void cce_gddr_read(
    uint32_t partition, uint32_t buffer_address, uint32_t l1_address, uint32_t num_words) {
    const volatile uint32_t* src =
        reinterpret_cast<volatile const uint32_t*>(cce_gddr_address(partition, buffer_address));
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
}

inline __attribute__((always_inline)) void cce_gddr_write(
    uint32_t partition, uint32_t buffer_address, uint32_t l1_address, uint32_t num_words) {
    const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile const tt_l1_ptr uint32_t*>(l1_address);
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(cce_gddr_address(partition, buffer_address));
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = src[i];
    }
}

}  // namespace experimental

#endif
