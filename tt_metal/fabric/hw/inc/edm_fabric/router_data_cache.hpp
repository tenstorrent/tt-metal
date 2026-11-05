// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_common.h"
#include "dev_mem_map.h"

// The router's own view of its L1 address @p addr; 0 (unset) stays 0. A Keraunos CCE (Quasar Ethernet) reaches its
// SRAM through the uncached alias: peers and the host change that memory by DMA, and the CCE cannot invalidate its
// data cache. Addresses the router hands to other cores or to its Ethernet peer stay plain offsets.
FORCE_INLINE constexpr uint32_t router_l1(uint32_t addr) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_ERISC)
    return addr == 0 ? 0 : addr + MEM_IERISC_UNCACHED_BASE;
#else
    return addr;
#endif
}

// The plain L1 offset of the router's own address @p addr, from either view. A DMA engine or a command buffer takes
// this, not router_l1(): the uncached alias is for CPU loads and stores only.
FORCE_INLINE constexpr uint32_t router_l1_offset(uint32_t addr) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_ERISC)
    static_assert(
        (MEM_IERISC_UNCACHED_BASE & (MEM_IERISC_UNCACHED_BASE - 1)) == 0 &&
            MEM_IERISC_UNCACHED_BASE >= MEM_IERISC_L1_SIZE,
        "Clearing the alias bit must leave every L1 offset unchanged");
    return addr & ~static_cast<uint32_t>(MEM_IERISC_UNCACHED_BASE);
#else
    return addr;
#endif
}

template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE void router_invalidate_l1_cache() {
    if constexpr (RISC_CPU_DATA_CACHE_ENABLED) {
        invalidate_l1_cache();
    }
}
