// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// C wrappers over unmodified tt-metal risc_common.h (ARCH_QUASAR && COMPILE_FOR_DM).

#include <cstddef>
#include <cstdint>

#include "internal/tt-2xx/risc_common.h"

extern "C" {

void tt_x280_flush_l1_dcache(uintptr_t addr) { flush_l1_dcache(addr); }
void tt_x280_invalidate_l1_dcache(uintptr_t addr) { invalidate_l1_dcache(addr); }
void tt_x280_invalidate_l1_icache(void) { invalidate_l1_icache(); }
void tt_x280_flush_l2_cache_line(uintptr_t addr) { flush_l2_cache_line(addr); }
void tt_x280_flush_l2_cache_range(uintptr_t addr, size_t size) { flush_l2_cache_range(addr, size); }
void tt_x280_flush_l2_cache_full(void) { flush_l2_cache_full(); }
void tt_x280_invalidate_l2_cache_line(uintptr_t addr) { invalidate_l2_cache_line(addr); }

}  // extern "C"
