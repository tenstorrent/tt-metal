// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// C view of tt-metal's X280 / Quasar DM cache primitives.
// Implemented by x280_cache_tt.cc, which includes the real tt-metal header.

#ifndef X280_CACHE_TT_H_
#define X280_CACHE_TT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void tt_x280_flush_l1_dcache(uintptr_t addr);
void tt_x280_invalidate_l1_dcache(uintptr_t addr);
void tt_x280_invalidate_l1_icache(void);

void tt_x280_flush_l2_cache_line(uintptr_t addr);
void tt_x280_flush_l2_cache_range(uintptr_t addr, size_t size);
void tt_x280_flush_l2_cache_full(void);
void tt_x280_invalidate_l2_cache_line(uintptr_t addr);

#ifdef __cplusplus
}
#endif

#endif  // X280_CACHE_TT_H_
