// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// C linkage wrappers around tt-metal's X280 / Quasar DM cache primitives.
//
// tt_metal/hw/inc/internal/tt-2xx/risc_common.h defines those primitives inside
// `#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)`, as always-inline C++
// functions. This translation unit is the only place in the demo that includes
// that header; it re-exports the handful of operations the freedom-metal C
// program needs so the C side does not have to drag in the whole tt-metal
// device-header world.
//
// The header is included unmodified, straight out of the tt-metal tree. That is
// the point of the demo: the same source that builds Quasar DM firmware also
// builds inside a freedom-e-sdk program.

#include <cstddef>
#include <cstdint>

#include "internal/tt-2xx/risc_common.h"

extern "C" {

// L1 D$ (4 KB, 2-way, write-back, private per DM core).
// Emits SiFive CFLUSH.D.L1 / CDISCARD.D.L1 (spelled tt.cache.* by sfpi).
void tt_x280_flush_l1_dcache(uintptr_t addr) { flush_l1_dcache(addr); }
void tt_x280_invalidate_l1_dcache(uintptr_t addr) { invalidate_l1_dcache(addr); }

// L1 I$ (4 KB, 2-way). FENCE.I.
void tt_x280_invalidate_l1_icache(void) { invalidate_l1_icache(); }

// L2 (128 KB, 4-way, shared across the 8 DM cores). Memory-mapped cache
// controller registers; a flush probes L1 D$ for dirty lines on the way out.
void tt_x280_flush_l2_cache_line(uintptr_t addr) { flush_l2_cache_line(addr); }
void tt_x280_flush_l2_cache_range(uintptr_t addr, size_t size) { flush_l2_cache_range(addr, size); }
void tt_x280_flush_l2_cache_full(void) { flush_l2_cache_full(); }
void tt_x280_invalidate_l2_cache_line(uintptr_t addr) { invalidate_l2_cache_line(addr); }

}  // extern "C"
