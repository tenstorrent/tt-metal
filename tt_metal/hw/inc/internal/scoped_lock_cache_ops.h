// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "risc_common.h"

// Cache primitives behind scoped_lock APIs for: CoreLocalMem / Scratchpad / LocalTensorAccessor,
// and DataflowBuffer's scoped_read_lock/scoped_write_lock.

inline __attribute__((always_inline)) void scoped_lock_acquire_cache_ops(uintptr_t addr, uint32_t num_bytes) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    invalidate_l2_cache_range(addr, num_bytes);
#else
    (void)addr;
    (void)num_bytes;
#endif
}

inline __attribute__((always_inline)) void scoped_lock_release_cache_ops(uintptr_t addr, uint32_t num_bytes) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    flush_l2_cache_range(addr, num_bytes);
#else
    (void)addr;
    (void)num_bytes;
#endif
}
