// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Invalidates Blackhole's entire L1 cache
// Blackhole L1 cache is a small write-through cache (4x16B L1 lines). The cache covers all of L1 (no
// MMU or range registers).
//  Writing an address on one proc and reading it from another proc only requires the reader to invalidate.
//  Need to invalidate any address written by noc that may have been previously read by riscv
inline __attribute__((always_inline)) void invalidate_l1_cache() {
#if defined(ARCH_BLACKHOLE)
    asm("fence");
#endif
}
