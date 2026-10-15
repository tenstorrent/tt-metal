// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Settings shared by the reader, compute and writer kernels, all passed by the host as JIT defines.

// Output block, in tiles. Each core accumulates a BLOCK_M x BLOCK_N patch of output tiles in the
// destination registers at once, so per step of K one column slice of A (BLOCK_M tiles) and one
// row slice of B (BLOCK_N tiles) feed BLOCK_M * BLOCK_N multiplies. 1x1 is tile at a time.
#if !defined(BLOCK_M) || !defined(BLOCK_N)
#error "BLOCK_M and BLOCK_N must be defined by the host"
#endif
constexpr uint32_t kBlockM = BLOCK_M;
constexpr uint32_t kBlockN = BLOCK_N;
constexpr uint32_t kBlockTiles = kBlockM * kBlockN;

// Every circular buffer holds this many slices or blocks, so the producer can fill the next one
// while the consumer works on the current one. Must match kCbBufferDepth in long_matmul.cpp.
constexpr uint32_t kCbBufferDepth = 2;

// LONG_MATMUL_MODE. "legacy" reproduces how the published tt-ember results were measured;
// "updated" (the default) fixes the effects found since. The README lists every difference.
#if defined(LONG_MATMUL_LEGACY)
constexpr bool kLegacyMode = true;
#else
constexpr bool kLegacyMode = false;
#endif

// In updated mode an idle reader still reads its first kCbBufferDepth slices of A and of B, and
// idle compute still computes and packs its first kCbBufferDepth output blocks, so every slot of
// their circular buffers holds real data that the idle loop then recycles. (An idle writer has no
// such path: it never writes.) In legacy mode idle kernels never do real work, and the buffers
// keep whatever L1 held before.
constexpr uint32_t kRealStepsWhenIdle = kLegacyMode ? 0 : kCbBufferDepth;
