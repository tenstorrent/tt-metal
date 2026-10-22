// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// cce_cache_ops kernel ABI. The kernel (cce_cache_ops.cpp) and the host (dram/test_mimir_dm_cache.cpp) both include it.
// Args, modes and codes are append-only: a later stage adds fields after the last one and never renumbers.
namespace cce_cache_ops_abi {

inline constexpr uint32_t kLineBytes = 64;

// Runtime args. kArgStatusOffset is a 64 B aligned offset in the kernel's own TL1. kArgDataOffset is a hart-local TL1
// offset in a data area. The host pre-fills word i with kArgOldValue + i; the kernel writes kArgValue + i. The store
// modes need 64 B aligned data and whole lines; the write-path modes need 8 B aligned data and an even word count.
// kArgFlagOffset (64 B aligned, own line) and kArgPollBudget* (rdcycle cycles, nonzero) are read by the hart-to-hart
// modes only.
inline constexpr uint32_t kArgMode = 0, kArgStatusOffset = 1, kArgDataOffset = 2, kArgNumWords = 3, kArgValue = 4,
                          kArgOldValue = 5, kArgFlagOffset = 6, kArgPollBudgetLo = 7, kArgPollBudgetHi = 8,
                          kNumArgs = 9;

// Store modes (risc_common.h cache ops; upstream l1_dcache_test / l2_flush_test): cached stores, a cached check
// (kStatusCachedBefore), an uncached-alias check (kStatusUncachedBefore), the mode's cache operation, then an
// uncached-alias check of the result (kStatusMatches counts words holding the new value).
//   kModeNoFlush: no cache operation. After the check it discards the lines from L1 D$ and L2, so no dirty line
//   outlives the kernel.
//   kModeL1FlushLine / kModeL1FlushFull: cflush.d.l1 per line / whole L1 D$, cdiscard.d.l1 per line, then the L2
//   range flush. The discard drops whatever the L1 flush left dirty, so a lost L1 flush shows as old values.
//   kModeL1InvalidateLine: cdiscard.d.l1 per line, then the L2 range flush (no write-back expected).
//   kModeL2FlushLine / kModeL2FlushRange / kModeL2FlushFull: the L2 FLUSH64 register per line / range / all of TL1.
//   kModeL2InvalidateLine: flush the whole L1 D$ to L2, the L2 INVALIDATE64 register per line, then the L2 range
//   flush, which would write back any line the invalidate kept.
//   kModeL1InvalidateFull: flush the whole L1 D$ before the stores, then cdiscard.d.l1 of the whole L1 D$ and the L2
//   range flush (no write-back expected).
// kModeInvalidateFreshRead (upstream mode 4): cached read of the old values (kStatusWarmOld), uncached store of the
// new values, cached re-read (kStatusStaleBefore counts words still old), invalidate each line in L1 D$ and L2, then
// a cached read (kStatusMatches).
// kModeL2InvalidateFreshRead: as kModeInvalidateFreshRead, with the L2 INVALIDATE64 register per line only.
// kModeProducer / kModeConsumer: the consumer reads the data cached (kStatusWarmOld counts old words) and sets the
// flag to kFlagWarmed. The producer waits for kFlagWarmed (bounded), stores, range-flushes, then sets the flag to
// kFlagReady through the uncached alias (`fence w, w` before it). The consumer waits for kFlagReady (bounded), reads
// the data cached again (kStatusStaleBefore counts words still old), invalidates each line in L1 D$ and L2 and reads
// the data cached (kStatusMatches).
// Write-path modes (upstream quasar_cache_perf, functional only): two passes (pass 0 writes the complement, pass 1 the
// new values), then an uncached-alias check. kModeWriteUncached1B / 8B: byte / 64-bit stores through the uncached
// alias, fence per pass. kModeWriteCachedRangeFlush: 64-bit cached stores and the L2 range flush per pass.
// kModeWriteCachedFastFlush (upstream mode 3): 64-bit cached stores, then bare FLUSH64 writes per line, per pass, and
// one fence at the end.
inline constexpr uint32_t kModeNoFlush = 0, kModeL1FlushLine = 1, kModeL1FlushFull = 2, kModeL1InvalidateLine = 3,
                          kModeL2FlushLine = 4, kModeL2FlushRange = 5, kModeL2FlushFull = 6, kModeL2InvalidateLine = 7,
                          kModeInvalidateFreshRead = 8, kModeProducer = 9, kModeConsumer = 10,
                          kModeWriteUncached1B = 11, kModeWriteUncached8B = 12, kModeWriteCachedRangeFlush = 13,
                          kModeWriteCachedFastFlush = 14, kModeL1InvalidateFull = 15, kModeL2InvalidateFreshRead = 16,
                          kNumModes = 17;
inline constexpr uint32_t kMaxWords = 16 * 1024;
inline constexpr uint32_t kFlagReady = 0xF1A60001u, kFlagWarmed = 0xF1A60002u;

// Status line, written once at exit: every field, `fence w, w`, then the magic. kStatusFirstMismatchWord and
// kStatusObserved describe the first word of the final check that did not hold the new value. kStatusFlagReady is 1
// once the hart saw the flag it waits for within its budget. kStatusCycles is the rdcycle count of the mode's work.
inline constexpr uint32_t kStatusMagic = 0, kStatusCode = 1, kStatusUncachedBefore = 2, kStatusStaleBefore = 3,
                          kStatusMatches = 4, kStatusFirstMismatchWord = 5, kStatusObserved = 6, kStatusFlagReady = 7,
                          kStatusCycles = 8, kStatusCachedBefore = 9, kStatusWarmOld = 10, kStatusWords = 11;
static_assert(kStatusWords * sizeof(uint32_t) <= kLineBytes);
inline constexpr uint32_t kDoneMagic = 0xCAC4E0DEu, kStatusPoison = 0xBADD0E00u;

inline constexpr uint32_t kCodeDone = 1, kCodeBadArgs = 2, kCodeFlagTimeout = 3;

}  // namespace cce_cache_ops_abi
