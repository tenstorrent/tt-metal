// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// cce_scoped_lock_{producer,consumer} kernel ABI. Both kernels and the host (dram/test_mimir_dm_scoped_lock.cpp)
// include it. Args, modes, codes, polls and status words are append-only: a later stage adds entries after the last one
// and never renumbers.
namespace cce_scoped_lock_abi {

inline constexpr uint32_t kLineBytes = 64;
inline constexpr uint32_t kWordsPerLine = kLineBytes / sizeof(uint32_t);

// The probe region is kNumLines lines of TL1; the first word of each line is its probe word. The host seeds line l's
// probe word with kOldBase + l, and the producer stores kNewBase + l (test_scoped_lock_cache.cpp sentinels).
inline constexpr uint32_t kNumLines = 8;
inline constexpr uint32_t kRegionBytes = kNumLines * kLineBytes;
inline constexpr uint32_t kOldBase = 0xBB00, kNewBase = 0xAA00;

// Runtime args, the same vector for both kernels except kArgStatusOffset. Every offset is a 64 B aligned TL1 offset.
// The lock covers lines [kArgLockFirstLine, kArgLockFirstLine + kArgLockNumLines). kArgWarmedFlagOffset (consumer ->
// producer) and kArgPublishedFlagOffset (producer -> consumer) are flag lines the host zeroes; each flag is set to 1.
// kArgResultOffset is the consumer's kNumResultLines result lines. kArgPollBudget* is rdcycle cycles per bounded wait
// or poll and must be nonzero.
inline constexpr uint32_t kArgMode = 0, kArgStatusOffset = 1, kArgPollBudgetLo = 2, kArgPollBudgetHi = 3,
                          kArgRegionOffset = 4, kArgLockFirstLine = 5, kArgLockNumLines = 6, kArgWarmedFlagOffset = 7,
                          kArgPublishedFlagOffset = 8, kArgResultOffset = 9, kNumArgs = 10;

// The values match ScopedLockCacheMode (scoped_lock_cache_common.h). The lock's acquire is
// scoped_lock_acquire_cache_ops and its release scoped_lock_release_cache_ops over the locked lines, as
// CoreLocalMem::scoped_lock does.
// kModeInvalidateOnAcquire: the consumer holds the lock. It reads every probe word through the cache, which caches OLD,
// and sets warmed. The producer stores NEW to every probe word through the uncached alias, which bypasses the caches,
// and sets published. The consumer acquires the lock, releases it, and reads every probe word through the cache again:
// locked lines must miss and read NEW, unlocked lines hit OLD.
// kModeFlushOnRelease: the producer holds the lock. It reads every probe word through the cache, acquires the lock,
// stores NEW to every probe word through the cache, releases the lock, reads the locked lines once
// (kFlagLandedAtFirstRead if they hold NEW), polls them, discards every probe line from its caches and sets published.
// The consumer reads every probe word through the uncached alias: locked lines must read NEW in TL1, unlocked lines
// OLD.
inline constexpr uint32_t kModeInvalidateOnAcquire = 0, kModeFlushOnRelease = 1;

// Consumer result lines, written through the uncached alias: word l of a line is line l's probe word.
// kResultWarm: the cached reads before the lock (kModeInvalidateOnAcquire). kResultCached: the cached reads after the
// lock (kModeInvalidateOnAcquire). kResultUncached: the uncached reads after the producer published (both modes).
// Lines a mode does not write keep the host's fill.
inline constexpr uint32_t kResultWarm = 0, kResultCached = 1, kResultUncached = 2, kNumResultLines = 3;

// Status line, one per hart, written once at exit: every field, `fence w, w`, then the magic. kStatusFlags holds the
// kFlag* bits of the waits and polls that succeeded. kStatusLastPoll is the last wait or poll that ran and
// kStatusLastPollReady is 1 if it succeeded, 0 if it used its budget. kStatusFlagObserved is the last flag value the
// hart's wait read. On a data timeout the mismatch words describe the last pass. The consumer has no data poll and
// writes 0 to kStatusPollCycles and the mismatch words.
inline constexpr uint32_t kStatusMagic = 0, kStatusCode = 1, kStatusFlags = 2, kStatusWaitCycles = 3,
                          kStatusPollCycles = 4, kStatusLastPoll = 5, kStatusLastPollReady = 6, kStatusFlagObserved = 7,
                          kStatusMismatchedWords = 8, kStatusFirstMismatchWord = 9, kStatusObserved = 10,
                          kStatusExpected = 11, kStatusWords = 12;
static_assert(kStatusWords * sizeof(uint32_t) <= kLineBytes);
inline constexpr uint32_t kProducerDoneMagic = 0x5C0FE0DEu, kConsumerDoneMagic = 0x5C0C50DEu,
                          kStatusPoison = 0xBADD0E00u;
inline constexpr uint32_t kFlagPeerSignalled = 1u << 0, kFlagLanded = 1u << 1, kFlagLandedAtFirstRead = 1u << 2;
// kPollPeer: the wait for the other hart's flag. kPollData: the producer's poll of the locked lines through the
// uncached alias after the release flush (kModeFlushOnRelease).
inline constexpr uint32_t kPollNone = 0, kPollPeer = 1, kPollData = 2;

// kCodeDone: every wait and poll succeeded. Otherwise the code names the first that used its budget, and the hart
// stops there.
inline constexpr uint32_t kCodeDone = 1, kCodePeerTimeout = 2, kCodeDataTimeout = 3, kCodeBadArgs = 4;

}  // namespace cce_scoped_lock_abi
