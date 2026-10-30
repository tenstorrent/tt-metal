// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "cce_test_status_abi.h"

// cce_scoped_lock_{producer,consumer} kernel ABI. Both kernels and the host (dram/test_mimir_dm_scoped_lock.cpp)
// include it. Args, modes, codes, polls and status words are append-only: a later stage adds entries after the last one
// and never renumbers.
namespace cce_scoped_lock_abi {

using cce_test_status_abi::kLineBytes;
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
// MM41 args. kArgConsumerCount and kArgConsumerIndex (kModeMultiConsumer): consumer c sets the warmed flag line
// kArgWarmedFlagOffset + c * kLineBytes. kArgRingSlots, kArgSlotLines and kArgRounds (kModeRingWrap): the ring is
// kArgRingSlots slots of kArgSlotLines lines from kArgRegionOffset, at most kNumLines lines, and kArgRounds rounds
// fill slot round % kArgRingSlots. kArgTargetCce (kModeNonSnoopingProducer): the consumer's CCE.
inline constexpr uint32_t kArgMode = 0, kArgStatusOffset = 1, kArgPollBudgetLo = 2, kArgPollBudgetHi = 3,
                          kArgRegionOffset = 4, kArgLockFirstLine = 5, kArgLockNumLines = 6, kArgWarmedFlagOffset = 7,
                          kArgPublishedFlagOffset = 8, kArgResultOffset = 9, kArgConsumerCount = 10,
                          kArgConsumerIndex = 11, kArgRingSlots = 12, kArgSlotLines = 13, kArgRounds = 14,
                          kArgTargetCce = 15, kNumArgs = 16;

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
// kModeMultiConsumer (MM41): kArgConsumerCount consumers on harts of the producer's CCE. Each consumer reads every
// probe word through the cache and sets its own warmed flag line. The producer waits for every warmed flag, acquires
// the lock, stores NEW to the locked lines' probe words through the cache, releases the lock, polls the locked lines
// through the uncached alias until they read NEW, and sets published. Each consumer then acquires and releases the
// lock and reads as in kModeInvalidateOnAcquire.
// kModeRingWrap (MM41): for each round r the producer waits until the consumer's read pointer (the warmed flag line)
// is at least r + 1 - kArgRingSlots, acquires slot r % kArgRingSlots, stores ring_word(r, w) to every word w of the
// slot through the cache, releases the slot, polls the slot through the uncached alias until it reads those values
// (the release flush has no completion register) and sets its write pointer (the published flag line) to r + 1. The
// consumer waits for the write pointer to pass r, acquires the slot, reads every word through the cache and counts the
// words that are not ring_word(r, w) (kStatusMismatchedWords), releases the slot and sets its read pointer to r + 1.
// kModeNonSnoopingProducer (MM41): the producer runs on another CCE and reaches the consumer's TL1 (CCE
// kArgTargetCce) by SPA only: it waits for warmed by SPA load, stores NEW to every probe word, `fence`, then sets
// published, all by SPA store, which the consumer's caches do not snoop by default. The consumer runs as in
// kModeInvalidateOnAcquire, except that after published it polls every probe word through the uncached alias until
// all read NEW (the producer's stores have landed) before it acquires the lock and before any store of its own.
inline constexpr uint32_t kModeInvalidateOnAcquire = 0, kModeFlushOnRelease = 1, kModeMultiConsumer = 2,
                          kModeRingWrap = 3, kModeNonSnoopingProducer = 4;
inline constexpr uint32_t kMaxConsumers = 7;

constexpr uint32_t ring_word(uint32_t round, uint32_t word) { return (kNewBase << 16) | (round << 8) | word; }

// Consumer result lines, written through the uncached alias: word l of a line is line l's probe word.
// kResultWarm: the cached reads before the lock. kResultCached: the cached reads after the lock. Both are written by
// every mode except kModeFlushOnRelease and kModeRingWrap. kResultUncached: the uncached reads after the producer
// published, written by every mode except kModeRingWrap. Lines a mode does not write keep the host's fill.
inline constexpr uint32_t kResultWarm = 0, kResultCached = 1, kResultUncached = 2, kNumResultLines = 3;

// Status line, one per hart, written once at exit: every field, `fence w, w`, then the magic. kStatusFlags holds the
// kFlag* bits of the waits and polls that succeeded. kStatusLastPoll is the last wait or poll that ran and
// kStatusLastPollReady is 1 if it succeeded, 0 if it used its budget. kStatusFlagObserved is the last flag value the
// hart's wait read. On a data timeout the mismatch words describe the last pass. The consumer has no data poll and
// writes 0 to kStatusPollCycles and the mismatch words, except in kModeRingWrap and kModeNonSnoopingProducer, where
// they describe its own check. kStatusRounds counts kModeRingWrap's completed rounds.
using cce_test_status_abi::kStatusCode;
using cce_test_status_abi::kStatusMagic;
inline constexpr uint32_t kStatusFlags = 2, kStatusWaitCycles = 3, kStatusPollCycles = 4, kStatusLastPoll = 5,
                          kStatusLastPollReady = 6, kStatusFlagObserved = 7, kStatusMismatchedWords = 8,
                          kStatusFirstMismatchWord = 9, kStatusObserved = 10, kStatusExpected = 11, kStatusRounds = 12,
                          kStatusWords = 13;
static_assert(kStatusWords * sizeof(uint32_t) <= kLineBytes);
inline constexpr uint32_t kProducerDoneMagic = 0x5C0FE0DEu, kConsumerDoneMagic = 0x5C0C50DEu;
using cce_test_status_abi::kStatusPoison;
inline constexpr uint32_t kFlagPeerSignalled = 1u << 0, kFlagLanded = 1u << 1, kFlagLandedAtFirstRead = 1u << 2;
// kPollPeer: the wait for the other hart's flag. kPollData: the producer's poll of the locked lines (kModeRingWrap: the
// slot) through the uncached alias after the release flush (kModeFlushOnRelease, kModeMultiConsumer, kModeRingWrap),
// or the consumer's poll of the probe words for the external producer's stores (kModeNonSnoopingProducer).
inline constexpr uint32_t kPollNone = 0, kPollPeer = 1, kPollData = 2;

// A status offset that is not a 64 B aligned line between the end of the harts' kernel text and the end of the
// hart-cacheable port gets no status line at all, so the host deadline reports it.
// kCodeDone: every wait and poll succeeded. Otherwise the code names the first that used its budget, and the hart
// stops there.
inline constexpr uint32_t kCodeDone = 1, kCodePeerTimeout = 2, kCodeDataTimeout = 3, kCodeBadArgs = 4;

}  // namespace cce_scoped_lock_abi
