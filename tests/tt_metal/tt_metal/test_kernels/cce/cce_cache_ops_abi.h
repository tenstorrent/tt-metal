// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "cce_test_status_abi.h"

// cce_cache_ops kernel ABI. The kernel (cce_cache_ops.cpp) and the host (dram/test_mimir_dm_cache.cpp) both include it.
// Args, modes and codes are append-only: a later stage adds fields after the last one and never renumbers.
namespace cce_cache_ops_abi {

using cce_test_status_abi::kLineBytes;

// Runtime args. kArgStatusOffset is a 64 B aligned offset in the kernel's own TL1. kArgDataOffset is a hart-local TL1
// offset in a data area. The host pre-fills word i with kArgOldValue + i; the kernel writes kArgValue + i. The store
// modes need 64 B aligned data and whole lines; the write-path modes need 8 B aligned data and an even word count.
// kArgFlagOffset (64 B aligned, own line) and kArgPollBudget* (rdcycle cycles, nonzero) are read by the hart-to-hart
// modes only.
// kArgStrideBytes, kArgCount and kArgAuxOffset (a hart-local TL1 offset, 64 B aligned) are read by the MM41 modes that
// name them. MM41 args (from kArgSeed) are described with the modes that read them. kArgTraceOffset is a 64 B aligned
// offset in the kernel's own TL1 that receives per-pass records after the mode's work, through the uncached alias.
// kArgTargetCce is the CCE whose TL1 an SPA mode reads or writes at the package SPA (experimental::cce_sram_address).
// kArgDataWindowCycles bounds a mode's second, shorter data poll (rdcycle cycles, nonzero). kArgPeerStatusOffset (own
// TL1, 64 B aligned, 0 for none) and kArgPeerMagic name another kernel's status line and done magic that
// kModeExternalReader waits for.
// Every region a mode touches (status line, data, flag lines, trace, aux lines up to aux + (count - 1) * stride, other
// words) must lie inside the test TL1 area, from the end of the harts' kernel text to the end of the hart-cacheable
// port (4 MiB), computed in 64 bits; flag and trace offsets a mode uses must be nonzero. Otherwise the code is
// kCodeBadArgs (MM41 modes). A status offset that is not a 64 B aligned line in that area gets no
// status line at all, so the host deadline reports it.
inline constexpr uint32_t kArgMode = 0, kArgStatusOffset = 1, kArgDataOffset = 2, kArgNumWords = 3, kArgValue = 4,
                          kArgOldValue = 5, kArgFlagOffset = 6, kArgPollBudgetLo = 7, kArgPollBudgetHi = 8,
                          kArgStrideBytes = 9, kArgCount = 10, kArgAuxOffset = 11, kArgSeed = 12, kArgPattern = 13,
                          kArgPasses = 14, kArgTraceOffset = 15, kArgHartSlot = 16, kArgNumHarts = 17,
                          kArgOtherOffset = 18, kArgOtherWords = 19, kArgOtherValue = 20, kArgTargetCce = 21,
                          kArgSteps = 22, kArgDataWindowCycles = 23, kArgPeerStatusOffset = 24, kArgPeerMagic = 25,
                          kNumArgs = 26;

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
// kModeNull: writes only its status line (MM41 EightHartLaunch); the data args are not read.
// kModeReadConfig: reads the CCE snoop-master config (NOC_SNOOP_TL_MASTER_CFG) into kStatusRegisterA and the L2
// CONFIGURATION register into kStatusRegisterB; the data args are not read.
//
// MM41 modes. "Old" word i is kArgOldValue + i and "new" word i is kArgValue + i unless a mode says otherwise.
// kModeDiscardFull: one inline-asm block cleans both caches (cflush.d.l1 x0, then FLUSH64 of every cacheable TL1 line
// and a drain load), stores the new values cached, counts them in a cached read (kStatusCachedBefore) and runs
// cdiscard.d.l1 x0, so no stack store falls between the clean and the discard. Then a bare FLUSH64 per line and an
// uncached read (kStatusMatches counts new words; the discard expects 0).
// kModeWipeFull: one inline-asm block cleans both caches as kModeDiscardFull, reads the old values cached
// (kStatusWarmOld), stores the new values through the uncached alias, reads cached again (kStatusStaleBefore counts
// old words), writes 0xFF to FULLINVALIDATE and polls it for 0 within kArgPollBudget* (kStatusRegisterA is 1 when it
// read 0), then cdiscard.d.l1 x0. A cached read follows (kStatusMatches). kCodeWipeTimeout when FULLINVALIDATE never
// read 0.
// kModeOrderingProducer / kModeOrderingConsumer (FlushFenceOrdering): kArgPasses passes over the data. The flag line
// (kArgFlagOffset) carries the producer's pass + 1, the next line the consumer's ack (the pass it waits for). Per pass
// the producer waits for the ack, stores kArgValue + (pass << 16) + i cached, runs leg kArgSteps (kOrderingLeg*), then
// stores the flag through the uncached alias. The consumer waits for the flag and reads the data through the uncached
// alias: kStatusMatches counts passes whose first read after the flag was complete, kStatusRegisterA passes that
// completed later within kArgDataWindowCycles, kStatusRegisterB passes never complete in the window, kStatusRegisterC
// the largest cycles from the flag to complete data. The producer flushes the data range after the last pass;
// kStatusRegisterA is its leg cycles summed over passes. The L1-then-L2 leg drains the range flush with an uncached
// read of its last line and a fence before the flag, as chippy does after FLUSH64.
// kModePartialLineEvict (DirtyEvictionPartialLine): per data line, a byte, halfword, word and doubleword store
// through the cache (kPartial* offsets, values from kArgValue + line), then cached reads of kArgCount lines at
// kArgAuxOffset + i * kArgStrideBytes (at most kMaxSweepLines) to evict them, then an uncached check per line:
// kStatusRegisterA counts lines whose partial bytes are in TL1, kStatusRegisterB lines with an untouched byte changed.
// Then an L2 range flush and a second check: kStatusMatches counts lines that are exactly the old line with the
// partial stores applied, kStatusRegisterC lines with an untouched byte changed.
// kModeL1FlushLineTarget / kModeL1FlushFullTarget (L1DcacheFlush*): cached stores, then cflush.d.l1 per line / x0,
// an uncached read before any L2 operation (kStatusRegisterA counts new words: the L1 flush reached TL1), cdiscard.d.l1
// per line, the L2 range flush and an uncached read (kStatusMatches).
// kModeSharedWriter / kModeSharedReader (MimirCacheSharedL2, one writer and one reader on one CCE): the reader reads
// its words (kArgDataOffset, kArgNumWords) cached (kStatusWarmOld) and sets the flag line (kArgFlagOffset) to
// kFlagWarmed. The writer waits for it, stores the new values to its words (kArgDataOffset, kArgNumWords) cached,
// `fence w, w`, then sets the flag to kFlagReady; kStatusUncachedBefore counts its words in TL1 then. It then runs
// cflush.d.l1 per line of its words, sets the third flag line to kFlagReady and waits for kFlagWarmed there. Then it
// range-flushes its words in L2, drains the flush with an uncached read and a fence, and sets the second flag line to
// kFlagReady. The reader waits for the first kFlagReady and reads the whole region (kArgAuxOffset, kArgCount words,
// old word i = kArgOldValue + i) cached with no cache operation (kStatusRegisterB counts words as expected: the
// writer's words (kArgOtherOffset, kArgOtherWords) new, every other word old; kStatusRegisterA counts the writer's
// words in TL1). It waits for the third line's kFlagReady, runs cdiscard.d.l1 per region line and reads the region
// cached through the shared L2 (kStatusRegisterC counts words as expected), then sets the third line to kFlagWarmed.
// It then waits for the second kFlagReady, invalidates every region line in L1 D$ and L2 and reads the region cached
// again (kStatusMatches counts words as expected).
// kModeSharedConcurrentWriter (8-hart form): hart slot s of kArgNumHarts stores kFlagReady to flag line s and waits
// for every flag line, then for kArgPasses passes stores kArgValue + (s << 16) + (pass << 8) + i to its kArgNumWords
// words at kArgDataOffset + s * kArgStrideBytes cached, then `fence`, the L2 range flush of its words;
// kStatusCachedBefore counts its words reading its last-pass values cached.
// kModeExternalReader (ExternalWriters, CrossCce, CrossD2d): reads the old values cached (kStatusWarmOld), sets the
// flag to kFlagWarmed when kArgFlagOffset is nonzero, then polls the data through the uncached alias until every word
// is new (kStatusUncachedBefore counts new words when the poll ended). When kArgPeerStatusOffset is nonzero it then
// waits, with uncached loads, for kArgPeerMagic there: a writer whose destination is larger than the data writes its
// done magic only after its whole destination has landed (kCodePeerTimeout otherwise). With kStepDiscriminate it reads
// cached (kStatusStaleBefore counts old words) and, if any was old, runs cdiscard.d.l1 per line and reads cached again
// (kStatusRegisterA counts new words; kStatusRegisterB is 1 when that re-read ran). With kStepL2Invalidate it runs
// INVALIDATE64 per line. A final cached read follows (kStatusMatches). It makes no store while the writer may be
// writing.
// kModeSpaWriter: waits for kFlagWarmed at kArgFlagOffset of CCE kArgTargetCce by SPA load (when kArgFlagOffset is
// nonzero), then stores the new values to kArgDataOffset of that CCE by SPA store, then `fence`.
// kModeSpaReader: stores kFlagWarmed to kArgFlagOffset of CCE kArgTargetCce by SPA store, waits for kFlagReady there
// by SPA load, then reads the data there by SPA load (kStatusMatches).
// kModeDirtyRace (DirtyLineRace): stores the new values to its words (kArgDataOffset, kArgNumWords, all on one line)
// cached, sets the flag to kFlagReady, then repeats action kArgSteps & kRaceActionMask (kRaceAction*) on the line,
// re-storing its words first when kRaceRedirty is set, until the external words (kArgOtherOffset, kArgOtherWords) read
// kArgOtherValue + i through the uncached alias, at most kArgPasses (<= kMaxRaceActions) times within
// kArgPollBudget*, which covers the launch of the external writer (kCodeRaceUnresolved otherwise). kStatusPasses
// counts the actions. A cached read follows (kStatusRegisterA counts its words reading new, kStatusRegisterB external
// words reading kArgOtherValue + i), then FLUSH64 of the line.
// kModeThrash (ReuseThrash): kArgPasses passes over the data in pattern kArgPattern (kThrash*; kArgStrideBytes is the
// conflict stride, kArgSeed the random permutation). Pass 0 stores thrash_word(0, i) cached; each later pass reads
// every word (kStatusRegisterA counts words that did not hold the previous pass's value) and stores
// thrash_word(pass, i). The rdcycle count of each pass goes to the trace. Then the L2 range flush and an uncached
// check of the last pass's values (kStatusMatches; kStatusRegisterB is their XOR).
// kModeChurnConsumer (PreloadChurn): reads the old values cached (kStatusWarmOld) and again (kStatusRegisterC: hit
// cycles). For each of kArgPasses rounds it sets the flag to round + 1, polls through the uncached alias until every
// word is churn_word(round, i), runs INVALIDATE64 per line when kStepL2Invalidate is set, reads cached
// (kStatusStaleBefore sums words not yet churn_word(round, i); kStatusRegisterA sums the read's cycles), then reads
// cached again (kStatusRegisterB sums those cycles). kStatusMatches counts rounds whose second read was complete.
// MM41 modes count completed passes or rounds in kStatusPasses. kModeWipeFull, kModePartialLineEvict,
// kModeSharedReader, kModeExternalReader, kModeDirtyRace (over its whole line) and kModeChurnConsumer first read their
// region cached and put the number of words that are not old (lines left cached by an earlier leg or launch) in
// kStatusCachedBefore, then drop the region from L1 D$ and L2 before their own cached reads.
inline constexpr uint32_t kModeNoFlush = 0, kModeL1FlushLine = 1, kModeL1FlushFull = 2, kModeL1InvalidateLine = 3,
                          kModeL2FlushLine = 4, kModeL2FlushRange = 5, kModeL2FlushFull = 6, kModeL2InvalidateLine = 7,
                          kModeInvalidateFreshRead = 8, kModeProducer = 9, kModeConsumer = 10,
                          kModeWriteUncached1B = 11, kModeWriteUncached8B = 12, kModeWriteCachedRangeFlush = 13,
                          kModeWriteCachedFastFlush = 14, kModeL1InvalidateFull = 15, kModeL2InvalidateFreshRead = 16,
                          kModeNull = 17, kModeReadConfig = 18, kModeDiscardFull = 20, kModeWipeFull = 21,
                          kModeOrderingProducer = 22, kModeOrderingConsumer = 23, kModePartialLineEvict = 24,
                          kModeL1FlushLineTarget = 25, kModeL1FlushFullTarget = 26, kModeSharedWriter = 27,
                          kModeSharedReader = 28, kModeSharedConcurrentWriter = 29, kModeExternalReader = 30,
                          kModeSpaWriter = 31, kModeSpaReader = 32, kModeDirtyRace = 33, kModeThrash = 34,
                          kModeChurnConsumer = 35, kNumModes = 36;
inline constexpr uint32_t kMaxEvictCount = 32;
// kModePartialLineEvict's eviction reads (1 MiB of lines) and kModeDirtyRace's actions.
inline constexpr uint32_t kMaxSweepLines = 16 * 1024;
inline constexpr uint32_t kMaxRaceActions = 1u << 24;
// 192 KiB: above the 128 KiB L2 and inside a hart's 248 KiB data area with room for a guard.
inline constexpr uint32_t kMaxWords = 48 * 1024;
inline constexpr uint32_t kFlagReady = 0xF1A60001u, kFlagWarmed = 0xF1A60002u;

// kModeOrdering* legs (kArgSteps): bare FLUSH64 per line and the flag with no fence; a fence and the flag with no
// flush; cflush.d.l1 per line then the L2 range flush; cflush.d.l1 x0 then flush_l2_cache_full(). The two cflush legs
// run cdiscard.d.l1 per data line before the L2 flush, so data a cflush left in L1 D$ is lost.
inline constexpr uint32_t kOrderingLegFlushNoFence = 0, kOrderingLegFenceNoFlush = 1, kOrderingLegL1ThenL2 = 2,
                          kOrderingLegFullHierarchy = 3;
// kModeExternalReader / kModeChurnConsumer steps (kArgSteps bits).
inline constexpr uint32_t kStepDiscriminate = 1u << 0, kStepL2Invalidate = 1u << 1;
// kModeDirtyRace actions (kArgSteps & kRaceActionMask): cached reads of kArgCount lines at kArgAuxOffset + i *
// kArgStrideBytes; FLUSH64 of the line; INVALIDATE64 of the line. kRaceRedirty re-stores the hart's words first.
inline constexpr uint32_t kRaceActionEvict = 0, kRaceActionFlush = 1, kRaceActionInvalidate = 2, kRaceActionMask = 0xF,
                          kRaceRedirty = 1u << 4;
// kModePartialLineEvict store offsets in a line and their widths: a byte, a halfword, a word and a doubleword.
inline constexpr uint32_t kPartialByteOffset = 1, kPartialHalfOffset = 6, kPartialWordOffset = 12,
                          kPartialDoubleOffset = 32;
// kModeThrash patterns: lines in order; lines kArgStrideBytes apart, then the next offset; a permutation of the lines
// from kArgSeed.
inline constexpr uint32_t kThrashSequential = 0, kThrashConflictStride = 1, kThrashRandom = 2;
inline constexpr uint32_t kMaxPasses = 16;

constexpr uint32_t thrash_word(uint32_t value, uint32_t seed, uint32_t pass, uint32_t word) {
    return ((value ^ seed) + word * 0x9E3779B9u) + pass;
}

constexpr uint32_t churn_word(uint32_t value, uint32_t round, uint32_t word) { return value + (round << 20) + word; }

// Status line, written once at exit: every field, `fence w, w`, then the magic. kStatusFirstMismatchWord and
// kStatusObserved describe the first word of the final check that did not hold the new value. kStatusFlagReady is 1
// once the hart saw the flag it waits for within its budget. kStatusCycles is the rdcycle count of the mode's work.
// kStatusRegisterA/B/C are per-mode values, described with the mode. kStatusPasses counts an MM41 mode's completed
// passes or rounds.
using cce_test_status_abi::kStatusCode;
using cce_test_status_abi::kStatusMagic;
inline constexpr uint32_t kStatusUncachedBefore = 2, kStatusStaleBefore = 3, kStatusMatches = 4,
                          kStatusFirstMismatchWord = 5, kStatusObserved = 6, kStatusFlagReady = 7, kStatusCycles = 8,
                          kStatusCachedBefore = 9, kStatusWarmOld = 10, kStatusRegisterA = 11, kStatusRegisterB = 12,
                          kStatusRegisterC = 13, kStatusPasses = 14, kStatusWords = 15;
static_assert(kStatusWords * sizeof(uint32_t) <= kLineBytes);
inline constexpr uint32_t kDoneMagic = 0xCAC4E0DEu;
using cce_test_status_abi::kStatusPoison;

// kCodeWipeTimeout: FULLINVALIDATE never read 0 (kModeWipeFull). kCodeDataTimeout: a data poll used its budget.
// kCodeRaceUnresolved: kModeDirtyRace never read the external words. kCodePeerTimeout: kModeExternalReader never read
// the writer's done magic, so the writer may still be writing.
inline constexpr uint32_t kCodeDone = 1, kCodeBadArgs = 2, kCodeFlagTimeout = 3, kCodeWipeTimeout = 4,
                          kCodeDataTimeout = 5, kCodeRaceUnresolved = 6, kCodePeerTimeout = 7;

}  // namespace cce_cache_ops_abi
