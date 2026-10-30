// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "cce_test_status_abi.h"

// cce_amo_ops kernel ABI. The kernel (cce_amo_ops.cpp) and the host (dram/test_mimir_atomics.cpp) both include it.
// Args, modes, codes and status words are append-only: a later stage adds entries after the last one and never
// renumbers.
namespace cce_amo_ops_abi {

using cce_test_status_abi::kLineBytes;
inline constexpr uint32_t kMaxParticipants = 8;

// Runtime args. kArgStatusOffset is a 64 B aligned offset in the kernel's own TL1. kArgTargetOffset is a 64 B aligned
// hart-local TL1 offset in a data area, accessed through the cache (AMOs are illegal on the uncached alias). Every
// participant of one launch passes the same kArgTargetOffset, kArgFlagOffset and kArgNumParticipants, and its own
// kArgParticipant (0 .. kArgNumParticipants - 1). kArgFlagOffset (64 B aligned) holds 2 * kMaxParticipants flag lines
// the host zeroes: ready flags first, then done flags, one line per participant. Every region a mode touches (target
// extent, flags, results, table, status line) must lie in the test TL1 area below the 4 MiB cacheable port, else
// kCodeBadArgs (no status store when the status line itself is outside). kArgPollBudget* is rdcycle cycles per
// bounded wait and must be nonzero.
// MM42 args (from kArgResultOffset). kArgResultOffset (64 B aligned, own TL1) receives per-participant results the
// status line has no room for; kArgTableOffset (64 B aligned, own TL1) is a host-written table of kArgNumPoints
// entries. kArgOperation is a kOp* code, kArgStrideBytes the distance between participants' words, kArgLeg selects a
// leg of a mode, kArgSeed seeds kModeStorm and kModeSoak, kArgPeerCce names the other CCE of kModeItnWriter.
// Modes from kModeOpMatrix share one budget, started at kernel entry, across every wait and loop of the launch.
inline constexpr uint32_t kArgMode = 0, kArgStatusOffset = 1, kArgPollBudgetLo = 2, kArgPollBudgetHi = 3,
                          kArgTargetOffset = 4, kArgCount = 5, kArgValue = 6, kArgFlagOffset = 7,
                          kArgNumParticipants = 8, kArgParticipant = 9, kArgResultOffset = 10, kArgTableOffset = 11,
                          kArgNumPoints = 12, kArgOperation = 13, kArgStrideBytes = 14, kArgLeg = 15, kArgSeed = 16,
                          kArgPeerCce = 17, kNumArgs = 18;

// kModeTwoHartIncrement (MM42 TwoHartAmo probe): every participant discards both lines from its L1 D$ (participant 0
// also from L2), sets its ready flag and waits for all ready flags, then does kArgCount amoadd.w of kArgValue on the
// word at kArgTargetOffset and kArgCount LR/SC increments of kArgValue on the word one line later (each bounded by
// kMaxScRetries attempts), then sets its done flag. Participant 0 waits for every done flag, reads both words with
// amoadd.w of 0 (kStatusAmoFinal, kStatusLrScFinal), then flushes L1 D$ and the two lines from L2 so the host sees
// them. kStatusAmoReturnSum / kStatusLrScReturnSum are the sums of the values each
// operation returned (the word before that hart's update).
// kModeOpMatrix (one participant): entry i of the kOpEntryWords-word table runs operation kOpEntryOperation on the
// doubleword (or word) at kArgTargetOffset + i * kLineBytes with kOpEntryOperand; the CAS operations compare with
// kOpEntryCompare. The kernel writes the value the operation returned and its LR/SC attempts back into the entry, then
// flushes L1 D$ and the target lines from L2.
// kModeSweep (Hotspot, Distributed, Wrap, LrScCas): table entry p is {value, count}. For each p, every participant
// waits at a barrier, then does count increments of value with kArgOperation (a kOp*Add* code) on the word at
// kArgTargetOffset + p * kSweepPointBytes + kArgParticipant * kArgStrideBytes (stride 0: one shared word). Participant
// p's result for point q is kSweepResultWords words at kArgResultOffset + (participant * kArgNumPoints + q) *
// kSweepResultBytes. Every participant flushes its L1 D$ before its done flag; participant 0 waits for every done
// flag and flushes the points from L2.
// kModeLineState (CleanDirtyLine, MixedPathControl): one participant runs leg kArgLeg (kLeg*) on the line at
// kArgTargetOffset; the control line is the next line. kStatusReturnedLo/Hi is what amoadd.w of kArgValue on word 0
// returned, kStatusObservedFirst word 0 read through the uncached alias before the AMO, kStatusObservedSecond the
// control line's word 0 read through the cache.
// kModeOrdering (two participants): kArgCount rounds of message passing. Participant 0 stores round data through the
// cache to the line at kArgTargetOffset, releases flag word kArgTargetOffset + kLineBytes with amoswap.w.rl and waits
// for the ack word one line later with amoadd.w.aq of 0; participant 1 waits for the flag the same way, discards the
// data line from its L1 D$, reads it through the cache and counts mismatches (kStatusMismatches,
// kStatusFirstMismatch, kStatusObservedFirst / Second for the first one), then releases the ack. kLegFlushRelease
// flushes the data line from L1 D$ before the release; kLegReleaseOnly does not.
// kModeNeighbourWrite: participant 0 does kArgCount amoadd.w of kArgValue on word 0 of the line at kArgTargetOffset
// while participant p > 0 stores neighbour_word(kArgSeed, p, i) through the cache to word p for i < kArgCount. Every
// participant flushes its L1 D$ before its done flag; participant 0 waits for every done flag and flushes the line
// from L2.
// kModeStorm / kModeSoak: kArgCount operations per participant drawn from storm_draw (below) on kStormClasses lines
// (kArgLeg kLegShared: one set at kArgTargetOffset; kLegDisjoint: participant p's set at kArgTargetOffset + p *
// kStormSetBytes). kModeStorm uses slot 0 of each width; kModeSoak picks the slot, and runs the operations in bursts
// from soak_burst with a barrier after a burst when soak_barrier says so. Results per participant: kStormResultWords
// words at kArgResultOffset + participant * kLineBytes. Flushes as kModeSweep.
// kModeItnWriter (MM42 ItnControl, on the writer CCE): waits for the ready flag line at kArgFlagOffset of CCE
// kArgPeerCce with SPA loads, stores kArgValue to word 0 at kArgTargetOffset of that CCE, full fence, loads the word
// back over SPA until it reads kArgValue (kCodeReadBackTimeout if it never does; no flag is stored then), full fence,
// then stores 1 to the next flag line. kModeItnReceiver (on CCE kArgPeerCce's hart): discards the target line from L1
// D$ and L2, reads it through the cache (kStatusObservedFirst), sets its ready flag, waits for the writer's flag with
// uncached loads, reads word 0 through the cache again (kStatusObservedSecond), then L2 INVALIDATE64 and cdiscard on
// the line, then amoadd.w of kArgValue; kStatusReturnedLo is what it returned. Then it flushes the line from L1 D$ and
// L2.
inline constexpr uint32_t kModeTwoHartIncrement = 0, kModeOpMatrix = 1, kModeSweep = 2, kModeLineState = 3,
                          kModeOrdering = 4, kModeNeighbourWrite = 5, kModeStorm = 6, kModeSoak = 7, kModeItnWriter = 8,
                          kModeItnReceiver = 9, kNumModes = 10;
inline constexpr uint32_t kMaxCount = 1024;
inline constexpr uint32_t kMaxScRetries = 1024;
inline constexpr uint32_t kMaxCasTries = 1u << 20;

// Operations (kArgOperation, kOpEntryOperation). .W returns are the low 32 bits. kOpLrScCas*: lr, compare, sc. The Add
// codes from kOpLrScAddW are increments built from LR/SC (lr, add, sc; bounded by kMaxScRetries attempts) and from CAS
// (plain load, then kOpLrScCas until it stores). A CAS increment has no forward-progress guarantee under contention, so
// its retries after a failed compare are bounded by the launch's budget (kCodeBudgetExpired) and a hard cap of
// kMaxCasTries; each CAS's failed sc is still bounded by kMaxScRetries.
inline constexpr uint32_t kOpAmoSwapW = 0, kOpAmoAddW = 1, kOpAmoXorW = 2, kOpAmoAndW = 3, kOpAmoOrW = 4,
                          kOpAmoMinW = 5, kOpAmoMaxW = 6, kOpAmoMinuW = 7, kOpAmoMaxuW = 8, kOpAmoSwapD = 9,
                          kOpAmoAddD = 10, kOpAmoXorD = 11, kOpAmoAndD = 12, kOpAmoOrD = 13, kOpAmoMinD = 14,
                          kOpAmoMaxD = 15, kOpAmoMinuD = 16, kOpAmoMaxuD = 17, kOpLrScCasW = 18, kOpLrScCasD = 19,
                          kOpLrScAddW = 20, kOpLrScAddD = 21, kOpCasAddW = 22, kOpCasAddD = 23, kOpAmoAddWAq = 24,
                          kOpAmoAddWRl = 25, kOpAmoAddWAqRl = 26, kOpAmoAddDAqRl = 27, kNumOps = 28;
constexpr bool is_doubleword_op(uint32_t op) {
    return (op >= kOpAmoSwapD && op <= kOpAmoMaxuD) || op == kOpLrScCasD || op == kOpLrScAddD || op == kOpCasAddD ||
           op == kOpAmoAddDAqRl;
}
constexpr bool is_increment_op(uint32_t op) {
    return op == kOpAmoAddW || op == kOpAmoAddD || (op >= kOpLrScAddW && op <= kOpCasAddD);
}

// kModeOpMatrix table entry.
inline constexpr uint32_t kOpEntryOperation = 0, kOpEntryOperandLo = 1, kOpEntryOperandHi = 2, kOpEntryCompareLo = 3,
                          kOpEntryCompareHi = 4, kOpEntryReturnedLo = 5, kOpEntryReturnedHi = 6, kOpEntryAttempts = 7,
                          kOpEntryWords = 8;
inline constexpr uint32_t kMaxOpEntries = 32;

// kModeSweep table entry and result.
inline constexpr uint32_t kSweepEntryValue = 0, kSweepEntryCount = 1, kSweepEntryWords = 2;
inline constexpr uint32_t kMaxSweepPoints = 16;
inline constexpr uint32_t kSweepPointBytes = kMaxParticipants * kLineBytes;
// kSweepResultInterleaved counts returned values that were not this participant's previous return plus the value:
// another participant updated the word in between, so the harts overlapped. kSweepResultMaxCasTries is the most CAS
// tries one CAS increment needed (the starvation measure; 0 for other operations).
inline constexpr uint32_t kSweepResultReturnSumLo = 0, kSweepResultReturnSumHi = 1, kSweepResultFailures = 2,
                          kSweepResultMaxAttempts = 3, kSweepResultCycles = 4, kSweepResultInterleaved = 5,
                          kSweepResultMaxCasTries = 6, kSweepResultWords = 8;
inline constexpr uint32_t kSweepResultBytes = kSweepResultWords * 4;

// kModeLineState legs. kLegClean: discard the line, read it through the cache, AMO. kLegDirty: discard the line, store
// words 1-15 through the cache (kArgSeed + word index), AMO with the line dirty. kLegFlushBeforeAtomic: kLegDirty's
// stores, flush the line from L1 D$ and L2, then AMO. Those three flush the line from L1 D$ and L2 at the end.
// kLegWarm: discard both lines, read both through the cache and return with them cached (no AMO, no flush).
// kLegAfterWrite: after an external write between launches, AMO with no invalidate. kLegInvalidateAfterWrite: the
// same after L2 INVALIDATE64 and cdiscard on the target line. Both flush the target line at the end.
inline constexpr uint32_t kLegClean = 0, kLegDirty = 1, kLegFlushBeforeAtomic = 2, kLegWarm = 3, kLegAfterWrite = 4,
                          kLegInvalidateAfterWrite = 5;
// kModeOrdering legs.
inline constexpr uint32_t kLegFlushRelease = 0, kLegReleaseOnly = 1;
// kModeStorm / kModeSoak layouts.
inline constexpr uint32_t kLegShared = 0, kLegDisjoint = 1;

constexpr uint32_t neighbour_word(uint32_t seed, uint32_t participant, uint32_t iteration) {
    return (seed & 0xFF000000u) | (participant << 16) | iteration;
}

// kModeStorm / kModeSoak. Each class has one line: four word slots at bytes 0-15, four doubleword slots at bytes
// 16-47, and bytes 48-63 untouched. Add and CAS classes increment by 1, so the values one participant gets back from
// one slot strictly increase. The draw sequence is the same on the host, which replays it to build its model.
inline constexpr uint32_t kStormAddDiscard = 0, kStormAddReturn = 1, kStormSwap = 2, kStormCas = 3, kStormXor = 4,
                          kStormOr = 5, kStormAnd = 6, kStormMax = 7, kStormMin = 8, kStormMaxu = 9, kStormMinu = 10,
                          kStormOps = 11;
inline constexpr uint32_t kStormClasses = 10;
inline constexpr uint32_t kStormSlots = 4;
inline constexpr uint32_t kStormSetBytes = 1024;
inline constexpr uint32_t kStormWordSlotBytes = 0, kStormDoublewordSlotBytes = 16;
inline constexpr uint32_t kSoakMaxBurst = 16;
static_assert(kStormClasses * kLineBytes <= kStormSetBytes);

constexpr uint32_t storm_class_of(uint32_t storm_op) { return storm_op == kStormAddDiscard ? 0 : storm_op - 1; }

constexpr uint32_t xorshift_next(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

constexpr uint32_t storm_state(uint32_t seed, uint32_t participant) {
    return (seed ^ (0x9E3779B9u * (participant + 1))) | 1u;
}

struct StormDraw {
    uint32_t op = 0;
    bool doubleword = false;
    uint32_t slot = 0;
    uint64_t operand = 0;
};

// One operation of one participant. Swap, xor, or, and, min and max take a random operand; add and CAS add 1.
constexpr StormDraw storm_draw(uint32_t& state, bool random_slot) {
    const uint32_t shape = xorshift_next(state);
    StormDraw draw;
    draw.op = shape % kStormOps;
    draw.doubleword = ((shape >> 8) & 1u) != 0;
    draw.slot = random_slot ? (shape >> 9) % kStormSlots : 0;
    const uint32_t low = xorshift_next(state);
    const uint32_t high = draw.doubleword ? xorshift_next(state) : 0;
    draw.operand = (static_cast<uint64_t>(high) << 32) | low;
    if (draw.op == kStormAddDiscard || draw.op == kStormAddReturn || draw.op == kStormCas) {
        draw.operand = 1;
    }
    return draw;
}

constexpr uint32_t storm_slot_offset(const StormDraw& draw) {
    return storm_class_of(draw.op) * kLineBytes +
           (draw.doubleword ? kStormDoublewordSlotBytes + draw.slot * 8 : kStormWordSlotBytes + draw.slot * 4);
}

// kModeSoak's burst schedule, the same for every participant.
constexpr uint32_t soak_schedule_state(uint32_t seed) { return (seed ^ 0x5A17C0DEu) | 1u; }
constexpr uint32_t soak_burst(uint32_t& schedule_state) { return 1 + xorshift_next(schedule_state) % kSoakMaxBurst; }
constexpr bool soak_barrier(uint32_t& schedule_state) { return xorshift_next(schedule_state) % 4 == 0; }

// kModeStorm / kModeSoak result line of one participant. Violations count values an add or CAS slot returned that did
// not exceed the last one this participant got from that slot. The swap XOR and sum are over every value swap
// returned, word slots zero-extended. kStormResultInterleaved counts add and CAS returns above the last one plus 1:
// another participant updated that slot in between. kStormResultMaxCasTries is the most CAS tries one CAS increment
// needed (the starvation measure). kStormResultProgress is the operations done, stored at each barrier
// and at the end.
inline constexpr uint32_t kStormResultProgress = 0, kStormResultAddViolations = 1, kStormResultCasViolations = 2,
                          kStormResultCasFailures = 3, kStormResultMaxAttempts = 4, kStormResultSwapXorLo = 5,
                          kStormResultSwapXorHi = 6, kStormResultSwapSumLo = 7, kStormResultSwapSumHi = 8,
                          kStormResultBarriers = 9, kStormResultMaxBarrierCycles = 10, kStormResultCycles = 11,
                          kStormResultInterleaved = 12, kStormResultMaxCasTries = 13, kStormResultWords = 14;
static_assert(kStormResultWords * 4 <= kLineBytes);

// Status line, written once at exit: every field, `fence w, w`, then the magic. kStatusScFailures counts failed sc.w
// attempts, kStatusMaxRetries is the most attempts one LR/SC increment took. kStatusPeersSeen counts the flags the
// last wait saw set. kStatusCycles is the rdcycle count of the operations. kStatusReturnedLo to kStatusFirstMismatch
// are described with the modes that set them.
using cce_test_status_abi::kStatusCode;
using cce_test_status_abi::kStatusMagic;
inline constexpr uint32_t kStatusAmoFinal = 2, kStatusLrScFinal = 3, kStatusAmoReturnSum = 4, kStatusLrScReturnSum = 5,
                          kStatusScFailures = 6, kStatusMaxRetries = 7, kStatusCycles = 8, kStatusPeersSeen = 9,
                          kStatusReturnedLo = 10, kStatusObservedFirst = 12, kStatusObservedSecond = 13,
                          kStatusMismatches = 14, kStatusFirstMismatch = 15, kStatusWords = 16;
static_assert(kStatusWords * sizeof(uint32_t) <= kLineBytes);
inline constexpr uint32_t kDoneMagic = 0xA70D0DE0u;
using cce_test_status_abi::kStatusPoison;

// kCodeDone: every wait succeeded and every LR/SC increment stored. Otherwise the code names the first failure.
// kCodeReadBackTimeout: kModeItnWriter's data word never read back as written. kCodeBudgetExpired: the launch's
// budget ran out during the operations of kModeOpMatrix, kModeSweep, kModeNeighbourWrite, kModeStorm or kModeSoak
// (a participant stops at the operation it was on and still stores its results and status).
inline constexpr uint32_t kCodeDone = 1, kCodeBadArgs = 2, kCodePeerTimeout = 3, kCodeScRetriesExhausted = 4,
                          kCodeReadBackTimeout = 6, kCodeBudgetExpired = 7;

}  // namespace cce_amo_ops_abi
