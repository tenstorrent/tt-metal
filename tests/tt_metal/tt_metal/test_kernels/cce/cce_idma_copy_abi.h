// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "cce_test_status_abi.h"

// cce_idma_copy kernel ABI. The kernels (cce_idma_copy.cpp, cce_idma_gddr_round_trip.cpp) and the host tests that run
// them (dram/test_mimir_{atomics,dm_cache,dm_idma,local_dram,trid}.cpp, context/test_mimir_emu.cpp) include it.
// Args, modes, kinds, polls, codes and status words are append-only: a later stage adds entries after the last one and
// never renumbers.
namespace cce_idma_copy_abi {

using cce_test_status_abi::kLineBytes;

// Runtime args. kArgStatusOffset is a 64 B aligned offset in the kernel's own TL1. kArgPollBudget* is rdcycle cycles
// per bounded poll and must be nonzero. kArgExpectedXor is the XOR of the source words (kModeIdmaCopyHostVerified).
// kArgNumElements is 1 except in kModeIdmaStrided, where kArgLenBytes is the element size and kArgSourceStrideBytes the
// distance between source elements.
// MM4x args (from kArgTrid). The batched modes (kModeTridBatch, kModeTridTable, kModeTridStream) read
// kArgNumElements as the command count (at most kMaxElements) and kArgSourceStrideBytes as the source stride; command
// i reads source + i * kArgSourceStrideBytes, writes destination + i * kArgDestinationStrideBytes and uses trid
// (kArgTrid + i * kArgTridStride) mod kNumTrids. kArgSnoop sets the snoop packet tag on every command. kArgTraceOffset
// (64 B aligned, own TL1, 0 for none) receives the trace after the last data poll. kArgHostVerified (batched modes):
// no data poll unless the destination is the kernel's own TL1, as kModeIdmaCopyHostVerified. kArgCmdTableOffset
// (kModeTridTable), kArgStopOffset and kArgProgressOffset (kModeTridStream), kArgSeed, kArgAccumulate* and
// kArgExpected* (kModeAccumulate), kArgPeerFlagOffset (kModeTridScopeObserver and kModeTridBatch), kArgPollRegion*
// (kModeTridBatch and kModeAccumulate; every other mode rejects a nonzero kArgPollRegionBytes) and kArgCheckAtAck
// (kModeTridBatch only) are described with their modes.
inline constexpr uint32_t kArgMode = 0, kArgBackendVc = 1, kArgStatusOffset = 2, kArgLenBytes = 3, kArgPollBudgetLo = 4,
                          kArgPollBudgetHi = 5, kArgSrcKind = 6, kArgSrcSelector = 7, kArgSrcOffsetLo = 8,
                          kArgSrcOffsetHi = 9, kArgDstKind = 10, kArgDstSelector = 11, kArgDstOffsetLo = 12,
                          kArgDstOffsetHi = 13, kArgExpectedXor = 14, kArgNumElements = 15, kArgSourceStrideBytes = 16,
                          kArgTrid = 17, kArgSnoop = 18, kArgDestinationStrideBytes = 19, kArgTridStride = 20,
                          kArgTraceOffset = 21, kArgCmdTableOffset = 22, kArgStopOffset = 23, kArgProgressOffset = 24,
                          kArgSeed = 25, kArgAccumulateFormat = 26, kArgAccumulateOp = 27,
                          kArgAccumulateDisableSaturation = 28, kArgExpectedLo = 29, kArgExpectedHi = 30,
                          kArgPeerFlagOffset = 31, kArgHostVerified = 32, kArgCommandBuffer = 34,
                          kArgPollRegionOffset = 35, kArgPollRegionBytes = 36, kArgCheckAtAck = 37, kNumArgs = 38;

// kModeIdmaCopy: one iDMA copy (MISC.idma_en=1) between package-global SPA operands, then a bounded poll of the
// destination data against the source. kModeIdmaCopyHostVerified: the same copy, and the kernel never loads a remote
// operand, so it cannot stall the hart; the host checks the data. When the destination is the kernel's own TL1 the
// data poll compares the XOR of the destination words with kArgExpectedXor; otherwise there is no data poll.
// kModeIdmaStrided (upstream IDMA_1D_Strided): address generator 0 feeds command buffer 0, one command per element.
// Element i is read at source + i * kArgSourceStrideBytes and written at destination + i * kArgLenBytes; the data poll
// checks every element.
// kModeTridBatch: kArgNumElements commands to one target with the strides above, each waiting for VC space, then a
// wait for every trid used to read 0 pending, then a data poll of every command (unless kArgHostVerified). When
// kArgPeerFlagOffset is nonzero (own TL1, 64 B aligned) the kernel first polls that word with uncached loads until it
// is nonzero and issues nothing if it never is (kCodePeerTimeout). One budget covers every wait of the launch.
// kStatusPendingHighWater is the largest pending count of any command's trid read right after that command's issue. The
// trace holds kTraceWordsPerTrid words per trid: the rank in which the trid
// reached 0 pending (kTraceNotUsed if unused, kTraceNotAcked if it never did) and the rdcycle cycles from the last
// issue to that point. When kArgPollRegionBytes is nonzero (host-verified, destination the kernel's own TL1 only) the
// data poll compares the XOR of [kArgPollRegionOffset, + kArgPollRegionBytes) of the kernel's own TL1 (64 B aligned)
// with kArgExpectedXor instead of the destination's, so harts that copy into disjoint blocks of one region can each
// poll every hart's result before any status store (MIM-560). When kArgCheckAtAck is nonzero (kernel-verified only)
// the kernel compares every command's destination with its source once, as soon as every trid reads 0 pending and
// before the data poll: kStatusProbeFirst holds the words that had not landed, and a nonzero count gives
// kCodeNotLandedAtAck.
// kModeAccumulate (MM42 Tier M): command buffer kArgCommandBuffer set up for iDMA L1 accumulate
// (idma_setup_as_atomic_accum_cmdbuf, then l1_atomic_instr_cmdbuf(kArgAccumulateFormat,
// kArgAccumulateDisableSaturation, kArgAccumulateOp)); otherwise kModeTridBatch's issue, trid and wait rules. The
// destination is the kernel's own TL1 (kKindOwnTl1). The data poll passes when the XOR of the destination words over
// [destination, destination + (n - 1) * kArgDestinationStrideBytes + kArgLenBytes) equals kArgExpectedLo and the
// first destination word equals kArgExpectedHi; the host computes both from its model, including every other hart's
// contribution when harts share a destination. When kArgPollRegionBytes is nonzero the XOR and the first word are
// taken over [kArgPollRegionOffset, + kArgPollRegionBytes) of the kernel's own TL1 (64 B aligned) instead, so harts
// with disjoint destinations can each poll every hart's result before any status store (MIM-560).
// kModeTridTable (MM43): kArgNumElements commands, each from its kTableWords-word entry at kArgCmdTableOffset (own
// TL1, 64 B aligned, written by the host): operands, backend VC, trid, length and a host-verified flag. The VC is
// rewritten per command with setup_wrapping_vcs_cmdbuf (start = end). Waits as kModeTridBatch; the data poll covers
// every entry not marked host-verified, and the trace as kModeTridBatch. Every entry is resolved before the first
// issue. kStatusProbeFirst is the largest gap in rdcycle cycles between consecutive issues. kStatusProbeSecond is the
// largest total pending count, summed over every trid the table has issued on, read right after each command's issue.
// kModeTridStream (MM43 WithPrefetcher): repeats kModeTridBatch rounds until it reads kStreamStop at kArgStopOffset
// (own TL1, uncached load, checked before each round) or the budget, shared by every round, runs out
// (kCodeStreamBudget). The destination is never the kernel's own TL1. After each round's data poll it stores the
// number of completed rounds to kArgProgressOffset (own TL1) through the uncached alias. kStatusProbeFirst is the
// number of completed rounds.
// kModeTridBatch and kModeAccumulate use command buffer kArgCommandBuffer (0 or 1); the other modes use command
// buffer 0.
// kModeTridScopeObserver: waits for 1 in the flag line at kArgPeerFlagOffset with uncached loads, reads its own pending
// count of kArgTrid (kStatusProbeFirst), then stores 1 to the next flag line. It issues nothing.
inline constexpr uint32_t kModeIdmaCopy = 0, kModeIdmaCopyHostVerified = 1, kModeIdmaStrided = 2, kModeTridBatch = 3,
                          kModeAccumulate = 4, kModeTridTable = 5, kModeTridStream = 6, kModeTridScopeObserver = 8;
inline constexpr uint32_t kMaxElements = 64;
inline constexpr uint32_t kNumTrids = 16;
inline constexpr uint32_t kTraceWordsPerTrid = 2, kTraceBytes = kNumTrids * kTraceWordsPerTrid * 4;
inline constexpr uint32_t kTraceNotUsed = 0xFFFFFFFFu, kTraceNotAcked = 0xFFFFFFFEu;
// kModeTridTable entry, kTableWords words per command; kinds and selectors as the kArgSrc* / kArgDst* args.
inline constexpr uint32_t kTableSrcKind = 0, kTableSrcSelector = 1, kTableSrcOffsetLo = 2, kTableSrcOffsetHi = 3,
                          kTableDstKind = 4, kTableDstSelector = 5, kTableDstOffsetLo = 6, kTableDstOffsetHi = 7,
                          kTableBackendVc = 8, kTableTrid = 9, kTableLenBytes = 10, kTableHostVerified = 11,
                          kTableWords = 16;
inline constexpr uint32_t kStreamStop = 0x57095709u;

// Operand kinds. kKindOwnTl1: the kernel's own TL1; the selector is the kernel's own CCE index, so the iDMA operand is
// that CCE's package SPA and the hart checks the data through its uncached alias. kKindSramSpa: another CCE's TL1 at
// its package SPA; the selector is that CCE's index. kKindGddr: GDDR at a chiplet offset; the selector is the target
// Mimir, and the CCE's host-written route table picks the D2D port.
inline constexpr uint32_t kKindOwnTl1 = 0, kKindSramSpa = 1, kKindGddr = 2;

// MIM-407 backend (request VC) per source/destination pair. A copy from local SRAM uses the destination's column; a
// copy from anywhere else uses kBackendFromRemote.
inline constexpr uint32_t kBackendLocalSram = 5, kBackendRemoteSram = 2, kBackendGddr0 = 0, kBackendGddr1 = 1,
                          kBackendD2d = 3, kBackendFromRemote = 5;
// NOC_OVERLAY_MAX_BYTES_IN_PACKET.
inline constexpr uint32_t kMaxBytesInPacket = 8 * 1024;
// Transaction ID of every command. Trid 0 is polled explicitly; all_*_acked_cmdbuf skips it.
inline constexpr uint32_t kTrid = 0;

// Status line, written once at exit: every field, `fence w, w`, then the magic. Each poll records the rdcycle cycles it
// spent. kStatusFlags holds the kFlag* bits of the polls that succeeded. kStatusLastPoll is the last poll that ran and
// kStatusLastPollReady is 1 if it succeeded, 0 if it used its budget. After an XOR poll that timed out,
// kStatusMismatchedWords is 1 and kStatusObserved / kStatusExpected hold the destination XOR and kArgExpectedXor.
// kStatusVcSpaceCycles sums the VC space polls, one per element. kStatusElementsIssued counts the commands issued.
using cce_test_status_abi::kStatusCode;
using cce_test_status_abi::kStatusMagic;
inline constexpr uint32_t kStatusFlags = 2, kStatusMismatchedWords = 3, kStatusFirstMismatchWord = 4,
                          kStatusObserved = 5, kStatusExpected = 6, kStatusVcSpaceCycles = 7, kStatusAckCycles = 8,
                          kStatusDataCycles = 9, kStatusLastPoll = 10, kStatusLastPollReady = 11,
                          kStatusElementsIssued = 12, kStatusPendingHighWater = 13, kStatusProbeFirst = 14,
                          kStatusProbeSecond = 15, kStatusWords = 16;
static_assert(kStatusWords * sizeof(uint32_t) <= kLineBytes);
inline constexpr uint32_t kDoneMagic = 0x1D3AC0DEu;
using cce_test_status_abi::kStatusPoison;
inline constexpr uint32_t kFlagVcSpace = 1u << 0, kFlagAcked = 1u << 1, kFlagLanded = 1u << 2;
inline constexpr uint32_t kPollNone = 0, kPollVcSpace = 1, kPollAck = 2, kPollData = 3, kPollPeer = 4;

// kCodeLanded (kModeIdmaCopy) and kCodeAcked (kModeIdmaCopyHostVerified): every poll succeeded. Otherwise the code
// names the first poll that used its budget. Later polls still run, except that no command is issued after
// kCodeVcSpaceTimeout.
// kCodePeerTimeout: the kModeTridScopeObserver flag wait or the kModeTridBatch start-flag wait used its budget.
// kCodeStreamBudget: kModeTridStream's budget ran out before it saw STOP. kCodeNotLandedAtAck: with kArgCheckAtAck,
// every trid read 0 pending before every command's data had landed.
inline constexpr uint32_t kCodeLanded = 1, kCodeAcked = 2, kCodeVcSpaceTimeout = 3, kCodeAckTimeout = 4,
                          kCodeDataTimeout = 5, kCodeBadArgs = 6, kCodePeerTimeout = 7, kCodeStreamBudget = 8,
                          kCodeNotLandedAtAck = 9;

}  // namespace cce_idma_copy_abi
