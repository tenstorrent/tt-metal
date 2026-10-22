// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// cce_idma_copy kernel ABI. The kernel (cce_idma_copy.cpp) and the host (dram/test_mimir_dm_idma.cpp) both include it.
// Args, modes, kinds, polls, codes and status words are append-only: a later stage adds entries after the last one and
// never renumbers.
namespace cce_idma_copy_abi {

inline constexpr uint32_t kLineBytes = 64;

// Runtime args. kArgStatusOffset is a 64 B aligned offset in the kernel's own TL1. kArgPollBudget* is rdcycle cycles
// per bounded poll and must be nonzero. kArgExpectedXor is the XOR of the source words (kModeIdmaCopyHostVerified).
// kArgNumElements is 1 except in kModeIdmaStrided, where kArgLenBytes is the element size and kArgSourceStrideBytes the
// distance between source elements.
inline constexpr uint32_t kArgMode = 0, kArgBackendVc = 1, kArgStatusOffset = 2, kArgLenBytes = 3, kArgPollBudgetLo = 4,
                          kArgPollBudgetHi = 5, kArgSrcKind = 6, kArgSrcSelector = 7, kArgSrcOffsetLo = 8,
                          kArgSrcOffsetHi = 9, kArgDstKind = 10, kArgDstSelector = 11, kArgDstOffsetLo = 12,
                          kArgDstOffsetHi = 13, kArgExpectedXor = 14, kArgNumElements = 15, kArgSourceStrideBytes = 16,
                          kNumArgs = 17;

// kModeIdmaCopy: one iDMA copy (MISC.idma_en=1) between package-global SPA operands, then a bounded poll of the
// destination data against the source. kModeIdmaCopyHostVerified: the same copy, and the kernel never loads a remote
// operand, so it cannot stall the hart; the host checks the data. When the destination is the kernel's own TL1 the
// data poll compares the XOR of the destination words with kArgExpectedXor; otherwise there is no data poll.
// kModeIdmaStrided (upstream IDMA_1D_Strided): address generator 0 feeds command buffer 0, one command per element.
// Element i is read at source + i * kArgSourceStrideBytes and written at destination + i * kArgLenBytes; the data poll
// checks every element.
inline constexpr uint32_t kModeIdmaCopy = 0, kModeIdmaCopyHostVerified = 1, kModeIdmaStrided = 2;
inline constexpr uint32_t kMaxElements = 64;

// Operand kinds. kKindOwnTl1: the kernel's own TL1; the selector is the kernel's own CCE index, so the iDMA operand is
// that CCE's package SPA and the hart checks the data through its uncached alias. kKindSramSpa: another CCE's TL1 at
// its package SPA; the selector is that CCE's index. kKindGddr: GDDR at a chiplet offset; the selector is
// pack_gddr_selector(local Mimir, target Mimir).
inline constexpr uint32_t kKindOwnTl1 = 0, kKindSramSpa = 1, kKindGddr = 2;
inline constexpr uint32_t kGddrLocalShift = 16, kGddrTargetMask = 0xFFFFu;
constexpr uint32_t pack_gddr_selector(uint32_t local_mimir, uint32_t target_mimir) {
    return (local_mimir << kGddrLocalShift) | target_mimir;
}
constexpr uint32_t gddr_selector_local(uint32_t selector) { return selector >> kGddrLocalShift; }
constexpr uint32_t gddr_selector_target(uint32_t selector) { return selector & kGddrTargetMask; }

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
inline constexpr uint32_t kStatusMagic = 0, kStatusCode = 1, kStatusFlags = 2, kStatusMismatchedWords = 3,
                          kStatusFirstMismatchWord = 4, kStatusObserved = 5, kStatusExpected = 6,
                          kStatusVcSpaceCycles = 7, kStatusAckCycles = 8, kStatusDataCycles = 9, kStatusLastPoll = 10,
                          kStatusLastPollReady = 11, kStatusElementsIssued = 12, kStatusWords = 13;
static_assert(kStatusWords * sizeof(uint32_t) <= kLineBytes);
inline constexpr uint32_t kDoneMagic = 0x1D3AC0DEu, kStatusPoison = 0xBADD0E00u;
inline constexpr uint32_t kFlagVcSpace = 1u << 0, kFlagAcked = 1u << 1, kFlagLanded = 1u << 2;
inline constexpr uint32_t kPollNone = 0, kPollVcSpace = 1, kPollAck = 2, kPollData = 3;

// kCodeLanded (kModeIdmaCopy) and kCodeAcked (kModeIdmaCopyHostVerified): every poll succeeded. Otherwise the code
// names the first poll that used its budget. Later polls still run, except that no command is issued after
// kCodeVcSpaceTimeout.
inline constexpr uint32_t kCodeLanded = 1, kCodeAcked = 2, kCodeVcSpaceTimeout = 3, kCodeAckTimeout = 4,
                          kCodeDataTimeout = 5, kCodeBadArgs = 6;

}  // namespace cce_idma_copy_abi
