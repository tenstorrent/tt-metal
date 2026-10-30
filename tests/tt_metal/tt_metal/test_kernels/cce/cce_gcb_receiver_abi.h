// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Kernel ABI for the MM-7 GlobalCircularBuffer test kernels: the smoke sender and the discard and validator
// receivers. The kernels, dram/mimir_prefetcher_common.hpp, dram/test_mimir_prefetcher.cpp and dram/test_mimir_trid.cpp
// include it.
namespace cce_gcb_receiver_abi {

inline constexpr uint32_t kLineBytes = 64;

// Runtime args, the same layout for every kernel. kArgGcbAddr is the GCB's config page on a receiver and its sender
// state block on the sender. The budget is rdcycle cycles per wait, 0 = infinite. kArgSeed is unused by the discard
// receiver; kArgMode and kArgStallPages are the discard receiver's only.
inline constexpr uint32_t kArgGcbAddr = 0, kArgBlockAddr = 1, kArgBudgetLo = 2, kArgBudgetHi = 3, kArgSeed = 4,
                          kArgMode = 5, kArgStallPages = 6, kArgNumRequests = 7, kArgScheduleBase = 8;

// One schedule entry per request, in order. The sender writes receiver r of its table slab `slab + r`; each
// receiver's own schedule names its slab directly.
inline constexpr uint32_t kRequestTensorId = 0, kRequestSlab = 1, kRequestPageBytes = 2, kRequestNumPages = 3,
                          kRequestWords = 4;
inline constexpr uint32_t kMaxRequests = 8;

constexpr uint32_t schedule_arg(uint32_t request, uint32_t field) {
    return kArgScheduleBase + kRequestWords * request + field;
}

// Discard receiver modes. DrainCount consumes the schedule; DrainUntilStopped consumes pages of the first request's
// size until the host writes kCommandStop; StallUntilReleased consumes the schedule as DrainCount does, then waits for
// kArgStallPages more pages of the last request's size, holds them, and on kCommandStop exits without acking them or
// writing rd_ptr back.
inline constexpr uint32_t kModeDrainCount = 0, kModeDrainUntilStopped = 1, kModeStallUntilReleased = 2;

// Each kernel's block, at kArgBlockAddr in its own TL1. The host carves it from the DRISC L1 receiver zone, at one
// offset on every CCE.
//   command line: host-written; kCommandStop stops a waiting receiver
//   status line:  kernel-written once at exit, magic last
//   holding line: the stalling receiver's held page count, written once it holds them
inline constexpr uint32_t kCommandLineOffset = 0, kStatusLineOffset = kLineBytes, kHoldingLineOffset = 2 * kLineBytes,
                          kBlockBytes = 3 * kLineBytes;
inline constexpr uint32_t kCommandRun = 0, kCommandStop = 0x5709u;

inline constexpr uint32_t kStatusMagic = 0, kStatusCode = 1, kStatusRequestsDone = 2, kStatusPagesDone = 3,
                          kStatusMismatchedWords = 4, kStatusFirstMismatchRequest = 5, kStatusFirstMismatchPage = 6,
                          kStatusFirstMismatchWord = 7, kStatusObserved = 8, kStatusExpected = 9,
                          kStatusWaitReceiver = 10, kStatusWaitPagesSent = 11, kStatusWaitPagesAcked = 12,
                          kStatusRechecked = 13, kStatusWords = 14;
inline constexpr uint32_t kDoneMagic = 0x6CB0D0E5u, kStatusPoison = 0xBADD0E00u;

// kCodeCreditTimeout: a sender's free-space wait or a receiver's page wait used its budget; the sender then records
// the short receiver's table slot and its pages_sent and pages_acked in the kStatusWait* words. kCodeDrainTimeout: the
// sender's final barrier did. kCodeStopped: a receiver saw kCommandStop while waiting for a page. kCodeReleased: the
// stalling receiver was released. kCodeReleaseTimeout: it held its pages for a whole budget without a release.
inline constexpr uint32_t kCodeOk = 1, kCodeCreditTimeout = 2, kCodeDrainTimeout = 3, kCodeMismatch = 4,
                          kCodeStopped = 5, kCodeReleased = 6, kCodeReleaseTimeout = 7, kCodeBadSchedule = 8;

// Word `word` of page `page` of a request. The sender writes it, the validator recomputes it, and the host can fill
// DRAM with it.
constexpr uint32_t gcb_page_word(uint32_t seed, uint32_t tensor_id, uint32_t slab, uint32_t page, uint32_t word) {
    uint32_t value =
        seed ^ (tensor_id * 0x9E3779B9u) ^ (slab * 0x85EBCA6Bu) ^ (page * 0xC2B2AE35u) ^ (word * 0x27D4EB2Fu);
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    return value;
}

}  // namespace cce_gcb_receiver_abi
