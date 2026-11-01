// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the NoC command-buffer microbenchmark's host test and kernel (test_erisc_load_overlap.cpp,
// NocCmdBufOverlap). The kernel runs once per group (NOC_CMDBUF_GROUP):
//   0  table 1: writes issued as the router's local writes (a ready read before each), rotating over 1, 2 or 4 buffers
//   1  table 2: the same writes with a ready mask: buffers known ready are used without a read; the mask is sampled as
//      a group (all the pool's ready registers read back to back) at the end of each group of 4 writes and, when the
//      next buffer's bit is clear, again before it (then a spin on that buffer). Fixed rotation, or any ready buffer
//   2  table 3: the cost of a grouped ready sample of 1 to 4 buffers (idle);
//      table 4: forward-like pairs, a posted data write and a posted inline write to a stream register on the same
//      core (the forward's credit), with the two on one buffer, on two buffers fully programmed, on two with presets,
//      and on two (or three, rotating the data) with presets and the ready mask.

#pragma once

#include <cstddef>
#include <cstdint>

namespace noc_cmdbuf_overlap {

// Shapes: a burst of 4 writes from an idle NoC interface (a scatter packet's chunks), and a stream of back-to-back
// writes (multicast forwards, or local writes under load).
enum Shape : uint32_t { BURST, STREAM, kNumShapes };
constexpr uint32_t kSizes[] = {64, 256, 1024, 2048, 4096};
constexpr uint32_t kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);
// Table 1: command buffers the writes rotate over (write k on buffer k % n).
constexpr uint32_t kBufCounts[] = {1, 2, 4};
constexpr uint32_t kNumBufCounts = sizeof(kBufCounts) / sizeof(kBufCounts[0]);
// Table 2: ready-mask variants: fixed rotation over 1, 2 or 4 buffers, or the lowest ready buffer of a pool of 2 or 4.
enum MaskVariant : uint32_t { MASK_ROT1, MASK_ROT2, MASK_ROT4, MASK_ANY2, MASK_ANY4, kNumMaskVariants };
constexpr uint32_t kBurstWrites = 4;
constexpr uint32_t kStreamWrites = 64;
constexpr uint32_t kBurstReps = 64;
constexpr uint32_t kStreamReps = 8;
// Table 3: grouped ready samples of 1..kMaxSample buffers, kSampleIters each.
constexpr uint32_t kMaxSample = 4;
constexpr uint32_t kSampleIters = 1024;
// Table 4: forward-like pairs.
enum PairMode : uint32_t {
    PAIR_SHARED,       // one buffer: each command fully programmed (the write first restores its source coordinate)
    PAIR_FULL,         // data on buffer 0 (preset), credit on buffer 3 fully programmed (several-downstream routers)
    PAIR_PRESET,       // data on buffer 0 (preset), credit on buffer 1 (preset) (one-downstream routers)
    PAIR_PRESET_MASK,  // PAIR_PRESET with the ready mask, sampled once per pair
    PAIR_PRESET_MASK_ROT,  // the same with the data rotating over buffers 0 and 2
    kNumPairModes,
};
constexpr uint32_t kPairSizes[] = {64, 4096};
constexpr uint32_t kNumPairSizes = sizeof(kPairSizes) / sizeof(kPairSizes[0]);
constexpr uint32_t kPairs = 32;
constexpr uint32_t kPairReps = 8;
constexpr uint32_t kCreditStream = 0;  // the destination core's stream register the credits increment

// The router's local writes: transaction ID tagged, static VC 3 (EDM_NOC_VC), destination a Tensix L1.
constexpr uint32_t kVc = 3;
constexpr uint32_t kSlots = 16;  // source and destination rings of kSlots x the largest size
constexpr uint32_t kSlotBytes = 4096;

// Results per case, in cycles, summed over the repetitions unless "min":
enum Field : uint32_t {
    ISSUE,      // first command programmed .. last command's go register written
    ISSUE_MIN,  // the same, fastest repetition
    TAG_WAIT,   // cycles in ready reads and waits (table 1: the transaction-ID call, which includes the tag write)
    FIRST_MIN,  // the first write (table 4: pair) alone, fastest repetition
    SENT,       // first command .. every write has left the NoC interface
    SENT_MIN,
    ACKED,      // first command .. every write acknowledged (non-posted writes: tables 1, 2)
    ACKED_MIN,
    READS,      // ready registers read (a grouped sample counts each buffer it reads)
    kNumFields,
};
constexpr uint32_t kNumGroups = 3;
constexpr uint32_t kT1 = 0;
constexpr uint32_t kT2 = kT1 + kNumShapes * kNumSizes * kNumBufCounts * kNumFields;
constexpr uint32_t kT3 = kT2 + kNumShapes * kNumSizes * kNumMaskVariants * kNumFields;
constexpr uint32_t kT4 = kT3 + kMaxSample;
constexpr uint32_t kDoneBase = kT4 + kNumPairSizes * kNumPairModes * kNumFields;  // a done flag per group
constexpr uint32_t kFailedBase = kDoneBase + kNumGroups;  // per group: the case whose wait timed out (0: none)
constexpr uint32_t kResultWords = kFailedBase + kNumGroups;
constexpr uint32_t kDone = 0xD0D1;
constexpr uint32_t t1_index(uint32_t shape, uint32_t size_i, uint32_t buf_i) {
    return kT1 + ((shape * kNumSizes + size_i) * kNumBufCounts + buf_i) * kNumFields;
}
constexpr uint32_t t2_index(uint32_t shape, uint32_t size_i, uint32_t variant) {
    return kT2 + ((shape * kNumSizes + size_i) * kNumMaskVariants + variant) * kNumFields;
}
constexpr uint32_t t4_index(uint32_t size_i, uint32_t mode) {
    return kT4 + (size_i * kNumPairModes + mode) * kNumFields;
}

}  // namespace noc_cmdbuf_overlap
