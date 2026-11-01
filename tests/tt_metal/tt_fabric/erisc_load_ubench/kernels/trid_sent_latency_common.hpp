// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the transaction-ID sent-latency microbenchmark's host test and kernel (test_erisc_load_overlap.cpp,
// TridSentLatency). How long after the router's receiver issues a packet's local writes does the NIU report every write
// of its transaction ID sent (the router's completion proof, writes_left)? Each case issues one packet's writes, as the
// receiver's local writes, and polls the ID's outgoing-write count until it reads 0.

#pragma once

#include <cstdint>

namespace trid_sent_latency {

// Packet kinds: one write of 64 B, 1 KiB, 2 KiB or 4352 B (a full fabric packet), or a scatter write of 4352 B in 2 or 4
// chunks, every write tagged with the packet's ID.
enum Kind : uint32_t { W64, W1K, W2K, W4352, SC2, SC4, kNumKinds };
constexpr uint32_t kKindWrites[kNumKinds] = {1, 1, 1, 1, 2, 4};
constexpr uint32_t kKindBytes[kNumKinds] = {64, 1024, 2048, 4352, 4352, 4352};
// Contexts: the NIU idle (every earlier write acknowledged), or the packet issued right behind one 4352 B write of
// another ID to another core (as a forward or another packet's write just before it).
enum Context : uint32_t { IDLE, BEHIND, kNumContexts };
constexpr uint32_t kNumCases = static_cast<uint32_t>(kNumKinds) * static_cast<uint32_t>(kNumContexts);
constexpr uint32_t kReps = 256;

// The router's local writes: static VC 3 (EDM_NOC_VC), a Tensix L1 destination; the probe's ID and the earlier write's.
constexpr uint32_t kVc = 3;
constexpr uint32_t kProbeTrid = 1;
constexpr uint32_t kOtherTrid = 2;
constexpr uint32_t kMaxBytes = 4352;

// Per case: the summary fields, then each repetition's sent time (cycles from the last command's go to the poll that
// read 0).
enum Field : uint32_t {
    ISSUE_SUM,  // first command programmed .. last command's go register written
    ISSUE_MIN,
    TOTAL_MIN,  // first command .. the poll that read 0
    POLLS,      // polls of the ID's register, summed over the repetitions
    kNumFields,
};
constexpr uint32_t kCaseWords = kNumFields + kReps;
constexpr uint32_t kPollCost = kNumCases * kCaseWords;  // one poll that reads 0 at once (nothing outstanding), min
constexpr uint32_t kDone = kPollCost + 1;
constexpr uint32_t kFailed = kDone + 1;  // the case whose wait timed out, + 1 (0: none)
constexpr uint32_t kResultWords = kFailed + 1;
constexpr uint32_t kDoneValue = 0xD0D2;
constexpr uint32_t case_index(uint32_t kind, uint32_t context) { return (context * kNumKinds + kind) * kCaseWords; }

}  // namespace trid_sent_latency
