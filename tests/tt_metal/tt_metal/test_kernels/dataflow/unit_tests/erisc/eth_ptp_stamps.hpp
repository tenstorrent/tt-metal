// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the 1588 stamping test (tests/tt_metal/tt_metal/eth/test_eth_ptp_stamps.cpp) and its kernel
// (eth_ptp_stamps.cpp): the L1 both ends of a link use, at the same offsets from the active eth unreserved base.

#pragma once

#include <cstdint>

namespace eth_ptp_stamps {

constexpr uint32_t kFrameBytes = 96;
constexpr uint32_t kPilotOffset = 64;
constexpr uint32_t kFrameOffset = kPilotOffset + kFrameBytes;
constexpr uint32_t kResultOffset = 512;
constexpr uint32_t kRounds = 256;
constexpr uint32_t kDone = 0xD0E5u;

// One end's report. stamps[i] is round i's pair in PTP ns: the initiator's frame egress and echo ingress, or the
// echoing end's frame ingress and echo egress. A missing stamp is a round whose FIFO gave none; an extra is a second
// entry under the round's tag or label.
struct Result {
    uint32_t done;
    uint32_t timer_ok;
    uint32_t ptp_offset_lo, ptp_offset_hi;
    uint32_t sel_before, sel_after;
    uint32_t no_match_before, no_match_after;
    uint32_t tx_missing, tx_extra, rx_missing, rx_extra;
    uint32_t rounds;
    uint32_t pad[3];
    uint64_t stamps[kRounds][2];
};
static_assert(sizeof(Result) % 16 == 0);

}  // namespace eth_ptp_stamps
