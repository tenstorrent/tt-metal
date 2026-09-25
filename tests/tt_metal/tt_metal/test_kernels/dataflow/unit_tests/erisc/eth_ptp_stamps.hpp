// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the 1588 stamping test (tests/tt_metal/tt_metal/eth/test_eth_ptp_stamps.cpp) and its kernel
// (eth_ptp_stamps.cpp): the L1 both ends of a link use, at the same offsets from the active eth unreserved base.

#pragma once

#include <cstdint>

namespace eth_ptp_stamps {

constexpr uint32_t kFrameBytes = 96;
constexpr uint32_t kFrameOffset = 64;
constexpr uint32_t kResultOffset = 512;
constexpr uint32_t kRounds = 256;
constexpr uint32_t kDone = 0xD0E5u;

// One end's report. stamps[i] is round i's pair in PTP ns for the frame this end received: the peer's egress stamp,
// carried in the frame, and this end's ingress stamp. A frame arrives unstamped when it carries no egress stamp; an
// ingress stamp is missing when the FIFO gave none, and extra when it gave more than one.
struct Result {
    uint32_t done;
    uint32_t timer_ok;
    uint32_t sel_before, sel_after;
    uint32_t no_match_before, no_match_after;
    uint32_t unstamped, rx_missing, rx_extra;
    uint32_t rounds;
    uint32_t pad[2];
    uint64_t stamps[kRounds][2];
};
static_assert(sizeof(Result) % 16 == 0);

}  // namespace eth_ptp_stamps
