// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The frame and result layout in the eth core's L1, shared by ActiveEthPtpStamps and its kernel.

#pragma once

#include <cstdint>

namespace eth_ptp_stamps {

constexpr uint32_t kFrameBytes = 32;
// The kernel's L1 base holds the start and stop handshake, and the frame and result follow at these offsets.
constexpr uint32_t kHandshakeBytes = 16;
constexpr uint32_t kFrameOffset = 64;
constexpr uint32_t kResultOffset = 512;
static_assert(kHandshakeBytes <= kFrameOffset && kFrameOffset + kFrameBytes <= kResultOffset);
constexpr uint32_t kRounds = 256;
constexpr uint32_t kTwoStepFrames = 16;
constexpr uint32_t kDone = 0xD0E5u;
// A refclk tick in PTP ns. Every stamp is a multiple of it.
constexpr uint32_t kStampTickNs = 20;

// The stamps, in PTP ns, of the frame this port received in one round. peer_egress is the peer's egress stamp, which
// the frame carries, and ingress is this port's ingress stamp.
struct RoundStamps {
    uint64_t peer_egress;
    uint64_t ingress;
};

struct Result {
    uint32_t done;
    uint32_t header_select_before, header_select_after;
    uint32_t no_match_before, no_match_after;
    // The PTP time minus the refclk time, in ns, read just after a refclk update.
    int32_t restart_error_ns;
    uint32_t unstamped;
    // Received frames whose RX stamp FIFO didn't hold exactly one entry with the stamp rule's label.
    uint32_t ingress_mismatched;
    uint32_t rounds;
    // Two-step frames whose TX stamp FIFO entry had the frame's tag and a time between the PTP times read before and
    // after the send.
    uint32_t two_step_matched;
    RoundStamps stamps[kRounds];
};

}  // namespace eth_ptp_stamps
