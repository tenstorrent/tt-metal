// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The streaming profiler's link half on the tile's 1588 hardware (internal/ethernet/eth_ptp.hpp): the session
// every sync kernel opens, and how a round's stamps are averaged and reported.

#pragma once

#include <cstdint>

#include "internal/ethernet/eth_ptp.hpp"

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kLinkTxq = 2;        // fabric routers send on queue 0
constexpr uint32_t kLinkHeaderRow = 3;  // firmware programs rows 0..2
constexpr uint32_t kLinkTcamRow = 63;
constexpr uint32_t kLinkLabel = 0x15;
using LinkSession = StampSession<kLinkTxq, kLinkHeaderRow, kLinkTcamRow, kLinkLabel>;

// A round is kTripsPerRound back-to-back exchanges whose stamps are averaged on each side: every stamp is quantised
// to the timer's 20 ns tick, the trips sit at different phases of it, so the mean's rounding noise falls by the
// square root of the count (8.8 -> ~0.4 ns per round at 256), and the averages are reported in quarter-ns units
// so that gain reaches the host whole. A round of 256 trips takes ~320 us of the link's 1 ms cadence.
constexpr uint32_t kTripsPerRound = 256;
constexpr uint32_t kHwUnitsPerNs = 4;
inline uint64_t link_hw_q(const LinkSession& st, int64_t ns_sum, uint32_t count) {
    const int64_t elapsed_sum = ns_sum - st.ns_minus_20cfr * static_cast<int64_t>(count);
    const int64_t c = static_cast<int64_t>(count);
    return static_cast<uint64_t>((elapsed_sum * kHwUnitsPerNs + c / 2) / c);
}

}  // namespace tt::tt_metal::eth_ptp
