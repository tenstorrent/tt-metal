// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(KERNEL_BUILD)
#include "api/dataflow/dataflow_api.h"
#endif

namespace sync_workload {

// A source role's numeric value is the NoC it multicasts on.
enum class MulticastRole : std::uint32_t { Noc0Source, Noc1Source, Receiver };

// A ping-pong role's numeric value is the parity of the rounds in which that end sends first.
enum class PingpongRole : std::uint32_t { EvenRoundSender, OddRoundSender };

// kRoundWord holds the latest round to reach the kernel, and kAckWord the host poke kernel's ack.
enum FlagWord : std::uint32_t { kRoundWord, kAckWord };

// How many times wait_for_round() polls before it gives up. It is long enough that a kernel gives up only once its peer
// or the host has stopped.
constexpr std::uint32_t kSpinLimit = 1u << 26;

#if defined(KERNEL_BUILD)
FORCE_INLINE bool wait_for_round(volatile tt_l1_ptr std::uint32_t* flag, std::uint32_t round) {
    for (std::uint32_t polls = 0; flag[kRoundWord] < round; polls++) {
        invalidate_l1_cache();
        if (polls == kSpinLimit) {
            return false;
        }
    }
    return true;
}
#endif

// The multicast's round values sit this far apart because an L1-to-L1 NoC write needs source and destination congruent
// mod 16.
constexpr std::uint32_t kRoundValueStrideBytes = 16;

}  // namespace sync_workload
