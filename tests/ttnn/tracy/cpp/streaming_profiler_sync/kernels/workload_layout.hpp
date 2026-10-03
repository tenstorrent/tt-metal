// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// A source's value is the NoC it multicasts on.
enum class MulticastRole : std::uint32_t { Noc0Source, Noc1Source, Receiver };

// The words of a kernel's flag: the round it has reached, the round it gave up waiting on (0 if none), and the host
// poke kernel's ack.
enum FlagWord : std::uint32_t { kRoundWord, kGaveUpRoundWord, kAckWord };

// The polls a kernel spends on a round before it records the round in kGaveUpRoundWord and exits.
constexpr std::uint32_t kSpinLimit = 1u << 26;

// The multicast's round values sit this far apart because an L1-to-L1 NoC write needs source and destination congruent
// mod 16.
constexpr std::uint32_t kRoundValueStrideBytes = 16;
