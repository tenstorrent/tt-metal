// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Transaction IDs for a router's receiver channels.
//
// A receiver channel tags every NoC command that reads one of its slots with the slot's completion group's
// transaction ID, and returns a group's credits once no command with that ID is in flight. All receiver channels of a
// router run on one ERISC and share that NoC's 16 IDs.
//
// assign_transaction_ids() is a pure function of the channels' final slot counts (the slot allocator's) and their
// first-level acks. Per channel it returns the completion group size, the number of IDs and the first ID:
//   1. Group size G: 1 under first-level acks (each slot is credited on its own), else 1 / 2 / 4 for fewer than 8 /
//      8-15 / 16 or more slots. Groups = ceil(slots / G): a partial last group counts.
//   2. When every channel's groups fit in the NoC's IDs together, each group has its own ID.
//   3. Otherwise the largest count shrinks until they fit: a count that is not a power of 2 is rounded down to one,
//      a power of 2 is halved. A channel with fewer IDs than groups shares them by mask, so a group's proof also waits
//      for the writes of the groups sharing its ID: credits come later, never early.
//   4. Every channel keeps at least one ID. Ranges are consecutive from 0, in receiver channel order; IDs left over
//      stay unused.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace tt::tt_fabric::trid_allocator {

inline constexpr std::size_t noc_transaction_ids = 16;  // NOC_MAX_TRANSACTION_ID + 1

// One receiver channel, as the rule reads it.
struct ReceiverChannel {
    std::size_t slots = 0;
    bool first_level_ack = false;
};

// A receiver channel's transaction IDs.
struct ChannelIds {
    std::size_t group_size = 1;  // slots per completion group: 1, 2 or 4
    std::size_t groups = 0;      // ceil(slots / group_size)
    std::size_t num_ids = 0;     // IDs in the channel's range; fewer than groups: shared by mask (a power of 2)
    std::size_t start = 0;       // the range's first ID
};

struct Assignment {
    std::vector<ChannelIds> channels;  // in receiver channel order
    std::size_t used_ids = 0;
    std::optional<std::string> error;
};

// Slots per completion group: 1 under first-level acks, else 1 / 2 / 4 for fewer than 8 / 8-15 / 16 or more slots.
std::size_t completion_group_size(std::size_t slots, bool first_level_ack);

Assignment assign_transaction_ids(const std::vector<ReceiverChannel>& channels);

}  // namespace tt::tt_fabric::trid_allocator
