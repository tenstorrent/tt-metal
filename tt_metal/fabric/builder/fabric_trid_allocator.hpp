// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Transaction IDs for a router's receiver channels.
//
// A receiver channel tags each NoC command that reads a slot with the slot's completion group's ID, and credits a group
// once no command with that ID is in flight. A router's receiver channels share one NoC's 16 IDs.
//
// assign_transaction_ids() is a pure function of each channel's final slot count, first-level acks and, on Wormhole,
// whether it forwards. Per channel it returns the group size, the number of IDs and the first ID:
//   1. Group size: 1 under first-level acks, and on a Wormhole channel that forwards (its credits share the one TX
//      queue with data, and a held group stalls a forwarding chain); else 1 / 2 / 4 for < 8 / 8-15 / >= 16 slots.
//      Groups = ceil(slots / group size).
//   2. If every channel's groups fit in the IDs, each group gets its own.
//   3. Otherwise the largest count shrinks until they fit (a power of 2 halves, any other count rounds down to one). A
//      channel with fewer IDs than groups shares them by mask, which can only delay a credit, never send it early.
//   4. Every channel keeps at least one ID; ranges are consecutive from 0, in channel order.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace tt::tt_fabric::trid_allocator {

inline constexpr std::size_t noc_transaction_ids = 16;  // NOC_MAX_TRANSACTION_ID + 1

// One receiver channel, as the rules read it.
struct ReceiverChannel {
    std::size_t slots = 0;
    bool first_level_ack = false;
    bool wormhole_forwards = false;  // a Wormhole router's channel that forwards (rule 1)
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

// Slots per completion group (rule 1).
std::size_t completion_group_size(const ReceiverChannel& channel);

Assignment assign_transaction_ids(const std::vector<ReceiverChannel>& channels);

}  // namespace tt::tt_fabric::trid_allocator
