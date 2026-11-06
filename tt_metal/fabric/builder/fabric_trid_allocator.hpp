// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Transaction IDs for a router's receiver channels (study prototype, fabric_exp/consolidation/TRID_ALLOCATION.md).
//
// A receiver channel tags every NoC command that reads a slot with its completion group's transaction ID, and credits
// a group back once no command with that ID is still in flight. All receiver channels of a router run on one ERISC
// and share that NoC's 16 IDs (0..15). The host owns the rule: it sizes each channel's range from its final slot count
// and hands the router, per channel, the completion group size, the number of IDs and the first ID.
// Standard library only, so fabric_exp's offline tool compiles this same file.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace tt::tt_fabric::trid_allocator {

inline constexpr std::size_t NOC_TRANSACTION_IDS = 16;  // NOC_MAX_TRANSACTION_ID + 1

struct ReceiverChannel {
    std::size_t slots = 0;
    bool first_level_ack = false;    // the channel credits every slot on its own (bubble flow control)
    std::size_t group_override = 0;  // study switch: 1, 2 or 4 slots per completion group (0: from the slot count)
};

struct ChannelIds {
    std::size_t group_size = 1;  // slots per completion group: 1, 2 or 4
    std::size_t groups = 0;      // ceil(slots / group_size): a partial last group counts
    std::size_t num_ids = 0;     // IDs in the channel's range; < groups means groups share them (a power of 2)
    std::size_t start = 0;       // the range's first ID
    std::string reason;
};

struct Assignment {
    std::vector<ChannelIds> channels;
    std::size_t used_ids = 0;
    std::optional<std::string> error;
};

// Slots per completion group: 1 under first-level acks, else the override when set, else 1 / 2 / 4 for fewer than 8 /
// 8-15 / 16 or more slots.
std::size_t completion_group_size(std::size_t slots, bool first_level_ack, std::size_t group_override = 0);

// The rule. One ID per completion group when the groups of every channel fit the NoC's IDs; otherwise, until they
// fit, the largest count shrinks: rounded down to a power of 2 if it is not one (20 -> 16), else halved. A channel with
// fewer IDs than groups shares them by mask.
// Every channel keeps at least one ID; leftover IDs stay unused. Ranges are consecutive from 0, in channel order.
Assignment assign_transaction_ids(const std::vector<ReceiverChannel>& channels);

// Today's uniform rule, for comparison and for the switch off: the same group sizes, and every channel the same
// count (16 for a lone channel without deadlock avoidance, else 8 with deadlock avoidance, else 4), ranges in order.
Assignment uniform_transaction_ids(const std::vector<ReceiverChannel>& channels, bool deadlock_avoidance);

}  // namespace tt::tt_fabric::trid_allocator
