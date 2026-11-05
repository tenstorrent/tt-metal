// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_trid_allocator.hpp"

namespace tt::tt_fabric::trid_allocator {

// Named rather than anonymous: the library builds this file in a unity batch with its neighbours.
namespace trid_detail {

bool is_power_of_2(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

std::size_t floor_power_of_2(std::size_t n) {
    std::size_t p = 1;
    while (p * 2 <= n) {
        p *= 2;
    }
    return p;
}

std::string describe(const ChannelIds& c, const ReceiverChannel& in) {
    return std::to_string(in.slots) + " slots in groups of " + std::to_string(c.group_size) +
           (in.first_level_ack ? " (first-level acks)" : "") + ": " + std::to_string(c.groups) + " groups, IDs " +
           std::to_string(c.start) + ".." + std::to_string(c.start + c.num_ids - 1) +
           (c.num_ids >= c.groups ? " (one per group)" : " (shared: group & " + std::to_string(c.num_ids - 1) + ")");
}

}  // namespace trid_detail

std::size_t completion_group_size(std::size_t slots, bool first_level_ack) {
    if (first_level_ack) {
        return 1;
    }
    return slots < 8 ? 1 : (slots < 16 ? 2 : 4);
}

Assignment assign_transaction_ids(const std::vector<ReceiverChannel>& channels) {
    Assignment result;
    std::size_t total = 0;
    for (const ReceiverChannel& in : channels) {
        ChannelIds c;
        c.group_size = completion_group_size(in.slots, in.first_level_ack);
        c.groups = (in.slots + c.group_size - 1) / c.group_size;
        c.num_ids = c.groups == 0 ? 1 : c.groups;
        total += c.num_ids;
        result.channels.push_back(c);
    }
    // Until the counts fit, the largest shrinks to the largest count that fits below it: a count that is not a power
    // of 2 rounds down to one (20 -> 16), a power of 2 halves (16 -> 8).
    while (total > noc_transaction_ids) {
        ChannelIds* largest = nullptr;
        for (ChannelIds& c : result.channels) {
            if (c.num_ids > 1 && (largest == nullptr || c.num_ids > largest->num_ids)) {
                largest = &c;
            }
        }
        if (largest == nullptr) {
            result.error = std::to_string(channels.size()) + " receiver channels need more than " +
                           std::to_string(noc_transaction_ids) + " transaction IDs at one each";
            return result;
        }
        const std::size_t next = trid_detail::is_power_of_2(largest->num_ids)
                                     ? largest->num_ids / 2
                                     : trid_detail::floor_power_of_2(largest->num_ids);
        total -= largest->num_ids - next;
        largest->num_ids = next;
    }
    std::size_t start = 0;
    for (std::size_t i = 0; i < result.channels.size(); i++) {
        ChannelIds& c = result.channels[i];
        c.start = start;
        start += c.num_ids;
        c.reason = trid_detail::describe(c, channels[i]);
    }
    result.used_ids = start;
    return result;
}

}  // namespace tt::tt_fabric::trid_allocator
