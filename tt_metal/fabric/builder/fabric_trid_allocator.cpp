// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_trid_allocator.hpp"

#include <fmt/format.h>

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

}  // namespace trid_detail

std::size_t completion_group_size(const ReceiverChannel& channel) {
    if (channel.credit_each_slot || channel.slots < 8) {
        return 1;
    }
    if (channel.slots < 16) {
        return 2;
    }
    return 4;
}

Assignment assign_transaction_ids(const std::vector<ReceiverChannel>& channels) {
    Assignment result;
    std::size_t total = 0;
    for (const ReceiverChannel& in : channels) {
        ChannelIds c;
        c.group_size = completion_group_size(in);
        c.groups = (in.slots + c.group_size - 1) / c.group_size;
        c.num_ids = c.groups == 0 ? 1 : c.groups;
        total += c.num_ids;
        result.channels.push_back(c);
    }
    // Rule 3: shrink the largest count until they fit (20 -> 16, 16 -> 8).
    while (total > noc_transaction_ids) {
        ChannelIds* largest = nullptr;
        for (ChannelIds& c : result.channels) {
            if (c.num_ids > 1 && (largest == nullptr || c.num_ids > largest->num_ids)) {
                largest = &c;
            }
        }
        if (largest == nullptr) {
            result.error = fmt::format(
                "{} receiver channels need more than {} transaction IDs at one each",
                channels.size(),
                noc_transaction_ids);
            return result;
        }
        const std::size_t next = trid_detail::is_power_of_2(largest->num_ids)
                                     ? largest->num_ids / 2
                                     : trid_detail::floor_power_of_2(largest->num_ids);
        total -= largest->num_ids - next;
        largest->num_ids = next;
    }
    std::size_t start = 0;
    for (ChannelIds& c : result.channels) {
        c.start = start;
        start += c.num_ids;
    }
    result.used_ids = start;
    return result;
}

}  // namespace tt::tt_fabric::trid_allocator
