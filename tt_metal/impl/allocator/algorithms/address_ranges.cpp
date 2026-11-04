// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/allocator/algorithms/address_ranges.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>
#include <sys/types.h>

namespace tt::tt_metal::allocator {

std::vector<AddressRange> coalesce(std::vector<AddressRange> ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<AddressRange> coalesced;
    coalesced.reserve(ranges.size());
    for (const auto& range : ranges) {
        // range.first == back().second means the two ranges touch, which also merges them.
        if (coalesced.empty() || range.first > coalesced.back().second) {
            coalesced.push_back(range);
        } else {
            coalesced.back().second = std::max(coalesced.back().second, range.second);
        }
    }
    return coalesced;
}

std::vector<AddressRange> set_intersection(const std::vector<AddressRange>& a, const std::vector<AddressRange>& b) {
    std::vector<AddressRange> result;
    size_t i = 0;
    size_t j = 0;
    while (i < a.size() && j < b.size()) {
        const DeviceAddr begin = std::max(a[i].first, b[j].first);
        const DeviceAddr end = std::min(a[i].second, b[j].second);
        if (begin < end) {
            result.emplace_back(begin, end);
        }
        // The range that ends first cannot overlap anything further in the other list.
        if (a[i].second < b[j].second) {
            ++i;
        } else {
            ++j;
        }
    }
    return result;
}

std::vector<AddressRange> set_difference(
    const std::vector<AddressRange>& ranges, const std::vector<AddressRange>& removed) {
    std::vector<AddressRange> result;
    result.reserve(ranges.size());
    // Monotonic pointer to scan all removed ranges; we don't reset the pointer after each range.
    // This relies on both lists being sorted and non-overlapping.
    size_t j = 0;
    for (const auto& range : ranges) {
        DeviceAddr range_start = range.first;
        DeviceAddr range_end = range.second;
        // 1) Skip removed ranges that are completely to the left of the range
        //    range:              [s --- e)
        //    removed: [s --- e)
        while (j < removed.size() && removed[j].second <= range_start) {
            ++j;
        }
        // Start of the part of the range that is not emitted yet
        DeviceAddr remainder_start = range_start;
        // 2) Scan removed ranges that might overlap the current range
        //    Loop until we hit removed ranges that are to the right of the range,
        //    that is, stop when we hit ranges like this:
        //    range:   [s --- e)
        //    removed:            [s --- e)
        while (j < removed.size() && removed[j].first < range_end) {
            DeviceAddr removed_start = removed[j].first;
            DeviceAddr removed_end = removed[j].second;
            // Case A: Gap exists before the removed range
            //   remainder: [s --- e)
            //   removed:     [s ---
            //   -> Keep gap: [remainder_start, removed_start)
            if (removed_start > remainder_start) {
                result.emplace_back(remainder_start, removed_start);
            }
            // Case B: The removed range overlaps with rest of the range
            //   remainder: [s ----- e)
            //   removed:       [s ----- e)
            //   -> Nothing remains in [remainder_start, e). Update remainder_start to range_end and break
            if (removed_end >= range_end) {
                remainder_start = range_end;
                break;
            }
            // Case C: The removed range ends before the end of the range
            //   remainder: [s ------- e)
            //   removed:     [s --- e)
            //   -> Update remainder_start to the end of the removed range
            //      We continue scanning the next removed range and emit any remainder at the tail
            remainder_start = removed_end;
            ++j;
        }
        // 3) Emit remaining tail (if any) after scanning all removed ranges
        if (remainder_start < range_end) {
            result.emplace_back(remainder_start, range_end);
        }
    }
    return result;
}

void trim_below(std::vector<AddressRange>& ranges, DeviceAddr limit) {
    if (limit == 0) {
        return;
    }
    for (auto& range : ranges) {
        range.first = std::max(range.first, limit);
    }
    std::erase_if(ranges, [](const auto& range) { return range.second <= range.first; });
}

std::optional<DeviceAddr> first_fit(const std::vector<AddressRange>& ranges, DeviceAddr size, bool bottom_up) {
    // Both directions test the window's width rather than forming range.first + size or
    // range.second - size: DeviceAddr is unsigned, so a large enough request wraps around, passes
    // the comparison, and the allocation "succeeds" at a nonsense address instead of reporting.
    if (bottom_up) {
        for (const auto& range : ranges) {
            if (range.second - range.first >= size) {
                return range.first;
            }
        }
        return std::nullopt;
    }
    for (ssize_t i = static_cast<ssize_t>(ranges.size()) - 1; i >= 0; --i) {
        const auto& range = ranges[static_cast<size_t>(i)];
        if (range.second - range.first >= size) {
            return range.second - size;
        }
    }
    return std::nullopt;
}

}  // namespace tt::tt_metal::allocator
