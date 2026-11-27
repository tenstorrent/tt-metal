// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "hal_types.hpp"

/// @file
/// Operations on lists of half-open address ranges [begin, end).
///
/// A list is sorted when its ranges are in begin address order and no two of them overlap; ranges
/// may touch. A list is coalesced when it is sorted and no two ranges touch either. Each function
/// states what it requires of its input.
namespace tt::tt_metal::allocator {

/// Half-open address range [first, second).
using AddressRange = std::pair<DeviceAddr, DeviceAddr>;

/// Coalesces any list: sorts ranges by begin address and merges ranges that overlap or touch.
/// Empty ranges are kept unless they overlap or touch another range.
std::vector<AddressRange> coalesce(std::vector<AddressRange> ranges);

/// Ranges covered by both lists, which must be sorted. O(|a| + |b|).
std::vector<AddressRange> set_intersection(const std::vector<AddressRange>& a, const std::vector<AddressRange>& b);

/// Ranges covered by `ranges` but not by `removed`; both must be sorted. O(|ranges| + |removed|).
/// Example: [(10, 30)] minus [(12, 15), (18, 25)] is [(10, 12), (15, 18), (25, 30)].
std::vector<AddressRange> set_difference(
    const std::vector<AddressRange>& ranges, const std::vector<AddressRange>& removed);

/// Removes the addresses below `limit` from any list, dropping ranges that become empty and keeping
/// the order of the rest. A limit of 0 leaves the list unchanged.
void trim_below(std::vector<AddressRange>& ranges, DeviceAddr limit);

/// The lowest address of a range that fits `size` bytes, or with bottom_up = false, the highest
/// address that leaves `size` bytes before the end of a range. Ranges are tried one at a time, in
/// list order or reverse list order, so a list that is not sorted gives the first fit in that order
/// and touching ranges are not combined. Returns nullopt when no range fits.
std::optional<DeviceAddr> first_fit(const std::vector<AddressRange>& ranges, DeviceAddr size, bool bottom_up);

}  // namespace tt::tt_metal::allocator
