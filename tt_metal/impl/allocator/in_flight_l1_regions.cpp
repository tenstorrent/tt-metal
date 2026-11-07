// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/allocator/in_flight_l1_regions.hpp"

#include <algorithm>
#include <iterator>

#include <tt-metalium/math.hpp>

namespace tt::tt_metal {

void InFlightL1Regions::mark(
    uint32_t cq_id,
    SubDeviceId sub_device_id,
    uint32_t event_id,
    const RegionsByCoreRange& regions,
    DeviceAddr alignment) {
    for (const auto& [core_range, intervals] : regions) {
        Intervals incoming;
        for (const auto& [begin, end] : intervals) {
            if (begin < end) {
                incoming.emplace_back(begin / alignment * alignment, tt::round_up(end, alignment));
            }
        }
        if (incoming.empty()) {
            continue;
        }
        // Programs enqueued between two events share a tag, so coalescing here keeps a mark's size
        // bounded by the distinct core ranges rather than by the number of enqueues.
        auto& merged = marks_[{cq_id, sub_device_id, event_id}][core_range];
        merged.insert(merged.end(), incoming.begin(), incoming.end());
        std::sort(merged.begin(), merged.end());
        Intervals coalesced;
        for (const auto& interval : merged) {
            if (coalesced.empty() || interval.first > coalesced.back().second) {
                coalesced.push_back(interval);
            } else {
                coalesced.back().second = std::max(coalesced.back().second, interval.second);
            }
        }
        merged = std::move(coalesced);
    }
}

void InFlightL1Regions::clear(uint32_t cq_id, SubDeviceId sub_device_id, uint32_t up_to_event_id) {
    marks_.erase(
        marks_.lower_bound({cq_id, sub_device_id, 0}), marks_.upper_bound({cq_id, sub_device_id, up_to_event_id}));
}

void InFlightL1Regions::clear(uint32_t cq_id, uint32_t up_to_event_id) {
    const auto end = marks_.lower_bound({cq_id + 1, SubDeviceId{0}, 0});
    for (auto it = marks_.lower_bound({cq_id, SubDeviceId{0}, 0}); it != end;) {
        it = std::get<2>(it->first) <= up_to_event_id ? marks_.erase(it) : std::next(it);
    }
}

void InFlightL1Regions::append_occupied_ranges(
    const std::optional<std::vector<CoreCoord>>& cores, Intervals& out) const {
    for (const auto& [key, regions] : marks_) {
        for (const auto& [core_range, intervals] : regions) {
            if (cores.has_value() && std::none_of(cores->begin(), cores->end(), [&](const CoreCoord& core) {
                    return core_range.contains(core);
                })) {
                continue;
            }
            out.insert(out.end(), intervals.begin(), intervals.end());
        }
    }
}

}  // namespace tt::tt_metal
