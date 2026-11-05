// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {

// Static L1 regions (CBs, DFBs) of programs enqueued on a command queue and sub-device whose
// completion the host has not observed yet. Each mark is tagged with the id of the first event
// recorded on that command queue after the program, so an event the host has seen complete
// retires every mark tagged at or before it.
//
// Not thread safe; the owning allocator serializes access.
class InFlightL1Regions {
public:
    using Intervals = std::vector<std::pair<DeviceAddr, DeviceAddr>>;
    using RegionsByCoreRange = std::map<CoreRange, Intervals>;

    // Widens each interval to `alignment` so placements ending or starting at it stay aligned.
    void mark(
        uint32_t cq_id,
        SubDeviceId sub_device_id,
        uint32_t event_id,
        const RegionsByCoreRange& regions,
        DeviceAddr alignment);
    void clear(uint32_t cq_id, SubDeviceId sub_device_id, uint32_t up_to_event_id);
    // Every sub-device of cq_id.
    void clear(uint32_t cq_id, uint32_t up_to_event_id);

    bool empty() const { return marks_.empty(); }

    // Appends the intervals of every mark on a core range holding any of `cores`; of every mark when
    // `cores` is nullopt.
    void append_occupied_ranges(const std::optional<std::vector<CoreCoord>>& cores, Intervals& out) const;

private:
    std::map<std::tuple<uint32_t, SubDeviceId, uint32_t>, RegionsByCoreRange> marks_;
};

}  // namespace tt::tt_metal
