// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_device_map.hpp"

namespace tt::tt_metal {

// Extracts all the pairs of noc multicast encodings given a set of core ranges
std::vector<multicast_transfer_info> extract_dst_noc_multicast_info(
    IDevice* device, const std::vector<CoreRange>& ranges, const CoreType core_type) {
    std::vector<multicast_transfer_info> dst_noc_multicast_info;
    dst_noc_multicast_info.reserve(ranges.size());
    for (const CoreRange& core_range : ranges) {
        CoreCoord virtual_start = device->virtual_core_from_logical_core(core_range.start_coord, core_type);
        CoreCoord virtual_end = device->virtual_core_from_logical_core(core_range.end_coord, core_type);

        uint32_t num_receivers = core_range.size();
        dst_noc_multicast_info.push_back(multicast_transfer_info{CoreRange(virtual_start, virtual_end), num_receivers});
    }
    return dst_noc_multicast_info;
}

}  // namespace tt::tt_metal
