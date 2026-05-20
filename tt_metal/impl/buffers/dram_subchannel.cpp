// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/dram_subchannel.hpp"

#include <set>

#include <tt_stl/assert.hpp>
#include <core_coord.hpp>
#include <device.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal::experimental {

uint32_t pick_unused_dram_subchannel(IDevice* device, uint32_t bank_id) {
    TT_FATAL(device != nullptr, "Device cannot be null");
    // Use build_id() instead of id() so MeshDevice (which has its own virtual id) routes to the
    // underlying chip's SOC descriptor.
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->build_id());
    const uint32_t num_banks = soc_desc.get_num_dram_views();
    TT_FATAL(bank_id < num_banks, "bank_id={} out of range (num_banks={})", bank_id, num_banks);

    std::set<std::pair<size_t, size_t>> reserved;
    for (const auto& c : soc_desc.dram_view_worker_cores.at(bank_id)) {
        reserved.emplace(c.x, c.y);
    }
    for (const auto& c : soc_desc.dram_view_eth_cores.at(bank_id)) {
        reserved.emplace(c.x, c.y);
    }

    const uint32_t num_subchannels = soc_desc.get_grid_size(tt::CoreType::DRAM).y;
    const size_t channel = soc_desc.get_channel_for_dram_view(static_cast<int>(bank_id));
    for (uint32_t sub = 0; sub < num_subchannels; ++sub) {
        tt::umd::CoreCoord coord = soc_desc.get_dram_core_for_channel(
            static_cast<int>(channel), static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
        if (reserved.find({coord.x, coord.y}) == reserved.end()) {
            return sub;
        }
    }
    TT_THROW(
        "No unused DRAM subchannel found for bank_id={}; all {} subchannels are reserved as worker/eth endpoints",
        bank_id,
        num_subchannels);
}

}  // namespace tt::tt_metal::experimental
