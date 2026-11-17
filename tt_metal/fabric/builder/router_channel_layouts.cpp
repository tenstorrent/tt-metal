// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_channel_layouts.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include "distributed_context.hpp"

#include <algorithm>

namespace tt::tt_fabric {

void RouterChannelLayouts::begin_launch() {
    std::lock_guard lock(mutex_);
    layouts_.clear();
}

void RouterChannelLayouts::publish(FabricNodeId node, chan_id_t eth_chan, RouterChannelLayout layout) {
    std::lock_guard lock(mutex_);
    const bool inserted = layouts_.try_emplace({node, eth_chan}, std::move(layout)).second;
    TT_FATAL(inserted, "Router {} channel {} published its channel layout twice", node, eth_chan);
}

std::vector<RouterChannelLayouts::Record> RouterChannelLayouts::pack(const Map& layouts) {
    std::vector<Record> records;
    for (const auto& [key, layout] : layouts) {
        Record record{.mesh_id = *key.first.mesh_id, .chip_id = key.first.chip_id, .eth_chan = key.second};
        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            for (const auto& channel : layout.receivers[vc]) {
                TT_FATAL(record.num_receivers < record.receivers.size(), "Router {} has too many receivers", key.first);
                record.receivers[record.num_receivers++] = {vc, channel.base_address, channel.num_slots};
            }
        }
        records.push_back(record);
    }
    return records;
}

void RouterChannelLayouts::merge(
    Map& layouts, const std::vector<Record>& gathered, const std::vector<uint32_t>& num_records) {
    const size_t stride = gathered.size() / num_records.size();
    for (size_t rank = 0; rank < num_records.size(); ++rank) {
        for (size_t i = 0; i < num_records[rank]; ++i) {
            const Record& record = gathered[(rank * stride) + i];
            RouterChannelLayout layout;
            for (uint32_t r = 0; r < record.num_receivers; ++r) {
                const auto& receiver = record.receivers[r];
                layout.receivers[receiver.vc].push_back({receiver.base_address, receiver.num_slots});
            }
            // a host's own routers are already here, with their senders
            layouts.try_emplace(
                {FabricNodeId(MeshId{record.mesh_id}, record.chip_id), static_cast<chan_id_t>(record.eth_chan)},
                std::move(layout));
        }
    }
}

void RouterChannelLayouts::exchange(const tt::tt_metal::distributed::multihost::DistributedContext& context) {
    const auto world_size = static_cast<size_t>(*context.size());
    if (world_size == 1) {
        return;
    }

    std::vector<Record> local_records;
    {
        std::lock_guard lock(mutex_);
        local_records = pack(layouts_);
    }

    // ranks publish different numbers of routers: gather the counts, then every rank's records padded to the largest
    auto num_local_records = static_cast<uint32_t>(local_records.size());
    std::vector<uint32_t> num_records(world_size);
    context.all_gather(
        ttsl::as_writable_bytes(ttsl::Span<uint32_t>{&num_local_records, 1}),
        ttsl::as_writable_bytes(ttsl::Span<uint32_t>{num_records.data(), num_records.size()}));
    const size_t stride = *std::max_element(num_records.begin(), num_records.end());
    local_records.resize(stride);
    std::vector<Record> gathered(stride * world_size);
    context.all_gather(
        ttsl::as_writable_bytes(ttsl::Span<Record>{local_records.data(), local_records.size()}),
        ttsl::as_writable_bytes(ttsl::Span<Record>{gathered.data(), gathered.size()}));

    std::lock_guard lock(mutex_);
    merge(layouts_, gathered, num_records);
}

const RouterChannelLayout* RouterChannelLayouts::find(FabricNodeId node, chan_id_t eth_chan) const {
    std::lock_guard lock(mutex_);
    const auto it = layouts_.find({node, eth_chan});
    return it == layouts_.end() ? nullptr : &it->second;
}

const RouterChannelLayout& RouterChannelLayouts::get(FabricNodeId node, chan_id_t eth_chan) const {
    const auto* layout = find(node, eth_chan);
    TT_FATAL(
        layout != nullptr,
        "No channel layout for the router at {} channel {}: its device's fabric build did not publish one",
        node,
        eth_chan);
    return *layout;
}

}  // namespace tt::tt_fabric
