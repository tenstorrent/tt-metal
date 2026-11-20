// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_channel_layouts.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>
#include "distributed_context.hpp"

#include <algorithm>
#include <exception>
#include <map>

namespace tt::tt_fabric {

void RouterChannelLayouts::begin_launch() {
    std::lock_guard lock(mutex_);
    layouts_.clear();
    ++launch_generation_;
}

void RouterChannelLayouts::publish(FabricNodeId node, chan_id_t eth_chan, const RouterChannelLayout& layout) {
    std::lock_guard lock(mutex_);
    const bool inserted = layouts_.try_emplace({node, eth_chan}, layout).second;
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
    Map& layouts, const std::vector<Record>& gathered, const std::vector<uint32_t>& num_records, size_t this_rank) {
    const size_t stride = num_records.empty() ? 0 : gathered.size() / num_records.size();
    std::map<Key, size_t> built_by;
    for (const auto& entry : layouts) {
        built_by.emplace(entry.first, this_rank);
    }
    for (size_t rank = 0; rank < num_records.size(); ++rank) {
        if (rank == this_rank) {
            continue;  // already here, with senders
        }
        for (size_t i = 0; i < num_records[rank]; ++i) {
            const Record& record = gathered[(rank * stride) + i];
            const Key key{
                FabricNodeId(MeshId{record.mesh_id}, record.chip_id), static_cast<chan_id_t>(record.eth_chan)};
            const auto [it, first] = built_by.try_emplace(key, rank);
            TT_FATAL(
                first,
                "fabric launch: ranks {} and {} both built router {} channel {}",
                it->second,
                rank,
                key.first,
                record.eth_chan);
            RouterChannelLayout layout;
            for (uint32_t r = 0; r < record.num_receivers; ++r) {
                const auto& receiver = record.receivers[r];
                layout.receivers[receiver.vc].push_back({receiver.base_address, receiver.num_slots});
            }
            layouts.emplace(key, std::move(layout));
        }
    }
}

void RouterChannelLayouts::exchange(
    const tt::tt_metal::distributed::multihost::DistributedContext& context,
    LaunchHeader header,
    const HeaderCheck& check) {
    const auto world_size = static_cast<size_t>(*context.size());
    std::vector<Record> local_records;
    {
        std::lock_guard lock(mutex_);
        header.launch_generation = launch_generation_;
    }
    if (header.status == 0) {
        try {
            std::lock_guard lock(mutex_);
            local_records = pack(layouts_);
        } catch (const std::exception& e) {
            if (world_size == 1) {
                throw;
            }
            // still go to the gather, so the other ranks don't hang
            log_error(tt::LogFabric, "fabric launch: {}", e.what());
            header.status = 1;
        }
    }
    header.num_records = static_cast<uint32_t>(local_records.size());
    if (world_size == 1) {
        check({header});
        return;
    }

    std::vector<LaunchHeader> headers(world_size);
    context.all_gather(
        ttsl::as_writable_bytes(ttsl::Span<LaunchHeader>{&header, 1}),
        ttsl::as_writable_bytes(ttsl::Span<LaunchHeader>{headers.data(), headers.size()}));
    check(headers);

    // ranks publish different numbers of routers: pad every rank's records to the largest count
    std::vector<uint32_t> num_records;
    num_records.reserve(headers.size());
    for (const auto& h : headers) {
        num_records.push_back(h.num_records);
    }
    const size_t stride = *std::max_element(num_records.begin(), num_records.end());
    if (stride == 0) {
        return;
    }
    local_records.resize(stride);
    std::vector<Record> gathered(stride * world_size);
    context.all_gather(
        ttsl::as_writable_bytes(ttsl::Span<Record>{local_records.data(), local_records.size()}),
        ttsl::as_writable_bytes(ttsl::Span<Record>{gathered.data(), gathered.size()}));

    std::lock_guard lock(mutex_);
    merge(layouts_, gathered, num_records, static_cast<size_t>(*context.rank()));
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
