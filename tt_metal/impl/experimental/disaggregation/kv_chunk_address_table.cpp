// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/api/tt-metalium/experimental/disaggregation/kv_chunk_address_table.hpp"

#include <algorithm>
#include <cstdio>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::disaggregation {

size_t KvChunkAddressTable::flat_index(uint32_t layer, uint32_t position_chunk, uint32_t slot) const {
    return (static_cast<size_t>(slot) * num_layers_x_chunks_) + (static_cast<size_t>(layer) * num_position_chunks_) +
           static_cast<size_t>(position_chunk);
}

uint32_t KvChunkAddressTable::to_chunk_index(uint32_t position) const { return position / config_.chunk_n_tokens; }

void KvChunkAddressTable::validate_args(uint32_t layer, uint32_t position, uint32_t slot) const {
    TT_FATAL(layer < config_.num_layers, "layer {} >= num_layers {}", layer, config_.num_layers);
    TT_FATAL(
        position < config_.max_sequence_length,
        "position {} >= max_sequence_length {}",
        position,
        config_.max_sequence_length);
    TT_FATAL(slot < config_.num_slots, "slot {} >= num_slots {}", slot, config_.num_slots);
    TT_FATAL(
        position % config_.chunk_n_tokens == 0,
        "position {} is not a multiple of chunk_n_tokens {}",
        position,
        config_.chunk_n_tokens);
}

KvChunkAddressTable::KvChunkAddressTable(const KvChunkAddressTableConfig& config) : config_(config) {
    TT_FATAL(config.chunk_n_tokens > 0, "chunk_n_tokens must be > 0");
    num_position_chunks_ = (config.max_sequence_length + config.chunk_n_tokens - 1) / config.chunk_n_tokens;
    num_layers_x_chunks_ = config.num_layers * num_position_chunks_;
    entries_.resize(static_cast<size_t>(config.num_slots) * config.num_layers * num_position_chunks_);
}

DeviceGroupIndex KvChunkAddressTable::add_device_group(std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids) {
    std::sort(fabric_node_ids.begin(), fabric_node_ids.end());

    // Check for existing identical group.
    for (uint32_t i = 0; i < device_groups_.size(); i++) {
        if (device_groups_[i].fabric_node_ids == fabric_node_ids) {
            return DeviceGroupIndex{i};
        }
    }

    uint32_t index = static_cast<uint32_t>(device_groups_.size());
    device_groups_.push_back(DeviceGroup{std::move(fabric_node_ids)});
    return DeviceGroupIndex{index};
}

const DeviceGroup& KvChunkAddressTable::get_device_group(DeviceGroupIndex index) const {
    TT_FATAL(
        *index < device_groups_.size(), "device_group_index {} >= num_device_groups {}", *index, device_groups_.size());
    return device_groups_[*index];
}

void KvChunkAddressTable::set(uint32_t layer, uint32_t position, uint32_t slot, KvCacheLocation location) {
    validate_args(layer, position, slot);
    entries_[flat_index(layer, to_chunk_index(position), slot)] = location;
}

void KvChunkAddressTable::set_fabric_node_host(
    const tt::tt_fabric::FabricNodeId& node_id, const std::string& host_name) {
    fabric_node_to_host_[node_id] = host_name;
}

const KvCacheLocation& KvChunkAddressTable::lookup(uint32_t layer, uint32_t position, uint32_t slot) const {
    validate_args(layer, position, slot);
    return entries_[flat_index(layer, to_chunk_index(position), slot)];
}

std::span<const KvCacheLocation> KvChunkAddressTable::lookup_range(
    uint32_t layer, uint32_t start_pos, uint32_t end_pos, uint32_t slot) const {
    validate_args(layer, start_pos, slot);
    TT_FATAL(
        end_pos <= config_.max_sequence_length,
        "end_pos {} > max_sequence_length {}",
        end_pos,
        config_.max_sequence_length);
    if (start_pos >= end_pos) {
        return {};
    }
    uint32_t start_chunk = to_chunk_index(start_pos);
    uint32_t end_chunk = to_chunk_index(end_pos + config_.chunk_n_tokens - 1);
    size_t base = flat_index(layer, start_chunk, slot);
    return std::span<const KvCacheLocation>(entries_.data() + base, end_chunk - start_chunk);
}

const std::string& KvChunkAddressTable::get_host(const tt::tt_fabric::FabricNodeId& node_id) const {
    auto it = fabric_node_to_host_.find(node_id);
    TT_FATAL(it != fabric_node_to_host_.end(), "FabricNodeId not found in host map");
    return it->second;
}

bool KvChunkAddressTable::has_host(const tt::tt_fabric::FabricNodeId& node_id) const {
    return fabric_node_to_host_.contains(node_id);
}

void KvChunkAddressTable::dump(const std::string& tag) const {
    const char* t = tag.empty() ? "" : tag.c_str();
    std::fprintf(
        stderr,
        "[KvChunkAddressTable.dump] tag='%s' config: num_layers=%u max_sequence_length=%u "
        "num_slots=%u chunk_n_tokens=%u chunk_size_bytes=%u num_position_chunks=%u total_entries=%zu\n",
        t,
        config_.num_layers,
        config_.max_sequence_length,
        config_.num_slots,
        config_.chunk_n_tokens,
        config_.chunk_size_bytes,
        num_position_chunks_,
        entries_.size());

    // Device groups
    std::fprintf(stderr, "[KvChunkAddressTable.dump] tag='%s' device_groups: %zu\n", t, device_groups_.size());
    for (size_t g = 0; g < device_groups_.size(); g++) {
        const auto& nodes = device_groups_[g].fabric_node_ids;
        std::fprintf(stderr, "[KvChunkAddressTable.dump] tag='%s'   group[%zu]: %zu nodes:", t, g, nodes.size());
        for (const auto& fnid : nodes) {
            std::fprintf(stderr, " (m=%u,c=%u)", *fnid.mesh_id, fnid.chip_id);
        }
        std::fprintf(stderr, "\n");
    }

    // Fabric-node host mapping
    std::fprintf(
        stderr,
        "[KvChunkAddressTable.dump] tag='%s' fabric_node_to_host: %zu entries\n",
        t,
        fabric_node_to_host_.size());
    for (const auto& [fnid, host] : fabric_node_to_host_) {
        std::fprintf(
            stderr,
            "[KvChunkAddressTable.dump] tag='%s'   (m=%u,c=%u) -> '%s'\n",
            t,
            *fnid.mesh_id,
            fnid.chip_id,
            host.c_str());
    }

    // Every entry. Iterate in storage order: [slot][layer][position_chunk].
    // Skip entries with size_bytes==0 (uninitialized — table.set never called
    // for that triple) so the dump stays compact for sparse tables.
    std::fprintf(
        stderr,
        "[KvChunkAddressTable.dump] tag='%s' entries (slot, layer, position) -> "
        "KvCacheLocation{noc_addr, size_bytes, device_group_index}:\n",
        t);
    size_t printed = 0;
    for (uint32_t slot = 0; slot < config_.num_slots; slot++) {
        for (uint32_t layer = 0; layer < config_.num_layers; layer++) {
            for (uint32_t chunk_idx = 0; chunk_idx < num_position_chunks_; chunk_idx++) {
                const auto& loc = entries_[flat_index(layer, chunk_idx, slot)];
                if (loc.size_bytes == 0) {
                    continue;
                }
                uint32_t position = chunk_idx * config_.chunk_n_tokens;
                std::fprintf(
                    stderr,
                    "[KvChunkAddressTable.dump] tag='%s'   (slot=%u, layer=%u, position=%u) -> "
                    "noc_addr=%llu (channel=%llu, local_addr=%llu) size_bytes=%u dg_idx=%u\n",
                    t,
                    slot,
                    layer,
                    position,
                    static_cast<unsigned long long>(loc.noc_addr),
                    static_cast<unsigned long long>(loc.noc_addr >> 32),
                    static_cast<unsigned long long>(loc.noc_addr & 0xFFFFFFFFull),
                    loc.size_bytes,
                    static_cast<uint32_t>(*loc.device_group_index));
                printed++;
            }
        }
    }
    std::fprintf(
        stderr,
        "[KvChunkAddressTable.dump] tag='%s' printed %zu populated entries (of %zu total slots)\n",
        t,
        printed,
        entries_.size());
    std::fflush(stderr);
}

void KvChunkAddressTable::for_each_populated_entry(const EntryVisitFn& fn) const {
    for (uint32_t slot = 0; slot < config_.num_slots; slot++) {
        for (uint32_t layer = 0; layer < config_.num_layers; layer++) {
            for (uint32_t chunk_idx = 0; chunk_idx < num_position_chunks_; chunk_idx++) {
                const auto& loc = entries_[flat_index(layer, chunk_idx, slot)];
                if (loc.size_bytes == 0) {
                    continue;
                }
                uint32_t position = chunk_idx * config_.chunk_n_tokens;
                const auto& group = device_groups_[*loc.device_group_index];
                fn(slot, layer, position, loc, group);
            }
        }
    }
}

}  // namespace tt::tt_metal::experimental::disaggregation
