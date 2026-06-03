// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/api/tt-metalium/experimental/disaggregation/kv_chunk_address_table.hpp"

#include <algorithm>
#include <cstring>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental::disaggregation {

namespace {

// noc_addr encoding (set by kv_cache_utils.py): (bank_id << 32) | local_addr
uint32_t addr_channel(uint64_t noc_addr) { return static_cast<uint32_t>(noc_addr >> 32); }
uint32_t addr_local(uint64_t noc_addr) { return static_cast<uint32_t>(noc_addr & 0xFFFFFFFFull); }

tt::tt_metal::IDevice* resolve_device(const tt::tt_fabric::FabricNodeId& node_id) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto chip_id = cp.get_physical_chip_id_from_fabric_node_id(node_id);
    auto* dev = tt::tt_metal::detail::GetActiveDevice(chip_id);
    TT_FATAL(dev != nullptr, "GetActiveDevice({}) returned null for {}", chip_id, node_id);
    return dev;
}

}  // namespace

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

std::vector<uint8_t> KvChunkAddressTable::read_device_chunk(uint32_t layer, uint32_t position, uint32_t slot) const {
    const auto& loc = lookup(layer, position, slot);
    const auto& dg = get_device_group(loc.device_group_index);
    TT_FATAL(
        !dg.fabric_node_ids.empty(), "DeviceGroup for (layer={}, pos={}, slot={}) is empty", layer, position, slot);

    std::vector<uint8_t> buf(loc.size_bytes);
    tt::tt_metal::detail::ReadFromDeviceDRAMChannel(
        resolve_device(dg.fabric_node_ids.front()),
        static_cast<int>(addr_channel(loc.noc_addr)),
        addr_local(loc.noc_addr),
        std::span<uint8_t>(buf));
    return buf;
}
}  // namespace tt::tt_metal::experimental::disaggregation
