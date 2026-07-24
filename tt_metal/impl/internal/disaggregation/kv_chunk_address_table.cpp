// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/internal/disaggregation/kv_chunk_address_table.hpp>

#include <algorithm>
#include <cstring>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/internal/disaggregation/noc_addr.hpp"

namespace tt::tt_metal::internal::disaggregation {

namespace {

tt::tt_metal::IDevice* resolve_device(const tt::tt_fabric::FabricNodeId& node_id) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto chip_id = cp.get_physical_chip_id_from_fabric_node_id(node_id);
    auto* dev = tt::tt_metal::detail::GetActiveDevice(chip_id);
    TT_FATAL(dev != nullptr, "GetActiveDevice({}) returned null for {}", chip_id, node_id);
    return dev;
}

}  // namespace

void KvChunkAddressTable::init_configs(
    std::span<const KvChunkAddressTableConfig> configs, std::vector<std::string> names) {
    TT_FATAL(!configs.empty(), "KvChunkAddressTable requires at least one config");
    TT_FATAL(configs.size() == names.size(), "internal: configs/names size mismatch");

    configs_.assign(configs.begin(), configs.end());
    config_names_ = std::move(names);
    num_position_chunks_.resize(configs_.size());
    num_layers_x_chunks_.resize(configs_.size());
    entries_.resize(configs_.size());

    for (uint32_t c = 0; c < configs_.size(); c++) {
        const auto& cfg = configs_[c];
        TT_FATAL(cfg.chunk_n_tokens > 0, "config[{}] chunk_n_tokens must be > 0", c);
        TT_FATAL(!config_names_[c].empty(), "config[{}] name must be non-empty", c);
        auto [it, inserted] = name_to_config_id_.emplace(config_names_[c], c);
        TT_FATAL(inserted, "duplicate config name '{}'", config_names_[c]);

        uint32_t npc = (cfg.max_sequence_length + cfg.chunk_n_tokens - 1) / cfg.chunk_n_tokens;
        num_position_chunks_[c] = npc;
        num_layers_x_chunks_[c] = cfg.num_layers * npc;
        entries_[c].resize(static_cast<size_t>(cfg.num_slots) * cfg.num_layers * npc);
    }
}

KvChunkAddressTable::KvChunkAddressTable(const KvChunkAddressTableConfig& config) {
    init_configs(std::span<const KvChunkAddressTableConfig>(&config, 1), {"0"});
}

KvChunkAddressTable::KvChunkAddressTable(std::span<const KvChunkAddressTableConfig> configs) {
    std::vector<std::string> names;
    names.reserve(configs.size());
    for (uint32_t i = 0; i < configs.size(); i++) {
        names.push_back(std::to_string(i));  // "0".."N-1"
    }
    init_configs(configs, std::move(names));
}

KvChunkAddressTable::KvChunkAddressTable(const std::map<std::string, KvChunkAddressTableConfig>& configs) {
    std::vector<KvChunkAddressTableConfig> cfgs;
    std::vector<std::string> names;
    cfgs.reserve(configs.size());
    names.reserve(configs.size());
    for (const auto& [name, cfg] : configs) {  // std::map iterates in sorted key order
        names.push_back(name);
        cfgs.push_back(cfg);
    }
    init_configs(cfgs, std::move(names));
}

uint32_t KvChunkAddressTable::resolve_config(const std::string& name) const {
    auto it = name_to_config_id_.find(name);
    TT_FATAL(it != name_to_config_id_.end(), "config name '{}' not found", name);
    return it->second;
}

void KvChunkAddressTable::validate_config_id(uint32_t config_id) const {
    TT_FATAL(config_id < configs_.size(), "config_id {} >= num_configs {}", config_id, configs_.size());
}

size_t KvChunkAddressTable::flat_index(uint32_t config_id, uint32_t layer, uint32_t position_chunk, uint32_t slot) const {
    return (static_cast<size_t>(slot) * num_layers_x_chunks_[config_id]) +
           (static_cast<size_t>(layer) * num_position_chunks_[config_id]) + static_cast<size_t>(position_chunk);
}

uint32_t KvChunkAddressTable::to_chunk_index(uint32_t config_id, uint32_t position) const {
    return position / configs_[config_id].chunk_n_tokens;
}

void KvChunkAddressTable::validate_args(uint32_t config_id, uint32_t layer, uint32_t position, uint32_t slot) const {
    validate_config_id(config_id);
    const auto& cfg = configs_[config_id];
    TT_FATAL(layer < cfg.num_layers, "layer {} >= num_layers {} (config {})", layer, cfg.num_layers, config_id);
    TT_FATAL(
        position < cfg.max_sequence_length,
        "position {} >= max_sequence_length {} (config {})",
        position,
        cfg.max_sequence_length,
        config_id);
    TT_FATAL(slot < cfg.num_slots, "slot {} >= num_slots {} (config {})", slot, cfg.num_slots, config_id);
    TT_FATAL(
        position % cfg.chunk_n_tokens == 0,
        "position {} is not a multiple of chunk_n_tokens {} (config {})",
        position,
        cfg.chunk_n_tokens,
        config_id);
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

void KvChunkAddressTable::set(
    uint32_t layer, uint32_t position, uint32_t slot, KvCacheLocation location, uint32_t config_id) {
    validate_args(config_id, layer, position, slot);
    entries_[config_id][flat_index(config_id, layer, to_chunk_index(config_id, position), slot)] = location;
}

void KvChunkAddressTable::set(
    uint32_t layer, uint32_t position, uint32_t slot, KvCacheLocation location, const std::string& config) {
    set(layer, position, slot, location, resolve_config(config));
}

void KvChunkAddressTable::set_fabric_node_host(
    const tt::tt_fabric::FabricNodeId& node_id, const std::string& host_name) {
    fabric_node_to_host_[node_id] = host_name;
}

const KvCacheLocation& KvChunkAddressTable::lookup(
    uint32_t layer, uint32_t position, uint32_t slot, uint32_t config_id) const {
    validate_args(config_id, layer, position, slot);
    return entries_[config_id][flat_index(config_id, layer, to_chunk_index(config_id, position), slot)];
}

const KvCacheLocation& KvChunkAddressTable::lookup(
    uint32_t layer, uint32_t position, uint32_t slot, const std::string& config) const {
    return lookup(layer, position, slot, resolve_config(config));
}

std::span<const KvCacheLocation> KvChunkAddressTable::lookup_range(
    uint32_t layer, uint32_t start_pos, uint32_t end_pos, uint32_t slot, uint32_t config_id) const {
    validate_args(config_id, layer, start_pos, slot);
    const auto& cfg = configs_[config_id];
    TT_FATAL(
        end_pos <= cfg.max_sequence_length,
        "end_pos {} > max_sequence_length {} (config {})",
        end_pos,
        cfg.max_sequence_length,
        config_id);
    if (start_pos >= end_pos) {
        return {};
    }
    uint32_t start_chunk = to_chunk_index(config_id, start_pos);
    uint32_t end_chunk = to_chunk_index(config_id, end_pos + cfg.chunk_n_tokens - 1);
    size_t base = flat_index(config_id, layer, start_chunk, slot);
    return std::span<const KvCacheLocation>(entries_[config_id].data() + base, end_chunk - start_chunk);
}

std::span<const KvCacheLocation> KvChunkAddressTable::lookup_range(
    uint32_t layer, uint32_t start_pos, uint32_t end_pos, uint32_t slot, const std::string& config) const {
    return lookup_range(layer, start_pos, end_pos, slot, resolve_config(config));
}

const std::string& KvChunkAddressTable::get_host(const tt::tt_fabric::FabricNodeId& node_id) const {
    auto it = fabric_node_to_host_.find(node_id);
    TT_FATAL(it != fabric_node_to_host_.end(), "FabricNodeId not found in host map");
    return it->second;
}

bool KvChunkAddressTable::has_host(const tt::tt_fabric::FabricNodeId& node_id) const {
    return fabric_node_to_host_.contains(node_id);
}

std::vector<uint8_t> KvChunkAddressTable::read_device_chunk(
    uint32_t layer, uint32_t position, uint32_t slot, uint32_t config_id) const {
    const auto& loc = lookup(layer, position, slot, config_id);
    const auto& dg = get_device_group(loc.device_group_index);
    TT_FATAL(
        !dg.fabric_node_ids.empty(),
        "DeviceGroup for (layer={}, pos={}, slot={}, config={}) is empty",
        layer,
        position,
        slot,
        config_id);

    std::vector<uint8_t> buf(loc.size_bytes);
    tt::tt_metal::detail::ReadFromDeviceDRAMChannel(
        resolve_device(dg.fabric_node_ids.front()),
        static_cast<int>(addr_channel(loc.noc_addr)),
        addr_local(loc.noc_addr),
        std::span<uint8_t>(buf));
    return buf;
}

std::vector<uint8_t> KvChunkAddressTable::read_device_chunk(
    uint32_t layer, uint32_t position, uint32_t slot, const std::string& config) const {
    return read_device_chunk(layer, position, slot, resolve_config(config));
}

const KvChunkAddressTableConfig& KvChunkAddressTable::config(uint32_t config_id) const {
    validate_config_id(config_id);
    return configs_[config_id];
}

const std::string& KvChunkAddressTable::config_name(uint32_t config_id) const {
    validate_config_id(config_id);
    return config_names_[config_id];
}

uint32_t KvChunkAddressTable::config_id_of(const std::string& name) const { return resolve_config(name); }

uint32_t KvChunkAddressTable::num_position_chunks(uint32_t config_id) const {
    validate_config_id(config_id);
    return num_position_chunks_[config_id];
}

size_t KvChunkAddressTable::total_entries() const {
    size_t total = 0;
    for (const auto& grid : entries_) {
        total += grid.size();
    }
    return total;
}
}  // namespace tt::tt_metal::internal::disaggregation
