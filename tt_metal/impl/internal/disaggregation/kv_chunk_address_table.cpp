// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/kv_chunk_address_table.hpp>

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
    maps_.resize(configs_.size());

    for (uint32_t c = 0; c < configs_.size(); c++) {
        const auto& cfg = configs_[c];
        TT_FATAL(cfg.chunk_n_tokens > 0, "config[{}] chunk_n_tokens must be > 0", c);
        TT_FATAL(!config_names_[c].empty(), "config[{}] name must be non-empty", c);
        auto [it, inserted] = name_to_config_id_.emplace(config_names_[c], c);
        TT_FATAL(inserted, "duplicate config name '{}'", config_names_[c]);

        const uint32_t npc = (cfg.max_sequence_length + cfg.chunk_n_tokens - 1) / cfg.chunk_n_tokens;
        num_position_chunks_[c] = npc;
        // All configs start UNROLLED; compressed maps are installed at import
        // (install_strided_map) once their runs arrive.
        UnrolledGrid grid;
        grid.num_layers = cfg.num_layers;
        grid.num_position_chunks = npc;
        grid.entries.resize(static_cast<size_t>(cfg.num_slots) * cfg.num_layers * npc);
        maps_[c] = std::move(grid);
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

// --- UnrolledGrid ---

size_t KvChunkAddressTable::UnrolledGrid::flat_index(uint32_t layer, uint32_t chunk, uint32_t slot) const {
    return (static_cast<size_t>(slot) * num_layers * num_position_chunks) +
           (static_cast<size_t>(layer) * num_position_chunks) + chunk;
}

const KvCacheLocation& KvChunkAddressTable::UnrolledGrid::lookup(uint32_t layer, uint32_t chunk, uint32_t slot)
    const {
    return entries[flat_index(layer, chunk, slot)];
}

std::span<const KvCacheLocation> KvChunkAddressTable::UnrolledGrid::lookup_range(
    uint32_t layer, uint32_t start_chunk, uint32_t end_chunk, uint32_t slot) const {
    const size_t base = flat_index(layer, start_chunk, slot);
    return std::span<const KvCacheLocation>(entries.data() + base, end_chunk - start_chunk);
}

// --- StridedRowMap ---

KvCacheLocation KvChunkAddressTable::StridedRowMap::lookup(uint32_t layer, uint32_t chunk, uint32_t slot) const {
    const Row& row = rows[static_cast<size_t>(slot) * num_layers + layer];
    if (row.step == 0) {
        return KvCacheLocation{};
    }
    return KvCacheLocation{
        .noc_addr =
            row.bases[chunk % row.step] + static_cast<uint64_t>(row.strides[chunk % row.step]) * (chunk / row.step),
        .size_bytes = row.size_bytes,
        .device_group_index = row.device_group_index,
    };
}

StridedRowRangeView KvChunkAddressTable::StridedRowMap::lookup_range(
    uint32_t layer, uint32_t start_chunk, uint32_t end_chunk, uint32_t slot) const {
    return StridedRowRangeView(this, layer, slot, start_chunk, end_chunk);
}

// --- StridedRowRangeView ---

StridedRowRangeView::StridedRowRangeView(
    const KvChunkAddressTable::StridedRowMap* map, uint32_t layer, uint32_t slot, uint32_t first, uint32_t last) :
    map_(map), layer_(layer), slot_(slot), first_(first), last_(last) {}

StridedRowRangeView::Iterator StridedRowRangeView::begin() const {
    return Iterator(map_, layer_, slot_, first_, 0);
}
StridedRowRangeView::Iterator StridedRowRangeView::end() const {
    return Iterator(map_, layer_, slot_, first_, last_ - first_);
}
size_t StridedRowRangeView::size() const { return last_ - first_; }

StridedRowRangeView::Iterator::Iterator(
    const KvChunkAddressTable::StridedRowMap* map, uint32_t layer, uint32_t slot, uint32_t first, uint32_t i) :
    map_(map), layer_(layer), slot_(slot), first_(first), i_(i) {}

KvCacheLocation StridedRowRangeView::Iterator::operator*() const {
    return map_->lookup(layer_, first_ + i_, slot_);
}
StridedRowRangeView::Iterator& StridedRowRangeView::Iterator::operator++() { return ++i_, *this; }
StridedRowRangeView::Iterator StridedRowRangeView::Iterator::operator++(int) {
    auto t = *this;
    ++i_;
    return t;
}
StridedRowRangeView::Iterator& StridedRowRangeView::Iterator::operator--() { return --i_, *this; }
StridedRowRangeView::Iterator& StridedRowRangeView::Iterator::operator+=(difference_type n) { return i_ += n, *this; }
StridedRowRangeView::Iterator& StridedRowRangeView::Iterator::operator-=(difference_type n) { return i_ -= n, *this; }
StridedRowRangeView::Iterator StridedRowRangeView::Iterator::operator+(difference_type n) const {
    return Iterator(map_, layer_, slot_, first_, i_ + n);
}
StridedRowRangeView::Iterator StridedRowRangeView::Iterator::operator-(difference_type n) const {
    return Iterator(map_, layer_, slot_, first_, i_ - n);
}
StridedRowRangeView::Iterator::difference_type StridedRowRangeView::Iterator::operator-(const Iterator& o) const {
    return i_ - o.i_;
}
bool StridedRowRangeView::Iterator::operator==(const Iterator& o) const { return i_ == o.i_; }
bool StridedRowRangeView::Iterator::operator!=(const Iterator& o) const { return i_ != o.i_; }
bool StridedRowRangeView::Iterator::operator<(const Iterator& o) const { return i_ < o.i_; }

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
    auto* grid = std::get_if<UnrolledGrid>(&maps_[config_id]);
    TT_FATAL(
        grid != nullptr,
        "set() on a compressed (STRIDED_ROWS) config {} — compressed maps are built at import; "
        "build unrolled and serialize, or install_strided_map()",
        config_id);
    grid->entries[grid->flat_index(layer, to_chunk_index(config_id, position), slot)] = location;
}

void KvChunkAddressTable::set(
    uint32_t layer, uint32_t position, uint32_t slot, KvCacheLocation location, const std::string& config) {
    set(layer, position, slot, location, resolve_config(config));
}

void KvChunkAddressTable::install_strided_map(uint32_t config_id, StridedRowMap map) {
    validate_config_id(config_id);
    TT_FATAL(
        map.num_slots == configs_[config_id].num_slots && map.num_layers == configs_[config_id].num_layers &&
            map.num_position_chunks == num_position_chunks_[config_id],
        "strided map dims do not match config {}",
        config_id);
    maps_[config_id] = std::move(map);
}

void KvChunkAddressTable::set_fabric_node_host(
    const tt::tt_fabric::FabricNodeId& node_id, const std::string& host_name) {
    fabric_node_to_host_[node_id] = host_name;
}

KvCacheLocation KvChunkAddressTable::lookup(
    uint32_t layer, uint32_t position, uint32_t slot, uint32_t config_id) const {
    validate_args(config_id, layer, position, slot);
    const uint32_t chunk = to_chunk_index(config_id, position);
    return visit_map(config_id, [&](const auto& map) { return map.lookup(layer, chunk, slot); });
}

KvCacheLocation KvChunkAddressTable::lookup(
    uint32_t layer, uint32_t position, uint32_t slot, const std::string& config) const {
    return lookup(layer, position, slot, resolve_config(config));
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
    const auto loc = lookup(layer, position, slot, config_id);
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

ChunkCompression KvChunkAddressTable::compression(uint32_t config_id) const {
    validate_config_id(config_id);
    return std::holds_alternative<UnrolledGrid>(maps_[config_id]) ? ChunkCompression::kUnrolled
                                                                  : ChunkCompression::kStridedRows;
}

size_t KvChunkAddressTable::total_entries() const {
    size_t total = 0;
    for (uint32_t c = 0; c < configs_.size(); c++) {
        // Grid-equivalent chunk count, independent of representation.
        total += static_cast<size_t>(configs_[c].num_slots) * configs_[c].num_layers * num_position_chunks_[c];
    }
    return total;
}
}  // namespace tt::tt_metal::internal::disaggregation
