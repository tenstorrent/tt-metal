// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::internal::disaggregation {

// Strongly-typed index into the device group side table.
using DeviceGroupIndex = ttsl::StrongType<uint32_t, struct DeviceGroupIndexTag>;

// A unique group of fabric nodes that hold replicas of a KV cache chunk.
// FabricNodeIds are stored sorted so that identical replica sets
// always produce the same DeviceGroup, enabling deduplication.
struct DeviceGroup {
    std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids;

    bool operator==(const DeviceGroup& other) const { return fabric_node_ids == other.fabric_node_ids; }
};

// Describes the physical location of a single KV cache chunk in device memory.
// 16 bytes — 4 entries per cache line.
struct KvCacheLocation {
    uint64_t noc_addr = 0;
    uint32_t size_bytes = 0;
    DeviceGroupIndex device_group_index{0};
};
static_assert(sizeof(KvCacheLocation) == 16, "KvCacheLocation must be 16 bytes for cache-line packing");

// Configuration for constructing a KvChunkAddressTable.
struct KvChunkAddressTableConfig {
    uint32_t num_layers = 0;
    uint32_t max_sequence_length = 0;  // in tokens
    uint32_t num_slots = 0;
    uint32_t chunk_n_tokens = 32;       // tokens per chunk (KV atomic block granularity)
    uint32_t chunk_size_bytes = 19584;  // physical size of one chunk in bytes (18 x 1088 bfp8 tiles)
};

// Lookup table mapping (layer, position, slot, config) -> KvCacheLocation.
//
// Describes how a KV cache is allocated/laid out across a multi-host,
// multi-chip, multi-memory system. Used by the migration layer to locate
// KV cache chunks for transfer.
//
// The table holds one or more configs ("groups"), each a distinct attention
// implementation / tensor representation a layer may use (e.g. dense KV in
// config 0, a sparse index_k representation in config 1). Each config has its
// own grid and may differ in num_layers / max_sequence_length / num_slots /
// chunk_n_tokens. Configs are addressed last on every accessor — by index
// (config_id, defaulting to 0) or by name (config). A single-config table
// names its lone config "0".
//
// Device replica groups are stored in a separate side table, shared across all
// configs, and referenced by index from each KvCacheLocation. Groups are
// deduplicated: identical sorted sets of FabricNodeIds share the same index.
// The side table is typically tiny (handful of entries) and stays in L1 cache.
//
// Per-config storage is ordered [slot][layer][position_chunk] so that the
// typical access pattern (fixed slot, iterate layers, iterate positions)
// gets contiguous position-chunk reads — the innermost loop.
//
// Position indices are in units of tokens and are converted internally
// to chunk indices via (position / chunk_n_tokens).
class KvChunkAddressTable {
public:
    // Single config — the whole table uses one configuration (config id 0, name "0").
    explicit KvChunkAddressTable(const KvChunkAddressTableConfig& config);

    // Indexed configs — config i is named "i" (its decimal index), so the string
    // accessors resolve "0".."N-1" to ids 0..N-1. Requires at least one config.
    explicit KvChunkAddressTable(std::span<const KvChunkAddressTableConfig> configs);

    // Named configs — config ids are assigned in the map's key order (std::map
    // iterates sorted), and each config's name is its key. Requires at least one config.
    explicit KvChunkAddressTable(const std::map<std::string, KvChunkAddressTableConfig>& configs);

    // --- Device Group Management ---

    // Register a device group (set of replica FabricNodeIds).
    // The FabricNodeIds are sorted internally for dedup.
    // Returns the index for this group. If an identical sorted group
    // already exists, returns the existing index.
    DeviceGroupIndex add_device_group(std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids);

    // Lookup a device group by index.
    const DeviceGroup& get_device_group(DeviceGroupIndex index) const;

    // Number of unique device groups registered.
    size_t num_device_groups() const { return device_groups_.size(); }

    // --- Mutators ---

    // Set the location for a specific (layer, position, slot, config).
    // `position` is in tokens and must be chunk-aligned (multiple of the config's chunk_n_tokens).
    // `config` is addressed by id (default 0) or by name.
    void set(uint32_t layer, uint32_t position, uint32_t slot, KvCacheLocation location, uint32_t config_id = 0);
    void set(uint32_t layer, uint32_t position, uint32_t slot, KvCacheLocation location, const std::string& config);

    // Register a mapping from FabricNodeId to its host name.
    void set_fabric_node_host(const tt::tt_fabric::FabricNodeId& node_id, const std::string& host_name);

    // --- Accessors ---

    // Lookup a single entry. `position` is in tokens (chunk-aligned). `config` by id (default 0) or name.
    const KvCacheLocation& lookup(uint32_t layer, uint32_t position, uint32_t slot, uint32_t config_id = 0) const;
    const KvCacheLocation& lookup(uint32_t layer, uint32_t position, uint32_t slot, const std::string& config) const;

    // Lookup a contiguous range of position chunks for a given (layer, slot, config).
    // Returns a span over the internal storage — zero-copy.
    // `start_pos` must be chunk-aligned. `end_pos` need not be aligned —
    // it is rounded up to include the enclosing chunk.
    // Returns entries for chunks covering positions [start_pos, end_pos).
    std::span<const KvCacheLocation> lookup_range(
        uint32_t layer, uint32_t start_pos, uint32_t end_pos, uint32_t slot, uint32_t config_id = 0) const;
    std::span<const KvCacheLocation> lookup_range(
        uint32_t layer, uint32_t start_pos, uint32_t end_pos, uint32_t slot, const std::string& config) const;

    // Resolve a FabricNodeId to its host name.
    const std::string& get_host(const tt::tt_fabric::FabricNodeId& node_id) const;

    // Check if a FabricNodeId has a registered host mapping.
    bool has_host(const tt::tt_fabric::FabricNodeId& node_id) const;

    // --- Device reads ---

    // Read a single chunk's raw bytes from the primary replica device
    // (first FabricNodeId in the chunk's DeviceGroup). Returns a buffer
    // of size loc.size_bytes. Resolves the device internally via the
    // global ControlPlane — no device list required from the caller.
    std::vector<uint8_t> read_device_chunk(uint32_t layer, uint32_t position, uint32_t slot, uint32_t config_id = 0) const;
    std::vector<uint8_t> read_device_chunk(
        uint32_t layer, uint32_t position, uint32_t slot, const std::string& config) const;

    // --- Config introspection ---

    // Number of configs ("groups") held by this table.
    size_t num_configs() const { return configs_.size(); }
    // Config by id (default 0, the lone config of a single-config table).
    const KvChunkAddressTableConfig& config(uint32_t config_id = 0) const;
    // Name of a config by id.
    const std::string& config_name(uint32_t config_id) const;
    // Resolve a config name to its id (throws if unknown).
    uint32_t config_id_of(const std::string& name) const;
    // Number of position chunks for a config.
    uint32_t num_position_chunks(uint32_t config_id = 0) const;
    // Total entries summed across all configs.
    size_t total_entries() const;

private:
    void init_configs(std::span<const KvChunkAddressTableConfig> configs, std::vector<std::string> names);
    uint32_t resolve_config(const std::string& name) const;
    void validate_config_id(uint32_t config_id) const;
    size_t flat_index(uint32_t config_id, uint32_t layer, uint32_t position_chunk, uint32_t slot) const;
    uint32_t to_chunk_index(uint32_t config_id, uint32_t position) const;
    void validate_args(uint32_t config_id, uint32_t layer, uint32_t position, uint32_t slot) const;

    std::vector<KvChunkAddressTableConfig> configs_;
    std::vector<std::string> config_names_;                        // config_id -> name
    std::unordered_map<std::string, uint32_t> name_to_config_id_;  // name -> config_id
    std::vector<uint32_t> num_position_chunks_;                    // per config
    std::vector<uint32_t> num_layers_x_chunks_;                    // per config: num_layers * num_position_chunks
    std::vector<std::vector<KvCacheLocation>> entries_;            // per-config grid [slot][layer][position_chunk]
    std::vector<DeviceGroup> device_groups_;                       // shared across configs
    std::unordered_map<tt::tt_fabric::FabricNodeId, std::string> fabric_node_to_host_;  // shared across configs
};

}  // namespace tt::tt_metal::internal::disaggregation
