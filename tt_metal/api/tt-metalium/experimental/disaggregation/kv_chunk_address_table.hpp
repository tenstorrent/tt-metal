// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental::disaggregation {

// Strongly-typed index into the device group side table.
using DeviceGroupIndex = tt::stl::StrongType<uint32_t, struct DeviceGroupIndexTag>;

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

// Lookup table mapping (layer, position, slot) -> KvCacheLocation.
//
// Describes how a KV cache is allocated/laid out across a multi-host,
// multi-chip, multi-memory system. Used by the migration layer to locate
// KV cache chunks for transfer.
//
// Device replica groups are stored in a separate side table and referenced
// by index from each KvCacheLocation. Groups are deduplicated: identical
// sorted sets of FabricNodeIds share the same index. The side table is
// typically tiny (handful of entries) and stays in L1 cache.
//
// Internal storage is ordered [slot][layer][position_chunk] so that the
// typical access pattern (fixed slot, iterate layers, iterate positions)
// gets contiguous position-chunk reads — the innermost loop.
//
// Position indices are in units of tokens and are converted internally
// to chunk indices via (position / chunk_n_tokens).
class KvChunkAddressTable {
public:
    explicit KvChunkAddressTable(const KvChunkAddressTableConfig& config);

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

    // Set the location for a specific (layer, position, slot).
    // `position` is in tokens and must be chunk-aligned (multiple of chunk_n_tokens).
    void set(uint32_t layer, uint32_t position, uint32_t slot, KvCacheLocation location);

    // Register a mapping from FabricNodeId to its host name.
    void set_fabric_node_host(const tt::tt_fabric::FabricNodeId& node_id, const std::string& host_name);

    // --- Accessors ---

    // Lookup a single entry. `position` is in tokens (chunk-aligned).
    const KvCacheLocation& lookup(uint32_t layer, uint32_t position, uint32_t slot) const;

    // Lookup a contiguous range of position chunks for a given (layer, slot).
    // Returns a span over the internal storage — zero-copy.
    // `start_pos` must be chunk-aligned. `end_pos` need not be aligned —
    // it is rounded up to include the enclosing chunk.
    // Returns entries for chunks covering positions [start_pos, end_pos).
    std::span<const KvCacheLocation> lookup_range(
        uint32_t layer, uint32_t start_pos, uint32_t end_pos, uint32_t slot) const;

    // Resolve a FabricNodeId to its host name.
    const std::string& get_host(const tt::tt_fabric::FabricNodeId& node_id) const;

    // Check if a FabricNodeId has a registered host mapping.
    bool has_host(const tt::tt_fabric::FabricNodeId& node_id) const;

    const KvChunkAddressTableConfig& config() const { return config_; }
    uint32_t num_position_chunks() const { return num_position_chunks_; }
    size_t total_entries() const { return entries_.size(); }

private:
    size_t flat_index(uint32_t layer, uint32_t position_chunk, uint32_t slot) const;
    uint32_t to_chunk_index(uint32_t position) const;
    void validate_args(uint32_t layer, uint32_t position, uint32_t slot) const;

    KvChunkAddressTableConfig config_;
    uint32_t num_position_chunks_;
    uint32_t num_layers_x_chunks_;  // cached: num_layers * num_position_chunks_
    std::vector<KvCacheLocation> entries_;
    std::vector<DeviceGroup> device_groups_;
    std::unordered_map<tt::tt_fabric::FabricNodeId, std::string> fabric_node_to_host_;
};

}  // namespace tt::tt_metal::experimental::disaggregation
