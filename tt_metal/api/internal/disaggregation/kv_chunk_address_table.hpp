// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/assert.hpp>
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

// How a config's chunk map is represented in memory (and on the wire; values must match
// ChunkCompression in kv_chunk_address_table.proto). Import instantiates the corresponding
// map type; an unrecognized wire value is rejected loudly (fail-closed versioning).
enum class ChunkCompression : uint32_t {
    kUnrolled = 0,     // UnrolledGrid
    kStridedRows = 1,  // StridedRowMap
};

class StridedRowRangeView;  // fwd — defined after KvChunkAddressTable

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
// Position indices are in units of tokens and are converted internally
// to chunk indices via (position / chunk_n_tokens).
class KvChunkAddressTable {
public:
    // --- Per-config map representations ---
    // Both store their config's dims and address chunks by CHUNK index
    // (position/chunk_n_tokens conversion and bounds validation stay in the table).

    struct UnrolledGrid {
        std::vector<KvCacheLocation> entries;  // [slot][layer][chunk]
        uint32_t num_layers = 0;
        uint32_t num_position_chunks = 0;

        size_t flat_index(uint32_t layer, uint32_t chunk, uint32_t slot) const;
        const KvCacheLocation& lookup(uint32_t layer, uint32_t chunk, uint32_t slot) const;
        std::span<const KvCacheLocation> lookup_range(
            uint32_t layer, uint32_t start_chunk, uint32_t end_chunk, uint32_t slot) const;
    };

    struct StridedRowMap {
        // One (slot, layer) row: chunk c at bases[c % step] + (c / step) * strides[c % step].
        // step == 0 marks a never-populated row (lookup returns a zeroed location, matching
        // an unset unrolled cell). A populated row is dense: residues 0..step-1 each present.
        struct Row {
            uint32_t step = 0;
            uint32_t size_bytes = 0;
            DeviceGroupIndex device_group_index{0};
            std::vector<uint64_t> bases;    // [step]
            std::vector<int64_t> strides;   // [step]
        };

        std::vector<Row> rows;  // [slot][layer]
        uint32_t num_layers = 0;
        uint32_t num_position_chunks = 0;
        uint32_t num_slots = 0;

        KvCacheLocation lookup(uint32_t layer, uint32_t chunk, uint32_t slot) const;
        StridedRowRangeView lookup_range(uint32_t layer, uint32_t start_chunk, uint32_t end_chunk, uint32_t slot) const;
    };

    // Single config — the whole table uses one configuration (config id 0, name "0").
    explicit KvChunkAddressTable(const KvChunkAddressTableConfig& config);

    // Indexed configs — config i is named "i" (its decimal index), so the string
    // accessors resolve "0".."N-1" to ids 0..N-1. Requires at least one config.
    explicit KvChunkAddressTable(std::span<const KvChunkAddressTableConfig> configs);

    // Named configs — config ids are assigned in the map's key order (std::map
    // iterates sorted), and each config's name is its key. Requires at least one config.
    explicit KvChunkAddressTable(const std::map<std::string, KvChunkAddressTableConfig>& configs);

    // Import/serializer path: like the named-configs constructor, but each config starts in its
    // declared representation. STRIDED_ROWS configs get an empty StridedRowMap (rows sized to
    // num_slots x num_layers, all step=0, filled by install_strided_map()) — importing a
    // runs-only table never materializes the unrolled grid that compression exists to avoid.
    struct NamedConfigInit {
        KvChunkAddressTableConfig config;
        ChunkCompression compression = ChunkCompression::kUnrolled;
    };
    explicit KvChunkAddressTable(const std::map<std::string, NamedConfigInit>& configs);

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
    // Only valid on UNROLLED configs — compressed maps are built at import; to produce one
    // from an unrolled table use install_strided_map() (the serializer does this).
    void set(uint32_t layer, uint32_t position, uint32_t slot, KvCacheLocation location, uint32_t config_id = 0);
    void set(uint32_t layer, uint32_t position, uint32_t slot, KvCacheLocation location, const std::string& config);

    // Replace a config's map with a compressed representation (import path).
    void install_strided_map(uint32_t config_id, StridedRowMap map);

    // Register a mapping from FabricNodeId to its host name.
    void set_fabric_node_host(const tt::tt_fabric::FabricNodeId& node_id, const std::string& host_name);

    // --- Accessors ---

    // Lookup a single entry, by value (uniform across representations).
    // `position` is in tokens (chunk-aligned). `config` by id (default 0) or name.
    // Hot paths: prefer visit_map()/visit_range() (dispatch once, then iterate the
    // representation-native range) over per-entry calls.
    KvCacheLocation lookup(uint32_t layer, uint32_t position, uint32_t slot, uint32_t config_id = 0) const;
    KvCacheLocation lookup(uint32_t layer, uint32_t position, uint32_t slot, const std::string& config) const;

    // Top-level dispatch on a config's map representation. fn is instantiated per
    // alternative (UnrolledGrid / StridedRowMap) and must return the same type
    // (e.g. void) for each — range objects do not escape the dispatch.
    template <typename F>
    decltype(auto) visit_map(uint32_t config_id, F&& fn) const {
        validate_config_id(config_id);
        if (const auto* grid = std::get_if<UnrolledGrid>(&maps_[config_id])) {
            return fn(*grid);
        }
        return fn(std::get<StridedRowMap>(maps_[config_id]));
    }

    // visit_map + position bookkeeping: fn receives the representation-native range
    // over chunks covering positions [start_pos, end_pos) — a std::span (unrolled) or
    // StridedRowRangeView (strided). Same single-return-type constraint.
    template <typename F>
    decltype(auto) visit_range(
        uint32_t layer, uint32_t start_pos, uint32_t end_pos, uint32_t slot, uint32_t config_id, F&& fn) const {
        validate_args(config_id, layer, start_pos, slot);
        const auto& cfg = configs_[config_id];
        TT_FATAL(
            end_pos <= cfg.max_sequence_length,
            "end_pos {} > max_sequence_length {} (config {})",
            end_pos,
            cfg.max_sequence_length,
            config_id);
        const uint32_t start_chunk = to_chunk_index(config_id, start_pos);
        // Empty/reversed interval: clamp to an empty range (the pre-visit_range API returned
        // an empty span here) rather than underflowing the unsigned chunk arithmetic.
        const uint32_t end_chunk =
            start_pos >= end_pos ? start_chunk : to_chunk_index(config_id, end_pos + cfg.chunk_n_tokens - 1);
        return visit_map(config_id, [&](const auto& map) -> decltype(auto) {
            return fn(map.lookup_range(layer, start_chunk, end_chunk, slot));
        });
    }

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
    // The representation of a config's chunk map.
    ChunkCompression compression(uint32_t config_id = 0) const;
    // Total entries summed across all configs (grid-equivalent chunk count).
    size_t total_entries() const;

private:
    void init_configs(
        std::span<const KvChunkAddressTableConfig> configs,
        std::vector<std::string> names,
        std::span<const ChunkCompression> compressions = {});  // empty => all UNROLLED
    uint32_t resolve_config(const std::string& name) const;
    void validate_config_id(uint32_t config_id) const;
    uint32_t to_chunk_index(uint32_t config_id, uint32_t position) const;
    void validate_args(uint32_t config_id, uint32_t layer, uint32_t position, uint32_t slot) const;

    std::vector<KvChunkAddressTableConfig> configs_;
    std::vector<std::string> config_names_;                        // config_id -> name
    std::unordered_map<std::string, uint32_t> name_to_config_id_;  // name -> config_id
    std::vector<uint32_t> num_position_chunks_;                    // per config
    std::vector<std::variant<UnrolledGrid, StridedRowMap>> maps_;  // per config
    std::vector<DeviceGroup> device_groups_;                       // shared across configs
    std::unordered_map<tt::tt_fabric::FabricNodeId, std::string> fabric_node_to_host_;  // shared across configs
};

// Random-access range over chunks [start_chunk, end_chunk) of one (slot, layer) row of a
// StridedRowMap, computing each location on dereference (compressed storage has nothing to
// reference, so iteration yields values). Namespace-scoped (not nested) so consumers can
// name and reuse the type without the map. Default-constructible; a default view is empty.
class StridedRowRangeView {
public:
    class Iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = KvCacheLocation;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;
        KvCacheLocation operator*() const;
        KvCacheLocation operator[](difference_type n) const;
        Iterator& operator++();
        Iterator operator++(int);
        Iterator& operator--();
        Iterator operator--(int);
        Iterator& operator-=(difference_type n);
        Iterator& operator+=(difference_type n);
        Iterator operator+(difference_type n) const;
        Iterator operator-(difference_type n) const;
        friend Iterator operator+(difference_type n, const Iterator& it) { return it + n; }
        difference_type operator-(const Iterator& o) const;
        bool operator==(const Iterator& o) const;
        bool operator!=(const Iterator& o) const;
        bool operator<(const Iterator& o) const;
        bool operator>(const Iterator& o) const;
        bool operator<=(const Iterator& o) const;
        bool operator>=(const Iterator& o) const;

    private:
        friend class StridedRowRangeView;
        Iterator(const KvChunkAddressTable::StridedRowMap* map, uint32_t layer, uint32_t slot, uint32_t first, uint32_t i);
        const KvChunkAddressTable::StridedRowMap* map_ = nullptr;
        uint32_t layer_ = 0;
        uint32_t slot_ = 0;
        uint32_t first_ = 0;
        uint32_t i_ = 0;
    };

    StridedRowRangeView() = default;
    Iterator begin() const;
    Iterator end() const;
    size_t size() const;
    // std::span parity, so generic consumers compile against either range type.
    bool empty() const;
    KvCacheLocation front() const;
    KvCacheLocation back() const;
    KvCacheLocation operator[](size_t i) const;

private:
    friend struct KvChunkAddressTable::StridedRowMap;
    StridedRowRangeView(
        const KvChunkAddressTable::StridedRowMap* map, uint32_t layer, uint32_t slot, uint32_t first, uint32_t last);
    const KvChunkAddressTable::StridedRowMap* map_ = nullptr;
    uint32_t layer_ = 0;
    uint32_t slot_ = 0;
    uint32_t first_ = 0;
    uint32_t last_ = 0;
};

}  // namespace tt::tt_metal::internal::disaggregation
