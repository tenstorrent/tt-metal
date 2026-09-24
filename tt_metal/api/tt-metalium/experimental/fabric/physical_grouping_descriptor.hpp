// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <filesystem>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

// Forward declaration
namespace tt::tt_metal {
class PhysicalSystemDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

namespace proto {
// Forward declare to avoid including the full protobuf header
class PhysicalGroupings;
class Grouping;
class GroupingItem;
class GroupingReference;
enum AsicLocation : int;
}  // namespace proto

// Grouping item information
struct GroupingItemInfo {
    enum class ItemType { ASIC_LOCATION, GROUPING_REF };
    enum class CornerOrientation { NW, NE, SW, SE };  // Corner orientation for mesh groupings

    ItemType type;
    tt::tt_metal::ASICLocation asic_location{0};  // Only valid if type == ASIC_LOCATION
    tt::tt_metal::TrayID tray_id{0};              // From optional instance tray_id (asic_location only); 0 = UNSET

    std::string grouping_name;   // Only valid if type == GROUPING_REF
    std::vector<CornerOrientation>
        corners;  // Corner orientations (can have multiple, e.g., 1D endpoints have 2, 1x1 has all 4)
    // Note: Counts are represented by having multiple items. Use items.size() to get the count.
    std::vector<std::string> grouping_path;  // Path through grouping hierarchy using grouping names
                                             // Includes ASIC location at the end (e.g., ["MESH", "hosts_0",
                                             // "tray_1", "ASIC_LOCATION_1"])
};

// Grouping information
struct GroupingInfo {
    std::string name;  // Unique identifier/name for this specific grouping instance
    std::string type;  // Type of grouping (e.g., "MESH", "tray", "meshes", "pods")
    // items[node_id] is the item for graph node node_id. Flattened meshes may use non-contiguous IDs;
    // in that case size is max_node_id+1 (only indices present in adjacency_graph are meaningful).
    std::vector<GroupingItemInfo> items;
    uint32_t asic_count = 0;  // Total ASICs provided by this grouping, calculated bottom-up during population

    // How PGD instances (sub-groupings) tile in row-major order, from row_major_mesh { dims: [...] }.
    // Empty for all_to_all/custom/no connection. Used when joining sub-meshes during flattening.
    std::vector<int32_t> instance_tile_layout_dims;

    // Row-major [rows, cols] of all ASIC nodes after flattening (e.g. [32, 4] for a 128-ASIC mesh).
    // Empty until flattening completes. Used for MGD topology matching and torus variant rebuild.
    std::vector<int32_t> flattened_node_grid_dims;

    // Adjacency graph over GroupingChipId nodes. For flattened groupings, items[node_id] matches
    // each node in the graph. Empty graph if no connection type is specified.
    AdjacencyGraph<GroupingChipId> adjacency_graph;

    // Stable per-descriptor handle of this resolved grouping, assigned in population order.
    PhysicalGroupingId id = 0;

    // Logical pinning for MESH groupings committed from a PGD<->MGD topology match in get_valid_groupings_for_mgd:
    // mesh-local chip id (row-major, 0..N-1) -> PGD slot (TrayID + ASICLocation). Populated at match time from
    // the MGD<->PGD pairing and this grouping's item labels. Empty when the grouping did not originate from a PGD
    // match (callers then assume row-major identity).
    std::map<LogicalChipId, tt::tt_metal::ASICPosition> mesh_node_to_asic_position;

    // PGD node -> host partition index from the matched MGD host_topology at PGD<->MGD commit time. A declared
    // host_topology of [1,1] is one partition, not an absent opinion, so it is populated here like any other and
    // held to the same containment rule. Empty only when the descriptor declared no host topology at all;
    // enumerate then uses a soft same-host preference instead of a required partition.
    std::map<LogicalChipId, uint32_t> mesh_node_to_host_group;

    // PGD node -> index of the descriptor's own declared host holding that chip, filled in while the mesh is
    // flattened. This is the descriptor's host level, not the machine's: it says which of the HOSTS groupings a
    // chip of this variant belongs to, so a declared rank can be held against it before a match is chosen.
    // Empty when the descriptor declares no hosts, or names chips no single declared host holds.
    std::map<LogicalChipId, uint32_t> mesh_node_to_pgd_host_group;

    GroupingInfo();
    ~GroupingInfo();
    GroupingInfo(const GroupingInfo&);
    GroupingInfo(GroupingInfo&&) noexcept;
    GroupingInfo& operator=(const GroupingInfo&);
    GroupingInfo& operator=(GroupingInfo&&) noexcept;
};

// LINE neighbors are always included. When ring_dims[d] is true, also wrap both ends of dimension d.
// Missing ring_dims entries are treated as LINE. RING wrap is skipped when dim < 3.
AdjacencyGraph<GroupingChipId> build_row_major_mesh_graph(
    const std::vector<GroupingChipId>& instance_ids,
    const std::vector<int32_t>& dims,
    const std::string& grouping_name = "",
    uint32_t connections_per_edge = 1,
    const std::vector<bool>& ring_dims = {});

// TORUSX wraps flattened_node_grid_dims[0], TORUSY wraps [1] (same as flatten variants).
// Size-1 and size-2 axes keep ordinary mesh links, so they do not raise the priority.
inline int effective_torus_variant_priority(const GroupingInfo& grouping) {
    const auto& dims = grouping.flattened_node_grid_dims;
    const std::string& type = grouping.type;
    return torus_variant_priority(
        (type == "TORUSX" || type == "TORUSXY") && !dims.empty() && is_genuine_torus_axis(dims[0]),
        (type == "TORUSY" || type == "TORUSXY") && dims.size() > 1 && is_genuine_torus_axis(dims[1]));
}

// One disjoint placement: the ASIC footprint it covers, plus the mesh-local
// (row-major) chip id -> ASIC position pinning (copied from the matched grouping's mesh_node_to_asic_position;
// empty when the grouping had no MGD pairing, where callers assume row-major identity). Only the pinning map is
// retained, not the full GroupingInfo, to avoid deep-copying its items + adjacency_graph per placement.
struct PsdPlacement {
    std::unordered_set<tt::tt_metal::AsicID> asics;
    std::map<LogicalChipId, tt::tt_metal::ASICPosition> mesh_node_to_asic_position;
};

// One seated mesh instance from SAT joint placement.
struct PlacedMesh {
    MeshId mesh_id;
    PsdPlacement placement;
    std::string grouping_name;
    std::string grouping_type;
};
using AssignedMeshes = std::vector<PlacedMesh>;

// Wall-clock and search counters for one SAT joint placement.
//
// Inner topology-solver enumerations (layer 1, per grouping variant) still dominate candidate
// generation; the master fields cover the MeshId→SeatId session.
struct PlacementSolveStats {
    bool success = false;
    std::size_t meshes_total = 0;
    std::size_t meshes_placed = 0;

    std::size_t candidates_generated = 0;  ///< Successful inner mappings turned into candidates

    // Inner topology-solver enumerations invoked while growing candidate pools
    std::size_t inner_solver_calls = 0;
    std::size_t inner_solver_sat_calls = 0;  ///< Auto backend chose SAT (n_target * n_global threshold)
    std::size_t inner_solver_dfs_calls = 0;
    std::size_t inner_solutions_found = 0;
    std::size_t inner_dfs_visits = 0;  ///< MappingResult::stats.dfs_calls summed over DFS-backend calls
    std::size_t inner_dfs_backtracks = 0;
    std::size_t inner_dfs_memoization_hits = 0;

    std::chrono::microseconds total_elapsed{};
    std::chrono::microseconds inner_solver_elapsed{};
    std::chrono::microseconds sat_elapsed{};
    std::chrono::microseconds dfs_elapsed{};

    std::chrono::microseconds slowest_inner_elapsed{};
    std::size_t slowest_inner_n_target = 0;
    std::size_t slowest_inner_n_global = 0;
    bool slowest_inner_used_sat = false;

    // Two-layer joint placement (Plan 4): per-variant candidate enumeration + one SAT master solve.
    // Populated only when that path ran; `candidate_lists_complete == false` means an UNSAT verdict from
    // the master solve is NOT trustworthy (some variant's list was truncated).
    bool master_solve_attempted = false;
    bool master_solve_success = false;
    std::size_t master_candidates_enumerated = 0;  ///< Distinct footprints across all definitions/variants
    std::size_t master_growth_rounds = 0;          ///< Column-generation rounds after the initial batch
    std::size_t master_sat_attempts = 0;           ///< Master SAT encode+solve attempts (tiers x rounds)
    bool candidate_lists_complete = false;
    std::size_t master_sat_vars = 0;     ///< Variables in the last master encoding
    std::size_t master_sat_clauses = 0;  ///< Clauses in the last master encoding
    std::chrono::microseconds master_enumeration_elapsed{};
    std::chrono::microseconds master_encode_elapsed{};
    std::chrono::microseconds master_solve_elapsed{};

    std::string to_string() const;
};

// Type aliases for valid groupings map structure
using InstanceType = std::string;  // Type of instance (e.g., "MESH", "FABRIC", "SUPER_FABRIC")
using InstanceName = std::string;  // Name of instance (e.g., "M0", "M1", "G0", "G1")
using ValidGroupingsMap = std::unordered_map<InstanceType, std::unordered_map<InstanceName, std::vector<GroupingInfo>>>;

// ValidGroupingsMap MESH key for a descriptor's instance name. Several descriptors can reuse the same
// instance name (e.g. "M0"), so each is tagged with its descriptor index; a single descriptor keeps the
// bare name. Every writer and reader of merged valid-groupings keys must use this spelling.
inline InstanceName merged_instance_key(
    std::size_t mgd_index, std::size_t mgd_count, const InstanceName& instance_name) {
    if (mgd_count <= 1) {
        return instance_name;
    }
    return "mgd" + std::to_string(mgd_index) + "_" + instance_name;
}

// PhysicalGroupingDescriptor - Interpreter class for physical grouping descriptor files
// Similar to MeshGraphDescriptor, provides validation and access to grouping definitions
class PhysicalGroupingDescriptor {
public:
    // Parse from textproto string
    explicit PhysicalGroupingDescriptor(const std::string& text_proto);

    // Parse from textproto file path
    explicit PhysicalGroupingDescriptor(const std::filesystem::path& text_proto_file_path);

    // Explicit path, then TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH, then cluster-name and
    // arch-specific files. The default descriptor is used only when none of those exist.
    // Returns nullopt when no descriptor file is present. Throws if an explicit path or env path
    // is set but the file is missing.
    static std::optional<PhysicalGroupingDescriptor> find_and_load(
        const std::optional<std::filesystem::path>& pgd_path = std::nullopt,
        const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor = nullptr);

    ~PhysicalGroupingDescriptor();

    // Check if a grouping exists
    bool has_grouping(const std::string& grouping_name) const;

    // Get all grouping names (including duplicates)
    std::vector<std::string> get_all_grouping_names() const;

    // Get all grouping types (preset_type or custom_type)
    std::vector<std::string> get_all_grouping_types() const;

    // Get total number of groupings (including duplicates)
    size_t get_grouping_count() const;

    // Get all groupings with a specific name (supports multiple definitions)
    // Returns grouping information without exposing proto objects
    std::vector<GroupingInfo> get_groupings_by_name(const std::string& grouping_name) const;

    // Get all groupings with a specific type (preset_type or custom_type)
    // Returns grouping information without exposing proto objects
    std::vector<GroupingInfo> get_groupings_by_type(const std::string& grouping_type) const;

    // Get all groupings
    std::vector<GroupingInfo> get_all_groupings() const;

    // PGD<->MGD matching for one descriptor: valid PGD groupings per MGD instance (PGD commits only).
    // Returns instance_type -> instance_name -> vector of GroupingInfo. pinnings use local mesh ids (same
    // shape as MeshGraphDescriptor::get_pinnings()). require_placement: when true (default), a mesh with
    // no placeable PGD match is fatal for footprint discovery; rank-bound pinning enrichment passes false.
    ValidGroupingsMap get_valid_groupings_for_mgd(
        const MeshGraphDescriptor& mesh_graph_descriptor,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings = std::nullopt,
        bool require_placement = true,
        const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank = {}) const;

    // MGD-native mesh groupings that embed on the PSD (torus wraps when the MGD uses RING dims).
    // Does not read PGD groupings; SAT uses these as seats when constructed without a PGD.
    static ValidGroupingsMap get_mgd_placement_fallbacks_for_mgd(
        const MeshGraphDescriptor& mesh_graph_descriptor,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings = std::nullopt);

    // Same as get_valid_groupings_for_mgd for every MGD, merged into one map. Keys are prefixed "mgd{i}_"
    // when there is more than one descriptor. per_mgd_pinnings[i] is forwarded to MGD i (local mesh ids).
    ValidGroupingsMap get_valid_groupings_for_mgds(
        const std::vector<MeshGraphDescriptor>& mesh_graph_descriptors,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const std::vector<std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>>& per_mgd_pinnings = {},
        bool require_placement = true) const;

    // Pair of get_valid_groupings_for_mgds; same key prefixing and deferred SAT pool insertion as above.
    ValidGroupingsMap get_mgd_placement_fallbacks_for_mgds(
        const std::vector<MeshGraphDescriptor>& mesh_graph_descriptors,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const std::vector<std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>>& per_mgd_pinnings = {})
        const;

    // Enumerate distinct embeddings of an already-flat grouping on the PSD (same helper SAT column
    // generation and the matcher PSD gate use). Flatten a hierarchical PGD grouping with
    // build_flattened_adjacency_mesh first. Returns up to `max_solutions` mappings; empty if none fit.
    std::vector<MappingResult<LogicalChipId, tt::tt_metal::AsicID>> enumerate_distinct_placements_for_grouping(
        const GroupingInfo& grouping,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        std::size_t max_solutions = 1) const;

    // Build flattened adjacency meshes - one per possibility based on possible groupings that can be formed
    // Returns vector of GroupingInfo objects, each with adjacency_graph populated and node metadata maps filled
    std::vector<GroupingInfo> build_flattened_adjacency_mesh(const GroupingInfo& grouping) const;

    // Same, with PSD validation/filtering when flattening.
    std::vector<GroupingInfo> build_flattened_adjacency_mesh(
        const GroupingInfo& grouping, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const;

    // Greedy minimum coverage over disjoint global groups (e.g. one set per host). Returns whether some single group
    // has enough capacity for all targets, and the union of the largest groups until target count is covered.
    template <typename TargetNode, typename GlobalNode>
    static std::pair<bool, std::set<GlobalNode>> find_minimum_coverage_group(
        const std::set<TargetNode>& all_targets, const std::vector<std::set<GlobalNode>>& global_groups) {
        std::pair<bool, std::set<GlobalNode>> out{false, {}};
        if (all_targets.empty() || global_groups.empty()) {
            return out;
        }
        const std::size_t target_count = all_targets.size();
        for (const auto& g : global_groups) {
            if (g.size() >= target_count) {
                out.first = true;
                break;
            }
        }
        std::vector<std::size_t> group_indices(global_groups.size());
        std::iota(group_indices.begin(), group_indices.end(), 0);
        std::sort(group_indices.begin(), group_indices.end(), [&](std::size_t a, std::size_t b) {
            return global_groups[a].size() > global_groups[b].size();
        });
        std::size_t covered = 0;
        for (std::size_t idx : group_indices) {
            const auto& g = global_groups[idx];
            out.second.insert(g.begin(), g.end());
            covered += g.size();
            if (covered >= target_count) {
                break;
            }
        }
        return out;
    }

private:
    // Data members
    std::shared_ptr<const proto::PhysicalGroupings> proto_;

    // Cache of resolved groupings with ASIC counts (populated bottom-up)
    // Two-tier structure: name -> type -> vector of GroupingInfo
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<GroupingInfo>>>
        resolved_groupings_cache_;

    // Internal helper to convert proto grouping to GroupingInfo
    GroupingInfo convert_grouping_to_info(const proto::Grouping& grouping) const;

    // Fills mesh_node_to_pgd_host_group for one flattened mesh variant from the descriptor's flattened
    // HOSTS groupings, by the chip slots each of them names.
    void assign_pgd_host_groups(
        GroupingInfo& flattened_mesh, const std::vector<GroupingInfo>& flattened_declared_hosts) const;

    // Helper to get ASIC count for a grouping name (from cache)
    uint32_t get_grouping_asic_count(const std::string& grouping_name) const;

    // Private helper that takes PSD pointer (used internally by public overloads)
    ValidGroupingsMap get_valid_groupings_for_mgd(
        const MeshGraphDescriptor& mesh_graph_descriptor,
        const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor,
        const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings = std::nullopt,
        bool require_placement = true,
        const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank = {}) const;

    // Private helper that takes PSD pointer (used internally by public overloads)
    std::vector<GroupingInfo> build_flattened_adjacency_mesh(
        const GroupingInfo& grouping, const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor) const;

    // Fast feasibility check for (tray_id, asic_location) slot counts vs. the PSD. Used by
    // build_flattened_adjacency_mesh to prune impossible flattened meshes before graph isomorphism.
    static bool can_map_to_psd(
        const GroupingInfo& grouping_info, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor);

    // Helper for reading files
    static std::string read_file_to_string(const std::filesystem::path& file_path);

    // Helper to get validation report from error vector
    static std::string get_validation_report(const std::vector<std::string>& errors);

    // Population method (called after validation passes)
    void populate();

    // Grouping validation (called after populate)
    void grouping_validate() const;

    // Instance validation - collects grouping validation errors without throwing (for reporting)
    void instance_validate(std::vector<std::string>& errors) const;

    // Internal validation helpers (used by grouping_validate)
    void validate_leaf_groupings(std::vector<std::string>& errors) const;
    void validate_asic_location_usage(std::vector<std::string>& errors) const;
    void validate_no_cycles(std::vector<std::string>& errors) const;
    void validate_instance_counts(std::vector<std::string>& errors) const;

    // Helper methods for populate()
    static uint32_t calculate_base_grouping_asic_count(const GroupingInfo& grouping);
    static uint32_t calculate_dependent_grouping_asic_count(
        const GroupingInfo& grouping,
        const std::unordered_map<std::string, std::vector<GroupingInfo>>& groupings_by_name);

    // Helper functions to access grouping name and type from proto
    static std::string get_grouping_name(const proto::Grouping& grouping);
    static std::string get_grouping_type_string(const proto::Grouping& grouping);

    // Helper function to assign corner orientations to grouping items based on mesh dimensions
    static void assign_corner_orientations_to_grouping(GroupingInfo& info, const std::vector<int32_t>& dims);

    // Helper function to convert MGD instances to GroupingInfo map (includes adjacency graphs and ASIC counts)
    // Calculates required ASIC counts bottom-up and builds adjacency graphs
    // Returns map: (type, name) -> GroupingInfo
    static std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>>
    build_mgd_to_grouping_info_map(const MeshGraphDescriptor& mesh_graph_descriptor);

    // Static validation - returns vector of error strings (similar to MeshGraphDescriptor)
    static std::vector<std::string> static_validate(const proto::PhysicalGroupings& proto);

    // Internal validation helpers (used by static_validate)
    static void validate_required_groupings(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_grouping_references(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_counts(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_grouping_structure(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
};

// Incremental SAT joint placement used by MultiMeshSolutionEnumerator.
// Physical identity of a seat is the ASIC footprint on PlacedMesh.
// The master SAT session stays live across next() calls and is advanced with next().
// It is rebuilt only when candidate pools grow, or when validation mode, the host-cap, or constraints change.
// Column growth stays inside next(); extra constraints applied on the subsequent next().
class SatPlacementEnumerationSession {
public:
    class Candidate;
    class CandidatePool;

    SatPlacementEnumerationSession(
        const PhysicalGroupingDescriptor& physical_grouping_descriptor,
        const MeshGraphDescriptor& mesh_graph_descriptor,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        PlacementSolveStats* stats,
        const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings = std::nullopt,
        const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank = {},
        bool unique_shapes = false,
        const std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>>& fabric_node_id_to_mesh_rank = {});

    // No PGD: seat from MGD placement fallbacks.
    SatPlacementEnumerationSession(
        const MeshGraphDescriptor& mesh_graph_descriptor,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        PlacementSolveStats* stats,
        const std::optional<tt::tt_metal::experimental::tt_fabric::PinningsByMesh>& pinnings = std::nullopt,
        const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank = {},
        bool unique_shapes = false);

    SatPlacementEnumerationSession(const SatPlacementEnumerationSession&) = delete;
    SatPlacementEnumerationSession& operator=(const SatPlacementEnumerationSession&) = delete;
    SatPlacementEnumerationSession(SatPlacementEnumerationSession&&) = delete;
    SatPlacementEnumerationSession& operator=(SatPlacementEnumerationSession&&) = delete;
    ~SatPlacementEnumerationSession();

    AssignedMeshes next();
    std::vector<AssignedMeshes> all();

    bool add_forbidden_constraint(MeshId mesh_id, const std::unordered_set<tt::tt_metal::AsicID>& asics);
    bool add_forbidden_constraint(const PlacedMesh& placed);
    bool add_required_constraint(MeshId mesh_id, const std::unordered_set<tt::tt_metal::AsicID>& asics);
    bool add_required_constraint(const PlacedMesh& placed);
    bool exclude_mapping(const AssignedMeshes& assigned);

private:
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor_ = nullptr;
    PlacementSolveStats* stats_ = nullptr;
    AdjacencyGraph<MeshId> mesh_level_graph_;
    AdjacencyGraph<tt::tt_metal::AsicID> physical_graph_;
    std::map<MeshId, std::vector<GroupingInfo>> global_mesh_groupings_;
    std::map<MeshId, GroupingInfo> mgd_fallback_by_mesh_;
    std::map<MeshId, ConnectionValidationMode> sat_intra_mesh_mode_by_mesh_;
    bool relaxed_inter_mesh_policy_ = false;
    bool unique_shapes_ = false;

    std::unique_ptr<std::map<MeshId, CandidatePool>> pools_;
    MappingConstraints<MeshId, const Candidate*> constraints_;
    std::size_t attempts_ = 0;
    std::size_t cycle_ = 0;
    bool fallbacks_in_ = false;
    bool ready_ = false;
    bool solved_ = false;
    std::vector<AssignedMeshes> pending_;
    std::size_t pending_index_ = 0;
    std::vector<std::pair<MeshId, std::unordered_set<tt::tt_metal::AsicID>>> extra_forbidden_;
    std::vector<std::pair<MeshId, std::unordered_set<tt::tt_metal::AsicID>>> extra_required_;
    std::vector<std::map<MeshId, std::unordered_set<tt::tt_metal::AsicID>>> yielded_footprints_;

    // Live SAT session. next() is another model from it when the mode and host cap still match.
    // The placement session resets it when the pools grow; next() rebuilds only then, or when the cap changes.
    struct MasterSolve {
        explicit MasterSolve(SatPlacementEnumerationSession* owner) : owner_(owner) {}

        void reset() { session.reset(); }

        // True when this session was encoded for this validation mode and host-cap setting.
        bool matches(bool relaxed_mode, bool drop_cap) const {
            return session != nullptr && relaxed == relaxed_mode && drop_host_cap == drop_cap;
        }

        // Rebuild the session for this mode and host cap. False when the seat constraints cannot be built.
        bool restart(bool relaxed_mode, bool drop_cap);

        // Another model from the live session. Restarts only when there is no session or the mode or host cap differs.
        MappingResult<MeshId, const Candidate*> next(bool drop_cap);

        void block_assigned(const AssignedMeshes& assigned);

        bool relaxed = false;
        bool drop_host_cap = false;

    private:
        SatPlacementEnumerationSession* owner_ = nullptr;
        std::unique_ptr<TopologyMappingEnumerationSession<MeshId, const Candidate*>> session;
    };
    std::unique_ptr<MasterSolve> master_solve_;

    void finish_init(const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank);
    void invalidate_pending_solve();
    std::set<const Candidate*> seats_matching(
        MeshId mesh_id, const std::unordered_set<tt::tt_metal::AsicID>& asics) const;
    bool apply_extra_constraints(MappingConstraints<MeshId, const Candidate*>& constraints) const;
    std::vector<std::map<MeshId, const Candidate*>> excluded_seat_maps() const;
    void remember_yielded(const AssignedMeshes& assigned);
};

}  // namespace tt::tt_fabric
