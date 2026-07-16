// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
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
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
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

    // Adjacency graph. For flattened groupings, items[node_id] matches each node in the graph.
    // Empty graph if no connection type is specified.
    AdjacencyGraph<uint32_t> adjacency_graph;

    // Logical pinning for MESH groupings committed from a PGD<->MGD topology match in get_valid_groupings_for_mgd:
    // mesh-local chip id (row-major, 0..N-1) -> PGD slot (TrayID + ASICLocation). Populated at match time from
    // the MGD<->PGD pairing and this grouping's item labels. Empty when the grouping did not originate from a PGD
    // match (callers then assume row-major identity).
    std::map<LogicalChipId, tt::tt_metal::ASICPosition> mesh_node_to_asic_position;

    GroupingInfo();
    ~GroupingInfo();
    GroupingInfo(const GroupingInfo&);
    GroupingInfo(GroupingInfo&&) noexcept;
    GroupingInfo& operator=(const GroupingInfo&);
    GroupingInfo& operator=(GroupingInfo&&) noexcept;
};

// One disjoint placement produced by find_all_in_psd: the ASIC footprint it covers, plus the mesh-local
// (row-major) chip id -> ASIC position pinning (copied from the matched grouping's mesh_node_to_asic_position;
// empty when the grouping had no MGD pairing, where callers assume row-major identity). Only the pinning map is
// retained, not the full GroupingInfo, to avoid deep-copying its items + adjacency_graph per placement.
struct PsdPlacement {
    std::unordered_set<tt::tt_metal::AsicID> asics;
    std::map<LogicalChipId, tt::tt_metal::ASICPosition> mesh_node_to_asic_position;
};

// Type aliases for valid groupings map structure
using InstanceType = std::string;  // Type of instance (e.g., "MESH", "FABRIC", "SUPER_FABRIC")
using InstanceName = std::string;  // Name of instance (e.g., "M0", "M1", "G0", "G1")
using ValidGroupingsMap = std::unordered_map<InstanceType, std::unordered_map<InstanceName, std::vector<GroupingInfo>>>;

// PhysicalGroupingDescriptor - Interpreter class for physical grouping descriptor files
// Similar to MeshGraphDescriptor, provides validation and access to grouping definitions
class PhysicalGroupingDescriptor {
public:
    // Parse from textproto string
    explicit PhysicalGroupingDescriptor(const std::string& text_proto);

    // Parse from textproto file path
    explicit PhysicalGroupingDescriptor(const std::filesystem::path& text_proto_file_path);

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

    // Main matching algorithm: Find valid groupings for MGD instances
    // Returns a nested map: instance_type -> instance_name -> vector of valid GroupingInfo matches
    // There can be multiple valid groupings for each MGD instance
    // Requires a PhysicalSystemDescriptor reference for validation/filtering
    // pinnings: optional (AsicPosition, FabricNodeId) constraints applied during PGD<->MGD topology matching
    ValidGroupingsMap get_valid_groupings_for_mgd(
        const MeshGraphDescriptor& mesh_graph_descriptor,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const std::optional<std::vector<std::pair<tt::tt_metal::ASICPosition, FabricNodeId>>>& pinnings =
            std::nullopt) const;

    // Build one GroupingInfo per MGD mesh instance for PSD placement fallback when PGD groupings fail to embed.
    // Includes torus wrap-around edges when the MGD device topology uses RING dimensions.
    // Intended for use by topology_mapper_utils when no PGD grouping successfully embeds into the PSD.
    static std::vector<GroupingInfo> get_mgd_mesh_groupings_for_placement(
        const MeshGraphDescriptor& mesh_graph_descriptor);

    // Find any valid mapping of a grouping to a physical system descriptor
    // Returns unordered_set of ASIC IDs that mark out the grouping in the PSD
    // Returns empty set if no valid mapping exists
    std::unordered_set<tt::tt_metal::AsicID> find_any_in_psd(
        const GroupingInfo& grouping, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const;

    // Find any valid mapping of a grouping to a physical system descriptor
    // Returns unordered_set of ASIC IDs that mark out the grouping in the PSD
    // Returns empty set if no valid mapping exists
    // errors_out will be populated with detailed error messages if mapping fails
    std::unordered_set<tt::tt_metal::AsicID> find_any_in_psd(
        const GroupingInfo& grouping,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        std::vector<std::string>& errors_out) const;

    // Find one maximal disjoint packing of the input `groupings` on the physical system descriptor.
    // Returns one PsdPlacement per placement: its ASIC footprint and the mesh-local (row-major, 0..N-1)
    // chip id -> ASIC position pinning, PsdPlacement::mesh_node_to_asic_position (copied from the matched
    // grouping at PGD<->MGD match commit time; empty for groupings that did not originate from a PGD match,
    // where callers assume row-major identity). No two placements share an ASIC. When multiple PGD grouping
    // variants are provided, the variant with the highest total ASIC coverage is chosen; alternatives are not
    // mixed in the same packing. Returns an empty vector if no valid packing exists.
    std::vector<PsdPlacement> find_all_in_psd(
        const std::vector<GroupingInfo>& groupings,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const;

    // Same as above, but uses a prebuilt flat ASIC adjacency graph from the PSD (from
    // build_flat_adjacency_map_from_psd). Callers that already built the graph can pass it to avoid a
    // duplicate O(|PSD|) scan and graph construction. When non-null, `errors_out` receives detailed
    // messages if no valid packing is found.
    std::vector<PsdPlacement> find_all_in_psd(
        const std::vector<GroupingInfo>& groupings,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const AdjacencyGraph<tt::tt_metal::AsicID>& physical_graph,
        std::vector<std::string>* errors_out = nullptr) const;

    // Build flattened adjacency meshes - one per possibility based on possible groupings that can be formed
    // Returns vector of GroupingInfo objects, each with adjacency_graph populated and node metadata maps filled
    std::vector<GroupingInfo> build_flattened_adjacency_mesh(const GroupingInfo& grouping) const;

    // Overload that accepts a PhysicalSystemDescriptor reference for validation/filtering
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

    // Helper to get ASIC count for a grouping name (from cache)
    uint32_t get_grouping_asic_count(const std::string& grouping_name) const;

    // Private helper that takes PSD pointer (used internally by public overloads)
    ValidGroupingsMap get_valid_groupings_for_mgd(
        const MeshGraphDescriptor& mesh_graph_descriptor,
        const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor,
        const std::optional<std::vector<std::pair<tt::tt_metal::ASICPosition, FabricNodeId>>>& pinnings =
            std::nullopt) const;

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

}  // namespace tt::tt_fabric
