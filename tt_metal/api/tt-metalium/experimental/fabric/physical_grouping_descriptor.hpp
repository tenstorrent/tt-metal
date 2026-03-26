// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ostream>
#include <string>
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
    tt::tt_metal::TrayID tray_id{0};              // Tray ID (1-4) if available, 0 otherwise

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
    std::string type;  // Type of grouping (e.g., "MESH", "TRAY_1", "meshes", "pods")
    // items[node_id] is the item for graph node node_id. Flattened meshes may use non-contiguous IDs;
    // in that case size is max_node_id+1 (only indices present in adjacency_graph are meaningful).
    std::vector<GroupingItemInfo> items;
    uint32_t asic_count = 0;  // Total ASICs provided by this grouping, calculated bottom-up during population

    // Adjacency graph. For flattened groupings, items[node_id] matches each node in the graph.
    // Empty graph if no connection type is specified.
    AdjacencyGraph<uint32_t> adjacency_graph;
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
    ValidGroupingsMap get_valid_groupings_for_mgd(
        const MeshGraphDescriptor& mesh_graph_descriptor,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const;

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

    // Find all possible ASIC IDs that could appear in any valid mapping of the input `groupings` to the physical
    // system descriptor.
    // Returns a vector of unordered_sets. Each element is one complete valid mapping: the set of ASIC IDs used
    // across all of the input groupings for that mapping (grouping type is not distinguished in the set).
    // Returns an empty vector if no valid combined mapping exists.
    std::vector<std::unordered_set<tt::tt_metal::AsicID>> find_all_in_psd(
        const std::vector<GroupingInfo>& groupings,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const;

    // Same semantics as the overload without `errors_out`.
    // Additionally, `errors_out` receives detailed messages when mapping fails or no valid combined mapping can be
    // formed.
    std::vector<std::unordered_set<tt::tt_metal::AsicID>> find_all_in_psd(
        const std::vector<GroupingInfo>& groupings,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        std::vector<std::string>& errors_out) const;

    // Same as find_all_in_psd above, but uses a prebuilt flat ASIC adjacency graph from the PSD (from
    // build_flat_adjacency_map_from_psd). Callers that already built the graph can pass it to avoid a
    // duplicate O(|PSD|) scan and graph construction.
    std::vector<std::unordered_set<tt::tt_metal::AsicID>> find_all_in_psd(
        const std::vector<GroupingInfo>& groupings,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const AdjacencyGraph<tt::tt_metal::AsicID>& physical_graph) const;

    std::vector<std::unordered_set<tt::tt_metal::AsicID>> find_all_in_psd(
        const std::vector<GroupingInfo>& groupings,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const AdjacencyGraph<tt::tt_metal::AsicID>& physical_graph,
        std::vector<std::string>& errors_out) const;

    // Build flattened adjacency meshes - one per possibility based on possible groupings that can be formed
    // Returns vector of GroupingInfo objects, each with adjacency_graph populated and node metadata maps filled
    std::vector<GroupingInfo> build_flattened_adjacency_mesh(const GroupingInfo& grouping) const;

    // Overload that accepts a PhysicalSystemDescriptor reference for validation/filtering
    std::vector<GroupingInfo> build_flattened_adjacency_mesh(
        const GroupingInfo& grouping, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const;

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
        const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor) const;

    // Private helper that takes PSD pointer (used internally by public overloads)
    std::vector<GroupingInfo> build_flattened_adjacency_mesh(
        const GroupingInfo& grouping, const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor) const;

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
