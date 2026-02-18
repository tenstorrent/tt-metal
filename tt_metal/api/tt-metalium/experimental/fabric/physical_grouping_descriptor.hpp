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

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

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
    uint32_t asic_location = 0;  // Only valid if type == ASIC_LOCATION
    std::string grouping_name;   // Only valid if type == GROUPING_REF
    std::vector<CornerOrientation>
        corners;  // Corner orientations (can have multiple, e.g., 1D endpoints have 2, 1x1 has all 4)
    // Note: Counts are represented by having multiple items. Use items.size() to get the count.
};

// Grouping information
struct GroupingInfo {
    std::string name;  // Unique identifier/name for this specific grouping instance
    std::string type;  // Type of grouping (e.g., "MESH", "TRAY_1", "meshes", "pods")
    std::vector<GroupingItemInfo> items;
    uint32_t asic_count = 0;  // Total ASICs provided by this grouping, calculated bottom-up during population

    // Adjacency graph representing the topology/connections between instances
    // Node IDs are instance IDs (uint32_t) from the grouping's instances list
    // Empty graph if no connection type is specified
    // Always uses 1 connection per edge (bidirectional)
    AdjacencyGraph<uint32_t> adjacency_graph;
};

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
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> get_valid_groupings_for_mgd(
        const MeshGraphDescriptor& mesh_graph_descriptor) const;

    // Validate predefined groupings (TRAYS and HOSTS) from PhysicalSystemDescriptor, making sure that they match
    bool validate_preformed_groups_from_physical_system_descriptor(
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) const;

    // Node metadata for flattened mesh nodes
    // Generic enough to be used throughout the flattened mesh representation
    struct FlattenedMeshNodeInfo {
        uint32_t unique_id;                      // Sequential unique ID assigned during flattening
        uint32_t asic_location = 0;              // ASIC location (1-8) if available, 0 otherwise
        uint32_t tray_id = 0;                    // Tray ID (1-4) if available, 0 otherwise
        std::vector<std::string> grouping_path;  // Path through grouping hierarchy using grouping names
                                                 // Includes ASIC location at the end (e.g., ["MESH", "hosts_0",
                                                 // "tray_1", "ASIC_LOCATION_1"])

        // Comparison operators for use in maps/sets
        bool operator==(const FlattenedMeshNodeInfo& other) const { return unique_id == other.unique_id; }
        bool operator<(const FlattenedMeshNodeInfo& other) const { return unique_id < other.unique_id; }
    };

    friend std::ostream& operator<<(std::ostream& os, const FlattenedMeshNodeInfo& info);

    // Build flattened adjacency graph forming one uniform mesh
    // Returns graph with FlattenedMeshNodeInfo as node type
    AdjacencyGraph<FlattenedMeshNodeInfo> build_flattened_adjacency_mesh(const GroupingInfo& grouping) const;

private:
    // Data members
    std::shared_ptr<const proto::PhysicalGroupings> proto_;

    // Cache of resolved groupings with ASIC counts (populated bottom-up)
    std::unordered_map<std::string, std::vector<GroupingInfo>> resolved_groupings_cache_;

    // Internal helper to convert proto grouping to GroupingInfo
    GroupingInfo convert_grouping_to_info(const proto::Grouping& grouping) const;

    // Helper to get ASIC count for a grouping name (from cache)
    uint32_t get_grouping_asic_count(const std::string& grouping_name) const;

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

    // Static validation - returns vector of error strings (similar to MeshGraphDescriptor)
    static std::vector<std::string> static_validate(const proto::PhysicalGroupings& proto);

    // Internal validation helpers (used by static_validate)
    static void uniquify_duplicate_names(proto::PhysicalGroupings& proto);
    static void validate_required_groupings(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_grouping_references(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_counts(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_grouping_structure(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_unique_names(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
};

}  // namespace tt::tt_fabric
