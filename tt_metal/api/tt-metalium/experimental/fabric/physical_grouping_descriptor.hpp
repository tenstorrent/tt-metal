// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <filesystem>
#include <memory>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>

// Forward declaration
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

    ItemType type;
    uint32_t asic_location = 0;  // Only valid if type == ASIC_LOCATION
    std::string grouping_name;   // Only valid if type == GROUPING_REF
    // Note: Counts are represented by having multiple items. Use items.size() to get the count.
};

// Grouping information
struct GroupingInfo {
    std::string name;
    std::vector<GroupingItemInfo> items;
    uint32_t asic_count = 0;  // Total ASICs provided by this grouping, calculated bottom-up during population
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

    // Get all groupings
    std::vector<GroupingInfo> get_all_groupings() const;

    // Main matching algorithm: Find valid groupings for MGD instances
    std::unordered_map<std::string, std::unordered_map<std::string, GroupingInfo>> get_valid_groupings_for_mgd(
        const MeshGraphDescriptor& mesh_graph_descriptor) const;

private:
    // Matching algorithm helper methods (internal use only)
    // Find best matching "meshes" grouping for a given chip count requirement
    GroupingInfo find_best_meshes_grouping(uint32_t required_chips) const;

    // Analyze how many of each lower-level grouping type a grouping contains
    std::unordered_map<std::string, uint32_t> analyze_grouping_composition(
        const GroupingInfo& grouping, const std::vector<std::string>& lower_level_names) const;

    // Find matching higher-level groupings based on all lower-level types
    std::vector<GroupingInfo> find_matching_higher_level_groupings(
        const std::string& grouping_name,
        const std::vector<std::string>& lower_level_names,
        const std::unordered_map<std::string, uint32_t>& required_counts) const;

    // Determine grouping hierarchy levels (which groupings reference which)
    // Returns a map from grouping name to the set of lower-level grouping names it references
    std::unordered_map<std::string, std::vector<std::string>> determine_grouping_hierarchy() const;

    // Count required lower-level groupings from MGD graph instances
    // graph_type: The type of graph instances we're matching (e.g., "POD", "CLUSTER")
    //             Used to count lower-level instances within those graph instances
    std::unordered_map<std::string, uint32_t> count_required_lower_level_groupings(
        const MeshGraphDescriptor& mesh_graph_descriptor,
        const std::vector<std::string>& lower_level_names,
        const std::string& graph_type = "") const;

    // Build composition requirements from MGD structure (Phase 2)
    // Returns: map of composition pattern -> list of graph instance IDs requiring that pattern
    // Composition pattern is a map: {lower_level_name -> count}
    std::unordered_map<std::string, std::vector<GlobalNodeId>> build_composition_requirements(
        const MeshGraphDescriptor& mesh_graph_descriptor) const;

    // Find groupings by composition (Phase 3) - NO name filtering
    // Returns best matching grouping that satisfies the required composition
    GroupingInfo find_groupings_by_composition(
        const std::unordered_map<std::string, uint32_t>& required_composition) const;

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

    // Static validation - returns vector of error strings (similar to MeshGraphDescriptor)
    static std::vector<std::string> static_validate(const proto::PhysicalGroupings& proto);

    // Helper to get validation report from error vector
    static std::string get_validation_report(const std::vector<std::string>& errors);

    // Population method (called after validation passes)
    void populate();

    // Helper methods for populate()
    static uint32_t calculate_base_grouping_asic_count(const GroupingInfo& grouping);
    static uint32_t calculate_dependent_grouping_asic_count(
        const GroupingInfo& grouping,
        const std::unordered_map<std::string, std::vector<GroupingInfo>>& groupings_by_name);

    // Internal validation helpers (used by static_validate)
    static void validate_required_groupings(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_grouping_references(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_counts(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_grouping_structure(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
};

}  // namespace tt::tt_fabric
