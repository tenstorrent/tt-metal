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
    };

    // Get all groupings with a specific name (supports multiple definitions)
    // Returns grouping information without exposing proto objects
    std::vector<GroupingInfo> get_groupings_by_name(const std::string& grouping_name) const;

    // Get all groupings
    std::vector<GroupingInfo> get_all_groupings() const;

private:
    std::shared_ptr<const proto::PhysicalGroupings> proto_;

    // Internal helper to convert proto grouping to GroupingInfo
    GroupingInfo convert_grouping_to_info(const proto::Grouping& grouping) const;

    // Helper for reading files
    static std::string read_file_to_string(const std::filesystem::path& file_path);

    // Static validation - returns vector of error strings (similar to MeshGraphDescriptor)
    static std::vector<std::string> static_validate(const proto::PhysicalGroupings& proto);

    // Helper to get validation report from error vector
    static std::string get_validation_report(const std::vector<std::string>& errors);

    // Population method (called after validation passes)
    void populate();

    // Internal validation helpers (used by static_validate)
    static void validate_required_groupings(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_grouping_references(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_counts(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
    static void validate_grouping_structure(const proto::PhysicalGroupings& proto, std::vector<std::string>& errors);
};

}  // namespace tt::tt_fabric
