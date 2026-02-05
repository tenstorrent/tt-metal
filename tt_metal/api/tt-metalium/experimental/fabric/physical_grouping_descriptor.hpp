// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <filesystem>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>

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

    // Access proto
    const proto::PhysicalGroupings& get_proto() const { return *proto_; }

    // Check if a grouping exists
    bool has_grouping(const std::string& grouping_name) const;

    // Get all groupings with a specific name (supports multiple definitions)
    std::vector<const proto::Grouping*> get_groupings_by_name(const std::string& grouping_name) const;

    // Get all grouping names (including duplicates)
    std::vector<std::string> get_all_grouping_names() const;

    // Validation result
    struct ValidationResult {
        std::vector<std::string> errors;
        std::vector<std::string> warnings;

        bool is_valid() const { return errors.empty(); }
        std::string get_report() const;
    };

    // Get validation result (already performed during construction)
    const ValidationResult& get_validation_result() const { return validation_result_; }

private:
    std::shared_ptr<const proto::PhysicalGroupings> proto_;
    ValidationResult validation_result_;

    // Helper for reading files
    static std::string read_file_to_string(const std::filesystem::path& file_path);

    // Validation methods
    static ValidationResult validate(const proto::PhysicalGroupings& proto);
    static void validate_required_groupings(const proto::PhysicalGroupings& proto, ValidationResult& result);
    static void validate_grouping_references(const proto::PhysicalGroupings& proto, ValidationResult& result);
    static void validate_counts(const proto::PhysicalGroupings& proto, ValidationResult& result);
    static void validate_grouping_structure(const proto::PhysicalGroupings& proto, ValidationResult& result);

    // Helper to get validation report
    static std::string get_validation_report(const ValidationResult& result);
};

}  // namespace tt::tt_fabric
