// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <ostream>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <queue>
#include <memory>
#include <cctype>
#include <functional>
#include <optional>
#include <tt_stl/assert.hpp>
#include <fmt/format.h>

#include "protobuf/physical_grouping_descriptor.pb.h"
#include "protobuf/mesh_graph_descriptor.pb.h"
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-logger/tt-logger.hpp>
#include <cctype>
#include <map>

#include <google/protobuf/text_format.h>

using namespace tt::tt_fabric;

namespace {

std::string read_file_to_string(const std::filesystem::path& file_path) {
    std::ifstream input(file_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

// Legacy function for backward compatibility - returns type string
std::string get_grouping_name_string(const proto::Grouping& grouping) {
    if (grouping.has_preset_type()) {
        switch (grouping.preset_type()) {
            case proto::TRAY_1: return "TRAY_1";
            case proto::TRAY_2: return "TRAY_2";
            case proto::TRAY_3: return "TRAY_3";
            case proto::TRAY_4: return "TRAY_4";
            case proto::HOSTS: return "HOSTS";
            case proto::MESH: return "MESH";
            default: return "";
        }
    } else if (grouping.has_custom_type()) {
        return grouping.custom_type();
    }
    return "";
}

bool grouping_exists(const proto::PhysicalGroupings& proto, const std::string& grouping_name) {
    for (const auto& grouping : proto.groupings()) {
        std::string name = get_grouping_name_string(grouping);
        if (name == grouping_name) {
            return true;
        }
    }
    return false;
}

}  // namespace

namespace tt::tt_fabric {

// Static helper functions to access grouping name and type from proto
std::string PhysicalGroupingDescriptor::get_grouping_name(const proto::Grouping& grouping) { return grouping.name(); }

std::string PhysicalGroupingDescriptor::get_grouping_type_string(const proto::Grouping& grouping) {
    if (grouping.has_preset_type()) {
        switch (grouping.preset_type()) {
            case proto::TRAY_1: return "TRAY_1";
            case proto::TRAY_2: return "TRAY_2";
            case proto::TRAY_3: return "TRAY_3";
            case proto::TRAY_4: return "TRAY_4";
            case proto::HOSTS: return "HOSTS";
            case proto::MESH: return "MESH";
            default: return "";
        }
    } else if (grouping.has_custom_type()) {
        return grouping.custom_type();
    }
    return "";
}

PhysicalGroupingDescriptor::PhysicalGroupingDescriptor(const std::string& text_proto) {
    proto::PhysicalGroupings temp_proto;
    google::protobuf::TextFormat::Parser parser;

    // Strict validation - don't allow unknown fields
    parser.AllowUnknownField(false);
    parser.AllowUnknownExtension(false);

    TT_FATAL(parser.ParseFromString(text_proto, &temp_proto), "Failed to parse PhysicalGroupingDescriptor textproto");

    // Uniquify duplicate names before validation
    uniquify_duplicate_names(temp_proto);

    // Validate the proto
    std::vector<std::string> all_errors = static_validate(temp_proto);

    TT_FATAL(
        all_errors.empty(),
        "Failed to validate PhysicalGroupingDescriptor textproto: \n{}",
        get_validation_report(all_errors));

    proto_ = std::make_shared<proto::PhysicalGroupings>(temp_proto);

    populate();

    // Collect grouping validation errors and add to the same error vector
    instance_validate(all_errors);

    TT_FATAL(
        all_errors.empty(),
        "Failed to validate PhysicalGroupingDescriptor textproto: \n{}",
        get_validation_report(all_errors));
}

PhysicalGroupingDescriptor::PhysicalGroupingDescriptor(const std::filesystem::path& text_proto_file_path) :
    PhysicalGroupingDescriptor(read_file_to_string(text_proto_file_path)) {}

PhysicalGroupingDescriptor::~PhysicalGroupingDescriptor() = default;

std::string PhysicalGroupingDescriptor::read_file_to_string(const std::filesystem::path& file_path) {
    return ::read_file_to_string(file_path);
}

bool PhysicalGroupingDescriptor::has_grouping(const std::string& grouping_name) const {
    return grouping_exists(*proto_, grouping_name);
}

size_t PhysicalGroupingDescriptor::get_grouping_count() const { return proto_->groupings_size(); }

uint32_t PhysicalGroupingDescriptor::get_grouping_asic_count(const std::string& grouping_name) const {
    auto name_it = resolved_groupings_cache_.find(grouping_name);
    if (name_it != resolved_groupings_cache_.end() && !name_it->second.empty()) {
        // Return the ASIC count from the first grouping with this name (any type)
        // (all groupings with same name should have same structure/count)
        const auto& first_type_map = name_it->second.begin()->second;
        if (!first_type_map.empty()) {
            return first_type_map[0].asic_count;
        }
    }
    return 0;
}

std::vector<GroupingInfo> PhysicalGroupingDescriptor::get_groupings_by_name(const std::string& grouping_name) const {
    auto name_it = resolved_groupings_cache_.find(grouping_name);
    if (name_it != resolved_groupings_cache_.end()) {
        std::vector<GroupingInfo> result;
        // Collect all groupings with this name (across all types)
        for (const auto& [type, groupings] : name_it->second) {
            result.insert(result.end(), groupings.begin(), groupings.end());
        }
        return result;
    }
    // Fallback: return empty vector if not found in cache
    return {};
}

std::vector<GroupingInfo> PhysicalGroupingDescriptor::get_groupings_by_type(const std::string& grouping_type) const {
    std::vector<GroupingInfo> result;
    // Search through all names to find groupings with the specified type
    for (const auto& [name, type_map] : resolved_groupings_cache_) {
        auto type_it = type_map.find(grouping_type);
        if (type_it != type_map.end()) {
            result.insert(result.end(), type_it->second.begin(), type_it->second.end());
        }
    }
    return result;
}

std::vector<GroupingInfo> PhysicalGroupingDescriptor::get_all_groupings() const {
    std::vector<GroupingInfo> result;
    for (const auto& [name, type_map] : resolved_groupings_cache_) {
        for (const auto& [type, groupings] : type_map) {
            for (const auto& grouping : groupings) {
                result.push_back(grouping);
            }
        }
    }
    return result;
}

std::vector<std::string> PhysicalGroupingDescriptor::get_all_grouping_names() const {
    std::vector<std::string> names;
    for (const auto& grouping : proto_->groupings()) {
        names.push_back(PhysicalGroupingDescriptor::get_grouping_name(grouping));
    }
    return names;
}

std::vector<std::string> PhysicalGroupingDescriptor::get_all_grouping_types() const {
    std::set<std::string> types_set;
    for (const auto& grouping : proto_->groupings()) {
        std::string type = PhysicalGroupingDescriptor::get_grouping_type_string(grouping);
        if (!type.empty()) {
            types_set.insert(type);
        }
    }
    return std::vector<std::string>(types_set.begin(), types_set.end());
}

std::string PhysicalGroupingDescriptor::get_validation_report(const std::vector<std::string>& errors) {
    if (errors.empty()) {
        return "No validation errors found.\n";
    }

    std::ostringstream report;
    report << "=== PhysicalGroupingDescriptor Validation Report ===\n\n";
    report << "Errors:\n";
    for (const auto& error : errors) {
        report << "  - " << error << "\n";
    }
    report << "\n";

    return report.str();
}

// Uniquify duplicate names in the proto by adding unique IDs
void PhysicalGroupingDescriptor::uniquify_duplicate_names(proto::PhysicalGroupings& proto) {
    std::unordered_map<std::string, uint32_t> name_counters;
    std::unordered_set<std::string> used_names;

    for (int i = 0; i < proto.groupings_size(); ++i) {
        auto* grouping = proto.mutable_groupings(i);
        std::string current_name = PhysicalGroupingDescriptor::get_grouping_name(*grouping);

        if (current_name.empty()) {
            continue;  // Skip if name is empty (will be caught by other validation)
        }

        // If this name is already used, uniquify it
        if (used_names.contains(current_name)) {
            // Generate unique name with ID suffix
            uint32_t& counter = name_counters[current_name];
            std::string unique_name;
            do {
                counter++;
                unique_name = fmt::format("{}_{}", current_name, counter);
            } while (used_names.contains(unique_name));

            // Update the proto with the unique name
            grouping->set_name(unique_name);
            used_names.insert(unique_name);
        } else {
            // First occurrence, keep as is
            used_names.insert(current_name);
            name_counters[current_name] = 0;  // Initialize counter
        }
    }
}

// Validate that all grouping names are unique (should be true after uniquification)
void PhysicalGroupingDescriptor::validate_unique_names(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    std::unordered_set<std::string> names;

    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = PhysicalGroupingDescriptor::get_grouping_name(grouping);

        if (name.empty()) {
            continue;  // Empty names are caught by other validation
        }

        if (names.contains(name)) {
            errors.push_back(fmt::format(
                "Grouping name '{}' appears multiple times (internal error: uniquification failed).", name));
        }
        names.insert(name);
    }
}

std::vector<std::string> PhysicalGroupingDescriptor::static_validate(const proto::PhysicalGroupings& proto) {
    std::vector<std::string> all_errors;

    // Run validation groups with early exit checkpoints
    {
        validate_required_groupings(proto, all_errors);
        if (!all_errors.empty()) {
            return all_errors;
        }
    }

    {
        validate_grouping_references(proto, all_errors);
        validate_counts(proto, all_errors);
        validate_grouping_structure(proto, all_errors);
        validate_unique_names(proto, all_errors);
        if (!all_errors.empty()) {
            return all_errors;
        }
    }

    return all_errors;
}

uint32_t PhysicalGroupingDescriptor::calculate_base_grouping_asic_count(const GroupingInfo& grouping) {
    uint32_t count = 0;
    for (const auto& item : grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
            count += 1;
        }
    }
    return count;
}

uint32_t PhysicalGroupingDescriptor::calculate_dependent_grouping_asic_count(
    const GroupingInfo& grouping, const std::unordered_map<std::string, std::vector<GroupingInfo>>& groupings_by_name) {
    uint32_t total_asics = 0;

    // Set of preset names that don't need to exist (can be auto-populated)
    std::unordered_set<std::string> preset_names = {"TRAY_1", "TRAY_2", "TRAY_3", "TRAY_4", "HOSTS", "MESH", "meshes"};

    for (const auto& item : grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
            total_asics += 1;
            continue;
        }

        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
            // Skip preset names that don't exist - they can be auto-populated later
            if (preset_names.contains(item.grouping_name)) {
                auto ref_it = groupings_by_name.find(item.grouping_name);
                if (ref_it == groupings_by_name.end() || ref_it->second.empty()) {
                    // Preset name doesn't exist yet - skip it (will be auto-populated)
                    continue;
                }
            }

            auto ref_it = groupings_by_name.find(item.grouping_name);
            if (ref_it == groupings_by_name.end() || ref_it->second.empty()) {
                TT_THROW("Grouping '{}' references non-existent grouping '{}'", grouping.name, item.grouping_name);
            }

            uint32_t ref_count = ref_it->second[0].asic_count;
            if (ref_count == 0) {
                TT_THROW(
                    "Grouping '{}' references '{}' which has zero ASIC count (not yet resolved)",
                    grouping.name,
                    item.grouping_name);
            }
            total_asics += ref_count;
        }
    }

    return total_asics;
}

void PhysicalGroupingDescriptor::populate() {
    // Step 1: Convert all proto groupings and build dependency graph
    std::unordered_map<std::string, std::vector<GroupingInfo>> groupings_by_name;
    std::unordered_map<std::string, std::set<std::string>> dependencies;

    // Set of preset names that don't need to exist (can be auto-populated)
    std::unordered_set<std::string> preset_names = {"TRAY_1", "TRAY_2", "TRAY_3", "TRAY_4", "HOSTS", "MESH", "meshes"};

    for (const auto& grouping : proto_->groupings()) {
        GroupingInfo info = convert_grouping_to_info(grouping);

        // Track dependencies (skip preset names that don't exist)
        std::set<std::string> deps;
        for (const auto& item : info.items) {
            if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                // Only track dependencies on groupings that exist or are not preset names
                // Preset names can be auto-populated, so don't treat them as blocking dependencies
                if (groupings_by_name.contains(item.grouping_name) || !preset_names.contains(item.grouping_name)) {
                    deps.insert(item.grouping_name);
                }
            }
        }
        dependencies[info.type] = deps;
        groupings_by_name[info.type].push_back(std::move(info));
    }

    // Step 2: Process base groupings (no dependencies)
    for (auto& [name, groupings] : groupings_by_name) {
        if (!dependencies[name].empty()) {
            continue;  // Skip dependent groupings for now
        }

        for (auto& grouping : groupings) {
            // Check if this grouping only references preset names
            bool only_preset_refs = true;
            bool has_any_refs = false;
            for (const auto& item : grouping.items) {
                if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                    has_any_refs = true;
                    if (!preset_names.contains(item.grouping_name)) {
                        only_preset_refs = false;
                        break;
                    }
                } else if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                    only_preset_refs = false;
                    break;
                }
            }

            if (only_preset_refs && has_any_refs) {
                // Grouping only references preset names - set count to 0 (will be populated later)
                grouping.asic_count = 0;
            } else {
                grouping.asic_count = calculate_base_grouping_asic_count(grouping);
                if (grouping.asic_count == 0) {
                    TT_THROW("Grouping '{}' has no ASIC_LOCATION items and cannot be resolved", name);
                }
            }
        }
    }

    // Step 3: Build topological sort data structures
    std::unordered_map<std::string, std::set<std::string>> incoming_edges;
    std::unordered_map<std::string, int> in_degree;

    for (const auto& [name, deps] : dependencies) {
        in_degree[name] = static_cast<int>(deps.size());
        for (const auto& dep : deps) {
            incoming_edges[dep].insert(name);
        }
    }

    // Step 4: Process dependent groupings in topological order
    std::queue<std::string> to_process;
    for (const auto& [name, deps] : dependencies) {
        if (deps.empty()) {
            to_process.push(name);
        }
    }

    std::vector<std::string> processed;
    while (!to_process.empty()) {
        std::string current = to_process.front();
        to_process.pop();
        processed.push_back(current);

        // Process all groupings that depend on current
        for (const auto& dependent : incoming_edges[current]) {
            in_degree[dependent]--;
            if (in_degree[dependent] > 0) {
                continue;  // Not ready yet
            }

            // All dependencies resolved, calculate ASIC count
            for (auto& grouping : groupings_by_name[dependent]) {
                grouping.asic_count = calculate_dependent_grouping_asic_count(grouping, groupings_by_name);
                if (grouping.asic_count == 0) {
                    TT_THROW(
                        "Grouping '{}' does not resolve to any ASIC locations (circular or missing references)",
                        dependent);
                }
            }

            to_process.push(dependent);
        }
    }

    // Step 5: Reorganize into two-tier cache structure: name -> type -> vector<GroupingInfo>
    // Note: Cycle detection is now handled by validate_no_cycles() in grouping_validate()
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<GroupingInfo>>> new_cache;
    for (const auto& [type, groupings] : groupings_by_name) {
        for (const auto& grouping : groupings) {
            new_cache[grouping.name][type].push_back(grouping);
        }
    }
    resolved_groupings_cache_ = std::move(new_cache);
}
void PhysicalGroupingDescriptor::grouping_validate() const {
    std::vector<std::string> errors;
    instance_validate(errors);

    // Throw if any errors found
    if (!errors.empty()) {
        std::string error_msg = "Grouping validation failed:\n";
        for (size_t i = 0; i < errors.size(); ++i) {
            error_msg += fmt::format("  {}. {}\n", i + 1, errors[i]);
        }
        TT_THROW("{}", error_msg);
    }
}

void PhysicalGroupingDescriptor::instance_validate(std::vector<std::string>& errors) const {
    validate_leaf_groupings(errors);
    validate_asic_location_usage(errors);
    validate_no_cycles(errors);
    validate_instance_counts(errors);
}

void PhysicalGroupingDescriptor::validate_leaf_groupings(std::vector<std::string>& errors) const {
    // Build dependency graph and identify leaf vs non-leaf groupings
    std::unordered_map<std::string, bool> has_asic_locations;  // grouping -> true if uses ASIC locations
    std::unordered_map<std::string, bool> has_grouping_refs;   // grouping -> true if uses grouping refs
    std::unordered_set<std::string> all_grouping_types;

    // First pass: identify all groupings and their characteristics
    for (const auto& [name, type_map] : resolved_groupings_cache_) {
        for (const auto& [type, groupings] : type_map) {
            all_grouping_types.insert(type);
            bool has_asic = false;
            bool has_refs = false;

            for (const auto& grouping : groupings) {
                for (const auto& item : grouping.items) {
                    if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                        has_asic = true;
                    } else if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                        has_refs = true;
                    }
                }
            }

            // Update flags for this type (may be set by multiple names with same type)
            has_asic_locations[type] = has_asic_locations[type] || has_asic;
            has_grouping_refs[type] = has_grouping_refs[type] || has_refs;
        }
    }

    // Validation: At least one leaf grouping uses ASIC locations
    // A leaf grouping is one that has ASIC locations and no grouping references
    bool has_leaf_with_asic = false;
    for (const auto& type : all_grouping_types) {
        if (has_asic_locations[type] && !has_grouping_refs[type]) {
            has_leaf_with_asic = true;
            break;
        }
    }
    if (!has_leaf_with_asic) {
        errors.push_back("At least one leaf grouping must use ASIC locations");
    }
}

void PhysicalGroupingDescriptor::validate_asic_location_usage(std::vector<std::string>& errors) const {
    // Build dependency graph and identify groupings
    std::unordered_map<std::string, bool> has_asic_locations;  // grouping -> true if uses ASIC locations
    std::unordered_map<std::string, bool> has_grouping_refs;   // grouping -> true if uses grouping refs
    std::unordered_set<std::string> all_grouping_types;

    // First pass: identify all groupings and their characteristics
    for (const auto& [name, type_map] : resolved_groupings_cache_) {
        for (const auto& [type, groupings] : type_map) {
            all_grouping_types.insert(type);
            bool has_asic = false;
            bool has_refs = false;

            for (const auto& grouping : groupings) {
                for (const auto& item : grouping.items) {
                    if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                        has_asic = true;
                    } else if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                        has_refs = true;
                    }
                }
            }

            // Update flags for this type (may be set by multiple names with same type)
            has_asic_locations[type] = has_asic_locations[type] || has_asic;
            has_grouping_refs[type] = has_grouping_refs[type] || has_refs;
        }
    }

    // Validation: Only leaf groupings should use ASIC locations, others should not
    // A grouping that has grouping references should not have ASIC locations
    for (const auto& type : all_grouping_types) {
        if (has_grouping_refs[type] && has_asic_locations[type]) {
            errors.push_back(fmt::format(
                "Grouping '{}' uses ASIC locations but also has grouping references. Only leaf groupings should use "
                "ASIC locations",
                type));
        }
    }
}

void PhysicalGroupingDescriptor::validate_no_cycles(std::vector<std::string>& errors) const {
    // Build dependency graph
    std::unordered_map<std::string, std::set<std::string>> dependencies;  // grouping -> set of dependencies
    std::unordered_set<std::string> all_grouping_types;

    for (const auto& [name, type_map] : resolved_groupings_cache_) {
        for (const auto& [type, groupings] : type_map) {
            all_grouping_types.insert(type);
            for (const auto& grouping : groupings) {
                for (const auto& item : grouping.items) {
                    if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                        dependencies[type].insert(item.grouping_name);
                    }
                }
            }
        }
    }

    // Use DFS to detect cycles
    std::unordered_map<std::string, int> color;  // 0 = white (unvisited), 1 = gray (visiting), 2 = black (visited)
    for (const auto& type : all_grouping_types) {
        color[type] = 0;
    }

    std::function<bool(const std::string&)> has_cycle = [&](const std::string& node) -> bool {
        if (color[node] == 1) {
            // Gray node - cycle detected
            return true;
        }
        if (color[node] == 2) {
            // Black node - already processed
            return false;
        }

        color[node] = 1;  // Mark as gray (visiting)

        // Check all dependencies
        auto deps_it = dependencies.find(node);
        if (deps_it != dependencies.end()) {
            for (const auto& dep : deps_it->second) {
                if (all_grouping_types.contains(dep)) {
                    if (has_cycle(dep)) {
                        return true;
                    }
                }
            }
        }

        color[node] = 2;  // Mark as black (visited)
        return false;
    };

    for (const auto& type : all_grouping_types) {
        if (color[type] == 0) {
            if (has_cycle(type)) {
                errors.push_back(
                    fmt::format("Circular dependencies detected in grouping hierarchy involving '{}'", type));
                break;  // Only report one cycle
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_instance_counts(std::vector<std::string>& errors) const {
    // Set of preset names that don't need to exist (can be auto-populated)
    std::unordered_set<std::string> preset_names = {"TRAY_1", "TRAY_2", "TRAY_3", "TRAY_4", "HOSTS", "MESH", "meshes"};

    // Validation: all groupings should have ASIC counts > 0
    // Exception: groupings that only reference preset names (which can be auto-populated) may have 0 count
    for (const auto& [name, type_map] : resolved_groupings_cache_) {
        for (const auto& [type, groupings] : type_map) {
            for (const auto& grouping : groupings) {
                if (grouping.asic_count == 0) {
                    // Check if this grouping only references preset names
                    bool only_preset_refs = true;
                    bool has_any_refs = false;
                    for (const auto& item : grouping.items) {
                        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF) {
                            has_any_refs = true;
                            if (preset_names.find(item.grouping_name) == preset_names.end()) {
                                only_preset_refs = false;
                                break;
                            }
                        } else if (item.type == GroupingItemInfo::ItemType::ASIC_LOCATION) {
                            only_preset_refs = false;
                            break;
                        }
                    }

                    // Allow zero count only if grouping only references preset names
                    if (!only_preset_refs || !has_any_refs) {
                        errors.push_back(fmt::format("Grouping '{}' has zero ASIC count after resolution", name));
                    }
                }
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_required_groupings(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    // Validate grouping names are non-empty and types are set
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = PhysicalGroupingDescriptor::get_grouping_name(grouping);
        std::string type = PhysicalGroupingDescriptor::get_grouping_type_string(grouping);

        if (name.empty()) {
            errors.push_back(
                fmt::format("Grouping at index {} has an empty name; grouping names must be non-empty", i));
        }

        if (type.empty()) {
            errors.push_back(fmt::format(
                "Grouping '{}' at index {} has no type set; exactly one of preset_type or custom_type must be set",
                name,
                i));
        }
    }

    // TRAY, HOST, and MESH definitions are no longer required - validation removed
}

void PhysicalGroupingDescriptor::validate_grouping_references(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    // Build set of all grouping types
    std::unordered_set<std::string> grouping_types;
    for (const auto& grouping : proto.groupings()) {
        grouping_types.insert(PhysicalGroupingDescriptor::get_grouping_type_string(grouping));
    }

    // Set of preset types that don't need to exist (can be auto-populated)
    std::unordered_set<std::string> preset_types = {"TRAY_1", "TRAY_2", "TRAY_3", "TRAY_4", "HOSTS", "MESH", "meshes"};

    // Validate all grouping references
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = PhysicalGroupingDescriptor::get_grouping_name(grouping);

        for (int j = 0; j < grouping.instances_size(); ++j) {
            const auto& instance = grouping.instances(j);

            if (instance.has_grouping_ref()) {
                const auto& ref = instance.grouping_ref();
                std::string ref_type;
                bool is_preset_type = false;

                if (ref.has_preset_type()) {
                    // Convert preset_type enum to string
                    switch (ref.preset_type()) {
                        case proto::TRAY_1: ref_type = "TRAY_1"; break;
                        case proto::TRAY_2: ref_type = "TRAY_2"; break;
                        case proto::TRAY_3: ref_type = "TRAY_3"; break;
                        case proto::TRAY_4: ref_type = "TRAY_4"; break;
                        case proto::HOSTS: ref_type = "HOSTS"; break;
                        case proto::MESH: ref_type = "MESH"; break;
                        default: ref_type = ""; break;
                    }
                    is_preset_type = true;
                } else if (ref.has_custom_type()) {
                    ref_type = ref.custom_type();
                    is_preset_type = false;
                }

                if (ref_type.empty()) {
                    errors.push_back(fmt::format("Grouping '{}' has a grouping_ref with empty grouping_type", name));
                    continue;
                }

                // Skip validation for preset types - they can be auto-populated
                if (is_preset_type || preset_types.contains(ref_type)) {
                    continue;
                }

                // For custom types, validate they exist
                if (!grouping_types.contains(ref_type)) {
                    errors.push_back(
                        fmt::format("Grouping '{}' references non-existent grouping type '{}'", name, ref_type));
                }
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_counts(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = PhysicalGroupingDescriptor::get_grouping_name(grouping);
        std::string type = PhysicalGroupingDescriptor::get_grouping_type_string(grouping);
        uint32_t instance_count = static_cast<uint32_t>(grouping.instances_size());

        // Validate instance count - all groupings must have at least 1 instance
        if (instance_count < 1) {
            errors.push_back(fmt::format(
                "Grouping '{}' (type '{}') has {} instances; all groupings must have at least 1 instance",
                name,
                type,
                instance_count));
        }
    }
}

void PhysicalGroupingDescriptor::validate_grouping_structure(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        std::string name = get_grouping_name_string(grouping);

        // Check that grouping has instances
        if (grouping.instances_size() == 0) {
            errors.push_back(fmt::format(
                "Grouping '{}' (type '{}') must have at least one instance",
                name,
                PhysicalGroupingDescriptor::get_grouping_type_string(grouping)));
            continue;
        }

        // Validate each instance
        for (int j = 0; j < grouping.instances_size(); ++j) {
            const auto& instance = grouping.instances(j);

            // Check that exactly one of asic_location or grouping_ref is set (enforced by oneof, but validate anyway)
            bool has_asic_location = instance.has_asic_location();
            bool has_grouping_ref = instance.has_grouping_ref();

            if (!has_asic_location && !has_grouping_ref) {
                errors.push_back(
                    fmt::format("Grouping '{}' instance {} must have either asic_location or grouping_ref", name, j));
            }

            // Validate ASIC location enum value
            if (has_asic_location) {
                proto::AsicLocation loc = instance.asic_location();
                if (loc == proto::ASIC_LOCATION_UNSPECIFIED) {
                    errors.push_back(fmt::format(
                        "Grouping '{}' instance {} uses ASIC_LOCATION_UNSPECIFIED; must use ASIC_LOCATION_1 through "
                        "ASIC_LOCATION_8",
                        name,
                        j));
                }
                if (static_cast<int>(loc) < 1 || static_cast<int>(loc) > 8) {
                    errors.push_back(fmt::format(
                        "Grouping '{}' instance {} uses invalid ASIC location value {}",
                        name,
                        j,
                        static_cast<int>(loc)));
                }
            }
        }
    }
}

}  // namespace tt::tt_fabric
