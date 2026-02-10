// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <tt_stl/assert.hpp>
#include <fmt/format.h>

#include "protobuf/physical_grouping_descriptor.pb.h"
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-logger/tt-logger.hpp>

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

bool grouping_exists(const proto::PhysicalGroupings& proto, const std::string& grouping_name) {
    for (const auto& grouping : proto.groupings()) {
        if (grouping.name() == grouping_name) {
            return true;
        }
    }
    return false;
}

}  // namespace

namespace tt::tt_fabric {

PhysicalGroupingDescriptor::PhysicalGroupingDescriptor(const std::string& text_proto) {
    proto::PhysicalGroupings temp_proto;
    google::protobuf::TextFormat::Parser parser;

    // Strict validation - don't allow unknown fields
    parser.AllowUnknownField(false);
    parser.AllowUnknownExtension(false);

    TT_FATAL(parser.ParseFromString(text_proto, &temp_proto), "Failed to parse PhysicalGroupingDescriptor textproto");

    // Validate the proto
    const auto errors = static_validate(temp_proto);
    TT_FATAL(
        errors.empty(), "Failed to validate PhysicalGroupingDescriptor textproto: \n{}", get_validation_report(errors));

    proto_ = std::make_shared<proto::PhysicalGroupings>(temp_proto);

    populate();
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

PhysicalGroupingDescriptor::GroupingInfo PhysicalGroupingDescriptor::convert_grouping_to_info(
    const proto::Grouping& grouping) const {
    GroupingInfo info;
    info.name = grouping.name();

    for (const auto& item : grouping.items()) {
        if (item.has_asic_location()) {
            GroupingItemInfo item_info;
            item_info.type = GroupingItemInfo::ItemType::ASIC_LOCATION;
            item_info.asic_location = static_cast<uint32_t>(item.asic_location());
            info.items.push_back(item_info);
        } else if (item.has_grouping_ref()) {
            // Expand grouping_ref with count N into N items
            const auto& ref = item.grouping_ref();
            const std::string& grouping_name = ref.grouping_name();
            uint32_t count = ref.count();

            for (uint32_t i = 0; i < count; ++i) {
                GroupingItemInfo item_info;
                item_info.type = GroupingItemInfo::ItemType::GROUPING_REF;
                item_info.grouping_name = grouping_name;
                info.items.push_back(item_info);
            }
        }
    }

    return info;
}

std::vector<PhysicalGroupingDescriptor::GroupingInfo> PhysicalGroupingDescriptor::get_groupings_by_name(
    const std::string& grouping_name) const {
    std::vector<GroupingInfo> result;
    for (const auto& grouping : proto_->groupings()) {
        if (grouping.name() == grouping_name) {
            result.push_back(convert_grouping_to_info(grouping));
        }
    }
    return result;
}

std::vector<PhysicalGroupingDescriptor::GroupingInfo> PhysicalGroupingDescriptor::get_all_groupings() const {
    std::vector<GroupingInfo> result;
    for (const auto& grouping : proto_->groupings()) {
        result.push_back(convert_grouping_to_info(grouping));
    }
    return result;
}

std::vector<std::string> PhysicalGroupingDescriptor::get_all_grouping_names() const {
    std::vector<std::string> names;
    for (const auto& grouping : proto_->groupings()) {
        names.push_back(grouping.name());
    }
    return names;
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
        if (!all_errors.empty()) {
            return all_errors;
        }
    }

    return all_errors;
}

void PhysicalGroupingDescriptor::populate() {
    // Currently no additional population needed beyond storing the proto
    // This method is here for consistency with MeshGraphDescriptor pattern
    // and can be extended in the future if needed (e.g., building lookup maps)
}

void PhysicalGroupingDescriptor::validate_required_groupings(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    // Rule 1: Check required "meshes" grouping
    // The "meshes" grouping MUST be defined - this is a hard requirement
    if (!grouping_exists(proto, "meshes")) {
        errors.push_back(
            "Required grouping 'meshes' is missing. At least one grouping with name 'meshes' must be defined.");
    }

    // Validate grouping names are non-empty
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        if (grouping.name().empty()) {
            errors.push_back(
                fmt::format("Grouping at index {} has an empty name; grouping names must be non-empty", i));
        }
    }
}

void PhysicalGroupingDescriptor::validate_grouping_references(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    // Build set of all grouping names
    std::unordered_set<std::string> grouping_names;
    for (const auto& grouping : proto.groupings()) {
        grouping_names.insert(grouping.name());
    }

    // Validate all grouping references
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        const std::string& name = grouping.name();

        for (int j = 0; j < grouping.items_size(); ++j) {
            const auto& item = grouping.items(j);

            if (item.has_grouping_ref()) {
                const auto& ref = item.grouping_ref();
                const std::string& ref_name = ref.grouping_name();

                if (ref_name.empty()) {
                    errors.push_back(fmt::format("Grouping '{}' has a grouping_ref with empty grouping_name", name));
                    continue;
                }

                if (!grouping_names.contains(ref_name)) {
                    errors.push_back(
                        fmt::format("Grouping '{}' references non-existent grouping '{}'", name, ref_name));
                }
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_counts(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        const std::string& name = grouping.name();

        // Calculate total logical item count (expanded counts)
        uint32_t total_logical_items = 0;
        for (int j = 0; j < grouping.items_size(); ++j) {
            const auto& item = grouping.items(j);
            if (item.has_asic_location()) {
                total_logical_items += 1;
            } else if (item.has_grouping_ref()) {
                total_logical_items += item.grouping_ref().count();
            }
        }

        // Validate counts for grouping_ref items
        for (int j = 0; j < grouping.items_size(); ++j) {
            const auto& item = grouping.items(j);

            if (item.has_grouping_ref()) {
                const auto& ref = item.grouping_ref();
                uint32_t count = ref.count();

                if (name == "meshes") {
                    // Meshes can have count >= 1, even if there's only one item
                    if (count < 1) {
                        errors.push_back(fmt::format(
                            "Grouping '{}' has grouping_ref with count {}; meshes must have count >= 1", name, count));
                    }
                } else {
                    // For non-meshes groupings:
                    // - If there's only one item total (any type), it must have count >= 2
                    // - If there are multiple items, each can have count >= 1
                    if (grouping.items_size() == 1 && count < 2) {
                        errors.push_back(fmt::format(
                            "Grouping '{}' has a single item with count {}; groupings other than meshes must "
                            "have count >= 2 when there is only one item",
                            name,
                            count));
                    } else if (count < 1) {
                        errors.push_back(fmt::format(
                            "Grouping '{}' has grouping_ref with count {}; count must be >= 1", name, count));
                    }
                }
            }
        }

        // Validate total logical item count for non-meshes groupings
        if (name != "meshes" && total_logical_items < 2) {
            errors.push_back(fmt::format(
                "Grouping '{}' has {} total logical items; groupings other than meshes must have at least 2 logical "
                "items",
                name,
                total_logical_items));
        }
    }
}

void PhysicalGroupingDescriptor::validate_grouping_structure(
    const proto::PhysicalGroupings& proto, std::vector<std::string>& errors) {
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        const std::string& name = grouping.name();

        // Check that grouping has items
        if (grouping.items_size() == 0) {
            errors.push_back(fmt::format("Grouping '{}' must have at least one item", name));
            continue;
        }

        // Validate each item
        for (int j = 0; j < grouping.items_size(); ++j) {
            const auto& item = grouping.items(j);

            // Check that exactly one of asic_location or grouping_ref is set (enforced by oneof, but validate anyway)
            bool has_asic_location = item.has_asic_location();
            bool has_grouping_ref = item.has_grouping_ref();

            if (!has_asic_location && !has_grouping_ref) {
                errors.push_back(
                    fmt::format("Grouping '{}' item {} must have either asic_location or grouping_ref", name, j));
            }

            // Validate ASIC location enum value
            if (has_asic_location) {
                proto::AsicLocation loc = item.asic_location();
                if (loc == proto::ASIC_LOCATION_UNSPECIFIED) {
                    errors.push_back(fmt::format(
                        "Grouping '{}' item {} uses ASIC_LOCATION_UNSPECIFIED; must use ASIC_LOCATION_1 through "
                        "ASIC_LOCATION_8",
                        name,
                        j));
                }
                if (static_cast<int>(loc) < 1 || static_cast<int>(loc) > 8) {
                    errors.push_back(fmt::format(
                        "Grouping '{}' item {} uses invalid ASIC location value {}", name, j, static_cast<int>(loc)));
                }
            }
        }
    }
}

}  // namespace tt::tt_fabric
