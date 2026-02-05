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
    validation_result_ = validate(temp_proto);
    TT_FATAL(
        validation_result_.is_valid(),
        "Failed to validate PhysicalGroupingDescriptor textproto: \n{}",
        PhysicalGroupingDescriptor::get_validation_report(validation_result_));

    proto_ = std::make_shared<proto::PhysicalGroupings>(temp_proto);
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
        GroupingItemInfo item_info;
        if (item.has_asic_location()) {
            item_info.type = GroupingItemInfo::ItemType::ASIC_LOCATION;
            item_info.asic_location = static_cast<uint32_t>(item.asic_location());
        } else if (item.has_grouping_ref()) {
            item_info.type = GroupingItemInfo::ItemType::GROUPING_REF;
            item_info.grouping_name = item.grouping_ref().grouping_name();
            item_info.count = item.grouping_ref().count();
        }
        info.items.push_back(item_info);
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

std::string PhysicalGroupingDescriptor::ValidationResult::get_report() const {
    return PhysicalGroupingDescriptor::get_validation_report(*this);
}

std::string PhysicalGroupingDescriptor::get_validation_report(const ValidationResult& result) {
    if (result.is_valid() && result.warnings.empty()) {
        return "No validation errors or warnings found.\n";
    }

    std::ostringstream report;
    report << "=== PhysicalGroupingDescriptor Validation Report ===\n\n";

    if (!result.errors.empty()) {
        report << "Errors:\n";
        for (const auto& error : result.errors) {
            report << "  - " << error << "\n";
        }
        report << "\n";
    }

    if (!result.warnings.empty()) {
        report << "Warnings:\n";
        for (const auto& warning : result.warnings) {
            report << "  - " << warning << "\n";
        }
        report << "\n";
    }

    return report.str();
}

PhysicalGroupingDescriptor::ValidationResult PhysicalGroupingDescriptor::validate(
    const proto::PhysicalGroupings& proto) {
    ValidationResult result;

    // Run validation groups
    validate_required_groupings(proto, result);
    if (!result.is_valid()) {
        return result;  // Early exit on critical errors
    }

    validate_grouping_references(proto, result);
    validate_counts(proto, result);
    validate_grouping_structure(proto, result);

    return result;
}

void PhysicalGroupingDescriptor::validate_required_groupings(
    const proto::PhysicalGroupings& proto, ValidationResult& result) {
    // Rule 1: Check required "meshes" grouping
    if (!grouping_exists(proto, "meshes")) {
        result.errors.push_back("Required grouping 'meshes' is missing");
    }

    // Optional: Warn about recommended groupings
    if (!grouping_exists(proto, "trays")) {
        result.warnings.push_back("Recommended grouping 'trays' is not defined");
    }
    if (!grouping_exists(proto, "hosts")) {
        result.warnings.push_back("Recommended grouping 'hosts' is not defined");
    }
}

void PhysicalGroupingDescriptor::validate_grouping_references(
    const proto::PhysicalGroupings& proto, ValidationResult& result) {
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
                    result.errors.push_back(
                        fmt::format("Grouping '{}' has a grouping_ref with empty grouping_name", name));
                    continue;
                }

                if (!grouping_names.count(ref_name)) {
                    result.errors.push_back(
                        fmt::format("Grouping '{}' references non-existent grouping '{}'", name, ref_name));
                }
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_counts(const proto::PhysicalGroupings& proto, ValidationResult& result) {
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        const std::string& name = grouping.name();

        for (int j = 0; j < grouping.items_size(); ++j) {
            const auto& item = grouping.items(j);

            if (item.has_grouping_ref()) {
                const auto& ref = item.grouping_ref();
                uint32_t count = ref.count();

                if (name == "meshes") {
                    // Meshes can have count >= 1
                    if (count < 1) {
                        result.errors.push_back(fmt::format(
                            "Grouping '{}' has grouping_ref with count {}; meshes must have count >= 1", name, count));
                    }
                } else {
                    // All other groupings must have count >= 2
                    if (count < 2) {
                        result.errors.push_back(fmt::format(
                            "Grouping '{}' has grouping_ref with count {}; groupings other than meshes must have count "
                            ">= 2",
                            name,
                            count));
                    }
                }
            }
        }
    }
}

void PhysicalGroupingDescriptor::validate_grouping_structure(
    const proto::PhysicalGroupings& proto, ValidationResult& result) {
    for (int i = 0; i < proto.groupings_size(); ++i) {
        const auto& grouping = proto.groupings(i);
        const std::string& name = grouping.name();

        // Check that grouping has items
        if (grouping.items_size() == 0) {
            result.errors.push_back(fmt::format("Grouping '{}' must have at least one item", name));
            continue;
        }

        // Validate each item
        for (int j = 0; j < grouping.items_size(); ++j) {
            const auto& item = grouping.items(j);

            // Check that exactly one of asic_location or grouping_ref is set (enforced by oneof, but validate anyway)
            bool has_asic_location = item.has_asic_location();
            bool has_grouping_ref = item.has_grouping_ref();

            if (!has_asic_location && !has_grouping_ref) {
                result.errors.push_back(
                    fmt::format("Grouping '{}' item {} must have either asic_location or grouping_ref", name, j));
            }

            // Validate ASIC location enum value
            if (has_asic_location) {
                proto::AsicLocation loc = item.asic_location();
                if (loc == proto::ASIC_LOCATION_UNSPECIFIED) {
                    result.errors.push_back(fmt::format(
                        "Grouping '{}' item {} uses ASIC_LOCATION_UNSPECIFIED; must use ASIC_LOCATION_1 through "
                        "ASIC_LOCATION_8",
                        name,
                        j));
                }
                if (static_cast<int>(loc) < 1 || static_cast<int>(loc) > 8) {
                    result.errors.push_back(fmt::format(
                        "Grouping '{}' item {} uses invalid ASIC location value {}", name, j, static_cast<int>(loc)));
                }
            }
        }
    }
}

}  // namespace tt::tt_fabric
