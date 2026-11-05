// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_json.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <string_view>
#include <utility>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::capture {

JsonReader::JsonReader(const json& document, std::string document_name) :
    document_(document), document_name_(std::move(document_name)) {}

const JsonReader::json& JsonReader::find_required(const json_pointer& pointer) const {
    TT_FATAL(document_.contains(pointer), "Fabric capture: the {} has no {}", document_name_, pointer.to_string());
    return document_.at(pointer);
}

bool JsonReader::is_null(const json_pointer& pointer) const { return find_required(pointer).is_null(); }

const JsonReader::json& JsonReader::read_object(const json_pointer& pointer) const {
    const json& value = find_required(pointer);
    TT_FATAL(value.is_object(), "Fabric capture: the {}'s {} isn't an object", document_name_, pointer.to_string());
    return value;
}

const JsonReader::json& JsonReader::read_array(const json_pointer& pointer) const {
    const json& value = find_required(pointer);
    TT_FATAL(value.is_array(), "Fabric capture: the {}'s {} isn't an array", document_name_, pointer.to_string());
    return value;
}

uint64_t JsonReader::read_unsigned(const json_pointer& pointer) const {
    const json& value = find_required(pointer);
    TT_FATAL(
        value.is_number_unsigned(),
        "Fabric capture: the {}'s {} is {}, not an unsigned integer",
        document_name_,
        pointer.to_string(),
        value.dump());
    return value.get<uint64_t>();
}

uint32_t JsonReader::read_uint32(const json_pointer& pointer) const {
    const uint64_t value = read_unsigned(pointer);
    TT_FATAL(
        value <= std::numeric_limits<uint32_t>::max(),
        "Fabric capture: the {}'s {} is {}, which doesn't fit 32 bits",
        document_name_,
        pointer.to_string(),
        value);
    return static_cast<uint32_t>(value);
}

std::string JsonReader::read_string(const json_pointer& pointer) const {
    const json& value = find_required(pointer);
    TT_FATAL(
        value.is_string(),
        "Fabric capture: the {}'s {} is {}, not a string",
        document_name_,
        pointer.to_string(),
        value.dump());
    return value.get<std::string>();
}

bool JsonReader::read_bool(const json_pointer& pointer) const {
    const json& value = find_required(pointer);
    TT_FATAL(
        value.is_boolean(),
        "Fabric capture: the {}'s {} is {}, not a boolean",
        document_name_,
        pointer.to_string(),
        value.dump());
    return value.get<bool>();
}

uint64_t JsonReader::read_asic_id(const json_pointer& pointer) const {
    const std::string text = read_string(pointer);
    const std::string_view digits = std::string_view(text).substr(text.starts_with("0x") ? 2 : 0);
    TT_FATAL(
        text.starts_with("0x") && !digits.empty() && digits.size() <= 16 &&
            std::ranges::all_of(digits, [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; }),
        "Fabric capture: the {}'s {} is \"{}\", not a hex asic id",
        document_name_,
        pointer.to_string(),
        text);
    return std::stoull(std::string(digits), nullptr, 16);
}

CoreXY JsonReader::read_core(const json_pointer& pointer) const {
    const json& core = read_array(pointer);
    TT_FATAL(core.size() == 2, "Fabric capture: the {}'s {} isn't an [x, y] pair", document_name_, pointer.to_string());
    return CoreXY{.x = read_uint32(pointer / 0), .y = read_uint32(pointer / 1)};
}

std::string make_asic_id_text(uint64_t asic_id) { return fmt::format("0x{:016x}", asic_id); }

nlohmann::ordered_json make_core_json(const CoreXY& core) { return nlohmann::ordered_json::array({core.x, core.y}); }

}  // namespace tt::tt_fabric::capture
