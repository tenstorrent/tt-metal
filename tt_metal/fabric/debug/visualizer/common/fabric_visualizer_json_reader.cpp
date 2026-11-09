// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_json_reader.hpp"

#include <limits>

#include <fmt/ranges.h>

#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::visualizer {

namespace {

// How errors name a value: its pointer, or "the top level" for the document itself.
std::string describe_pointer(const JsonValueReader::json_pointer& pointer) {
    return pointer.empty() ? "the top level" : pointer.to_string();
}

// A value as an error quotes it: its JSON text, cut short when long, since it may be a whole object.
std::string describe_value(const JsonValueReader::json& value) {
    constexpr size_t k_max_length = 80;
    std::string text = value.dump();
    if (text.size() > k_max_length) {
        text.resize(k_max_length);
        text += "...";
    }
    return text;
}

}  // namespace

// ============ JsonValueReader ============

JsonValueReader::JsonValueReader(const json& document, std::string_view document_name) :
    JsonValueReader(document, json_pointer(), document_name) {}

JsonValueReader::JsonValueReader(const json& value, json_pointer pointer, std::string_view document_name) :
    value_(&value), pointer_(std::move(pointer)), document_name_(document_name) {}

void JsonValueReader::throw_error(const std::string& problem) const {
    TT_THROW("{}: {} {}", document_name_, describe_pointer(pointer_), problem);
}

void JsonValueReader::require_type(bool has_type, std::string_view type_name) const {
    if (!has_type) {
        fail("is {}, not {}", describe_value(*value_), type_name);
    }
}

bool JsonValueReader::read_bool() const {
    require_type(value_->is_boolean(), "a boolean");
    return value_->get<bool>();
}

std::string JsonValueReader::read_string() const {
    require_type(value_->is_string(), "a string");
    return value_->get<std::string>();
}

// A document parsed from text holds a non-negative integer as unsigned, but one built in memory may hold it as
// signed, so both are taken.
uint64_t JsonValueReader::read_unsigned_integer(size_t bits) const {
    const bool is_non_negative_integer =
        value_->is_number_unsigned() || (value_->is_number_integer() && value_->get<int64_t>() >= 0);
    require_type(is_non_negative_integer, "an unsigned integer");
    const uint64_t value = value_->get<uint64_t>();
    const uint64_t max = bits >= 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << bits) - 1;
    if (value > max) {
        fail("is {}, which doesn't fit {} bits", value, bits);
    }
    return value;
}

int64_t JsonValueReader::read_signed_integer(size_t bits) const {
    require_type(value_->is_number_integer(), "an integer");
    if (value_->is_number_unsigned() && value_->get<uint64_t>() > uint64_t{std::numeric_limits<int64_t>::max()}) {
        fail("is {}, which doesn't fit {} signed bits", value_->get<uint64_t>(), bits);
    }
    const int64_t value = value_->get<int64_t>();
    const int64_t max = bits >= 64 ? std::numeric_limits<int64_t>::max() : (int64_t{1} << (bits - 1)) - 1;
    const int64_t min = -max - 1;
    if (value < min || value > max) {
        fail("is {}, which doesn't fit {} signed bits", value, bits);
    }
    return value;
}

// ============ JsonObjectReader ============

JsonObjectReader::JsonObjectReader(const JsonValueReader& object) : object_(object) {
    object_.require_type(object_.value_->is_object(), "an object");
}

JsonValueReader JsonObjectReader::make_key_reader(const json& value, std::string_view key) const {
    return JsonValueReader(value, pointer() / std::string(key), object_.document_name_);
}

const JsonObjectReader::json* JsonObjectReader::take_value(std::string_view key) {
    if (!taken_keys_.emplace(key).second) {
        make_key_reader(*object_.value_, key).fail("is read twice");
    }
    const auto it = object_.value_->find(std::string(key));
    return it == object_.value_->end() ? nullptr : &*it;
}

JsonValueReader JsonObjectReader::take(std::string_view key) {
    const json* value = take_value(key);
    if (value == nullptr) {
        make_key_reader(*object_.value_, key).fail("is missing");
    }
    return make_key_reader(*value, key);
}

std::optional<JsonValueReader> JsonObjectReader::take_optional(std::string_view key) {
    const json* value = take_value(key);
    return value == nullptr ? std::nullopt : std::optional<JsonValueReader>(make_key_reader(*value, key));
}

void JsonObjectReader::skip(std::string_view key) { take(key); }

void JsonObjectReader::check(std::string_view key, const json& expected) {
    const JsonValueReader value = take(key);
    if (*value.value_ != expected) {
        value.fail("is {}, not {}", describe_value(*value.value_), describe_value(expected));
    }
}

void JsonObjectReader::finish() const {
    std::vector<std::string> untaken;
    for (const auto& entry : object_.value_->items()) {
        if (!taken_keys_.contains(entry.key())) {
            untaken.push_back((pointer() / entry.key()).to_string());
        }
    }
    if (!untaken.empty()) {
        object_.fail("has keys this reader doesn't read: {}", fmt::join(untaken, ", "));
    }
}

}  // namespace tt::tt_fabric::visualizer
