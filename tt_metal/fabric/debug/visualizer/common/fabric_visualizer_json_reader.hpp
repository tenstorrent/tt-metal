// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A strict reader for the JSON files the fabric visualizer's stages write, such as the manifest. A reader built on it
// fails when the file and the reader disagree about which keys there are, whatever their values:
//   - every key it takes must be there, unless it takes it as optional;
//   - every key of an object must be taken, skipped or checked before the object is done.
// So a key the writer adds or renames without its reader, or a key the reader expects that the writer doesn't write,
// fails even when its value is a default such as false, 0 or []. Each failure names the value by its JSON pointer,
// such as "Fabric manifest: /meshes/M0/chips/C1/routers/E0/shape/num_vcs is missing".
//
// Objects are read only through read_object, which fails on any key the read left untaken once it returns, so no
// object can be read partly by mistake. Arrays are read whole. docs/JSON_READER.md walks through how to use it.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace tt::tt_fabric::visualizer {

class JsonObjectReader;

// One value of a JSON document and its JSON pointer.
class JsonValueReader {
public:
    using json = nlohmann::ordered_json;
    using json_pointer = json::json_pointer;

    // The document's top level. Every error starts with `document_name`, such as "Fabric manifest", which must outlive
    // the reader and every reader made from it.
    JsonValueReader(const json& document, std::string_view document_name);

    const json_pointer& pointer() const { return pointer_; }

    bool is_null() const { return value_->is_null(); }

    // The value as T: bool, std::string, or an integer type the value fits. Fails on any other JSON type, such as a
    // number written as a string, a fraction, or a negative number read as unsigned.
    template <typename T>
    T read() const;

    // Null as nullopt, otherwise read<T>().
    template <typename T>
    std::optional<T> read_nullable() const {
        return is_null() ? std::nullopt : std::optional<T>(read<T>());
    }

    // Reads an object with `read(JsonObjectReader&)`, then fails on any key `read` left untaken. Returns what `read`
    // returns.
    template <typename Read>
    auto read_object(Read&& read) const;

    // Null as nullopt, otherwise read_object(read).
    template <typename Read>
    auto read_nullable_object(Read&& read) const -> std::optional<std::invoke_result_t<Read&, JsonObjectReader&>> {
        if (is_null()) {
            return std::nullopt;
        }
        return read_object(std::forward<Read>(read));
    }

    // Reads an array, calling `read_element(JsonValueReader)` on each element in order.
    template <typename ReadElement>
    void read_array(ReadElement&& read_element) const;

    // An array with each element read<T>().
    template <typename T>
    std::vector<T> read_array_of() const {
        std::vector<T> out;
        read_array([&out](const JsonValueReader& element) { out.push_back(element.read<T>()); });
        return out;
    }

    // Fails, naming this value: "<document_name>: <pointer> <problem>". For a value of the right JSON type that the
    // reader still can't take, such as the path of a router the document doesn't have.
    template <typename... Args>
    [[noreturn]] void fail(fmt::format_string<Args...> problem, Args&&... args) const {
        throw_error(fmt::format(problem, std::forward<Args>(args)...));
    }

private:
    friend class JsonObjectReader;

    JsonValueReader(const json& value, json_pointer pointer, std::string_view document_name);

    [[noreturn]] void throw_error(const std::string& problem) const;
    // Fails unless the value's JSON type is the one `type_name` names, such as "an object".
    void require_type(bool has_type, std::string_view type_name) const;

    bool read_bool() const;
    std::string read_string() const;
    uint64_t read_unsigned_integer(size_t bits) const;
    int64_t read_signed_integer(size_t bits) const;

    const json* value_;
    json_pointer pointer_;
    std::string_view document_name_;
};

// One object being read. Only read_object makes one, and it fails once the read returns if any key is left untaken.
// Each key is taken once: taking, skipping or checking a key twice fails.
class JsonObjectReader {
public:
    using json = JsonValueReader::json;
    using json_pointer = JsonValueReader::json_pointer;

    JsonObjectReader(const JsonObjectReader&) = delete;
    JsonObjectReader& operator=(const JsonObjectReader&) = delete;

    const json_pointer& pointer() const { return object_.pointer(); }

    // A key the writer always writes. Fails when it's missing. Where the writer has nothing to write, it writes null,
    // so a value that can be absent is read with read_nullable rather than taken as optional.
    JsonValueReader take(std::string_view key);

    template <typename T>
    T take(std::string_view key) {
        return take(key).read<T>();
    }

    // A key the writer leaves out where it doesn't apply, such as torus on a mesh that isn't one. Nullopt when the
    // key is missing.
    std::optional<JsonValueReader> take_optional(std::string_view key);

    // A key the writer derives from what else it writes, which the reader has no use for, such as the manifest's
    // vocabulary. It must still be there.
    void skip(std::string_view key);

    // A key whose value the reader knows, such as the file's version: fails unless the value equals `expected`.
    void check(std::string_view key, const json& expected);

    // Every key not yet taken, in the document's order, calling `read_entry(std::string_view key, JsonValueReader
    // value)` on each. For an object keyed by id, such as meshes keyed M<mesh_id>.
    template <typename ReadEntry>
    void take_each(ReadEntry&& read_entry);

private:
    friend class JsonValueReader;

    explicit JsonObjectReader(const JsonValueReader& object);

    // Fails on any key not taken, naming each by its pointer.
    void finish() const;

    // Marks `key` taken, failing if it already was. The value, or null when the object has no such key.
    const json* take_value(std::string_view key);

    // A reader for `value` at `key`'s pointer. When `key` is missing, `value` is the object, so the reader can still
    // fail naming the key.
    JsonValueReader make_key_reader(const json& value, std::string_view key) const;

    JsonValueReader object_;
    std::set<std::string, std::less<>> taken_keys_;
};

template <typename T>
T JsonValueReader::read() const {
    if constexpr (std::is_same_v<T, bool>) {
        return read_bool();
    } else if constexpr (std::is_same_v<T, std::string>) {
        return read_string();
    } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
        return static_cast<T>(read_unsigned_integer(sizeof(T) * 8));
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        return static_cast<T>(read_signed_integer(sizeof(T) * 8));
    } else {
        static_assert(sizeof(T) == 0, "JsonValueReader reads bool, std::string and integer types");
    }
}

template <typename Read>
auto JsonValueReader::read_object(Read&& read) const {
    JsonObjectReader object(*this);
    if constexpr (std::is_void_v<std::invoke_result_t<Read&, JsonObjectReader&>>) {
        read(object);
        object.finish();
    } else {
        auto result = read(object);
        object.finish();
        return result;
    }
}

template <typename ReadElement>
void JsonValueReader::read_array(ReadElement&& read_element) const {
    require_type(value_->is_array(), "an array");
    for (size_t index = 0; index < value_->size(); ++index) {
        read_element(JsonValueReader((*value_)[index], pointer_ / index, document_name_));
    }
}

template <typename ReadEntry>
void JsonObjectReader::take_each(ReadEntry&& read_entry) {
    for (const auto& entry : object_.value_->items()) {
        if (!taken_keys_.contains(entry.key())) {
            read_entry(std::string_view(entry.key()), take(entry.key()));
        }
    }
}

}  // namespace tt::tt_fabric::visualizer
