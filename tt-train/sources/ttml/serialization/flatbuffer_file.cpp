// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer_file.hpp"

#include <flatbuffers/flatbuffers.h>
#include <fmt/format.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// Include generated FlatBuffer code
#include "tar_reader.hpp"
#include "tar_writer.hpp"
#include "ttml_tensor_generated.h"

namespace {

// Compile-time strlen for constexpr strings (consteval ensures compile-time evaluation)
consteval size_t constexpr_strlen(const char* str) {
    size_t len = 0;
    while (str[len] != '\0') {
        ++len;
    }
    return len;
}

}  // namespace

namespace ttml::serialization {

class FlatBufferFile::Impl {
public:
    explicit Impl(bool use_tarball) : m_use_tarball(use_tarball) {
    }

    void set_use_tarball(bool use_tarball) {
        m_use_tarball = use_tarball;
    }

    bool get_use_tarball() const {
        return m_use_tarball;
    }

    // Methods to store different types
    void put(std::string_view key, bool value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, char value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, int value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, float value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, double value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, uint32_t value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, size_t value) {
        m_data[std::string(key)] = value;
    }

    void put(std::string_view key, std::string_view value) {
        m_data[std::string(key)] = std::string(value);
    }

    // Overloads for std::span
    void put(std::string_view key, std::span<const char> value) {
        m_data[std::string(key)] = std::vector<char>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const uint8_t> value) {
        m_data[std::string(key)] = std::vector<uint8_t>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const int> value) {
        m_data[std::string(key)] = std::vector<int>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const float> value) {
        m_data[std::string(key)] = std::vector<float>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const double> value) {
        m_data[std::string(key)] = std::vector<double>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const uint32_t> value) {
        m_data[std::string(key)] = std::vector<uint32_t>(value.begin(), value.end());
    }

    void put(std::string_view key, std::span<const std::string> value) {
        m_data[std::string(key)] = std::vector<std::string>(value.begin(), value.end());
    }

    void put(std::string_view key, const ValueType& value) {
        m_data[std::string(key)] = value;
    }

    // Helper function to build a flatbuffer from a subset of data
    std::vector<uint8_t> build_flatbuffer(const std::unordered_map<std::string, ValueType>& data_subset) const {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<flatbuffers::Offset<ttml::flatbuffer::KeyValuePair>> kv_pairs;

        for (const auto& [key, value] : data_subset) {
            auto key_str = builder.CreateString(key);
            flatbuffers::Offset<void> union_offset;
            ttml::flatbuffer::SerializableType union_type;

            std::visit(
                [&builder, &union_offset, &union_type](const auto& val) {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (std::is_same_v<T, bool>) {
                        auto offset = ttml::flatbuffer::CreateBoolValue(builder, val);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::BoolValue;
                    } else if constexpr (std::is_same_v<T, char>) {
                        auto offset = ttml::flatbuffer::CreateCharValue(builder, static_cast<int8_t>(val));
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::CharValue;
                    } else if constexpr (std::is_same_v<T, int>) {
                        auto offset = ttml::flatbuffer::CreateIntValue(builder, val);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::IntValue;
                    } else if constexpr (std::is_same_v<T, float>) {
                        auto offset = ttml::flatbuffer::CreateFloatValue(builder, val);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::FloatValue;
                    } else if constexpr (std::is_same_v<T, double>) {
                        auto offset = ttml::flatbuffer::CreateDoubleValue(builder, val);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::DoubleValue;
                    } else if constexpr (std::is_same_v<T, uint32_t>) {
                        auto offset = ttml::flatbuffer::CreateUInt32Value(builder, val);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::UInt32Value;
                    } else if constexpr (std::is_same_v<T, size_t>) {
                        auto offset = ttml::flatbuffer::CreateSizeTValue(builder, static_cast<uint64_t>(val));
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::SizeTValue;
                    } else if constexpr (std::is_same_v<T, std::string>) {
                        auto str_offset = builder.CreateString(val);
                        auto offset = ttml::flatbuffer::CreateStringValue(builder, str_offset);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::StringValue;
                    } else if constexpr (std::is_same_v<T, std::vector<char>>) {
                        std::vector<int8_t> int8_vec(val.begin(), val.end());
                        auto vec_offset = builder.CreateVector(int8_vec);
                        auto offset = ttml::flatbuffer::CreateVectorChar(builder, vec_offset);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::VectorChar;
                    } else if constexpr (std::is_same_v<T, std::vector<int>>) {
                        auto vec_offset = builder.CreateVector(val);
                        auto offset = ttml::flatbuffer::CreateVectorInt(builder, vec_offset);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::VectorInt;
                    } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                        auto vec_offset = builder.CreateVector(val);
                        auto offset = ttml::flatbuffer::CreateVectorFloat(builder, vec_offset);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::VectorFloat;
                    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                        auto vec_offset = builder.CreateVector(val);
                        auto offset = ttml::flatbuffer::CreateVectorDouble(builder, vec_offset);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::VectorDouble;
                    } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                        auto vec_offset = builder.CreateVector(val);
                        auto offset = ttml::flatbuffer::CreateVectorUInt8(builder, vec_offset);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::VectorUInt8;
                    } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
                        auto vec_offset = builder.CreateVector(val);
                        auto offset = ttml::flatbuffer::CreateVectorUInt32(builder, vec_offset);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::VectorUInt32;
                    } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                        std::vector<flatbuffers::Offset<flatbuffers::String>> string_offsets;
                        for (const auto& s : val) {
                            string_offsets.push_back(builder.CreateString(s));
                        }
                        auto vec_offset = builder.CreateVector(string_offsets);
                        auto offset = ttml::flatbuffer::CreateVectorString(builder, vec_offset);
                        union_offset = offset.Union();
                        union_type = ttml::flatbuffer::SerializableType::VectorString;
                    }
                },
                value);

            auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(builder, key_str, union_type, union_offset);
            kv_pairs.push_back(kv_pair);
        }

        auto pairs_vector = builder.CreateVector(kv_pairs);
        auto root = ttml::flatbuffer::CreateTTMLData(builder, pairs_vector);
        builder.Finish(root);

        // Copy flatbuffer data to vector
        const void* flatbuffer_data = builder.GetBufferPointer();
        size_t flatbuffer_size = builder.GetSize();
        return std::vector<uint8_t>(
            reinterpret_cast<const uint8_t*>(flatbuffer_data),
            reinterpret_cast<const uint8_t*>(flatbuffer_data) + flatbuffer_size);
    }

    // Extract top-level prefix from a key (everything before first '/')
    std::string get_prefix(std::string_view key) const {
        size_t pos = key.find('/');
        if (pos != std::string::npos) {
            return std::string(key.substr(0, pos));
        }
        return "data";  // Default prefix for keys without '/'
    }

    // Helper function to get base filename without extension
    std::string get_base_filename(std::string_view filename) const {
        std::string fname(filename);
        // Remove .tar extension if present
        if (fname.size() >= 4 && fname.substr(fname.size() - 4) == ".tar") {
            return fname.substr(0, fname.size() - 4);
        }
        return fname;
    }

    // Helper function to group data by prefix and build flatbuffer files
    std::vector<std::pair<std::string, std::vector<uint8_t>>> build_flatbuffer_files() const {
        // Group data by top-level prefix
        std::unordered_map<std::string, std::unordered_map<std::string, ValueType>> grouped_data;

        for (const auto& [key, value] : m_data) {
            std::string prefix = get_prefix(key);
            std::string suffix = key;
            size_t pos = key.find('/');
            if (pos != std::string::npos) {
                suffix = key.substr(pos + 1);  // Remove prefix from key
            }
            grouped_data[prefix][suffix] = value;
        }

        // Build a flatbuffer file for each prefix group
        std::vector<std::pair<std::string, std::vector<uint8_t>>> flatbuffer_files;

        for (const auto& [prefix, data_subset] : grouped_data) {
            if (data_subset.empty()) {
                continue;  // Skip empty groups
            }
            auto flatbuffer_data = build_flatbuffer(data_subset);
            if (flatbuffer_data.empty()) {
                // This shouldn't happen if data_subset is not empty, but check anyway
                continue;
            }
            std::string file_name = prefix + ".flatbuffer";
            flatbuffer_files.emplace_back(file_name, std::move(flatbuffer_data));
        }

        return flatbuffer_files;
    }

    // Serialization method - tarball version
    void serialize_tarball(
        std::string_view filename, const std::vector<std::pair<std::string, std::vector<uint8_t>>>& flatbuffer_files) {
        // Create tarball in memory with multiple files
        TarWriter tar_writer;

        for (const auto& [file_name, flatbuffer_data] : flatbuffer_files) {
            tar_writer.add_file(file_name, flatbuffer_data.data(), flatbuffer_data.size());
        }

        // Write tarball to file (single write operation)
        tar_writer.write_to_file(std::string(filename));
    }

    // Serialization method - non-tarball version
    void serialize_non_tarball(
        std::string_view filename, const std::vector<std::pair<std::string, std::vector<uint8_t>>>& flatbuffer_files) {
        // Write individual files
        std::string base_filename = get_base_filename(filename);

        for (const auto& [file_name, flatbuffer_data] : flatbuffer_files) {
            // Construct individual filename: base_filename_prefix.flatbuffer
            std::string individual_filename = base_filename + "_" + file_name;

            std::ofstream file(individual_filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error(fmt::format("Failed to open file for writing: {}", individual_filename));
            }

            file.write(reinterpret_cast<const char*>(flatbuffer_data.data()), flatbuffer_data.size());
            if (!file.good()) {
                throw std::runtime_error(
                    fmt::format("Failed to write flatbuffer data to file: {}", individual_filename));
            }
        }
    }

    // Serialization method - dispatches to tarball or non-tarball version
    void serialize(std::string_view filename) {
        auto flatbuffer_files = build_flatbuffer_files();

        // Ensure we have at least one file (should always be true if m_data is not empty)
        if (flatbuffer_files.empty() && !m_data.empty()) {
            throw std::runtime_error("Failed to create any flatbuffer files during serialization");
        }

        if (m_use_tarball) {
            serialize_tarball(filename, flatbuffer_files);
        } else {
            serialize_non_tarball(filename, flatbuffer_files);
        }
    }

    // Helper function to deserialize a flatbuffer and merge into m_data with prefix
    void deserialize_flatbuffer(const std::vector<uint8_t>& buffer, std::string_view prefix) {
        if (buffer.empty()) {
            return;
        }

        // Verify and get the root
        flatbuffers::Verifier verifier(buffer.data(), buffer.size());
        if (!ttml::flatbuffer::VerifyTTMLDataBuffer(verifier)) {
            throw std::runtime_error("Invalid FlatBuffer data in tarball");
        }

        auto* ttml_data = ttml::flatbuffer::GetTTMLData(buffer.data());

        if (!ttml_data || !ttml_data->pairs()) {
            throw std::runtime_error("Invalid FlatBuffer structure");
        }

        // Deserialize each key-value pair and add prefix
        // Note: If prefix is "data" (default for keys without '/') or empty (non-tarball mode),
        // don't add it back to preserve the original key structure
        for (const auto* kv_pair : *ttml_data->pairs()) {
            if (!kv_pair || !kv_pair->key()) {
                continue;
            }

            std::string suffix(kv_pair->key()->c_str());
            std::string key;
            if (prefix.empty() || prefix == "data") {
                // Keys without '/' were stored with default prefix "data" or empty prefix (non-tarball)
                // Don't add prefix back to preserve original key structure
                key = suffix;
            } else {
                // Keys with '/' were stored with their actual prefix
                key = std::string(prefix) + "/" + suffix;
            }

            switch (kv_pair->value_type()) {
                case ttml::flatbuffer::SerializableType::BoolValue: {
                    auto* val = kv_pair->value_as_BoolValue();
                    if (val)
                        m_data[key] = val->value();
                    break;
                }
                case ttml::flatbuffer::SerializableType::CharValue: {
                    auto* val = kv_pair->value_as_CharValue();
                    if (val)
                        m_data[key] = static_cast<char>(val->value());
                    break;
                }
                case ttml::flatbuffer::SerializableType::IntValue: {
                    auto* val = kv_pair->value_as_IntValue();
                    if (val)
                        m_data[key] = val->value();
                    break;
                }
                case ttml::flatbuffer::SerializableType::FloatValue: {
                    auto* val = kv_pair->value_as_FloatValue();
                    if (val)
                        m_data[key] = val->value();
                    break;
                }
                case ttml::flatbuffer::SerializableType::DoubleValue: {
                    auto* val = kv_pair->value_as_DoubleValue();
                    if (val)
                        m_data[key] = val->value();
                    break;
                }
                case ttml::flatbuffer::SerializableType::UInt32Value: {
                    auto* val = kv_pair->value_as_UInt32Value();
                    if (val)
                        m_data[key] = val->value();
                    break;
                }
                case ttml::flatbuffer::SerializableType::SizeTValue: {
                    auto* val = kv_pair->value_as_SizeTValue();
                    if (val)
                        m_data[key] = static_cast<size_t>(val->value());
                    break;
                }
                case ttml::flatbuffer::SerializableType::StringValue: {
                    auto* val = kv_pair->value_as_StringValue();
                    if (val && val->value()) {
                        m_data[key] = std::string(val->value()->c_str());
                    }
                    break;
                }
                case ttml::flatbuffer::SerializableType::VectorChar: {
                    auto* val = kv_pair->value_as_VectorChar();
                    if (val && val->values()) {
                        std::vector<char> result;
                        result.reserve(val->values()->size());
                        for (int8_t v : *val->values()) {
                            result.push_back(static_cast<char>(v));
                        }
                        m_data[key] = result;
                    }
                    break;
                }
                case ttml::flatbuffer::SerializableType::VectorInt: {
                    auto* val = kv_pair->value_as_VectorInt();
                    if (val && val->values()) {
                        m_data[key] = std::vector<int>(val->values()->begin(), val->values()->end());
                    }
                    break;
                }
                case ttml::flatbuffer::SerializableType::VectorFloat: {
                    auto* val = kv_pair->value_as_VectorFloat();
                    if (val && val->values()) {
                        m_data[key] = std::vector<float>(val->values()->begin(), val->values()->end());
                    }
                    break;
                }
                case ttml::flatbuffer::SerializableType::VectorDouble: {
                    auto* val = kv_pair->value_as_VectorDouble();
                    if (val && val->values()) {
                        m_data[key] = std::vector<double>(val->values()->begin(), val->values()->end());
                    }
                    break;
                }
                case ttml::flatbuffer::SerializableType::VectorUInt8: {
                    auto* val = kv_pair->value_as_VectorUInt8();
                    if (val && val->values()) {
                        m_data[key] = std::vector<uint8_t>(val->values()->begin(), val->values()->end());
                    }
                    break;
                }
                case ttml::flatbuffer::SerializableType::VectorUInt32: {
                    auto* val = kv_pair->value_as_VectorUInt32();
                    if (val && val->values()) {
                        m_data[key] = std::vector<uint32_t>(val->values()->begin(), val->values()->end());
                    }
                    break;
                }
                case ttml::flatbuffer::SerializableType::VectorString: {
                    auto* val = kv_pair->value_as_VectorString();
                    if (val && val->values()) {
                        std::vector<std::string> result;
                        result.reserve(val->values()->size());
                        for (const auto* s : *val->values()) {
                            if (s) {
                                result.push_back(s->str());
                            }
                        }
                        m_data[key] = result;
                    }
                    break;
                }
                default: throw std::runtime_error(fmt::format("Unknown serializable type for key: {}", key));
            }
        }
    }

    // Deserialization method - tarball version
    void deserialize_tarball(std::string_view filename) {
        // Read tarball
        TarReader tar_reader;
        try {
            tar_reader.read_from_file(std::string(filename));
        } catch (const std::runtime_error& e) {
            // Re-throw tar reading errors (invalid tarball, empty file, etc.)
            throw;
        }

        // Get list of all files in tarball
        auto files = tar_reader.list_files();

        // If tarball is empty, that's okay only if it was intentionally empty
        // (i.e., serialized with no data). But if the file exists and is not empty,
        // an empty file list might indicate an invalid tarball.
        // For now, allow empty tarballs (they represent empty serialized data)
        if (files.empty()) {
            return;
        }

        // Deserialize each flatbuffer file
        bool found_flatbuffer = false;
        constexpr const char* flatbuffer_ext = ".flatbuffer";
        constexpr size_t flatbuffer_ext_len = constexpr_strlen(flatbuffer_ext);

        for (const auto& file_name : files) {
            // Extract prefix from filename (remove .flatbuffer extension)
            // Filenames should already be trimmed by TarReader, but be defensive
            std::string clean_name = file_name;
            while (!clean_name.empty() && (clean_name.back() == ' ' || clean_name.back() == '\0')) {
                clean_name.pop_back();
            }

            // Check if filename ends with .flatbuffer using find
            size_t ext_pos = clean_name.rfind(flatbuffer_ext);
            if (ext_pos != std::string::npos && ext_pos + flatbuffer_ext_len == clean_name.size()) {
                found_flatbuffer = true;
                std::string prefix = clean_name.substr(0, ext_pos);

                // Read and deserialize the flatbuffer file (use original filename for lookup)
                std::vector<uint8_t> buffer = tar_reader.get_file(file_name);
                deserialize_flatbuffer(buffer, prefix);
            }
        }

        if (!found_flatbuffer) {
            // Provide more helpful error message
            std::string file_list;
            for (const auto& f : files) {
                if (!file_list.empty())
                    file_list += ", ";
                file_list += "'" + f + "'";
            }
            throw std::runtime_error(
                "Tarball does not contain any .flatbuffer files: " + std::string(filename) +
                ". Files found: " + (files.empty() ? "(empty)" : file_list));
        }
    }

    // Deserialization method - non-tarball version
    void deserialize_non_tarball(std::string_view filename) {
        // Read individual flatbuffer files
        std::string base_filename = get_base_filename(filename);
        std::string base_path = std::filesystem::path(filename).parent_path().string();
        if (base_path.empty()) {
            base_path = ".";
        }

        std::string base_name = std::filesystem::path(base_filename).filename().string();
        std::string search_pattern = base_name + "_";
        constexpr const char* flatbuffer_ext = ".flatbuffer";

        bool found_any_files = false;

        // Search for files matching the pattern: base_name_*.flatbuffer
        try {
            for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                std::string entry_name = entry.path().filename().string();

                // Check if filename starts with search_pattern and ends with .flatbuffer
                if (entry_name.size() >= search_pattern.size() + constexpr_strlen(flatbuffer_ext) &&
                    entry_name.substr(0, search_pattern.size()) == search_pattern &&
                    entry_name.substr(entry_name.size() - constexpr_strlen(flatbuffer_ext)) == flatbuffer_ext) {
                    // Extract prefix from filename: remove base_name_ prefix and .flatbuffer extension
                    std::string prefix = entry_name.substr(
                        search_pattern.size(),
                        entry_name.size() - search_pattern.size() - constexpr_strlen(flatbuffer_ext));

                    // Read the file
                    std::ifstream file(entry.path(), std::ios::binary | std::ios::ate);
                    if (!file.is_open()) {
                        continue;  // Skip files we can't open
                    }

                    size_t file_size = file.tellg();
                    file.seekg(0, std::ios::beg);

                    std::vector<uint8_t> buffer(file_size);
                    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
                    if (file.good() || file_size == 0) {
                        deserialize_flatbuffer(buffer, prefix);
                        found_any_files = true;
                    }
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            throw std::runtime_error(fmt::format("Failed to read directory for flatbuffer files: {}", e.what()));
        }

        if (!found_any_files) {
            throw std::runtime_error(
                fmt::format("No flatbuffer files found matching pattern: {}_*.flatbuffer in {}", base_name, base_path));
        }
    }

    // Deserialization method - dispatches to tarball or non-tarball version
    void deserialize(std::string_view filename) {
        // Clear existing data
        m_data.clear();

        if (m_use_tarball) {
            deserialize_tarball(filename);
        } else {
            deserialize_non_tarball(filename);
        }
    }

    // Methods to get values
    bool get_bool(std::string_view key) const {
        return get_value<bool>(key);
    }

    char get_char(std::string_view key) const {
        return get_value<char>(key);
    }

    int get_int(std::string_view key) const {
        return get_value<int>(key);
    }

    float get_float(std::string_view key) const {
        return get_value<float>(key);
    }

    double get_double(std::string_view key) const {
        return get_value<double>(key);
    }

    uint32_t get_uint32(std::string_view key) const {
        return get_value<uint32_t>(key);
    }

    size_t get_size_t(std::string_view key) const {
        return get_value<size_t>(key);
    }

    std::string get_string(std::string_view key) const {
        return get_value<std::string>(key);
    }

    std::vector<char> get_vector_char(std::string_view key) const {
        return get_value<std::vector<char>>(key);
    }

    std::vector<uint8_t> get_vector_uint8(std::string_view key) const {
        return get_value<std::vector<uint8_t>>(key);
    }

    std::vector<int> get_vector_int(std::string_view key) const {
        return get_value<std::vector<int>>(key);
    }

    std::vector<float> get_vector_float(std::string_view key) const {
        return get_value<std::vector<float>>(key);
    }

    std::vector<double> get_vector_double(std::string_view key) const {
        return get_value<std::vector<double>>(key);
    }

    std::vector<uint32_t> get_vector_uint32(std::string_view key) const {
        return get_value<std::vector<uint32_t>>(key);
    }

    std::vector<std::string> get_vector_string(std::string_view key) const {
        return get_value<std::vector<std::string>>(key);
    }

    ValueType get_value_type(std::string_view key) const {
        auto it = m_data.find(std::string(key));
        if (it != m_data.end()) {
            return it->second;
        } else {
            throw std::runtime_error(fmt::format("Key not found: {}", key));
        }
    }

private:
    std::unordered_map<std::string, ValueType> m_data;
    bool m_use_tarball;

    // Helper function to get value from m_data
    template <typename T>
    T get_value(std::string_view key) const {
        auto it = m_data.find(std::string(key));
        if (it != m_data.end()) {
            if (const auto* pval = std::get_if<T>(&(it->second))) {
                return *pval;
            } else {
                throw std::runtime_error(fmt::format("Type mismatch for key: {}", key));
            }
        } else {
            throw std::runtime_error(fmt::format("Key not found: {}", key));
        }
    }
};

FlatBufferFile::FlatBufferFile(bool use_tarball) : m_impl(std::make_unique<Impl>(use_tarball)) {
}

FlatBufferFile::~FlatBufferFile() {
    // Destructor must be defined here (not defaulted) because Impl is forward-declared
    // and the compiler needs to see the full definition of Impl to destroy unique_ptr
}

FlatBufferFile::FlatBufferFile(FlatBufferFile&&) noexcept = default;

void FlatBufferFile::put(std::string_view key, bool value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, char value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, int value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, float value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, double value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, uint32_t value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, size_t value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, std::string_view value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, const char* value) {
    put(key, std::string_view(value));
}

void FlatBufferFile::put(std::string_view key, std::span<const char> value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, std::span<const uint8_t> value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, std::span<const int> value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, std::span<const float> value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, std::span<const double> value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, std::span<const uint32_t> value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, std::span<const std::string> value) {
    m_impl->put(key, value);
}

void FlatBufferFile::put(std::string_view key, const ValueType& value) {
    m_impl->put(key, value);
}

void FlatBufferFile::set_use_tarball(bool use_tarball) {
    m_impl->set_use_tarball(use_tarball);
}

bool FlatBufferFile::get_use_tarball() const {
    return m_impl->get_use_tarball();
}

void FlatBufferFile::serialize(std::string_view filename) {
    m_impl->serialize(filename);
}

void FlatBufferFile::deserialize(std::string_view filename) {
    m_impl->deserialize(filename);
}

bool FlatBufferFile::get_bool(std::string_view key) const {
    return m_impl->get_bool(key);
}

char FlatBufferFile::get_char(std::string_view key) const {
    return m_impl->get_char(key);
}

int FlatBufferFile::get_int(std::string_view key) const {
    return m_impl->get_int(key);
}

float FlatBufferFile::get_float(std::string_view key) const {
    return m_impl->get_float(key);
}

double FlatBufferFile::get_double(std::string_view key) const {
    return m_impl->get_double(key);
}

uint32_t FlatBufferFile::get_uint32(std::string_view key) const {
    return m_impl->get_uint32(key);
}

size_t FlatBufferFile::get_size_t(std::string_view key) const {
    return m_impl->get_size_t(key);
}

std::string FlatBufferFile::get_string(std::string_view key) const {
    return m_impl->get_string(key);
}

std::vector<char> FlatBufferFile::get_vector_char(std::string_view key) const {
    return m_impl->get_vector_char(key);
}

std::vector<uint8_t> FlatBufferFile::get_vector_uint8(std::string_view key) const {
    return m_impl->get_vector_uint8(key);
}

std::vector<int> FlatBufferFile::get_vector_int(std::string_view key) const {
    return m_impl->get_vector_int(key);
}

std::vector<float> FlatBufferFile::get_vector_float(std::string_view key) const {
    return m_impl->get_vector_float(key);
}

std::vector<double> FlatBufferFile::get_vector_double(std::string_view key) const {
    return m_impl->get_vector_double(key);
}

std::vector<uint32_t> FlatBufferFile::get_vector_uint32(std::string_view key) const {
    return m_impl->get_vector_uint32(key);
}

std::vector<std::string> FlatBufferFile::get_vector_string(std::string_view key) const {
    return m_impl->get_vector_string(key);
}

ValueType FlatBufferFile::get_value_type(std::string_view key) const {
    return m_impl->get_value_type(key);
}

}  // namespace ttml::serialization
