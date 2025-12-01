// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Define FLATBUFFERS_LARGE_SIZE before any flatbuffers includes
// This must be defined before flatbuffers.h is included (including in generated headers)
// Define it as 1 to ensure it's treated as defined
#ifndef FLATBUFFERS_LARGE_SIZE
#define FLATBUFFERS_LARGE_SIZE 1
#endif

#include "flatbuffer_file.hpp"

#include <flatbuffers/flatbuffers.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tt-metalium/bfloat16.hpp>
#include <unordered_map>
#include <variant>
#include <vector>

// Include generated FlatBuffer code
#include "ttml_metadata_generated.h"

namespace {

constexpr const char flatbuffer_ext[] = ".flatbuffer";
constexpr size_t flatbuffer_ext_len = sizeof(flatbuffer_ext) - 1;
static_assert(flatbuffer_ext_len == 11);

}  // namespace

namespace ttml::serialization {
FlatBufferFile::~FlatBufferFile() {
}

void FlatBufferFile::put(std::string_view key, bool value) {
    m_data[std::string(key)] = value;
}

void FlatBufferFile::put(std::string_view key, char value) {
    m_data[std::string(key)] = value;
}

void FlatBufferFile::put(std::string_view key, int value) {
    m_data[std::string(key)] = value;
}

void FlatBufferFile::put(std::string_view key, float value) {
    m_data[std::string(key)] = value;
}

void FlatBufferFile::put(std::string_view key, double value) {
    m_data[std::string(key)] = value;
}

void FlatBufferFile::put(std::string_view key, uint32_t value) {
    m_data[std::string(key)] = value;
}

void FlatBufferFile::put(std::string_view key, size_t value) {
    m_data[std::string(key)] = value;
}

void FlatBufferFile::put(std::string_view key, bfloat16 value) {
    m_data[std::string(key)] = value;
}

void FlatBufferFile::put(std::string_view key, std::string_view value) {
    m_data[std::string(key)] = std::string(value);
}

// Overloads for std::span
void FlatBufferFile::put(std::string_view key, std::span<const char> value) {
    m_data[std::string(key)] = std::vector<char>(value.begin(), value.end());
}

void FlatBufferFile::put(std::string_view key, std::span<const uint8_t> value) {
    m_data[std::string(key)] = std::vector<uint8_t>(value.begin(), value.end());
}

void FlatBufferFile::put(std::string_view key, std::span<const int> value) {
    m_data[std::string(key)] = std::vector<int>(value.begin(), value.end());
}

void FlatBufferFile::put(std::string_view key, std::span<const float> value) {
    m_data[std::string(key)] = std::vector<float>(value.begin(), value.end());
}

void FlatBufferFile::put(std::string_view key, std::span<const double> value) {
    m_data[std::string(key)] = std::vector<double>(value.begin(), value.end());
}

void FlatBufferFile::put(std::string_view key, std::span<const uint32_t> value) {
    m_data[std::string(key)] = std::vector<uint32_t>(value.begin(), value.end());
}

void FlatBufferFile::put(std::string_view key, std::span<const bfloat16> value) {
    m_data[std::string(key)] = std::vector<bfloat16>(value.begin(), value.end());
}

void FlatBufferFile::put(std::string_view key, std::span<const std::string> value) {
    m_data[std::string(key)] = std::vector<std::string>(value.begin(), value.end());
}

void FlatBufferFile::put(std::string_view key, const char* value) {
    m_data[std::string(key)] = value;
}

void FlatBufferFile::put(std::string_view key, const ValueType& value) {
    m_data[std::string(key)] = value;
}

// Helper function to build a flatbuffer from a subset of data
std::vector<uint8_t> FlatBufferFile::build_flatbuffer(
    const std::unordered_map<std::string, ValueType>& data_subset) const {
    // Use a smaller initial buffer size to avoid hitting the 2GB limit
    // FlatBuffers has a hard limit of ~2GB due to 32-bit signed offsets
    // If data exceeds this, we need to split it further upstream
    constexpr size_t INITIAL_BUFFER_SIZE = 512UL * 1024 * 1024;  // 512MB initial size
    flatbuffers::FlatBufferBuilder builder(INITIAL_BUFFER_SIZE);
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
                } else if constexpr (std::is_same_v<T, bfloat16>) {
                    uint16_t bf16_bits = std::bit_cast<uint16_t>(val);
                    auto offset = ttml::flatbuffer::CreateBFloat16Value(builder, bf16_bits);
                    union_offset = offset.Union();
                    union_type = ttml::flatbuffer::SerializableType::BFloat16Value;
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
                } else if constexpr (std::is_same_v<T, std::vector<bfloat16>>) {
                    std::vector<uint16_t> uint16_vec;
                    uint16_vec.reserve(val.size());
                    for (bfloat16 bf16 : val) {
                        uint16_vec.push_back(std::bit_cast<uint16_t>(bf16));
                    }
                    auto vec_offset = builder.CreateVector(uint16_vec);
                    auto offset = ttml::flatbuffer::CreateVectorBFloat16(builder, vec_offset);
                    union_offset = offset.Union();
                    union_type = ttml::flatbuffer::SerializableType::VectorBFloat16;
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
std::string FlatBufferFile::get_prefix(std::string_view key) const {
    size_t pos = key.find('/');
    if (pos != std::string::npos) {
        return std::string(key.substr(0, pos));
    }
    return "data";  // Default prefix for keys without '/'
}

// Helper function to group data by prefix and build flatbuffer files
std::vector<std::pair<std::string, std::vector<uint8_t>>> FlatBufferFile::build_flatbuffer_files() const {
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
    // Split large groups to avoid exceeding FlatBuffers' 2GB limit
    std::vector<std::pair<std::string, std::vector<uint8_t>>> flatbuffer_files;
    constexpr size_t MAX_ESTIMATED_SIZE = 1500UL * 1024 * 1024;  // ~1.5GB safety limit

    for (const auto& [prefix, data_subset] : grouped_data) {
        if (data_subset.empty()) {
            continue;  // Skip empty groups
        }

        // Estimate total size of data_subset
        size_t estimated_size = 0;
        for (const auto& [key, value] : data_subset) {
            estimated_size += key.size() + 100;  // Key + overhead
            if (std::holds_alternative<std::vector<uint8_t>>(value)) {
                estimated_size += std::get<std::vector<uint8_t>>(value).size();
            } else if (std::holds_alternative<std::vector<int>>(value)) {
                estimated_size += std::get<std::vector<int>>(value).size() * sizeof(int);
            } else if (std::holds_alternative<std::vector<float>>(value)) {
                estimated_size += std::get<std::vector<float>>(value).size() * sizeof(float);
            } else if (std::holds_alternative<std::vector<double>>(value)) {
                estimated_size += std::get<std::vector<double>>(value).size() * sizeof(double);
            } else if (std::holds_alternative<std::vector<uint32_t>>(value)) {
                estimated_size += std::get<std::vector<uint32_t>>(value).size() * sizeof(uint32_t);
            } else if (std::holds_alternative<std::vector<bfloat16>>(value)) {
                estimated_size += std::get<std::vector<bfloat16>>(value).size() * sizeof(uint16_t);
            } else if (std::holds_alternative<std::string>(value)) {
                estimated_size += std::get<std::string>(value).size();
            }
        }

        // If estimated size is too large, split into chunks
        if (estimated_size > MAX_ESTIMATED_SIZE) {
            std::vector<std::pair<std::string, ValueType>> items(data_subset.begin(), data_subset.end());
            size_t chunk_index = 0;
            size_t current_chunk_size = 0;
            std::unordered_map<std::string, ValueType> chunk;

            for (const auto& [key, value] : items) {
                size_t item_size = key.size() + 100;
                if (std::holds_alternative<std::vector<uint8_t>>(value)) {
                    item_size += std::get<std::vector<uint8_t>>(value).size();
                } else if (std::holds_alternative<std::vector<int>>(value)) {
                    item_size += std::get<std::vector<int>>(value).size() * sizeof(int);
                } else if (std::holds_alternative<std::vector<float>>(value)) {
                    item_size += std::get<std::vector<float>>(value).size() * sizeof(float);
                } else if (std::holds_alternative<std::vector<double>>(value)) {
                    item_size += std::get<std::vector<double>>(value).size() * sizeof(double);
                } else if (std::holds_alternative<std::vector<uint32_t>>(value)) {
                    item_size += std::get<std::vector<uint32_t>>(value).size() * sizeof(uint32_t);
                } else if (std::holds_alternative<std::string>(value)) {
                    item_size += std::get<std::string>(value).size();
                }

                // If adding this item would exceed limit, finalize current chunk
                if (!chunk.empty() && current_chunk_size + item_size > MAX_ESTIMATED_SIZE) {
                    auto chunk_data = build_flatbuffer(chunk);
                    if (!chunk_data.empty()) {
                        flatbuffer_files.emplace_back(
                            fmt::format("{}_cunk{}", prefix, chunk_index), std::move(chunk_data));
                        chunk_index++;
                    }
                    chunk.clear();
                    current_chunk_size = 0;
                }

                chunk[key] = value;
                current_chunk_size += item_size;
            }

            // Finalize last chunk
            if (!chunk.empty()) {
                auto chunk_data = build_flatbuffer(chunk);
                if (!chunk_data.empty()) {
                    flatbuffer_files.emplace_back(
                        fmt::format("{}_chunk{}", prefix, chunk_index), std::move(chunk_data));
                }
            }
        } else {
            // Size is acceptable, build normally
            auto flatbuffer_data = build_flatbuffer(data_subset);
            if (flatbuffer_data.empty()) {
                // This shouldn't happen if data_subset is not empty, but check anyway
                continue;
            }
            flatbuffer_files.emplace_back(fmt::format("{}{}", prefix, flatbuffer_ext), std::move(flatbuffer_data));
        }
    }

    return flatbuffer_files;
}

void FlatBufferFile::do_serialize(
    std::string_view dirname, const std::vector<std::pair<std::string, std::vector<uint8_t>>>& flatbuffer_files) {
    // Extract directory from the command-line dirname
    std::filesystem::path dirname_path(dirname);
    if (dirname_path.empty()) {
        dirname_path = std::filesystem::current_path();
    }

    // Write individual files
    std::string base_filename = dirname_path.string();
    if (base_filename.size() >= flatbuffer_ext_len &&
        base_filename.substr(base_filename.size() - flatbuffer_ext_len) == flatbuffer_ext) {
        base_filename = base_filename.substr(0, base_filename.size() - flatbuffer_ext_len);
    }

    // Write flatbuffer metadata files
    for (const auto& [file_name, flatbuffer_data] : flatbuffer_files) {
        // Construct individual filename: base_filename_prefix.flatbuffer
        const std::string individual_filename = fmt::format("{}_{}", base_filename, file_name);

        {
            std::ofstream file(individual_filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error(fmt::format("Failed to open file for writing: {}", individual_filename));
            }

            file.write(reinterpret_cast<const char*>(flatbuffer_data.data()), flatbuffer_data.size());
            if (!file.good()) {
                throw std::runtime_error(
                    fmt::format("Failed to write flatbuffer data to file: {}", individual_filename));
            }
        }  // File automatically flushed and closed here
    }
}

void FlatBufferFile::serialize(std::string_view filename) {
    // Set output directory first (in case tensors are written after this call)
    // This ensures tensor files written before serialize are in the right place

    auto flatbuffer_files = build_flatbuffer_files();

    // If no files were created (empty serializer), create an empty file with default "data" prefix
    if (flatbuffer_files.empty()) {
        if (!m_data.empty()) {
            throw std::runtime_error("Failed to create any flatbuffer files during serialization");
        }
        // Create an empty flatbuffer file for empty serializer
        std::unordered_map<std::string, ValueType> empty_data;
        auto empty_flatbuffer = build_flatbuffer(empty_data);
        flatbuffer_files.emplace_back(fmt::format("data{}", flatbuffer_ext), std::move(empty_flatbuffer));
    }

    do_serialize(filename, flatbuffer_files);
}

// Helper function to deserialize a flatbuffer and merge into m_data with prefix
void FlatBufferFile::deserialize_flatbuffer(const std::vector<uint8_t>& buffer, std::string_view prefix) {
    if (buffer.empty()) {
        return;
    }

    // Verify and get the root
    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!ttml::flatbuffer::VerifyTTMLDataBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data");
    }

    auto* ttml_data = ttml::flatbuffer::GetTTMLData(buffer.data());

    if (!ttml_data || !ttml_data->pairs()) {
        throw std::runtime_error("Invalid FlatBuffer structure");
    }

    for (const auto* kv_pair : *ttml_data->pairs()) {
        if (!kv_pair || !kv_pair->key()) {
            continue;
        }

        std::string suffix(kv_pair->key()->c_str());
        std::string key;
        if (prefix.empty() || prefix == "data") {
            // Keys without '/' were stored with default prefix "data" or empty prefix
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
            case ttml::flatbuffer::SerializableType::BFloat16Value: {
                auto* val = kv_pair->value_as_BFloat16Value();
                if (val) {
                    uint16_t bf16_bits = val->value();
                    m_data[key] = std::bit_cast<bfloat16>(bf16_bits);
                }
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
            case ttml::flatbuffer::SerializableType::VectorBFloat16: {
                auto* val = kv_pair->value_as_VectorBFloat16();
                if (val && val->values()) {
                    std::vector<bfloat16> result;
                    result.reserve(val->values()->size());
                    for (uint16_t bf16_bits : *val->values()) {
                        result.push_back(std::bit_cast<bfloat16>(bf16_bits));
                    }
                    m_data[key] = result;
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

void FlatBufferFile::do_deserialize(std::string_view filename) {
    // Read individual flatbuffer files
    const auto path = std::filesystem::path(filename);
    std::string base_path = path.parent_path().string();
    if (base_path.empty()) {
        base_path = ".";
    }

    // Extract just the filename part (without extension) for pattern matching
    std::string base_name = path.string();
    // Remove .flatbuffer extension if present
    if (base_name.size() >= flatbuffer_ext_len &&
        base_name.substr(base_name.size() - flatbuffer_ext_len) == flatbuffer_ext) {
        base_name = base_name.substr(0, base_name.size() - flatbuffer_ext_len);
    }
    const std::string base_filename = std::filesystem::path(base_name).filename().string();
    const std::string search_pattern = fmt::format("{}_", base_filename);

    bool found_any_files = false;

    // Search for files matching the pattern: base_name_*.flatbuffer
    try {
        for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            std::string entry_name = entry.path().filename().string();

            // Check if filename starts with search_pattern and ends with .flatbuffer
            if (entry_name.size() >= search_pattern.size() + flatbuffer_ext_len &&
                entry_name.substr(0, search_pattern.size()) == search_pattern &&
                entry_name.substr(entry_name.size() - flatbuffer_ext_len) == flatbuffer_ext) {
                // Extract prefix from filename: remove base_name_ prefix and .flatbuffer extension
                std::string prefix = entry_name.substr(
                    search_pattern.size(), entry_name.size() - search_pattern.size() - flatbuffer_ext_len);

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
            fmt::format("No flatbuffer files found matching pattern: {}_*.flatbuffer in {}", base_filename, base_path));
    }
}

void FlatBufferFile::deserialize(std::string_view filename) {
    // Clear existing data
    m_data.clear();

    do_deserialize(filename);
}

// Methods to get values
bool FlatBufferFile::get_bool(std::string_view key) const {
    return get_value<bool>(key);
}

char FlatBufferFile::get_char(std::string_view key) const {
    return get_value<char>(key);
}

int FlatBufferFile::get_int(std::string_view key) const {
    return get_value<int>(key);
}

float FlatBufferFile::get_float(std::string_view key) const {
    return get_value<float>(key);
}

double FlatBufferFile::get_double(std::string_view key) const {
    return get_value<double>(key);
}

uint32_t FlatBufferFile::get_uint32(std::string_view key) const {
    return get_value<uint32_t>(key);
}

size_t FlatBufferFile::get_size_t(std::string_view key) const {
    return get_value<size_t>(key);
}

bfloat16 FlatBufferFile::get_bfloat16(std::string_view key) const {
    return get_value<bfloat16>(key);
}

std::string FlatBufferFile::get_string(std::string_view key) const {
    return get_value<std::string>(key);
}

std::vector<uint8_t> FlatBufferFile::get_vector_uint8(std::string_view key) const {
    return get_value<std::vector<uint8_t>>(key);
}

ValueType FlatBufferFile::get_value_type(std::string_view key) const {
    auto it = m_data.find(std::string(key));
    if (it != m_data.end()) {
        return it->second;
    } else {
        throw std::runtime_error(fmt::format("Key not found: {}", key));
    }
}

}  // namespace ttml::serialization
