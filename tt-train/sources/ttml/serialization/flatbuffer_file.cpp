// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Define FLATBUFFERS_LARGE_SIZE before any flatbuffers includes
// This must be defined before flatbuffers.h is included (including in generated headers)
#ifndef FLATBUFFERS_LARGE_SIZE
#define FLATBUFFERS_LARGE_SIZE 1
#endif

#include "flatbuffer_file.hpp"

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tt-metalium/bfloat16.hpp>
#include <ttnn/tensor/serialization.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

// Include generated FlatBuffer code
#include "ttml_metadata_generated.h"

namespace {

template <typename T, typename... Ts>
static constexpr bool is_one_of_v = (std::is_same_v<T, Ts> || ...);

constexpr auto DOT_TENSORBIN = ".tensorbin";

}  // namespace

namespace ttml::serialization {
FlatBufferFile::~FlatBufferFile() {
    // Builder will be cleared automatically
}

FlatBufferFile::FlatBufferFile(FlatBufferFile&& other) noexcept :
    m_data(std::move(other.m_data)),
    m_tensors(std::move(other.m_tensors)),
    m_builder(std::move(other.m_builder)),
    m_pairs(std::move(other.m_pairs)) {
}

void FlatBufferFile::put(std::string_view key, bool value) {
    // Build directly into flatbuffer - no need to store in m_data
    auto key_offset = m_builder.CreateString(key);
    auto bool_val = ttml::flatbuffer::CreateBoolValue(m_builder, value);
    auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(
        m_builder, key_offset, ttml::flatbuffer::SerializableType::BoolValue, bool_val.Union());
    m_pairs.push_back(kv_pair);
}

void FlatBufferFile::put(std::string_view key, char value) {
    auto key_offset = m_builder.CreateString(key);
    auto char_val = ttml::flatbuffer::CreateCharValue(m_builder, static_cast<int8_t>(value));
    auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(
        m_builder, key_offset, ttml::flatbuffer::SerializableType::CharValue, char_val.Union());
    m_pairs.push_back(kv_pair);
}

void FlatBufferFile::put(std::string_view key, int value) {
    auto key_offset = m_builder.CreateString(key);
    auto int_val = ttml::flatbuffer::CreateIntValue(m_builder, value);
    auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(
        m_builder, key_offset, ttml::flatbuffer::SerializableType::IntValue, int_val.Union());
    m_pairs.push_back(kv_pair);
}

void FlatBufferFile::put(std::string_view key, float value) {
    auto key_offset = m_builder.CreateString(key);
    auto float_val = ttml::flatbuffer::CreateFloatValue(m_builder, value);
    auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(
        m_builder, key_offset, ttml::flatbuffer::SerializableType::FloatValue, float_val.Union());
    m_pairs.push_back(kv_pair);
}

void FlatBufferFile::put(std::string_view key, double value) {
    auto key_offset = m_builder.CreateString(key);
    auto double_val = ttml::flatbuffer::CreateDoubleValue(m_builder, value);
    auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(
        m_builder, key_offset, ttml::flatbuffer::SerializableType::DoubleValue, double_val.Union());
    m_pairs.push_back(kv_pair);
}

void FlatBufferFile::put(std::string_view key, uint32_t value) {
    auto key_offset = m_builder.CreateString(key);
    auto uint32_val = ttml::flatbuffer::CreateUInt32Value(m_builder, value);
    auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(
        m_builder, key_offset, ttml::flatbuffer::SerializableType::UInt32Value, uint32_val.Union());
    m_pairs.push_back(kv_pair);
}

void FlatBufferFile::put(std::string_view key, size_t value) {
    auto key_offset = m_builder.CreateString(key);
    auto size_t_val = ttml::flatbuffer::CreateSizeTValue(m_builder, static_cast<uint64_t>(value));
    auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(
        m_builder, key_offset, ttml::flatbuffer::SerializableType::SizeTValue, size_t_val.Union());
    m_pairs.push_back(kv_pair);
}

void FlatBufferFile::put(std::string_view key, bfloat16 value) {
    auto key_offset = m_builder.CreateString(key);
    uint16_t bf16_bits = std::bit_cast<uint16_t>(value);
    auto bf16_val = ttml::flatbuffer::CreateBFloat16Value(m_builder, bf16_bits);
    auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(
        m_builder, key_offset, ttml::flatbuffer::SerializableType::BFloat16Value, bf16_val.Union());
    m_pairs.push_back(kv_pair);
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
    // Dispatch to appropriate put() method based on variant type
    // Scalars are written directly, strings/vectors are stored in m_data
    std::visit(
        [this, key](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (is_one_of_v<T, bool, char, int, float, double, uint32_t, size_t, bfloat16>) {
                this->put(key, arg);
            } else {
                // Strings and vectors - store in m_data for later serialization
                m_data[std::string(key)] = arg;
            }
        },
        value);
}

void FlatBufferFile::put(std::string_view key, const tt::tt_metal::Tensor& tensor) {
    m_tensors[std::string(key)] = tensor;
}

void FlatBufferFile::serialize(std::string_view file_path) {
    // Always treat file_path as a directory path
    // Create metadata.flatbuffer inside it
    std::filesystem::path tensor_dir(file_path);
    std::filesystem::path metadata_file = tensor_dir / "metadata.flatbuffer";

    // Create directory if needed
    std::error_code ec;
    std::filesystem::create_directories(tensor_dir, ec);
    if (ec) {
        throw std::runtime_error(fmt::format("Failed to create directory {}: {}", tensor_dir.string(), ec.message()));
    }

    // Write each tensor to its own file
    for (const auto& [key, tensor] : m_tensors) {
        // Sanitize key for filename (replace / with _, remove invalid chars)
        std::string sanitized_key = std::string(key);
        std::replace(sanitized_key.begin(), sanitized_key.end(), '/', '_');
        std::replace(sanitized_key.begin(), sanitized_key.end(), '\\', '_');

        std::filesystem::path tensor_file = tensor_dir / (sanitized_key + DOT_TENSORBIN);

        // Convert tensor to CPU if needed
        tt::tt_metal::Tensor cpu_tensor = tensor.cpu();

        // Write tensor to file using tt-metal's dump function
        tt::tt_metal::dump_tensor_flatbuffer(tensor_file.string(), cpu_tensor);
    }

    // Build KeyValuePair for each entry in m_data (strings and vectors only)
    for (const auto& [key, value] : m_data) {
        auto key_offset = m_builder.CreateString(key);
        flatbuffers::Offset<void> value_offset;
        ttml::flatbuffer::SerializableType value_type;

        // Create the appropriate value type based on the variant
        std::visit(
            [&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    auto str_offset = m_builder.CreateString(arg);
                    auto str_val = ttml::flatbuffer::CreateStringValue(m_builder, str_offset);
                    value_offset = str_val.Union();
                    value_type = ttml::flatbuffer::SerializableType::StringValue;
                } else if constexpr (std::is_same_v<T, std::vector<char>>) {
                    std::vector<int8_t> int8_vec(arg.begin(), arg.end());
                    auto vec_offset = m_builder.CreateVector(int8_vec);
                    auto vec_val = ttml::flatbuffer::CreateVectorChar(m_builder, vec_offset);
                    value_offset = vec_val.Union();
                    value_type = ttml::flatbuffer::SerializableType::VectorChar;
                } else if constexpr (std::is_same_v<T, std::vector<int>>) {
                    auto vec_offset = m_builder.CreateVector(arg);
                    auto vec_val = ttml::flatbuffer::CreateVectorInt(m_builder, vec_offset);
                    value_offset = vec_val.Union();
                    value_type = ttml::flatbuffer::SerializableType::VectorInt;
                } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                    auto vec_offset = m_builder.CreateVector(arg);
                    auto vec_val = ttml::flatbuffer::CreateVectorFloat(m_builder, vec_offset);
                    value_offset = vec_val.Union();
                    value_type = ttml::flatbuffer::SerializableType::VectorFloat;
                } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                    auto vec_offset = m_builder.CreateVector(arg);
                    auto vec_val = ttml::flatbuffer::CreateVectorDouble(m_builder, vec_offset);
                    value_offset = vec_val.Union();
                    value_type = ttml::flatbuffer::SerializableType::VectorDouble;
                } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                    auto vec_offset = m_builder.CreateVector(arg);
                    auto vec_val = ttml::flatbuffer::CreateVectorUInt8(m_builder, vec_offset);
                    value_offset = vec_val.Union();
                    value_type = ttml::flatbuffer::SerializableType::VectorUInt8;
                } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
                    auto vec_offset = m_builder.CreateVector(arg);
                    auto vec_val = ttml::flatbuffer::CreateVectorUInt32(m_builder, vec_offset);
                    value_offset = vec_val.Union();
                    value_type = ttml::flatbuffer::SerializableType::VectorUInt32;
                } else if constexpr (std::is_same_v<T, std::vector<bfloat16>>) {
                    std::vector<uint16_t> bf16_vec;
                    bf16_vec.reserve(arg.size());
                    for (const auto& bf16 : arg) {
                        bf16_vec.push_back(std::bit_cast<uint16_t>(bf16));
                    }
                    auto vec_offset = m_builder.CreateVector(bf16_vec);
                    auto vec_val = ttml::flatbuffer::CreateVectorBFloat16(m_builder, vec_offset);
                    value_offset = vec_val.Union();
                    value_type = ttml::flatbuffer::SerializableType::VectorBFloat16;
                } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    std::vector<flatbuffers::Offset<flatbuffers::String>> str_offsets;
                    str_offsets.reserve(arg.size());
                    for (const auto& s : arg) {
                        str_offsets.push_back(m_builder.CreateString(s));
                    }
                    auto vec_offset = m_builder.CreateVector(str_offsets);
                    auto vec_val = ttml::flatbuffer::CreateVectorString(m_builder, vec_offset);
                    value_offset = vec_val.Union();
                    value_type = ttml::flatbuffer::SerializableType::VectorString;
                }
            },
            value);

        auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(m_builder, key_offset, value_type, value_offset);
        m_pairs.push_back(kv_pair);
    }

    // Build KeyValuePair for each tensor - store file path as string
    for (const auto& [key, tensor] : m_tensors) {
        auto key_offset = m_builder.CreateString(key);

        // Sanitize key for filename (same as above)
        std::string sanitized_key = std::string(key);
        std::replace(sanitized_key.begin(), sanitized_key.end(), '/', '_');
        std::replace(sanitized_key.begin(), sanitized_key.end(), '\\', '_');

        // Store relative path to tensor file
        std::string tensor_file_path = sanitized_key + DOT_TENSORBIN;
        auto tensor_path_offset = m_builder.CreateString(tensor_file_path);
        auto str_val = ttml::flatbuffer::CreateStringValue(m_builder, tensor_path_offset);
        auto kv_pair = ttml::flatbuffer::CreateKeyValuePair(
            m_builder, key_offset, ttml::flatbuffer::SerializableType::StringValue, str_val.Union());
        m_pairs.push_back(kv_pair);
    }

    // Create the TTMLData table and finish the builder
    auto pairs_offset = m_builder.CreateVector(m_pairs);
    auto ttml_data = ttml::flatbuffer::CreateTTMLData(m_builder, pairs_offset);
    m_builder.Finish(ttml_data);

    // Write flatbuffer to file
    std::ofstream file(metadata_file.string(), std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open file for writing: {}", metadata_file.string()));
    }

    auto data = m_builder.GetBufferSpan();
    file.write(reinterpret_cast<const char*>(data.data()), data.size());

    if (!file.good()) {
        throw std::runtime_error(fmt::format("Failed to write flatbuffer data to file: {}", metadata_file.string()));
    }

    // Clear builder and pairs for next use
    m_builder.Clear();
    m_pairs.clear();
}

void FlatBufferFile::deserialize_flatbuffer(std::span<const uint8_t> buffer) {
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

        const std::string key(kv_pair->key()->c_str());
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
                    const auto& values = val->values();
                    result.reserve(values->size());
                    std::for_each(values->begin(), values->end(), [&result](auto* s) {
                        if (s) {
                            result.push_back(s->str());
                        }
                    });
                    m_data[key] = result;
                }
                break;
            }
            default: throw std::runtime_error(fmt::format("Unknown serializable type for key: {}", key));
        }
    }
}

void FlatBufferFile::deserialize(std::string_view filename) {
    m_data.clear();
    m_tensors.clear();
    m_builder.Clear();
    m_pairs.clear();

    // Determine metadata file path - can be directory or file
    std::filesystem::path path(filename);
    std::filesystem::path metadata_file;
    std::filesystem::path tensor_dir;

    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        // Path is a directory, look for metadata.flatbuffer
        tensor_dir = path;
        metadata_file = tensor_dir / "metadata.flatbuffer";
        if (!std::filesystem::exists(metadata_file)) {
            // Try to find any .flatbuffer file in the directory
            for (const auto& entry : std::filesystem::directory_iterator(tensor_dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".flatbuffer") {
                    metadata_file = entry.path();
                    break;
                }
            }
        }
    } else {
        // Path is a file, use it as metadata file
        metadata_file = path;
        tensor_dir = metadata_file.parent_path();
        if (tensor_dir.empty()) {
            tensor_dir = std::filesystem::current_path();
        }
    }

    if (!std::filesystem::exists(metadata_file)) {
        throw std::runtime_error(fmt::format("Metadata file not found: {}", metadata_file.string()));
    }

    std::ifstream file(metadata_file.string(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open file for reading: {}", metadata_file.string()));
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size == 0) {
        return;
    }

    // Read entire flatbuffer (new format doesn't use header_size prefix)
    std::vector<uint8_t> buffer(file_size);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    // Deserialize flatbuffer metadata
    deserialize_flatbuffer(buffer);

    // Load tensors from separate files
    // Get TTMLData to find tensor file paths stored as StringValue
    auto* ttml_data = ttml::flatbuffer::GetTTMLData(buffer.data());
    if (ttml_data && ttml_data->pairs()) {
        for (const auto* kv_pair : *ttml_data->pairs()) {
            if (!kv_pair || !kv_pair->key()) {
                continue;
            }

            const std::string key(kv_pair->key()->c_str());

            // Check if this is a tensor file path (stored as StringValue ending in .tensorbin)
            if (kv_pair->value_type() == ttml::flatbuffer::SerializableType::StringValue) {
                auto* str_val = kv_pair->value_as_StringValue();
                if (str_val && str_val->value()) {
                    std::string file_path_str(str_val->value()->c_str());
                    // Check if it's a tensor file (ends with .tensorbin)
                    if (file_path_str.ends_with(DOT_TENSORBIN)) {
                        // Resolve tensor file path relative to metadata file directory
                        std::filesystem::path tensor_file = tensor_dir / file_path_str;

                        if (!std::filesystem::exists(tensor_file)) {
                            throw std::runtime_error(fmt::format(
                                "Tensor file not found: {} (resolved from {})", tensor_file.string(), file_path_str));
                        }

                        // Load tensor from file using tt-metal's load function
                        tt::tt_metal::Tensor tensor =
                            tt::tt_metal::load_tensor_flatbuffer(tensor_file.string(), nullptr);
                        m_tensors[key] = tensor;
                    }
                }
            }
        }
    }
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

tt::tt_metal::Tensor FlatBufferFile::get_tensor(std::string_view key) const {
    auto it = m_tensors.find(std::string(key));
    if (it != m_tensors.end()) {
        return it->second;
    } else {
        throw std::runtime_error(fmt::format("Tensor key not found: {}", key));
    }
}

}  // namespace ttml::serialization
