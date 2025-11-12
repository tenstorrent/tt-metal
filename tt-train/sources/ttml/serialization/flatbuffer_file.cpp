// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "flatbuffer_file.hpp"

#include <flatbuffers/flatbuffers.h>
#include <fmt/format.h>

#include <cstdint>
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

namespace ttml::serialization {

class FlatBufferFile::Impl {
public:
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

    void put(std::string_view key, const std::string& value) {
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

    // Serialization method
    void serialize(const std::string& filename) {
        flatbuffers::FlatBufferBuilder builder;

        std::vector<flatbuffers::Offset<ttml::flatbuffer::KeyValuePair>> kv_pairs;

        for (const auto& [key, value] : m_data) {
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

        // Build flatbuffer in memory
        const void* flatbuffer_data = builder.GetBufferPointer();
        size_t flatbuffer_size = builder.GetSize();

        // Create tarball in memory
        TarWriter tar_writer;
        tar_writer.add_file("data.flatbuffer", flatbuffer_data, flatbuffer_size);

        // Write tarball to file (single write operation)
        tar_writer.write_to_file(filename);
    }

    // Deserialization method
    void deserialize(const std::string& filename) {
        // Read tarball and extract flatbuffer data
        TarReader tar_reader;
        tar_reader.read_from_file(filename);

        if (!tar_reader.has_file("data.flatbuffer")) {
            throw std::runtime_error("Tarball does not contain data.flatbuffer: " + filename);
        }

        std::vector<uint8_t> buffer = tar_reader.get_file("data.flatbuffer");

        if (buffer.empty()) {
            throw std::runtime_error("FlatBuffer data is empty in tarball: " + filename);
        }

        // Verify and get the root
        flatbuffers::Verifier verifier(buffer.data(), buffer.size());
        if (!ttml::flatbuffer::VerifyTTMLDataBuffer(verifier)) {
            throw std::runtime_error("Invalid FlatBuffer data in tarball: " + filename);
        }

        auto* ttml_data = ttml::flatbuffer::GetTTMLData(buffer.data());

        if (!ttml_data || !ttml_data->pairs()) {
            throw std::runtime_error("Invalid FlatBuffer structure in file: " + filename);
        }

        // Clear existing data
        m_data.clear();

        // Deserialize each key-value pair
        for (const auto* kv_pair : *ttml_data->pairs()) {
            if (!kv_pair || !kv_pair->key()) {
                continue;
            }

            std::string key(kv_pair->key()->c_str());

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

    // Methods to get values
    void get(std::string_view key, bool& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, char& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, int& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, float& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, double& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, uint32_t& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, size_t& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, std::string& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, std::vector<char>& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, std::vector<uint8_t>& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, std::vector<int>& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, std::vector<float>& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, std::vector<double>& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, std::vector<uint32_t>& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, std::vector<std::string>& value) const {
        get_value(key, value);
    }

    void get(std::string_view key, ValueType& value) const {
        get_value(key, value);
    }

private:
    std::unordered_map<std::string, ValueType> m_data;

    // Helper function to get value from m_data
    template <typename T>
    void get_value(std::string_view key, T& value) const {
        auto it = m_data.find(std::string(key));
        if (it != m_data.end()) {
            if (const auto* pval = std::get_if<T>(&(it->second))) {
                value = *pval;
            } else {
                throw std::runtime_error(fmt::format("Type mismatch for key: {}", key));
            }
        } else {
            throw std::runtime_error(fmt::format("Key not found: {}", key));
        }
    }

    void get_value(std::string_view key, ValueType& value) const {
        auto it = m_data.find(std::string(key));
        if (it != m_data.end()) {
            value = it->second;
        } else {
            throw std::runtime_error(fmt::format("Key not found: {}", key));
        }
    }
};

FlatBufferFile::FlatBufferFile() : m_impl(std::make_unique<Impl>()) {
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

void FlatBufferFile::serialize(const std::string& filename) {
    m_impl->serialize(filename);
}

void FlatBufferFile::deserialize(const std::string& filename) {
    m_impl->deserialize(filename);
}

void FlatBufferFile::get(std::string_view key, bool& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, char& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, int& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, float& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, double& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, uint32_t& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, size_t& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, std::string& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, std::vector<char>& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, std::vector<uint8_t>& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, std::vector<int>& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, std::vector<float>& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, std::vector<double>& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, std::vector<uint32_t>& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, std::vector<std::string>& value) const {
    m_impl->get(key, value);
}

void FlatBufferFile::get(std::string_view key, ValueType& value) const {
    m_impl->get(key, value);
}

}  // namespace ttml::serialization
