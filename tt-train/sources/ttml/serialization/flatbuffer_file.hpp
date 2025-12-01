// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tt-metalium/bfloat16.hpp>
#include <variant>
#include <vector>

namespace ttml::serialization {

using ValueType = std::variant<
    bool,
    char,
    int,
    float,
    double,
    uint32_t,
    size_t,
    bfloat16,
    std::string,
    std::vector<char>,
    std::vector<int>,
    std::vector<float>,
    std::vector<double>,
    std::vector<uint8_t>,
    std::vector<uint32_t>,
    std::vector<bfloat16>,
    std::vector<std::string>>;

class FlatBufferFile {
public:
    explicit FlatBufferFile() = default;

    ~FlatBufferFile();

    // Copy constructor
    FlatBufferFile(const FlatBufferFile& other) = delete;

    // Copy assignment operator
    FlatBufferFile& operator=(const FlatBufferFile& other) = delete;

    // Move constructor
    FlatBufferFile(FlatBufferFile&& other) noexcept;

    // Move assignment operator
    FlatBufferFile& operator=(FlatBufferFile&& other) = delete;

    // Methods to put different types
    void put(std::string_view key, bool value);
    void put(std::string_view key, char value);
    void put(std::string_view key, int value);
    void put(std::string_view key, float value);
    void put(std::string_view key, double value);
    void put(std::string_view key, uint32_t value);
    void put(std::string_view key, size_t value);
    void put(std::string_view key, bfloat16 value);
    void put(std::string_view key, std::string_view value);
    void put(std::string_view key, const char* value);

    // Overloads for std::span
    void put(std::string_view key, std::span<const char> value);
    void put(std::string_view key, std::span<const uint8_t> value);
    void put(std::string_view key, std::span<const int> value);
    void put(std::string_view key, std::span<const float> value);
    void put(std::string_view key, std::span<const double> value);
    void put(std::string_view key, std::span<const uint32_t> value);
    void put(std::string_view key, std::span<const bfloat16> value);
    void put(std::string_view key, std::span<const std::string> value);

    void put(std::string_view key, const ValueType& value);

    // Serialization method
    void serialize(std::string_view filename);

    // Deserialization method
    void deserialize(std::string_view filename);

    // Methods to get values
    [[nodiscard]] bool get_bool(std::string_view key) const;
    [[nodiscard]] char get_char(std::string_view key) const;
    [[nodiscard]] int get_int(std::string_view key) const;
    [[nodiscard]] float get_float(std::string_view key) const;
    [[nodiscard]] double get_double(std::string_view key) const;
    [[nodiscard]] uint32_t get_uint32(std::string_view key) const;
    [[nodiscard]] size_t get_size_t(std::string_view key) const;
    [[nodiscard]] bfloat16 get_bfloat16(std::string_view key) const;
    [[nodiscard]] std::string get_string(std::string_view key) const;

    // Methods to get vectors
    [[nodiscard]] std::vector<uint8_t> get_vector_uint8(std::string_view key) const;

    [[nodiscard]] ValueType get_value_type(std::string_view key) const;

    [[nodiscard]] const std::filesystem::path& get_base_dir() const;

private:
    std::vector<uint8_t> build_flatbuffer(const std::unordered_map<std::string, ValueType>& data_subset) const;
    std::string get_prefix(std::string_view key) const;
    std::vector<std::pair<std::string, std::vector<uint8_t>>> build_flatbuffer_files() const;
    void do_serialize(
        std::string_view dirname, const std::vector<std::pair<std::string, std::vector<uint8_t>>>& flatbuffer_files);
    void deserialize_flatbuffer(const std::vector<uint8_t>& buffer, std::string_view prefix);
    void do_deserialize(std::string_view filename);
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

    std::unordered_map<std::string, ValueType> m_data;
};

}  // namespace ttml::serialization
