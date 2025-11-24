// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
    // Constructor with optional tarballing and compression (defaults to true for backward compatibility)
    explicit FlatBufferFile(bool use_tarball = true, bool compress = false);
    ~FlatBufferFile();

    // Copy constructor
    FlatBufferFile(const FlatBufferFile& other) = delete;

    // Copy assignment operator
    FlatBufferFile& operator=(const FlatBufferFile& other) = delete;

    // Move constructor
    FlatBufferFile(FlatBufferFile&& other) noexcept;

    // Move assignment operator
    FlatBufferFile& operator=(FlatBufferFile&& other) = delete;

    // Set whether to use tarball format (can be changed at runtime)
    void set_use_tarball(bool use_tarball);

    // Get whether tarball format is enabled
    [[nodiscard]] bool get_use_tarball() const;

    // Set whether to use compression (only applies when tarball is enabled)
    void set_compress(bool compress);

    // Get whether compression is enabled
    [[nodiscard]] bool get_compress() const;

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

    // added it to prevent implicit casts from const char* to bool
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

    // Add tensor file data to be included in tarball during serialization
    void add_tensor_file(std::string_view filename, std::vector<uint8_t>&& data);

    // Get tensor file data (for reading from tarball)
    std::vector<uint8_t> get_tensor_file(std::string_view filename) const;

    // Check if tensor file exists
    bool has_tensor_file(std::string_view filename) const;

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

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ttml::serialization
