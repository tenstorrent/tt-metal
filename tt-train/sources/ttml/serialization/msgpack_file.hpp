// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
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
    std::string,
    std::vector<char>,
    std::vector<int>,
    std::vector<float>,
    std::vector<double>,
    std::vector<uint8_t>,
    std::vector<uint32_t>,
    std::vector<std::string>>;

class MsgPackFile {
public:
    MsgPackFile();
    ~MsgPackFile();

    // Copy constructor
    MsgPackFile(const MsgPackFile& other) = delete;

    // Copy assignment operator
    MsgPackFile& operator=(const MsgPackFile& other) = delete;

    // Move constructor
    MsgPackFile(MsgPackFile&& other) noexcept;

    // Move assignment operator
    MsgPackFile& operator=(MsgPackFile&& other) = delete;

    // Methods to put different types
    void put(std::string_view key, bool value);
    void put(std::string_view key, char value);
    void put(std::string_view key, int value);
    void put(std::string_view key, float value);
    void put(std::string_view key, double value);
    void put(std::string_view key, uint32_t value);
    void put(std::string_view key, size_t value);
    void put(std::string_view key, std::string_view value);

    // added it to prevent implicit casts from const char* to bool
    void put(std::string_view key, const char* value);

    // Overloads for std::span
    void put(std::string_view key, std::span<const uint8_t> value);
    void put(std::string_view key, std::span<const int> value);
    void put(std::string_view key, std::span<const float> value);
    void put(std::string_view key, std::span<const double> value);
    void put(std::string_view key, std::span<const uint32_t> value);
    void put(std::string_view key, std::span<const std::string> value);

    void put(std::string_view key, const ValueType& value);
    // Serialization method
    void serialize(const std::string& filename);

    // Deserialization method
    void deserialize(const std::string& filename);

    // Methods to get values
    void get(std::string_view key, bool& value) const;
    void get(std::string_view key, char& value) const;
    void get(std::string_view key, int& value) const;
    void get(std::string_view key, float& value) const;
    void get(std::string_view key, double& value) const;
    void get(std::string_view key, uint32_t& value) const;
    void get(std::string_view key, size_t& value) const;
    void get(std::string_view key, std::string& value) const;

    // Methods to get vectors (from spans)
    void get(std::string_view key, std::vector<uint8_t>& value) const;
    void get(std::string_view key, std::vector<int>& value) const;
    void get(std::string_view key, std::vector<float>& value) const;
    void get(std::string_view key, std::vector<double>& value) const;
    void get(std::string_view key, std::vector<uint32_t>& value) const;
    void get(std::string_view key, std::vector<std::string>& value) const;

    void get(std::string_view key, ValueType& type) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace ttml::serialization
