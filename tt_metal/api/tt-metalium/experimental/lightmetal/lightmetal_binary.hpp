// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental::lightmetal {

// Public interface wrapper for serialized LightMetalBinary data
class LightMetalBinary {
private:
    std::vector<uint8_t> data_;

public:
    // Default constructor, and constructor from raw data vector.
    LightMetalBinary() = default;
    explicit LightMetalBinary(std::vector<uint8_t> data) : data_(std::move(data)) {}

    // Public accessors for the binary data
    const std::vector<uint8_t>& get_data() const { return data_; }
    void set_data(std::vector<uint8_t> data) { data_ = std::move(data); }
    size_t size() const { return data_.size(); }
    bool is_empty() const { return data_.empty(); }

    // Save binary data to a file
    void save_to_file(const std::string& filename) const;

    // Load binary data from a file
    static LightMetalBinary load_from_file(const std::string& filename);
};

}  // namespace tt::tt_metal::experimental::lightmetal
