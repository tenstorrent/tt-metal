// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace tt::tt_metal {

// Public interface wrapper for serialized LightMetalBinary data
struct LightMetalBinary {
    std::vector<uint8_t> data;

    // Constructor
    LightMetalBinary() = default;

    // Construct from raw data
    explicit LightMetalBinary(std::vector<uint8_t> data) : data(std::move(data)) {}

    // Size of the binary data
    size_t Size() const { return data.size(); }

    // Check if the binary data is empty
    bool IsEmpty() const { return data.empty(); }

    // Save binary data to a file
    void SaveToFile(const std::string& filename) const {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile.is_open()) {
            throw std::ios_base::failure("Failed to open file: " + filename);
        }
        outFile.write(reinterpret_cast<const char*>(data.data()), data.size());
        if (!outFile.good()) {
            throw std::ios_base::failure("Failed to write to file: " + filename);
        }
    }

    // Load binary data from a file
    static LightMetalBinary LoadFromFile(const std::string& filename) {
        std::ifstream inFile(filename, std::ios::binary | std::ios::ate);
        if (!inFile.is_open()) {
            throw std::ios_base::failure("Failed to open file: " + filename);
        }
        auto fileSize = inFile.tellg();
        if (fileSize <= 0) {
            throw std::runtime_error("File is empty or error occurred while reading: " + filename);
        }
        std::vector<uint8_t> buffer(fileSize);
        inFile.seekg(0, std::ios::beg);
        inFile.read(reinterpret_cast<char*>(buffer.data()), fileSize);
        if (!inFile.good()) {
            throw std::ios_base::failure("Failed to read file: " + filename);
        }
        return LightMetalBinary(std::move(buffer));
    }
};

}  // namespace tt::tt_metal
