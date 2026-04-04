// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/lightmetal/lightmetal_binary.hpp>

#include <fstream>
#include <stdexcept>
#include <vector>

namespace tt::tt_metal::experimental::lightmetal {

void LightMetalBinary::save_to_file(const std::string& filename) const {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        throw std::ios_base::failure("Failed to open file: " + filename);
    }
    outFile.write(reinterpret_cast<const char*>(data_.data()), data_.size());
    if (!outFile.good()) {
        throw std::ios_base::failure("Failed to write to file: " + filename);
    }
}

LightMetalBinary LightMetalBinary::load_from_file(const std::string& filename) {
    std::ifstream in_file(filename, std::ios::binary | std::ios::ate);
    if (!in_file.is_open()) {
        throw std::ios_base::failure("Failed to open file: " + filename);
    }
    auto file_size = in_file.tellg();
    if (file_size <= 0) {
        throw std::runtime_error("File is empty or error occurred while reading: " + filename);
    }
    std::vector<uint8_t> buffer(file_size);
    in_file.seekg(0, std::ios::beg);
    in_file.read(reinterpret_cast<char*>(buffer.data()), file_size);
    if (!in_file.good()) {
        throw std::ios_base::failure("Failed to read file: " + filename);
    }
    return LightMetalBinary(std::move(buffer));
}

}  // namespace tt::tt_metal::experimental::lightmetal
