// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ttml::serialization {

// Simple tar reader for extracting files from tarball
class TarReader {
public:
    TarReader();
    ~TarReader() = default;

    // Read tarball from file
    void read_from_file(const std::string& filename);

    // Get file data by filename
    std::vector<uint8_t> get_file(std::string_view filename) const;

    // Check if file exists in tarball
    bool has_file(std::string_view filename) const;

    // Get list of all filenames in tarball
    std::vector<std::string> list_files() const;

private:
    std::unordered_map<std::string, std::vector<uint8_t>> m_files;

    // Parse tar header
    bool parse_header(const uint8_t* header, std::string& filename, size_t& size) const;

    // Read octal number from tar header
    uint64_t read_octal(const char* str, size_t len) const;
};

}  // namespace ttml::serialization
