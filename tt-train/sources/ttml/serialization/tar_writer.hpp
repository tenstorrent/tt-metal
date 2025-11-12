// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "tar_format.hpp"

namespace ttml::serialization {

// Simple in-memory tar writer
class TarWriter {
public:
    TarWriter();
    ~TarWriter() = default;

    // Add a file to the tarball
    void add_file(std::string_view filename, const void* data, size_t size);

    // Get the complete tarball as a byte vector
    std::vector<uint8_t> get_tarball() const;

    // Write tarball to file
    void write_to_file(std::string_view filename) const;

private:
    struct FileEntry {
        std::string filename;
        std::vector<uint8_t> data;
    };

    std::vector<FileEntry> m_files;

    // Write tar header for a file
    void write_header(std::vector<uint8_t>& tarball, const FileEntry& entry) const;

    // Pad data to 512-byte boundary
    void pad_to_block(std::vector<uint8_t>& data) const;

    // Convert number to octal string (for tar header)
    std::string to_octal(uint64_t value, size_t width) const;
};

}  // namespace ttml::serialization
