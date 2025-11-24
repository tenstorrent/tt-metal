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

    // Add a file to the tarball (takes ownership of data)
    void add_file(std::string_view filename, std::vector<uint8_t>&& data);

    // Get the complete tarball as a byte vector
    std::vector<uint8_t> get_tarball() const;

    // Write files to disk
    // If use_tarball is true, creates a tarball (compressed if compress is true, uncompressed otherwise)
    // If use_tarball is false, writes individual files to a directory (filename is treated as directory path)
    void write_to_file(std::string_view filename, bool use_tarball = false, bool compress = false) const;

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
