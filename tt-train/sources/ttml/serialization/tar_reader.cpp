// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tar_reader.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

// Include zstd for decompression
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include "tar_format.hpp"

namespace ttml::serialization {

TarReader::TarReader() = default;

uint64_t TarReader::read_octal(const char* str, size_t len) const {
    uint64_t value = 0;
    for (size_t i = 0; i < len && str[i] != '\0' && str[i] != ' '; ++i) {
        if (str[i] >= '0' && str[i] <= '7') {
            value = value * 8 + (str[i] - '0');
        }
    }
    return value;
}

bool TarReader::parse_header(const uint8_t* header, std::string& filename, size_t& size) const {
    using namespace tar_format;
    using namespace tar_format::header;

    // Check if header is all zeros (end of archive)
    bool all_zeros = true;
    for (size_t i = 0; i < HEADER_SIZE; ++i) {
        if (header[i] != 0) {
            all_zeros = false;
            break;
        }
    }
    if (all_zeros) {
        return false;
    }

    // Read filename (null-terminated)
    filename.clear();
    for (size_t i = 0; i < FILENAME_SIZE && header[FILENAME_OFFSET + i] != 0; ++i) {
        filename += static_cast<char>(header[FILENAME_OFFSET + i]);
    }
    // Trim trailing spaces (tar format may pad with spaces before null)
    while (!filename.empty() && filename.back() == ' ') {
        filename.pop_back();
    }

    // Read file size (octal)
    char size_str[SIZE_SIZE + 1] = {0};
    std::memcpy(size_str, header + SIZE_OFFSET, SIZE_SIZE);
    size = static_cast<size_t>(read_octal(size_str, SIZE_SIZE));

    return true;
}

void TarReader::read_from_file(std::string_view filename) {
    std::ifstream ifs(std::string(filename), std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Unable to open file for reading: " + std::string(filename));
    }

    std::vector<uint8_t> file_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    if (file_data.empty()) {
        throw std::runtime_error("Tarball file is empty: " + std::string(filename));
    }

    std::vector<uint8_t> tarball;

    // Check if file is zstd compressed by checking magic number
    // ZSTD magic number is 0xFD2FB528 (little-endian)
    constexpr uint32_t ZSTD_MAGIC = 0xFD2FB528;
    bool is_zstd_compressed = false;
    if (file_data.size() >= sizeof(uint32_t)) {
        uint32_t magic = *reinterpret_cast<const uint32_t*>(file_data.data());
        if (magic == ZSTD_MAGIC) {
            is_zstd_compressed = true;
        }
    }

    if (is_zstd_compressed) {
        // Decompress zstd data
        // First, get the decompressed size from the frame header
        unsigned long long decompressed_size = ZSTD_getFrameContentSize(file_data.data(), file_data.size());

        if (decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
            throw std::runtime_error("Invalid zstd compressed file: " + std::string(filename));
        }
        if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
            // Content size unknown, need to decompress in streaming mode or estimate
            // For simplicity, use a reasonable estimate and resize if needed
            decompressed_size = file_data.size() * 4;  // Conservative estimate
        }

        tarball.resize(decompressed_size);
        size_t actual_decompressed_size =
            ZSTD_decompress(tarball.data(), decompressed_size, file_data.data(), file_data.size());

        if (ZSTD_isError(actual_decompressed_size)) {
            throw std::runtime_error(
                "ZSTD decompression failed: " + std::string(ZSTD_getErrorName(actual_decompressed_size)));
        }

        // Resize to actual decompressed size
        tarball.resize(actual_decompressed_size);
    } else {
        // Not compressed, use data as-is
        tarball = std::move(file_data);
    }

    // Check if file is too small to be a valid tarball (must be at least one header block)
    using namespace tar_format;
    if (tarball.size() < BLOCK_SIZE) {
        throw std::runtime_error("Invalid tarball: file too small to contain tar header: " + std::string(filename));
    }

    m_files.clear();

    using namespace tar_format;
    using namespace tar_format::header;

    size_t offset = 0;

    while (offset + HEADER_SIZE <= tarball.size()) {
        const uint8_t* header = tarball.data() + offset;
        std::string file_name;
        size_t file_size;

        if (!parse_header(header, file_name, file_size)) {
            // Two consecutive empty blocks indicate end of archive
            break;
        }

        offset += HEADER_SIZE;

        if (offset + file_size > tarball.size()) {
            throw std::runtime_error("Invalid tarball: file extends beyond archive");
        }

        // Read file data
        std::vector<uint8_t> file_data(file_size);
        std::memcpy(file_data.data(), tarball.data() + offset, file_size);
        m_files[file_name] = std::move(file_data);

        offset += file_size;

        // Pad to block boundary
        size_t remainder = file_size % BLOCK_SIZE;
        if (remainder != 0) {
            offset += BLOCK_SIZE - remainder;
        }
    }
}

std::vector<uint8_t> TarReader::get_file(std::string_view filename) const {
    std::string key(filename);
    auto it = m_files.find(key);
    if (it == m_files.end()) {
        throw std::runtime_error("File not found in tarball: " + std::string(filename));
    }
    return it->second;
}

bool TarReader::has_file(std::string_view filename) const {
    std::string key(filename);
    return m_files.find(key) != m_files.end();
}

std::vector<std::string> TarReader::list_files() const {
    std::vector<std::string> files;
    files.reserve(m_files.size());
    for (const auto& [name, _] : m_files) {
        files.push_back(name);
    }
    return files;
}

}  // namespace ttml::serialization
