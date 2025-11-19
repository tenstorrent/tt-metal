// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tar_writer.hpp"

#include <tar.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

// Include zstd for compression
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

namespace ttml::serialization {

TarWriter::TarWriter() = default;

void TarWriter::add_file(std::string_view filename, const void* data, size_t size) {
    FileEntry entry;
    entry.filename = std::string(filename);
    entry.data.resize(size);
    std::memcpy(entry.data.data(), data, size);
    m_files.push_back(std::move(entry));
}

std::string TarWriter::to_octal(uint64_t value, size_t width) const {
    std::ostringstream oss;
    oss << std::oct << std::setfill('0') << std::setw(width) << value;
    return oss.str();
}

void TarWriter::write_header(std::vector<uint8_t>& tarball, const FileEntry& entry) const {
    using namespace tar_format;
    using namespace tar_format::header;
    using namespace tar_format::defaults;

    std::vector<uint8_t> header(HEADER_SIZE, 0);

    // Filename
    std::string filename = entry.filename;
    if (filename.size() > FILENAME_SIZE) {
        throw std::runtime_error("Filename too long for tar format: " + filename);
    }
    std::memcpy(header.data() + FILENAME_OFFSET, filename.c_str(), filename.size());

    // File mode (null-terminated, padded with nulls)
    size_t mode_len = std::min(strlen(MODE), MODE_SIZE - 1);
    std::memcpy(header.data() + MODE_OFFSET, MODE, mode_len);
    header[MODE_OFFSET + MODE_SIZE - 1] = '\0';

    // UID (null-terminated, padded with nulls)
    size_t uid_len = std::min(strlen(UID), UID_SIZE - 1);
    std::memcpy(header.data() + UID_OFFSET, UID, uid_len);
    header[UID_OFFSET + UID_SIZE - 1] = '\0';

    // GID (null-terminated, padded with nulls)
    size_t gid_len = std::min(strlen(GID), GID_SIZE - 1);
    std::memcpy(header.data() + GID_OFFSET, GID, gid_len);
    header[GID_OFFSET + GID_SIZE - 1] = '\0';

    // File size (octal, null-terminated, padded with nulls)
    std::string size_str = to_octal(entry.data.size(), SIZE_SIZE - 1);
    size_t size_len = std::min(size_str.size(), SIZE_SIZE - 1);
    std::memcpy(header.data() + SIZE_OFFSET, size_str.c_str(), size_len);
    // Ensure null termination and padding
    header[SIZE_OFFSET + SIZE_SIZE - 1] = '\0';

    // Modification time (octal, null-terminated, padded with nulls)
    std::string mtime_str = to_octal(MTIME, MTIME_SIZE - 1);
    size_t mtime_len = std::min(mtime_str.size(), MTIME_SIZE - 1);
    std::memcpy(header.data() + MTIME_OFFSET, mtime_str.c_str(), mtime_len);
    // Ensure null termination and padding
    header[MTIME_OFFSET + MTIME_SIZE - 1] = '\0';

    // Link name - empty (already zero-initialized)

    // Magic (use TMAGIC and TMAGLEN directly from tar.h)
    std::memcpy(header.data() + MAGIC_OFFSET, TMAGIC, TMAGLEN - 1);
    header[MAGIC_OFFSET + TMAGLEN - 1] = '\0';

    // Version (use TVERSION and TVERSLEN directly from tar.h)
    // TVERSLEN is 2, copy exactly 2 bytes (no null terminator needed)
    std::memcpy(header.data() + VERSION_OFFSET, TVERSION, TVERSLEN);

    // User name, Group name, Device major/minor, Prefix - empty (already zero-initialized)

    // Type flag - regular file (use REGTYPE directly from tar.h)
    // Must be set BEFORE checksum calculation
    header[TYPE_FLAG_OFFSET] = REGTYPE;

    // Calculate checksum (checksum field bytes are treated as spaces during calculation)
    uint32_t checksum = 0;
    for (size_t i = 0; i < HEADER_SIZE; ++i) {
        if (i >= CHECKSUM_OFFSET && i < CHECKSUM_OFFSET + CHECKSUM_SIZE) {
            // Checksum field treated as spaces during calculation
            checksum += static_cast<uint8_t>(' ');
        } else {
            checksum += static_cast<uint8_t>(header[i]);
        }
    }

    // Write checksum (octal, null-terminated, space-terminated)
    // Format: 6 octal digits + null + space = 8 bytes total
    std::string checksum_str = to_octal(checksum, CHECKSUM_SIZE - 2);
    size_t checksum_len = std::min(checksum_str.size(), CHECKSUM_SIZE - 2);
    std::memcpy(header.data() + CHECKSUM_OFFSET, checksum_str.c_str(), checksum_len);
    // Ensure proper formatting: null terminator at position CHECKSUM_SIZE - 2, space at CHECKSUM_SIZE - 1
    header[CHECKSUM_OFFSET + CHECKSUM_SIZE - 2] = '\0';
    header[CHECKSUM_OFFSET + CHECKSUM_SIZE - 1] = ' ';  // Space after checksum

    tarball.insert(tarball.end(), header.begin(), header.end());
}

void TarWriter::pad_to_block(std::vector<uint8_t>& data) const {
    using namespace tar_format;
    size_t remainder = data.size() % BLOCK_SIZE;
    if (remainder != 0) {
        size_t padding = BLOCK_SIZE - remainder;
        data.insert(data.end(), padding, 0);
    }
}

std::vector<uint8_t> TarWriter::get_tarball() const {
    std::vector<uint8_t> tarball;

    // Write each file
    for (const auto& entry : m_files) {
        // Write header
        write_header(tarball, entry);

        // Write file data
        tarball.insert(tarball.end(), entry.data.begin(), entry.data.end());

        // Pad to 512-byte boundary
        pad_to_block(tarball);
    }

    // Write two empty blocks at the end
    using namespace tar_format;
    tarball.insert(tarball.end(), BLOCK_SIZE * END_BLOCKS, 0);

    return tarball;
}

void TarWriter::write_to_file(std::string_view filename) const {
    auto tarball = get_tarball();

    // Compress tarball with zstd
    const size_t compressed_bound = ZSTD_compressBound(tarball.size());
    std::vector<uint8_t> compressed_data(compressed_bound);

    const int compression_level = 3;  // Default compression level (balanced speed/ratio)
    size_t compressed_size =
        ZSTD_compress(compressed_data.data(), compressed_bound, tarball.data(), tarball.size(), compression_level);

    if (ZSTD_isError(compressed_size)) {
        throw std::runtime_error("ZSTD compression failed: " + std::string(ZSTD_getErrorName(compressed_size)));
    }

    // Resize to actual compressed size
    compressed_data.resize(compressed_size);

    // Write compressed data to file
    std::ofstream ofs(std::string(filename), std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + std::string(filename));
    }
    ofs.write(
        reinterpret_cast<const char*>(compressed_data.data()), static_cast<std::streamsize>(compressed_data.size()));
    if (!ofs.good()) {
        throw std::runtime_error("Error writing compressed tarball to file: " + std::string(filename));
    }
}

}  // namespace ttml::serialization
