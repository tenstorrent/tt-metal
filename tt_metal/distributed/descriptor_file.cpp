// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/descriptor_file.hpp>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::internal {

void write_file_atomically(const std::string& path, ttsl::Span<const uint8_t> bytes) {
    const std::string tmp_path = path + ".tmp";
    std::ofstream ofs(tmp_path, std::ios::binary);
    TT_FATAL(ofs.is_open(), "Failed to open descriptor file for writing: {}", tmp_path);
    ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    ofs.close();
    TT_FATAL(!ofs.fail(), "Failed to write descriptor file: {}", tmp_path);
    TT_FATAL(
        std::rename(tmp_path.c_str(), path.c_str()) == 0,
        "Failed to rename descriptor file from {} to {}: {}",
        tmp_path,
        path,
        std::strerror(errno));
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    TT_FATAL(ifs.is_open(), "Failed to open descriptor file for reading: {}", path);
    const auto pos = ifs.tellg();
    TT_FATAL(pos > 0, "Descriptor file is empty or unreadable: {}", path);
    std::vector<uint8_t> bytes(static_cast<std::size_t>(pos));
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    TT_FATAL(!ifs.fail(), "Failed to read descriptor file: {}", path);
    return bytes;
}

std::vector<uint8_t> wait_and_read_file(const std::string& path, uint32_t timeout_ms) {
    const auto start = std::chrono::steady_clock::now();
    while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(timeout_ms)) {
            TT_THROW("Timeout waiting for descriptor file: {}", path);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return read_file(path);
}

}  // namespace tt::tt_metal::internal
