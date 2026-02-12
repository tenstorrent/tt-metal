// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, bool verbose);
void create_file(const std::string& file_path_str);

// Need a stable hash for things to persist across runs.
// std::hash is not guaranteed to be stable across runs / implementations.
class FNV1a {
public:
    // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
    static constexpr uint64_t FNV_PRIME = 0x100000001b3;
    static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325;

    FNV1a(uint64_t offset = FNV_OFFSET) : hash_(offset) {}

    void update(uint64_t data) {
        hash_ ^= data;
        hash_ *= FNV_PRIME;
    }

    template <typename ForwardIterator>
    void update(ForwardIterator begin, ForwardIterator end) {
        for (auto it = begin; it != end; ++it) {
            update(static_cast<uint64_t>(*it));
        }
    }

    uint64_t digest() const { return hash_; }

private:
    uint64_t hash_;
};

// An RAII wrapper that generates a temporary filename and renames the file on destruction.
// This is to allow multiple processes to write to the same target file without clobbering each other.
class FileRenamer {
public:
    FileRenamer(const std::string& target_path);
    FileRenamer(const FileRenamer&) = delete;
    FileRenamer& operator=(const FileRenamer&) = delete;
    FileRenamer(FileRenamer&&) = default;
    FileRenamer& operator=(FileRenamer&&) = default;
    ~FileRenamer();

    const std::string& path() const { return temp_path_; }
    static std::string generate_temp_path(const std::filesystem::path& target_path);

private:
    std::string temp_path_;
    std::string target_path_;
    static uint64_t unique_id_;
};

}  // namespace tt::jit_build::utils
