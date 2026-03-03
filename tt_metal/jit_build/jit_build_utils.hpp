// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "common/filesystem_utils.hpp"

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, bool verbose);
void create_file(const std::string& file_path_str);

// Re-export filesystem utilities for backwards compatibility
using tt::filesystem::is_estale_error;
using tt::filesystem::is_not_found_error;
using tt::filesystem::kFsRetryDelayMs;
using tt::filesystem::kMaxFsRetries;
using tt::filesystem::safe_create_directories;
using tt::filesystem::safe_hard_link_or_copy;
using tt::filesystem::safe_remove;
using tt::filesystem::safe_remove_all;
using tt::filesystem::safe_rename;

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
