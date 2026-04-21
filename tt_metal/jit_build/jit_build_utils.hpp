// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::filesystem::path& log_file, bool verbose);

// Like run_command but bypasses the shell entirely by using posix_spawn with an explicit
// argument vector.  Immune to shell metacharacter injection.
// |working_dir| is passed as the cwd for the child process (empty = inherit parent cwd).
bool exec_command(
    const std::vector<std::string>& args,
    const std::filesystem::path& working_dir,
    const std::filesystem::path& log_file);

// Split a whitespace-delimited string into tokens (no shell quoting support).
std::vector<std::string> tokenize_flags(const std::string& flags);

// Creates an empty file (and parent directories). Returns false if the file could not be created.
bool create_file(const std::filesystem::path& file_path_str);

// An RAII wrapper that generates a temporary filename and renames the file on destruction.
// This is to allow multiple processes to write to the same target file without clobbering each other.
class FileRenamer {
public:
    FileRenamer(const std::filesystem::path& target_path);
    FileRenamer(const FileRenamer&) = delete;
    FileRenamer& operator=(const FileRenamer&) = delete;
    FileRenamer(FileRenamer&&) = default;
    FileRenamer& operator=(FileRenamer&&) = default;
    ~FileRenamer();

    const std::filesystem::path& path() const { return temp_path_; }
    static std::filesystem::path generate_temp_path(const std::filesystem::path& target_path);

private:
    std::filesystem::path temp_path_;
    std::filesystem::path target_path_;
    static uint64_t unique_id_;
};

}  // namespace tt::jit_build::utils
