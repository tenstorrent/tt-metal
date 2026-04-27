// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "jit_build/types.hpp"

namespace tt::jit_build::utils {

bool run_command(const std::string& cmd, const std::string& log_file, bool verbose);

// Like run_command but bypasses the shell entirely by using posix_spawn with an explicit
// argument vector.  Immune to shell metacharacter injection.
// |working_dir| is passed as the cwd for the child process (empty = inherit parent cwd).
bool exec_command(const std::vector<std::string>& args, const std::string& working_dir, const std::string& log_file);

// Split a whitespace-delimited string into tokens (no shell quoting support).
std::vector<std::string> tokenize_flags(const std::string& flags);

void create_file(const std::string& file_path_str);

// Read the entire contents of a binary file into a byte vector.
// Throws std::runtime_error if the file cannot be read or if the read is incomplete.
std::vector<std::uint8_t> read_file_bytes(const std::string& path);

// Read regular files in |dir| and return them as (filename, content) entries.
// When |extensions| is non-empty, only files whose extension matches one of the
// entries (e.g. ".h", ".cpp") are included.
// Returns an empty vector if |dir| does not exist or is not a directory.
std::vector<tt::jit_build::GeneratedFile> read_directory_files(
    const std::string& dir, std::span<const std::string> extensions = {});

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
