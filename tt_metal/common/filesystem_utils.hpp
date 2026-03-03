// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <system_error>
#include <vector>

namespace tt::filesystem {

// Maximum number of retries for filesystem operations on NFS
inline constexpr int kMaxFsRetries = 5;
// Base delay between retries (in milliseconds), multiplied by attempt number
inline constexpr int kFsRetryDelayMs = 500;

// Check if error code is ESTALE (stale file handle) - NFS specific error
inline bool is_estale_error(const std::error_code& ec) { return ec.value() == ESTALE; }

// Check if error is ENOENT (file/directory not found)
inline bool is_not_found_error(const std::error_code& ec) { return ec == std::errc::no_such_file_or_directory; }

// Safe remove that ignores ENOENT and retries on ESTALE.
// Returns true if file was removed or didn't exist, false on other errors.
bool safe_remove(const std::filesystem::path& path);

// Safe remove_all that ignores ENOENT/ENOTEMPTY and retries on ESTALE.
// Returns true if directory was removed or didn't exist, false on other errors.
bool safe_remove_all(const std::filesystem::path& path);

// Safe rename that retries on ESTALE errors.
// If ignore_missing is true, ENOENT errors are ignored (returns true).
// Returns true on success, false on non-retryable errors.
bool safe_rename(const std::filesystem::path& src, const std::filesystem::path& dst, bool ignore_missing = false);

// Safe create_hard_link with fallback to copy_file, retrying on ESTALE.
// Returns true on success, false on failure.
bool safe_hard_link_or_copy(const std::filesystem::path& target, const std::filesystem::path& link);

// Safe create_directories that ignores "already exists" and retries on ESTALE.
// Returns true on success (directory exists or was created), false on other errors.
bool safe_create_directories(const std::filesystem::path& path);

// Safe exists check with ESTALE retry.
// Returns true if path exists, false if it doesn't exist or on unrecoverable error.
bool safe_exists(const std::filesystem::path& path);

// Safe directory check with ESTALE retry.
// Returns true if path exists and is a directory, false otherwise.
bool safe_is_directory(const std::filesystem::path& path);

// Safe regular file check with ESTALE retry.
// Returns true if path exists and is a regular file, false otherwise.
bool safe_is_regular_file(const std::filesystem::path& path);

// Safe file size query with ESTALE retry.
// Returns file size on success, std::nullopt on error (including file not found).
std::optional<std::uintmax_t> safe_file_size(const std::filesystem::path& path);

// Safe last_write_time query with ESTALE retry.
// Returns file time on success, std::nullopt on error (including file not found).
std::optional<std::filesystem::file_time_type> safe_last_write_time(const std::filesystem::path& path);

// Safe directory iteration with ESTALE retry.
// Returns vector of directory entries, skipping entries that disappear during iteration.
// Returns empty vector on error or if directory doesn't exist.
std::vector<std::filesystem::directory_entry> safe_directory_entries(const std::filesystem::path& path);

}  // namespace tt::filesystem
