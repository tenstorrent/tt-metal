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

// NFS-Specific Filesystem Utilities
// =================================
//
// These utilities provide wrappers around std::filesystem operations with
// built-in handling for NFS-specific failure modes. They are designed for
// use in multi-process and multi-host environments where NFS filesystems
// are shared across many clients.
//
// Why ESTALE occurs
// -----------------
// ESTALE ("Stale file handle") is an NFS-specific error that occurs when
// the NFS client holds a file handle that is no longer valid on the server.
// Common causes include:
//
//   - Attribute caching: NFS clients cache file attributes for performance.
//     When one client modifies a file, other clients may have stale cached
//     handles until their cache expires.
//
//   - Rapid file operations: When one process creates and immediately renames
//     a file while another process attempts to access it, the file handle
//     may become invalid between the lookup and the operation.
//
//   - Multi-host contention: In distributed environments with multiple hosts
//     accessing the same NFS share, file handles can become stale when other
//     hosts perform operations that invalidate client-side caches.
//
//   - Directory reorganization: When directories are modified (e.g., entries
//     added or removed), existing file handles to entries in those directories
//     may become stale.
//
// The utilities below automatically retry operations that fail with ESTALE,
// giving the NFS cache time to refresh before retrying.
//
// Sync() behavior
// ---------------
// Write operations (remove, rename, create_directories, hard_link/copy) call
// ::sync() after successful completion. This ensures:
//
//   - Data durability: All written data is flushed to stable storage on
//     the NFS server before returning.
//
//   - Cross-client visibility: Other NFS clients will see the changes
//     immediately, rather than waiting for their cache to expire.
//
//   - Corruption prevention: In crash scenarios, data is less likely to be
//     lost or corrupted.
//
// Performance implications:
//   - ::sync() is a process-wide flush that affects ALL pending I/O across
//     the entire process, not just the file being operated on.
//   - This can impact concurrent I/O operations in other threads.
//   - Use these utilities when correctness and durability matter more than
//     raw performance. For high-throughput scenarios, use std::filesystem
//     directly with your own error handling.
//
// Blocking behavior
// -----------------
// All functions in this namespace are blocking and may sleep during retry:
//
//   - Maximum retry attempts: 5 (kMaxFsRetries)
//   - Base delay per attempt: 500ms * attempt_number (linear backoff)
//   - Jitter: Write operations add up to 100ms of random jitter to prevent
//     thundering herd issues when multiple processes retry simultaneously
//
//   - Maximum total sleep time for write operations: ~7.5 seconds
//     (500 + 1000 + 1500 + 2000 + 2500 = 7500ms, plus up to 100ms jitter)
//
//   - Maximum total sleep time for read-only operations: ~7.5 seconds
//     (500 + 1000 + 1500 + 2000 + 2500 = 7500ms, no jitter)
//
//   Note: Read-only operations (exists, is_directory, file_size, etc.) do not
//   include jitter since they don't modify state. Write operations (remove,
//   rename, create_directories, hard_link/copy) include random jitter to
//   reduce contention in high-concurrency scenarios.
//
//   If all retries are exhausted, the function returns failure (false or
//   std::nullopt) and logs a warning.

namespace tt::filesystem {

// Maximum number of retries for filesystem operations on NFS
inline constexpr int kMaxFsRetries = 5;
// Base delay between retries (in milliseconds), multiplied by attempt number
inline constexpr int kFsRetryDelayMs = 250;
// Maximum random jitter for write operations (in milliseconds) to prevent thundering herd
inline constexpr int kFsRetryJitterMs = 100;

// Check if error code is ESTALE (stale file handle) - NFS specific error
inline bool is_estale_error(const std::error_code& ec) { return ec.value() == ESTALE; }

// Check if error is ENOENT (file/directory not found)
inline bool is_not_found_error(const std::error_code& ec) { return ec == std::errc::no_such_file_or_directory; }

// Safe remove that ignores ENOENT and retries on ESTALE.
// Calls ::sync() after successful removal. Includes jitter on retries.
// Returns true if file was removed or didn't exist, false on other errors.
bool safe_remove(const std::filesystem::path& path);

// Safe remove_all that ignores ENOENT/ENOTEMPTY and retries on ESTALE.
// Calls ::sync() after successful removal. Includes jitter on retries.
// Returns true if directory was removed or didn't exist, false on other errors.
bool safe_remove_all(const std::filesystem::path& path);

// Safe rename that retries on ESTALE errors.
// Calls ::sync() after successful rename. Includes jitter on retries.
// If ignore_missing is true, ENOENT errors are ignored (returns true).
// Returns true on success, false on non-retryable errors.
bool safe_rename(const std::filesystem::path& src, const std::filesystem::path& dst, bool ignore_missing = false);

// Safe create_hard_link with fallback to copy_file, retrying on ESTALE.
// Calls ::sync() after successful link or copy. Includes jitter on retries.
// Returns true on success, false on failure.
bool safe_hard_link_or_copy(const std::filesystem::path& target, const std::filesystem::path& link);

// Safe create_directories that ignores "already exists" and retries on ESTALE.
// Calls ::sync() after successful creation. Includes jitter on retries.
// Returns true on success (directory exists or was created), false on other errors.
bool safe_create_directories(const std::filesystem::path& path);

// Safe exists check with ESTALE retry (no sync, no jitter).
// Returns true if path exists, false if it doesn't exist, std::nullopt on error.
std::optional<bool> safe_exists(const std::filesystem::path& path);

// Safe directory check with ESTALE retry (no sync, no jitter).
// Returns true if path exists and is a directory, false if it doesn't exist or isn't a directory,
// std::nullopt on error.
std::optional<bool> safe_is_directory(const std::filesystem::path& path);

// Safe regular file check with ESTALE retry (no sync, no jitter).
// Returns true if path exists and is a regular file, false if it doesn't exist or isn't a regular file,
// std::nullopt on error.
std::optional<bool> safe_is_regular_file(const std::filesystem::path& path);

// Safe file size query with ESTALE retry (no sync, no jitter).
// Returns file size on success, std::nullopt on error (including file not found).
std::optional<std::uintmax_t> safe_file_size(const std::filesystem::path& path);

// Safe last_write_time query with ESTALE retry (no sync, no jitter).
// Returns file time on success, std::nullopt on error (including file not found).
std::optional<std::filesystem::file_time_type> safe_last_write_time(const std::filesystem::path& path);

// Safe directory iteration with ESTALE retry (no sync, no jitter).
// Returns vector of directory entries, skipping entries that disappear during iteration.
// Returns empty vector on error or if directory doesn't exist.
std::vector<std::filesystem::directory_entry> safe_directory_entries(const std::filesystem::path& path);

}  // namespace tt::filesystem
