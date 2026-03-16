// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <system_error>
#include <thread>
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
// Sync behavior
// -------------
// The safe_* functions do NOT automatically sync the filesystem after writes.
// Callers that need cross-client visibility (e.g. after merging JIT build
// artifacts to an NFS cache) should call sync_filesystem() explicitly at
// the appropriate boundary.  This avoids the severe performance cost of
// flushing all pending I/O on every individual file operation.
//
// sync_filesystem() ensures:
//   - Data durability: All written data is flushed to stable storage on
//     the NFS server before returning.
//   - Cross-client visibility: Other NFS clients will see the changes
//     immediately, rather than waiting for their cache to expire.
//
// Implementation:
//   - On Linux: Uses syncfs() to sync only the specific filesystem containing
//     the target path, minimizing impact on concurrent I/O in other filesystems.
//   - On other platforms: Falls back to process-wide sync().
//
// Performance implications:
//   - Even with syncfs(), the flush affects all pending I/O on that filesystem.
//   - Call sync_filesystem() only at batch boundaries (e.g. after a set of
//     merges completes), not after each individual file operation.
//
// Blocking behavior
// -----------------
// All functions in this namespace are blocking and may sleep during retry:
//
//   - Maximum retry attempts: 5 (kMaxFsRetries)
//   - Base delay: kFsRetryDelayMs * attempt_number (linear backoff)
//   - Jitter: Write operations add up to kFsRetryJitterMs of random jitter
//     per retry to prevent thundering herd issues
//
//   - Sleep occurs before retries 1-4 (not before first attempt or after last)
//   - Maximum total sleep time for write operations: ~5.4 seconds
//     (500 + 1000 + 1500 + 2000 = 5000ms, plus up to 400ms total jitter)
//
//   - Maximum total sleep time for read-only operations: ~5 seconds
//     (500 + 1000 + 1500 + 2000 = 5000ms, no jitter)
//
//   Note: Read-only operations (exists, is_directory, file_size, etc.) do not
//   include jitter since they don't modify state. Write operations (remove,
//   rename, create_directories, hard_link/copy) include random jitter to
//   reduce contention in high-concurrency scenarios.
//
//   If all retries are exhausted, the function returns failure (false or
//   std::nullopt) and logs a warning.

namespace tt::filesystem {

// NFS safety mode control.
// When enabled, safe_* wrappers use ESTALE retry loops with backoff and jitter.
// When disabled (default), safe_* wrappers delegate directly to std::filesystem
// with a single attempt -- no retry loops, no sleep, no jitter.
// Coupled to scratch-dir configuration: call set_nfs_safety(true) when
// TT_METAL_JIT_SCRATCH is set (i.e. multihost NFS operation).
bool nfs_safety_enabled();
void set_nfs_safety(bool enabled);

// Maximum number of retries for filesystem operations on NFS
inline constexpr int kMaxFsRetries = 5;
// Base delay between retries (in milliseconds), multiplied by attempt number
inline constexpr int kFsRetryDelayMs = 500;
// Maximum random jitter for write operations (in milliseconds) to prevent thundering herd
inline constexpr int kFsRetryJitterMs = 100;

// Check if error code is ESTALE (stale file handle) - NFS specific error
inline bool is_estale_error(const std::error_code& ec) {
    return ec.category() == std::system_category() && ec.value() == ESTALE;
}

// Check if error is ENOENT (file/directory not found)
inline bool is_not_found_error(const std::error_code& ec) { return ec == std::errc::no_such_file_or_directory; }

// Get random jitter (0 to kFsRetryJitterMs) for retry delays.
// Uses thread-local RNG for thread safety.
int get_retry_jitter_ms();

// Generic ESTALE retry helper for low-level C-style syscalls using errno.
// Calls `operation()` up to kMaxFsRetries times, retrying only if
// errno == ESTALE after a failed attempt. Sleeps between retries with
// random jitter to prevent thundering herd issues.
//
// IMPORTANT: This template is designed for C-style syscalls that set errno.
// It should NOT be used with std::filesystem operations (which use std::error_code).
// For std::filesystem operations, use the safe_* wrappers instead.
//
// The operation should:
// - Clear errno before the syscall (errno = 0)
// - Return true on success, false to trigger retry check
// - The syscall should set errno on failure
//
// Returns true if operation eventually succeeded, false if all retries failed.
template <typename Operation>
bool retry_on_estale(Operation&& operation) {
    if (!nfs_safety_enabled()) {
        return operation();
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        if (operation()) {
            return true;
        }
        if (errno != ESTALE) {
            break;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1) + get_retry_jitter_ms()));
        }
    }
    return false;
}

// ESTALE retry helper for C++ operations that report errors via std::error_code.
// Use this instead of retry_on_estale when the operation uses std::filesystem or
// C++ streams (which do not reliably set errno).
//
// The operation lambda receives a std::error_code& and should:
// - Clear or ignore the incoming ec (it is pre-cleared before each attempt)
// - Populate ec on failure (e.g. ec.assign(errno, std::system_category()))
// - Return true on success, false on failure
//
// Returns true if operation eventually succeeded, false if all retries failed.
// On failure, ec contains the error from the last attempt.
template <typename Operation>
bool retry_on_estale_ec(Operation&& operation, std::error_code& ec) {
    ec.clear();
    if (!nfs_safety_enabled()) {
        return operation(ec);
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        ec.clear();
        if (operation(ec)) {
            return true;
        }
        if (!is_estale_error(ec)) {
            break;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1) + get_retry_jitter_ms()));
        }
    }
    return false;
}

// Flush all pending writes on the filesystem containing `path` to stable storage.
// On Linux, uses syncfs() scoped to that filesystem; elsewhere falls back to sync().
// Call this at batch boundaries (e.g. after merging build artifacts) rather than
// after every individual file operation.
void sync_filesystem(const std::filesystem::path& path);

// Safe remove that ignores ENOENT and retries on ESTALE.
// Works on both files and empty directories (mirrors std::filesystem::remove).
// Does NOT sync -- call sync_filesystem() at the appropriate boundary.
// Returns true if the entry was removed or didn't exist, false on other errors.
bool safe_remove(const std::filesystem::path& path);

// Safe remove_all that retries on ESTALE and ENOTEMPTY (transient race conditions).
// ENOTEMPTY is retried because it can occur when:
//   - Another process creates files while remove_all is running
//   - NFS "silly-rename" files (.nfsXXXX) appear when deleting files held open by other processes
//   - Mount points exist inside the directory
// Does NOT sync -- call sync_filesystem() at the appropriate boundary.
// Returns true if directory was removed or didn't exist, false on other errors.
bool safe_remove_all(const std::filesystem::path& path);

// Safe rename that retries on ESTALE errors.
// Does NOT sync -- call sync_filesystem() at the appropriate boundary.
// If ignore_missing is true, ENOENT errors are ignored (returns true).
// Returns true on success, false on non-retryable errors.
bool safe_rename(const std::filesystem::path& src, const std::filesystem::path& dst, bool ignore_missing = false);

// Safe create_hard_link with fallback to copy_file, retrying on ESTALE.
// Does NOT sync -- call sync_filesystem() at the appropriate boundary.
// Returns true on success, false on failure.
//
// @warning NFS safety: Callers MUST write to a unique temporary path first
// (e.g. via jit_build::utils::FileRenamer::generate_temp_path()), then
// atomically rename to the final destination. This function operates
// directly on the destination path and does not provide atomicity guarantees.
// See the "NFS write safety" comment in tt_metal/jit_build/build.cpp for
// the full pattern and rationale.
bool safe_hard_link_or_copy(const std::filesystem::path& target, const std::filesystem::path& link);

// Safe create_directories that ignores "already exists" and retries on ESTALE.
// Does NOT sync -- call sync_filesystem() at the appropriate boundary.
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
//
// Consistency guarantee: If ESTALE occurs mid-iteration, the entire operation is
// retried (up to kMaxFsRetries times) with the entries vector cleared before each
// retry. This ensures callers always receive a consistent snapshot of the directory
// at a single point in time, at the cost of potentially higher latency when
// ESTALE errors occur frequently.
std::vector<std::filesystem::directory_entry> safe_directory_entries(const std::filesystem::path& path);

// Targeted fsync of a single file and its parent directory entry.
// Much cheaper than sync_filesystem() (which flushes the entire FS).
// Use for individual small files (e.g. markers) where full-filesystem flush is overkill.
void fsync_file(const std::filesystem::path& path);

// Asynchronously flush the filesystem containing `path`.
// Launches sync_filesystem() on a background thread.  The next call to
// async_sync_filesystem() or wait_for_pending_sync() blocks until the
// previous async sync completes.
void async_sync_filesystem(const std::filesystem::path& path);

// Block until any pending async_sync_filesystem() completes.
void wait_for_pending_sync();

// Remove empty directories from the given path, stopping at the first non-empty directory.
// This is useful for cleaning up scratch directories after successful JIT builds.
// Returns the number of directories removed.
//
// Example: if path is /tmp/tt-jit-build/123/kernels/my_kernel/brisc/
// and only 'brisc' and 'my_kernel' are empty, removes those two directories
// leaving /tmp/tt-jit-build/123/kernels/ intact.
//
// Safety: Only removes empty directories. Non-empty directories act as a barrier.
// Does NOT remove the root of the path if it has no parent (path is at filesystem root).
size_t remove_empty_parent_directories(const std::filesystem::path& path);

}  // namespace tt::filesystem
