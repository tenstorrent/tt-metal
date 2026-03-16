// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/filesystem_utils.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <mutex>
#include <random>
#include <system_error>
#include <thread>

#include <tt-logger/tt-logger.hpp>
#include <fcntl.h>
#include <unistd.h>

namespace tt::filesystem {

static std::atomic<bool> g_nfs_safety{false};

bool nfs_safety_enabled() { return g_nfs_safety.load(std::memory_order_relaxed); }
void set_nfs_safety(bool enabled) { g_nfs_safety.store(enabled, std::memory_order_seq_cst); }

int get_retry_jitter_ms() {
    thread_local std::mt19937 rng{std::random_device{}()};
    thread_local std::uniform_int_distribution<> jitter_dist(0, kFsRetryJitterMs);
    return jitter_dist(rng);
}

}  // namespace tt::filesystem

namespace tt::filesystem {

void sync_filesystem(const std::filesystem::path& path) {
#ifdef __linux__
    // Use explicit string conversion for portability (path::c_str() may be wchar_t* on Windows)
    std::string path_str = path.string();

    // Try to open the path as a directory directly to avoid TOCTOU race.
    // Using O_DIRECTORY ensures we only succeed if it's actually a directory.
    int fd = ::open(path_str.c_str(), O_RDONLY | O_DIRECTORY);

    if (fd == -1 && errno == ENOTDIR) {
        // Path exists but is not a directory - try to open its parent.
        std::filesystem::path parent = path.parent_path();

        // Handle empty parent_path() case (e.g., relative path "file.txt" with no directory)
        if (parent.empty()) {
            parent = std::filesystem::current_path();
        }

        fd = ::open(parent.string().c_str(), O_RDONLY | O_DIRECTORY);
    }

    if (fd == -1) {
        log_debug(tt::LogMetal, "Failed to open path for syncfs: {}: {}", path_str, ::strerror(errno));
        return;
    }
    if (::syncfs(fd) != 0) {
        log_debug(tt::LogMetal, "syncfs failed for {}: {}", path_str, ::strerror(errno));
    }
    if (::close(fd) != 0) {
        log_debug(tt::LogMetal, "close failed after syncfs for {}: {}", path_str, ::strerror(errno));
    }
#else
    ::sync();
#endif
}

bool safe_remove(const std::filesystem::path& path) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        std::filesystem::remove(path, ec);
        if (!ec || is_not_found_error(ec)) {
            return true;
        }
        log_warning(tt::LogMetal, "Failed to remove {}: {}", path.string(), ec.message());
        return false;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::remove(path, ec);
        if (!ec) {
            return true;
        }
        if (is_not_found_error(ec)) {
            return true;
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to remove {}: {}", path.string(), ec.message());
            return false;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds((kFsRetryDelayMs * (attempt + 1)) + get_retry_jitter_ms()));
        }
    }
    log_warning(tt::LogMetal, "Failed to remove {} after {} retries: {}", path.string(), kMaxFsRetries, ec.message());
    return false;
}

bool safe_remove_all(const std::filesystem::path& path) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        std::filesystem::remove_all(path, ec);
        if (!ec || is_not_found_error(ec)) {
            return true;
        }
        log_warning(tt::LogMetal, "Failed to remove_all {}: {}", path.string(), ec.message());
        return false;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::remove_all(path, ec);
        if (!ec) {
            return true;
        }
        if (is_not_found_error(ec)) {
            return true;
        }
        // Retry on ESTALE (stale NFS handle) or ENOTEMPTY (transient race condition).
        // ENOTEMPTY can occur when:
        //   - Another process creates files while remove_all is running
        //   - NFS "silly-rename" files (.nfsXXXX) appear when deleting files held open by other processes
        //   - Mount points exist inside the directory
        // Retrying gives concurrent operations time to complete.
        bool is_retryable = is_estale_error(ec) || ec == std::errc::directory_not_empty;
        if (!is_retryable) {
            log_warning(tt::LogMetal, "Failed to remove_all {}: {}", path.string(), ec.message());
            return false;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds((kFsRetryDelayMs * (attempt + 1)) + get_retry_jitter_ms()));
        }
    }
    log_warning(
        tt::LogMetal, "Failed to remove_all {} after {} retries: {}", path.string(), kMaxFsRetries, ec.message());
    return false;
}

bool safe_rename(const std::filesystem::path& src, const std::filesystem::path& dst, bool ignore_missing) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        std::filesystem::rename(src, dst, ec);
        if (!ec) {
            return true;
        }
        if (ignore_missing && is_not_found_error(ec)) {
            return true;
        }
        log_warning(tt::LogMetal, "Failed to rename {} to {}: {}", src.string(), dst.string(), ec.message());
        return false;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::rename(src, dst, ec);
        if (!ec) {
            return true;
        }
        if (ignore_missing && is_not_found_error(ec)) {
            return true;
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to rename {} to {}: {}", src.string(), dst.string(), ec.message());
            return false;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds((kFsRetryDelayMs * (attempt + 1)) + get_retry_jitter_ms()));
        }
    }
    log_warning(
        tt::LogMetal,
        "Failed to rename {} to {} after {} retries: {}",
        src.string(),
        dst.string(),
        kMaxFsRetries,
        ec.message());
    return false;
}

bool safe_hard_link_or_copy(const std::filesystem::path& target, const std::filesystem::path& link) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        std::filesystem::create_hard_link(target, link, ec);
        if (!ec) {
            return true;
        }
        // Hard link failed -- fall back to copy unconditionally.  The old
        // implementation always fell back to copy on any hard-link error and
        // restricting the fallback to specific error codes caused regressions
        // in environments where unexpected errno values are returned.
        ec.clear();
        std::filesystem::copy_file(target, link, std::filesystem::copy_options::overwrite_existing, ec);
        if (!ec) {
            return true;
        }
        log_warning(
            tt::LogMetal, "Failed to hard_link or copy {} to {}: {}", target.string(), link.string(), ec.message());
        return false;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::create_hard_link(target, link, ec);
        if (!ec) {
            return true;
        }
        if (ec == std::errc::file_exists) {
            if (safe_remove(link)) {
                continue;
            }
            return false;
        }
        if (is_estale_error(ec)) {
            if (attempt < kMaxFsRetries - 1) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds((kFsRetryDelayMs * (attempt + 1)) + get_retry_jitter_ms()));
            }
            continue;
        }
        // Hard link failed for a non-transient reason -- fall back to copy
        // unconditionally (matches pre-NFS-safety behavior).
        ec.clear();
        std::filesystem::copy_file(target, link, std::filesystem::copy_options::overwrite_existing, ec);
        if (!ec) {
            return true;
        }
        if (is_estale_error(ec)) {
            if (attempt < kMaxFsRetries - 1) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds((kFsRetryDelayMs * (attempt + 1)) + get_retry_jitter_ms()));
            }
            continue;
        }
        log_warning(
            tt::LogMetal, "Failed to hard_link or copy {} to {}: {}", target.string(), link.string(), ec.message());
        return false;
    }
    log_warning(
        tt::LogMetal,
        "Failed to hard_link or copy {} to {} after {} retries: {}",
        target.string(),
        link.string(),
        kMaxFsRetries,
        ec.message());
    return false;
}

bool safe_create_directories(const std::filesystem::path& path) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        std::filesystem::create_directories(path, ec);
        if (!ec) {
            return true;
        }
        if (ec == std::errc::file_exists && std::filesystem::is_directory(path)) {
            return true;
        }
        log_warning(tt::LogMetal, "Failed to create directories {}: {}", path.string(), ec.message());
        return false;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::create_directories(path, ec);
        if (!ec) {
            return true;
        }
        if (ec == std::errc::file_exists) {
            std::error_code type_ec;
            bool is_dir = std::filesystem::is_directory(path, type_ec);
            if (!type_ec && is_dir) {
                return true;
            }
            if (type_ec && is_estale_error(type_ec)) {
                ec = type_ec;
            } else {
                if (type_ec) {
                    log_warning(
                        tt::LogMetal,
                        "Path {} already exists but is not a directory: {}",
                        path.string(),
                        type_ec.message());
                } else {
                    log_warning(tt::LogMetal, "Path {} already exists but is not a directory", path.string());
                }
                return false;
            }
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to create directories {}: {}", path.string(), ec.message());
            return false;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds((kFsRetryDelayMs * (attempt + 1)) + get_retry_jitter_ms()));
        }
    }
    log_warning(
        tt::LogMetal,
        "Failed to create directories {} after {} retries: {}",
        path.string(),
        kMaxFsRetries,
        ec.message());
    return false;
}

std::optional<bool> safe_exists(const std::filesystem::path& path) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        bool result = std::filesystem::exists(path, ec);
        if (!ec) {
            return result;
        }
        log_warning(tt::LogMetal, "Failed to check existence of {}: {}", path.string(), ec.message());
        return std::nullopt;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        bool result = std::filesystem::exists(path, ec);
        if (!ec) {
            return result;
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to check existence of {}: {}", path.string(), ec.message());
            return std::nullopt;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1)));
        }
    }
    log_warning(
        tt::LogMetal,
        "Failed to check existence of {} after {} retries: {}",
        path.string(),
        kMaxFsRetries,
        ec.message());
    return std::nullopt;
}

std::optional<bool> safe_is_directory(const std::filesystem::path& path) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        bool result = std::filesystem::is_directory(path, ec);
        if (!ec) {
            return result;
        }
        if (is_not_found_error(ec)) {
            return false;
        }
        log_warning(tt::LogMetal, "Failed to check if {} is directory: {}", path.string(), ec.message());
        return std::nullopt;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        bool result = std::filesystem::is_directory(path, ec);
        if (!ec) {
            return result;
        }
        if (is_not_found_error(ec)) {
            return false;
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to check if {} is directory: {}", path.string(), ec.message());
            return std::nullopt;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1)));
        }
    }
    log_warning(
        tt::LogMetal,
        "Failed to check if {} is directory after {} retries: {}",
        path.string(),
        kMaxFsRetries,
        ec.message());
    return std::nullopt;
}

std::optional<bool> safe_is_regular_file(const std::filesystem::path& path) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        bool result = std::filesystem::is_regular_file(path, ec);
        if (!ec) {
            return result;
        }
        if (is_not_found_error(ec)) {
            return false;
        }
        log_warning(tt::LogMetal, "Failed to check if {} is regular file: {}", path.string(), ec.message());
        return std::nullopt;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        bool result = std::filesystem::is_regular_file(path, ec);
        if (!ec) {
            return result;
        }
        if (is_not_found_error(ec)) {
            return false;
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to check if {} is regular file: {}", path.string(), ec.message());
            return std::nullopt;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1)));
        }
    }
    log_warning(
        tt::LogMetal,
        "Failed to check if {} is regular file after {} retries: {}",
        path.string(),
        kMaxFsRetries,
        ec.message());
    return std::nullopt;
}

std::optional<std::uintmax_t> safe_file_size(const std::filesystem::path& path) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        std::uintmax_t size = std::filesystem::file_size(path, ec);
        if (!ec) {
            return size;
        }
        if (is_not_found_error(ec)) {
            return std::nullopt;
        }
        log_warning(tt::LogMetal, "Failed to get file size of {}: {}", path.string(), ec.message());
        return std::nullopt;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::uintmax_t size = std::filesystem::file_size(path, ec);
        if (!ec) {
            return size;
        }
        if (is_not_found_error(ec)) {
            return std::nullopt;
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to get file size of {}: {}", path.string(), ec.message());
            return std::nullopt;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1)));
        }
    }
    log_warning(
        tt::LogMetal, "Failed to get file size of {} after {} retries: {}", path.string(), kMaxFsRetries, ec.message());
    return std::nullopt;
}

std::optional<std::filesystem::file_time_type> safe_last_write_time(const std::filesystem::path& path) {
    std::error_code ec;
    if (!nfs_safety_enabled()) {
        auto time = std::filesystem::last_write_time(path, ec);
        if (!ec) {
            return time;
        }
        if (is_not_found_error(ec)) {
            return std::nullopt;
        }
        log_warning(tt::LogMetal, "Failed to get last write time of {}: {}", path.string(), ec.message());
        return std::nullopt;
    }
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        auto time = std::filesystem::last_write_time(path, ec);
        if (!ec) {
            return time;
        }
        if (is_not_found_error(ec)) {
            return std::nullopt;
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to get last write time of {}: {}", path.string(), ec.message());
            return std::nullopt;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1)));
        }
    }
    log_warning(
        tt::LogMetal,
        "Failed to get last write time of {} after {} retries: {}",
        path.string(),
        kMaxFsRetries,
        ec.message());
    return std::nullopt;
}

std::vector<std::filesystem::directory_entry> safe_directory_entries(const std::filesystem::path& path) {
    std::vector<std::filesystem::directory_entry> entries;
    std::error_code ec;

    if (!nfs_safety_enabled()) {
        auto it =
            std::filesystem::directory_iterator(path, std::filesystem::directory_options::skip_permission_denied, ec);
        if (ec) {
            if (!is_not_found_error(ec)) {
                log_warning(tt::LogMetal, "Failed to iterate directory {}: {}", path.string(), ec.message());
            }
            return entries;
        }
        while (it != std::filesystem::directory_iterator()) {
            entries.push_back(*it);
            it.increment(ec);
            if (ec) {
                break;
            }
        }
        return entries;
    }

    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        entries.clear();
        auto it =
            std::filesystem::directory_iterator(path, std::filesystem::directory_options::skip_permission_denied, ec);

        if (ec) {
            if (is_not_found_error(ec)) {
                return entries;
            }
            if (!is_estale_error(ec)) {
                log_warning(tt::LogMetal, "Failed to iterate directory {}: {}", path.string(), ec.message());
                return entries;
            }
            if (attempt < kMaxFsRetries - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1)));
            }
            continue;
        }

        bool estale_during_iteration = false;
        while (it != std::filesystem::directory_iterator()) {
            try {
                entries.push_back(*it);
            } catch (const std::filesystem::filesystem_error& e) {
                log_debug(tt::LogMetal, "Skipping stale directory entry in {}: {}", path.string(), e.what());
                if (is_estale_error(e.code())) {
                    estale_during_iteration = true;
                    break;
                }
            }
            it.increment(ec);
            if (ec) {
                if (is_estale_error(ec)) {
                    estale_during_iteration = true;
                    break;
                }
                // Other errors during iteration: skip remaining entries
                break;
            }
        }

        if (!estale_during_iteration) {
            return entries;
        }

        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1)));
        }
    }

    log_warning(
        tt::LogMetal,
        "Failed to iterate directory {} after {} retries: {}",
        path.string(),
        kMaxFsRetries,
        ec.message());
    return entries;
}

void fsync_file(const std::filesystem::path& path) {
#ifdef __linux__
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        log_debug(tt::LogMetal, "Failed to open {} for fsync: {}", path.string(), ::strerror(errno));
        return;
    }
    ::fsync(fd);
    ::close(fd);

    auto parent = path.parent_path();
    if (!parent.empty()) {
        fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
        if (fd >= 0) {
            ::fsync(fd);
            ::close(fd);
        }
    }
#else
    (void)path;
#endif
}

namespace {
std::mutex g_async_sync_mutex;
std::future<void> g_pending_sync;
}  // namespace

void async_sync_filesystem(const std::filesystem::path& path) {
    std::lock_guard lk(g_async_sync_mutex);
    if (g_pending_sync.valid()) {
        g_pending_sync.get();
    }
    g_pending_sync = std::async(std::launch::async, [p = path] { sync_filesystem(p); });
}

void wait_for_pending_sync() {
    std::lock_guard lk(g_async_sync_mutex);
    if (g_pending_sync.valid()) {
        g_pending_sync.get();
    }
}

size_t remove_empty_parent_directories(const std::filesystem::path& path) {
    size_t removed_count = 0;
    std::error_code ec;

    std::filesystem::path current = path;
    while (true) {
        // Check if directory is empty
        // A directory is considered empty if it has no entries other than . and ..
        bool is_empty = std::filesystem::is_empty(current, ec);
        if (ec || !is_empty) {
            // Directory is not empty or error occurred - stop here
            break;
        }

        // Get parent before removing
        std::filesystem::path parent = current.parent_path();

        // Don't remove if we're at the filesystem root (no parent)
        if (parent.empty() || parent == current) {
            break;
        }

        // Try to remove the empty directory.
        // Use std::filesystem::remove (not remove_all) so that if another process
        // adds files between the is_empty check and this call, the remove fails
        // with ENOTEMPTY instead of recursively deleting another process's files.
        if (nfs_safety_enabled()) {
            bool removed = false;
            for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
                std::filesystem::remove(current, ec);
                if (!ec || is_not_found_error(ec)) {
                    removed = true;
                    break;
                }
                if (!is_estale_error(ec)) {
                    break;
                }
                if (attempt < kMaxFsRetries - 1) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds((kFsRetryDelayMs * (attempt + 1)) + get_retry_jitter_ms()));
                }
            }
            if (!removed) {
                break;
            }
        } else {
            std::filesystem::remove(current, ec);
            if (ec) {
                break;  // Failed to remove, stop here
            }
        }

        ++removed_count;
        current = parent;
    }

    return removed_count;
}

}  // namespace tt::filesystem
