// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/filesystem_utils.hpp"

#include <chrono>
#include <filesystem>
#include <random>
#include <system_error>
#include <thread>

#include <tt-logger/tt-logger.hpp>
#include <fcntl.h>
#include <unistd.h>

namespace {
// Thread-local random number generator for jitter in retry delays.
// Using thread_local ensures thread-safety (avoids data races on rand()).
thread_local std::mt19937 rng{std::random_device{}()};
thread_local std::uniform_int_distribution<> jitter_dist(0, tt::filesystem::kFsRetryJitterMs);

int get_retry_jitter_ms() { return jitter_dist(rng); }

// Sync the filesystem containing the given path.
// Uses syncfs() when available (Linux) to limit scope to specific filesystem,
// falling back to sync() on other platforms. This reduces impact on concurrent
// I/O operations in other threads compared to process-wide sync().
void sync_filesystem(const std::filesystem::path& path) {
#ifdef __linux__
    // Try to open the path (or its parent directory if it's not a directory itself)
    // and sync just that filesystem using syncfs().
    std::error_code ec;
    std::filesystem::path sync_target = path;
    if (!std::filesystem::is_directory(path, ec)) {
        sync_target = path.parent_path();
    }
    // Open directory to get file descriptor for syncfs
    int fd = ::open(sync_target.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd != -1) {
        ::syncfs(fd);
        ::close(fd);
        return;
    }
#endif
    // Fallback to process-wide sync() on non-Linux platforms or if open fails
    ::sync();
}
}  // namespace

namespace tt::filesystem {

bool safe_remove(const std::filesystem::path& path) {
    std::error_code ec;
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::remove(path, ec);
        if (!ec) {
            sync_filesystem(path);
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
                std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1) + get_retry_jitter_ms()));
        }
    }
    log_warning(tt::LogMetal, "Failed to remove {} after {} retries: {}", path.string(), kMaxFsRetries, ec.message());
    return false;
}

bool safe_remove_all(const std::filesystem::path& path) {
    std::error_code ec;
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::remove_all(path, ec);
        if (!ec) {
            sync_filesystem(path);
            return true;
        }
        if (is_not_found_error(ec) || ec == std::errc::directory_not_empty) {
            return true;
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to remove_all {}: {}", path.string(), ec.message());
            return false;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1) + get_retry_jitter_ms()));
        }
    }
    log_warning(
        tt::LogMetal, "Failed to remove_all {} after {} retries: {}", path.string(), kMaxFsRetries, ec.message());
    return false;
}

bool safe_rename(const std::filesystem::path& src, const std::filesystem::path& dst, bool ignore_missing) {
    std::error_code ec;
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::rename(src, dst, ec);
        if (!ec) {
            sync_filesystem(dst);
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
                std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1) + get_retry_jitter_ms()));
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
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::create_hard_link(target, link, ec);
        if (!ec) {
            sync_filesystem(link);
            return true;
        }
        if (!is_estale_error(ec)) {
            ec.clear();
            std::filesystem::copy_file(target, link, std::filesystem::copy_options::overwrite_existing, ec);
            if (!ec) {
                sync_filesystem(link);
                return true;
            }
            if (!is_estale_error(ec)) {
                log_warning(
                    tt::LogMetal,
                    "Failed to hard_link or copy {} to {}: {}",
                    target.string(),
                    link.string(),
                    ec.message());
                return false;
            }
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1) + get_retry_jitter_ms()));
        }
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
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        std::filesystem::create_directories(path, ec);
        if (!ec || ec == std::errc::file_exists) {
            sync_filesystem(path);
            return true;
        }
        if (!is_estale_error(ec)) {
            log_warning(tt::LogMetal, "Failed to create directories {}: {}", path.string(), ec.message());
            return false;
        }
        if (attempt < kMaxFsRetries - 1) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kFsRetryDelayMs * (attempt + 1) + get_retry_jitter_ms()));
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
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        bool result = std::filesystem::is_directory(path, ec);
        if (!ec) {
            return result;
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
    for (int attempt = 0; attempt < kMaxFsRetries; ++attempt) {
        bool result = std::filesystem::is_regular_file(path, ec);
        if (!ec) {
            return result;
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
                // Entry may have disappeared during iteration (ESTALE, ENOENT), skip it.
                // Only ignore filesystem errors; other exceptions propagate.
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

}  // namespace tt::filesystem
