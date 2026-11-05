// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Disk-space-driven eviction for the JIT compile server's on-disk cache.
//
// The cache grows without bound: every new kernel variant adds a
// <root>/<build_key>/kernels/<kernel_name>/ directory and nothing ever removes one. When pruning is
// enabled, a background thread checks free space on the cache filesystem every check_interval.
// Once free space drops below start_below_free_pct it deletes kernel directories, least recently
// used first across all build keys, until free space is back at stop_at_free_pct.
//
// Deleting a kernel directory is safe because the server treats a missing .o/.elf as a cache miss
// and recompiles. What must not happen is deleting one while a request is using it. Every compile
// request holds a Lease on its kernel directory for its whole duration, and the pruner checks for
// leases and renames the directory away under the same mutex, so a request either sees the old
// directory intact or starts from an empty one.
//
// A Lease also stamps the directory's mtime, so last use is exact for every request this server
// handles. Directories last used under an older server only have file timestamps to go on, so the
// pruner takes the newest mtime or atime of anything beneath them.
//
// Firmware (<build_key>/firmware/) and the shared PCH directory are never deleted: firmware is
// uploaded once per client process, and a running client whose firmware vanished would fail every
// later compile until it restarted.

namespace tt::tt_metal::jit_server {

inline constexpr const char* kCacheMinFreePctEnv = "TT_METAL_JIT_SERVER_CACHE_MIN_FREE_PCT";
inline constexpr const char* kCacheTargetFreePctEnv = "TT_METAL_JIT_SERVER_CACHE_TARGET_FREE_PCT";
inline constexpr const char* kCacheMinIdleEnv = "TT_METAL_JIT_SERVER_CACHE_MIN_IDLE_S";
inline constexpr const char* kCacheCheckIntervalEnv = "TT_METAL_JIT_SERVER_CACHE_CHECK_INTERVAL_S";

struct CachePruneConfig {
    // Pruning starts when free space falls below this percentage of the filesystem; 0 disables it.
    double start_below_free_pct = 0.0;
    // A run deletes until free space reaches this percentage.
    double stop_at_free_pct = 0.0;
    // Directories used more recently than this are never deleted.
    std::chrono::seconds min_idle{3600};
    std::chrono::seconds check_interval{300};

    bool enabled() const { return start_below_free_pct > 0.0; }

    // Pruning is opt-in: it is enabled only when kCacheMinFreePctEnv is set to a positive value.
    // Throws std::invalid_argument on a malformed or out-of-range value, so that a typo stops the
    // server at startup instead of silently leaving the disk to fill.
    static CachePruneConfig from_env(
        const std::function<const char*(const char*)>& get_env = [](const char* name) -> const char* {
            return std::getenv(name);
        }) {
        auto read = [&](const char* name) -> std::string {
            const char* value = get_env(name);
            return value != nullptr ? std::string(value) : std::string();
        };
        auto parse_number = [](const std::string& text, const char* name) {
            std::size_t used = 0;
            double value = 0.0;
            try {
                value = std::stod(text, &used);
            } catch (const std::exception&) {
                used = 0;
            }
            if (used == 0 || used != text.size() || !(value >= 0.0)) {
                throw std::invalid_argument(std::string(name) + " must be a non-negative number, got '" + text + "'");
            }
            return value;
        };

        CachePruneConfig config;
        const std::string min_free = read(kCacheMinFreePctEnv);
        if (!min_free.empty()) {
            config.start_below_free_pct = parse_number(min_free, kCacheMinFreePctEnv);
        }
        if (!config.enabled()) {
            return config;
        }
        if (config.start_below_free_pct > 100.0) {
            throw std::invalid_argument(std::string(kCacheMinFreePctEnv) + " must be at most 100");
        }

        const std::string target = read(kCacheTargetFreePctEnv);
        config.stop_at_free_pct = target.empty() ? std::min(config.start_below_free_pct + 10.0, 100.0)
                                                 : parse_number(target, kCacheTargetFreePctEnv);
        if (config.stop_at_free_pct < config.start_below_free_pct || config.stop_at_free_pct > 100.0) {
            throw std::invalid_argument(
                std::string(kCacheTargetFreePctEnv) + " must be between " + kCacheMinFreePctEnv + " and 100");
        }

        const std::string min_idle = read(kCacheMinIdleEnv);
        if (!min_idle.empty()) {
            config.min_idle = std::chrono::seconds(static_cast<std::int64_t>(parse_number(min_idle, kCacheMinIdleEnv)));
        }
        const std::string interval = read(kCacheCheckIntervalEnv);
        if (!interval.empty()) {
            config.check_interval =
                std::chrono::seconds(static_cast<std::int64_t>(parse_number(interval, kCacheCheckIntervalEnv)));
            if (config.check_interval.count() < 1) {
                throw std::invalid_argument(std::string(kCacheCheckIntervalEnv) + " must be at least 1");
            }
        }
        return config;
    }
};

struct DiskSpace {
    std::uint64_t capacity = 0;
    std::uint64_t available = 0;
};

inline DiskSpace statvfs_disk_space(const std::filesystem::path& path) {
    struct statvfs stats{};
    if (::statvfs(path.c_str(), &stats) != 0) {
        throw std::system_error(errno, std::generic_category(), "statvfs " + path.string());
    }
    return {
        static_cast<std::uint64_t>(stats.f_blocks) * stats.f_frsize,
        static_cast<std::uint64_t>(stats.f_bavail) * stats.f_frsize};
}

struct CachePruneResult {
    bool ran = false;  // false when free space was already at or above the start threshold
    std::size_t deleted = 0;
    std::uint64_t freed_bytes = 0;
    double free_pct_before = 0.0;
    double free_pct_after = 0.0;
    bool reached_target = true;
};

class ServerCachePruner {
public:
    using LogFn = std::function<void(const std::string&)>;
    using DiskSpaceFn = std::function<DiskSpace(const std::filesystem::path&)>;

    // Marks one kernel directory as in use for as long as it lives.
    class Lease {
    public:
        Lease() = default;
        Lease(Lease&& other) noexcept : owner_(std::exchange(other.owner_, nullptr)), key_(std::move(other.key_)) {}
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();
                owner_ = std::exchange(other.owner_, nullptr);
                key_ = std::move(other.key_);
            }
            return *this;
        }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease() { release(); }

    private:
        friend class ServerCachePruner;
        Lease(ServerCachePruner* owner, std::string key) : owner_(owner), key_(std::move(key)) {}
        void release() {
            if (owner_ != nullptr) {
                owner_->release(key_);
                owner_ = nullptr;
            }
        }

        ServerCachePruner* owner_ = nullptr;
        std::string key_;
    };

    ServerCachePruner(
        std::filesystem::path cache_root,
        CachePruneConfig config,
        LogFn log = {},
        DiskSpaceFn disk_space = statvfs_disk_space) :
        root_(normalize(std::move(cache_root))),
        config_(config),
        log_(std::move(log)),
        disk_space_(std::move(disk_space)) {}

    ServerCachePruner(const ServerCachePruner&) = delete;
    ServerCachePruner& operator=(const ServerCachePruner&) = delete;
    ~ServerCachePruner() { stop(); }

    const CachePruneConfig& config() const { return config_; }

    // Call before touching kernel_dir. Returns an empty lease when pruning is disabled.
    Lease acquire(const std::filesystem::path& kernel_dir) {
        if (!config_.enabled()) {
            return Lease();
        }
        std::string key = normalize(kernel_dir).string();
        {
            std::lock_guard lock(mutex_);
            ++in_use_[key];
        }
        // Record the use. A directory that does not exist yet gets a fresh mtime when it is created.
        std::error_code ec;
        std::filesystem::last_write_time(key, std::filesystem::file_time_type::clock::now(), ec);
        return Lease(this, std::move(key));
    }

    // Starts the background thread. It first removes deletions an earlier server left half done,
    // then checks free space immediately and every check_interval after that.
    void start() {
        if (!config_.enabled() || worker_.joinable()) {
            return;
        }
        {
            std::lock_guard lock(stop_mutex_);
            stopping_ = false;
        }
        worker_ = std::thread([this] { run(); });
    }

    void stop() {
        {
            std::lock_guard lock(stop_mutex_);
            stopping_ = true;
        }
        stop_cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    // Removes trash directories left by a pruner that died mid-delete. Returns the bytes freed.
    std::uint64_t remove_leftover_trash() {
        std::uint64_t freed = 0;
        for (const auto& build_key : child_dirs(root_)) {
            for (const auto& kernel : child_dirs(build_key / "kernels", /*include_trash=*/true)) {
                for (const auto& entry : child_dirs(kernel, /*include_trash=*/true)) {
                    if (is_trash(entry)) {
                        freed += measure(entry).bytes;
                        std::error_code ec;
                        std::filesystem::remove_all(entry, ec);
                    }
                }
            }
        }
        return freed;
    }

    // One check: does nothing unless free space is below the start threshold.
    CachePruneResult prune_once() {
        std::lock_guard prune_lock(prune_mutex_);
        CachePruneResult result;
        std::error_code ec;
        if (!config_.enabled() || !std::filesystem::is_directory(root_, ec)) {
            return result;
        }
        const DiskSpace before = disk_space_(root_);
        result.free_pct_before = result.free_pct_after = free_pct(before);
        if (result.free_pct_before >= config_.start_below_free_pct) {
            return result;
        }
        result.ran = true;

        const auto target =
            static_cast<std::uint64_t>(static_cast<long double>(before.capacity) * config_.stop_at_free_pct / 100.0L);
        const std::uint64_t needed = target > before.available ? target - before.available : 0;

        const std::time_t now = std::time(nullptr);
        std::vector<Candidate> candidates = scan(now);
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.last_used < b.last_used;
        });

        std::time_t oldest = 0;
        std::time_t newest = 0;
        for (const auto& candidate : candidates) {
            if (result.freed_bytes >= needed) {
                break;
            }
            const std::uint64_t freed = remove(candidate.path, now);
            if (freed > 0) {
                if (result.deleted == 0) {
                    oldest = candidate.last_used;
                }
                newest = candidate.last_used;
                ++result.deleted;
                result.freed_bytes += freed;
            }
        }

        result.reached_target = result.freed_bytes >= needed;
        try {
            result.free_pct_after = free_pct(disk_space_(root_));
        } catch (const std::exception&) {
            result.free_pct_after = result.free_pct_before;
        }
        log(summary(result, needed, now, oldest, newest));
        return result;
    }

private:
    struct Candidate {
        std::time_t last_used = 0;
        std::filesystem::path path;
    };

    struct Usage {
        std::time_t last_used = 0;
        std::uint64_t bytes = 0;
    };

    static constexpr std::string_view kTrashPrefix = ".jit-prune-trash.";

    static std::filesystem::path normalize(const std::filesystem::path& path) {
        std::filesystem::path normal = path.lexically_normal();
        if (!normal.has_filename() && normal.has_parent_path() && normal != normal.root_path()) {
            normal = normal.parent_path();
        }
        return normal;
    }

    static bool is_trash(const std::filesystem::path& path) {
        return path.filename().string().rfind(kTrashPrefix, 0) == 0;
    }

    static bool is_build_key(const std::filesystem::path& path) {
        const std::string name = path.filename().string();
        return !name.empty() &&
               std::all_of(name.begin(), name.end(), [](unsigned char c) { return c >= '0' && c <= '9'; });
    }

    // True when one path is the other or lies inside it.
    static bool overlaps(const std::filesystem::path& a, const std::filesystem::path& b) {
        auto ai = a.begin();
        auto bi = b.begin();
        for (; ai != a.end() && bi != b.end(); ++ai, ++bi) {
            if (*ai != *bi) {
                return false;
            }
        }
        return true;
    }

    static double free_pct(const DiskSpace& space) {
        return space.capacity == 0 ? 100.0 : 100.0 * static_cast<double>(space.available) / space.capacity;
    }

    // Real subdirectories of path, not following symlinks. Entries that vanish mid-scan are skipped.
    static std::vector<std::filesystem::path> child_dirs(
        const std::filesystem::path& path, bool include_trash = false) {
        std::vector<std::filesystem::path> dirs;
        std::error_code ec;
        std::filesystem::directory_iterator it(path, ec);
        for (; !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
            std::error_code type_ec;
            if (it->is_directory(type_ec) && !it->is_symlink(type_ec) && (include_trash || !is_trash(it->path()))) {
                dirs.push_back(it->path());
            }
        }
        return dirs;
    }

    // Newest timestamp and allocated size of a directory tree. The directory mtimes count, and so do
    // each file's mtime and atime; directory atimes are skipped because the scan itself updates them.
    static Usage measure(const std::filesystem::path& path) {
        Usage usage;
        struct stat info{};
        if (::lstat(path.c_str(), &info) != 0) {
            return usage;
        }
        usage.bytes = static_cast<std::uint64_t>(info.st_blocks) * 512;
        usage.last_used = info.st_mtime;
        if (!S_ISDIR(info.st_mode)) {
            usage.last_used = std::max(info.st_mtime, info.st_atime);
            return usage;
        }
        std::error_code ec;
        std::filesystem::directory_iterator it(path, ec);
        for (; !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
            const Usage child = measure(it->path());
            usage.bytes += child.bytes;
            usage.last_used = std::max(usage.last_used, child.last_used);
        }
        return usage;
    }

    std::vector<Candidate> scan(std::time_t now) const {
        std::vector<Candidate> candidates;
        for (const auto& build_key : child_dirs(root_)) {
            if (!is_build_key(build_key)) {
                continue;  // e.g. the shared pch/ directory
            }
            for (const auto& kernel : child_dirs(build_key / "kernels")) {
                for (const auto& variant : child_dirs(kernel)) {
                    const Usage usage = measure(variant);
                    if (idle_enough(usage.last_used, now)) {
                        candidates.push_back({usage.last_used, variant});
                    }
                }
            }
        }
        return candidates;
    }

    bool idle_enough(std::time_t last_used, std::time_t now) const {
        return now - last_used >= static_cast<std::time_t>(config_.min_idle.count());
    }

    bool in_use_locked(const std::filesystem::path& path) const {
        return std::any_of(in_use_.begin(), in_use_.end(), [&](const auto& entry) {
            return overlaps(std::filesystem::path(entry.first), path);
        });
    }

    // Deletes one kernel directory unless a request holds it or used it since the scan. Returns the
    // bytes freed.
    std::uint64_t remove(const std::filesystem::path& path, std::time_t now) {
        std::error_code ec;
        std::filesystem::path trash;
        std::uint64_t bytes = 0;
        {
            std::lock_guard lock(mutex_);
            if (in_use_locked(path)) {
                return 0;
            }
            const Usage usage = measure(path);
            if (usage.bytes == 0 || !idle_enough(usage.last_used, now)) {
                return 0;
            }
            bytes = usage.bytes;
            // Renaming within the same directory needs no new disk blocks, so it still works on a
            // full disk, and it takes the directory out of every request's reach at once.
            trash = path.parent_path() / (std::string(kTrashPrefix) + path.filename().string() + "." +
                                          std::to_string(::getpid()) + "." + std::to_string(++trash_counter_));
            std::filesystem::rename(path, trash, ec);
            if (ec) {
                // Fall back to deleting in place; holding the mutex keeps new requests out meanwhile.
                ec.clear();
                std::filesystem::remove_all(path, ec);
                return ec ? 0 : bytes;
            }
        }
        std::filesystem::remove_all(trash, ec);
        {
            // Drop the kernel's own directory once its last variant is gone. remove() only deletes
            // an empty directory, and the lease check stops it racing a request that is creating one.
            std::lock_guard lock(mutex_);
            const std::filesystem::path kernel = path.parent_path();
            if (!in_use_locked(kernel)) {
                std::filesystem::remove(kernel, ec);
            }
        }
        return bytes;
    }

    void release(const std::string& key) {
        std::lock_guard lock(mutex_);
        auto it = in_use_.find(key);
        if (it != in_use_.end() && --it->second == 0) {
            in_use_.erase(it);
        }
    }

    void run() {
        try {
            const std::uint64_t freed = remove_leftover_trash();
            if (freed > 0) {
                log("Cache prune: removed " + megabytes(freed) + " MB left over from an interrupted prune");
            }
        } catch (const std::exception& e) {
            log(std::string("Cache prune: removing leftover trash failed: ") + e.what());
        }
        while (true) {
            try {
                prune_once();
            } catch (const std::exception& e) {
                log(std::string("Cache prune failed: ") + e.what());
            }
            std::unique_lock lock(stop_mutex_);
            if (stop_cv_.wait_for(lock, config_.check_interval, [this] { return stopping_; })) {
                return;
            }
        }
    }

    void log(const std::string& message) const {
        if (!log_) {
            return;
        }
        try {
            log_(message);
        } catch (...) {
            // A failed log write (e.g. a full disk under a redirected log) must not stop pruning.
        }
    }

    static std::string megabytes(std::uint64_t bytes) { return std::to_string(bytes / (1024 * 1024)); }

    static std::string hours_ago(std::time_t then, std::time_t now) {
        return std::to_string((now - then) / 3600) + " h";
    }

    std::string summary(
        const CachePruneResult& result,
        std::uint64_t needed,
        std::time_t now,
        std::time_t oldest,
        std::time_t newest) const {
        auto pct = [](double value) {
            const auto tenths = static_cast<long long>(value * 10.0 + 0.5);
            return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + "%";
        };
        std::string text = "Cache prune: free space " + pct(result.free_pct_before) + " < " +
                           pct(config_.start_below_free_pct) + "; deleted " + std::to_string(result.deleted) +
                           " kernel directories, " + megabytes(result.freed_bytes) + " MB";
        if (result.deleted > 0) {
            text += " (last used " + hours_ago(oldest, now) + " to " + hours_ago(newest, now) + " ago)";
        }
        text += "; free space now " + pct(result.free_pct_after);
        if (!result.reached_target) {
            text += ". Could not reach " + pct(config_.stop_at_free_pct) + ": " +
                    megabytes(needed - result.freed_bytes) + " MB more is needed, but nothing else has been idle for " +
                    std::to_string(config_.min_idle.count()) + " s (firmware and PCH are never deleted)";
        }
        return text;
    }

    const std::filesystem::path root_;
    const CachePruneConfig config_;
    const LogFn log_;
    const DiskSpaceFn disk_space_;

    std::mutex mutex_;  // guards in_use_ and every rename of a kernel directory
    std::unordered_map<std::string, int> in_use_;
    std::uint64_t trash_counter_ = 0;

    std::mutex prune_mutex_;

    std::mutex stop_mutex_;
    std::condition_variable stop_cv_;
    bool stopping_ = false;
    std::thread worker_;
};

}  // namespace tt::tt_metal::jit_server
