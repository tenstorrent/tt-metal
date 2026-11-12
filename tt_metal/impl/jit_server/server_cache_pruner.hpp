// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Disk-space-driven eviction for the JIT compile server's on-disk cache.
//
// The cache grows without bound: every new kernel variant adds a
// <root>/<build_key>/kernels/<kernel_name>/ directory and nothing ever removes one. When pruning is
// enabled, a background thread checks free space on the cache filesystem every check_interval_seconds.
// Once free space drops below start_below_free_pct it deletes kernel directories, least recently
// used first across all build keys, until free space is back at stop_at_free_pct.
//
// Deleting a kernel directory is safe because the server treats a missing .o/.elf as a cache miss
// and recompiles. What must not happen is deleting one while a request is using it. Every compile
// request holds a Lease on its kernel directory for its whole duration, and the pruner checks for
// leases and renames the directory away under the same mutex, so a request either sees the old
// directory intact or starts from an empty one.
//
// A Lease also stamps the directory's mtime, and the pruner ranks directories by that mtime alone:
// one lstat per directory, not a walk of every file beneath it. A directory no request has stamped,
// such as one an older server wrote, keeps the mtime it got when it was compiled, so it counts as
// last used then.
//
// Firmware (<build_key>/firmware/) and the shared PCH directory are never deleted: firmware is
// uploaded once per client process, and a running client whose firmware vanished would fail every
// later compile until it restarted.
//
// Leases only cover this server's own requests, so the cache root must not double as anyone else's
// cache. validate_cache_root() refuses roots that a client's local TT_METAL_CACHE can resolve to,
// since that layout is identical and the pruner would delete ELFs a client is about to load.

namespace tt::tt_metal::jit_server {

inline constexpr const char* kCacheMinFreePctEnv = "TT_METAL_JIT_SERVER_CACHE_MIN_FREE_PCT";
inline constexpr const char* kCacheTargetFreePctEnv = "TT_METAL_JIT_SERVER_CACHE_TARGET_FREE_PCT";
inline constexpr const char* kCacheMinIdleEnv = "TT_METAL_JIT_SERVER_CACHE_MIN_IDLE_S";
inline constexpr const char* kCacheCheckIntervalEnv = "TT_METAL_JIT_SERVER_CACHE_CHECK_INTERVAL_S";
inline constexpr const char* kServerCacheRootEnv = "TT_METAL_JIT_SERVER_CACHE_ROOT";
inline constexpr const char* kDefaultServerCacheRoot = "/tmp/tt-metal-cache/";
// When kCacheTargetFreePctEnv is unset, a run stops this many percentage points above the start
// threshold. The gap is how much of the disk each run frees, so it sets how often the cache is
// scanned: a wider gap means fewer, larger runs.
inline constexpr double kDefaultPruneTargetGapPct = 15.0;
// After a run that frees nothing, checks back off exponentially up to this.
inline constexpr std::chrono::seconds kMaxPruneBackoffSeconds{3600};

struct CachePruneConfig {
    // Pruning starts when free space falls below this percentage of the filesystem; 0 disables it.
    double start_below_free_pct = 0.0;
    // A run deletes until free space reaches this percentage.
    double stop_at_free_pct = 0.0;
    // Directories used more recently than this are never deleted.
    std::chrono::seconds min_idle_seconds{3600};
    std::chrono::seconds check_interval_seconds{300};
    // Prune variables that are set but have no effect because pruning is disabled.
    std::vector<std::string> ignored_settings;

    bool enabled() const { return start_below_free_pct > 0.0; }

    using GetEnvFn = std::function<const char*(const char*)>;

    // Pruning is opt-in: it is enabled only when kCacheMinFreePctEnv is set to a positive value.
    // Every variable is validated whether or not pruning ends up enabled. Throws
    // std::invalid_argument on a malformed or out-of-range value, so that a typo stops the server at
    // startup instead of silently leaving the disk to fill.
    static CachePruneConfig from_env(const GetEnvFn& get_env);
    static CachePruneConfig from_env();

    // When pruning is enabled, throws std::invalid_argument unless kServerCacheRootEnv is set and
    // names a directory that is not a possible client cache: /tmp/tt-metal-cache/,
    // $HOME/.cache/tt-metal-cache/ or $TT_METAL_CACHE/tt-metal-cache/.
    void validate_cache_root(const std::filesystem::path& root, const GetEnvFn& get_env) const;
    void validate_cache_root(const std::filesystem::path& root) const;
};

// Sizes as df reports them: capacity excludes blocks reserved for root, so free space can reach 100%.
struct DiskSpace {
    std::uint64_t capacity = 0;
    std::uint64_t available = 0;
};

DiskSpace disk_space_from_counts(
    std::uint64_t blocks, std::uint64_t free_blocks, std::uint64_t available_blocks, std::uint64_t block_size);
DiskSpace statvfs_disk_space(const std::filesystem::path& path);

struct CachePruneResult {
    bool ran = false;  // false when free space was already at or above the start threshold
    std::size_t deleted = 0;
    std::uint64_t freed_bytes = 0;
    double free_pct_before = 0.0;
    double free_pct_after = 0.0;
    bool reached_target = true;
    // Idle candidates the run did not delete, by reason.
    std::size_t skipped_in_use = 0;
    std::size_t skipped_used_since_scan = 0;
    std::size_t failed = 0;
    std::string first_error;
    std::uint64_t still_needed_bytes = 0;  // measured after the run
    bool interrupted = false;              // stop() was called mid-run
};

// How long to wait before the next check: the configured interval, doubled after each run that
// had to prune but freed nothing (so an unreachable target does not rescan the cache every few
// minutes), capped at kMaxPruneBackoffSeconds.
std::chrono::seconds next_prune_delay(
    std::chrono::seconds previous_delay_seconds, const CachePruneResult& result, const CachePruneConfig& config);

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
        const std::filesystem::path& cache_root,
        CachePruneConfig config,
        LogFn log = {},
        DiskSpaceFn disk_space = statvfs_disk_space);
    ServerCachePruner(const ServerCachePruner&) = delete;
    ServerCachePruner& operator=(const ServerCachePruner&) = delete;
    ~ServerCachePruner();

    const CachePruneConfig& config() const { return config_; }

    // Call before touching kernel_dir. Returns an empty lease when pruning is disabled.
    Lease acquire(const std::filesystem::path& kernel_dir);

    // Starts the background thread. It checks free space immediately and then as next_prune_delay
    // says.
    void start();
    // Stops the background thread, interrupting a run in progress between directories.
    void stop();

    // One check: does nothing unless free space is below the start threshold. A run first removes
    // trash an earlier failed or interrupted delete left behind, provided the process that made it
    // has exited.
    CachePruneResult prune_once();

private:
    struct Candidate {
        std::int64_t last_used_epoch_seconds = 0;
        std::uint64_t bytes = 0;
        std::filesystem::path path;
    };

    struct ScanResult {
        std::vector<Candidate> candidates;
        std::vector<std::filesystem::path> trash;
    };

    enum class RemoveOutcome { Deleted, Gone, InUse, UsedSinceScan, Failed };

    ScanResult scan(std::int64_t now_epoch_seconds) const;
    RemoveOutcome remove(
        const Candidate& candidate, std::int64_t now_epoch_seconds, std::uint64_t& freed, std::string& error);
    bool in_use_locked(const std::filesystem::path& path) const;
    bool idle_enough(std::int64_t last_used_epoch_seconds, std::int64_t now_epoch_seconds) const;
    void release(const std::string& key);
    void run();
    void log(const std::string& message) const;
    std::string summary(const CachePruneResult& result) const;
    bool stopping() const { return stopping_.load(std::memory_order_relaxed); }

    const std::filesystem::path root_;
    const CachePruneConfig config_;
    const LogFn log_;
    const DiskSpaceFn disk_space_;

    std::mutex mutex_;                   // guards in_use_ and every rename of a kernel directory
    std::map<std::string, int> in_use_;  // ordered, so leases under a directory are one range
    std::uint64_t trash_counter_ = 0;

    std::mutex prune_mutex_;  // one run at a time

    std::mutex stop_mutex_;
    std::condition_variable stop_cv_;
    std::atomic<bool> stopping_{false};
    std::thread worker_;
};

}  // namespace tt::tt_metal::jit_server
