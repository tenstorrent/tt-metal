// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Bounded on-disk cache.
//
// Metal writes compiled kernels to disk and never deleted them, so on a shared machine the
// cache grew until the disk filled. This keeps a cache root under a size bound by evicting
// whole entries -- the direct child directories of the root, i.e. <build_key> trees -- least
// recently used first.
//
// Bookkeeping lives in dot-prefixed names, which are never treated as entries:
//
//   <root>/<entry>/.inuse       created by any lock-aware build; locked while in use
//   <root>/<entry>/.last_used   recency stamp, rewritten at most once per kTouchInterval
//   <root>/.trim.lock           trim mutual exclusion between processes
//   <root>/.last_trim           when this root was last swept
//   <root>/.trash/<unique>      entries staged for deletion
//
// Only entries carrying .inuse are eviction candidates. That file is created by
// DiskCacheEntryLock the first time a lock-aware build claims the tree, so its presence means
// "a build that participates in the locking protocol has used this". Trees left by binaries
// that predate this component have none and are therefore never touched; they are adopted as
// current builds reuse them. That is also the guard against a mistyped root: an arbitrary
// directory's children carry no .inuse, so nothing is a candidate.
//
// Recency comes from the .last_used stamp, not atime, which is unusable on machines mounted
// relatime or noatime and unreliable on NFS.
//
// Concurrency, which is why this belongs in Metal rather than a cleanup script:
//
//   - trim() takes an exclusive flock on .trim.lock and gives up at once if another process
//     holds it, so only one process trims a root at a time.
//   - Every live process holds a shared flock on its <entry>/.inuse for its whole lifetime.
//     A trim skips any entry it cannot lock exclusively. flock is owned by the open file
//     description, so the kernel releases it when the holder dies, including under SIGKILL --
//     there are no stale claims to time out.
//   - Eviction renames the entry into .trash, releases the entry lock, and only then removes
//     the staged tree. A partly deleted tree is therefore never visible as a cache hit, and a
//     waiting process blocks on a rename rather than on a recursive delete.
//
// Nothing here throws on filesystem failure: a cache that cannot be trimmed is a disk-space
// problem, not a correctness problem, and must never fail a compile.

namespace tt::tt_metal {

// How stale a .last_used stamp must be before a cache hit rewrites it.
inline constexpr std::chrono::seconds kDiskCacheTouchInterval{3600};

// Minimum spacing between automatic trims of one root. Sizing a cache means walking every file
// in it, too expensive to repeat per process start.
inline constexpr std::chrono::seconds kDiskCacheTrimInterval{3600};

// Do not evict an entry used this recently: a guard against racing a writer that has created
// and stamped its tree and is now compiling into it.
inline constexpr std::chrono::seconds kDiskCacheMinEvictionAge{300};

struct DiskCacheConfig {
    // Cache root. Its direct child directories are entries.
    std::filesystem::path root;

    // Byte cap on the sum of all entries. 0, the default, means no cap and so no eviction.
    uint64_t max_size_bytes = 0;

    std::chrono::seconds min_eviction_age = kDiskCacheMinEvictionAge;
};

struct DiskCacheEntry {
    std::string name;
    std::filesystem::path path;

    // Disk usage from allocated blocks rather than apparent size, so sparse files and block
    // rounding count the way the filesystem counts them. A file reachable by several hard
    // links inside one entry is counted once.
    uint64_t size_bytes = 0;

    std::chrono::system_clock::time_point last_used;

    // True when another live process holds the shared .inuse lock.
    bool in_use = false;
};

struct DiskCacheStats {
    uint64_t total_size_bytes = 0;
    // Least recently used first, which is eviction order.
    std::vector<DiskCacheEntry> entries;
};

struct DiskCacheTrimResult {
    uint64_t bytes_before = 0;
    uint64_t bytes_after = 0;
    // Only counts entries whose removal completed, so it can be trusted against du.
    uint64_t bytes_reclaimed = 0;
    size_t entries_removed = 0;
    size_t entries_skipped_in_use = 0;
    size_t entries_skipped_too_young = 0;
    // Set when the pass stopped rather than empty the cache; the bound is then still exceeded.
    size_t entries_kept_as_working_set = 0;

    bool skipped = false;
    std::string skip_reason;
};

enum class DiskCacheEviction : uint8_t {
    Evict,
    KeepInUse,
    KeepTooYoung,
    // Evicting this would empty the cache. An entry larger than the bound can never satisfy
    // it, so evicting would only have the next run regenerate it and the next trim remove it
    // again. Keeping the last one costs one entry's overshoot and buys termination.
    KeepWorkingSet,
    // Within the bound; since entries arrive least recently used first, so is everything after.
    StopScan,
};

// The single definition of eviction policy, shared so no caller can reimplement and drift.
// Feed entries in DiskCacheStats order with the bytes still live ahead of each one and how
// many entries remain on disk. `honor_limits` false means "evict everything not in use".
DiskCacheEviction disk_cache_decide_eviction(
    const DiskCacheEntry& entry,
    const DiskCacheConfig& config,
    uint64_t live_bytes,
    size_t entries_remaining,
    std::chrono::system_clock::time_point now,
    bool honor_limits);

// Kernel cache root as the runtime resolves it: TT_METAL_CACHE with a tt-metal-cache
// subdirectory appended if set, else $HOME/.cache/tt-metal-cache/, else
// /tmp/tt-metal-cache-<uid>/. Always ends in a separator.
std::filesystem::path default_kernel_cache_root();

// Apply TT_METAL_CACHE_MAX_SIZE. Deliberately forgiving: an empty or unparseable value warns
// and leaves the default, because a cache tuning knob must never stop a process running.
void disk_cache_apply_env(DiskCacheConfig& config, const char* max_size);

// Parse "50G", "512M", "1024K", "2T" or a bare byte count. Suffixes are binary and case
// insensitive, and one redundant trailing 'B' is allowed. nullopt if not a valid size.
std::optional<uint64_t> parse_disk_cache_size(std::string_view text);

// Render a byte count for logs, e.g. "3.4 GiB".
std::string format_disk_cache_size(uint64_t bytes);

// Scan the root. Never throws; an unreadable root yields empty stats.
DiskCacheStats disk_cache_stat(const DiskCacheConfig& config);

// Evict least recently used entries until the root is under max_size_bytes, skipping entries
// in use or used more recently than min_eviction_age. Drains leftover .trash. Never throws.
DiskCacheTrimResult disk_cache_trim(const DiskCacheConfig& config);

// Evict every entry not in use, ignoring size and min_eviction_age. Never throws.
DiskCacheTrimResult disk_cache_clear(const DiskCacheConfig& config);

// Run disk_cache_trim() on a detached background thread, at most once per root per process and
// at most once per kDiskCacheTrimInterval per root. Safe to call from device init: it takes no
// Metal locks. Because eviction renames before removing, an abrupt exit leaves only .trash
// entries for the next trim to drain.
void disk_cache_trim_in_background(const DiskCacheConfig& config);

// Rewrite <entry>/.last_used if the stamp is older than kDiskCacheTouchInterval. One stat in
// the common case. Never throws.
void disk_cache_touch(const std::filesystem::path& entry_path);

// Process-lifetime shared lock on <entry>/.inuse, announcing that this entry must not be
// evicted. Acquisition is non-blocking with bounded retries, because a conflicting exclusive
// lock is held only across a rename and device init must never stall on the filesystem.
class DiskCacheEntryLock {
public:
    explicit DiskCacheEntryLock(const std::filesystem::path& entry_path);
    ~DiskCacheEntryLock();

    DiskCacheEntryLock(const DiskCacheEntryLock&) = delete;
    DiskCacheEntryLock& operator=(const DiskCacheEntryLock&) = delete;
    DiskCacheEntryLock(DiskCacheEntryLock&&) = delete;
    DiskCacheEntryLock& operator=(DiskCacheEntryLock&&) = delete;

    bool held() const { return fd_ >= 0; }

private:
    int fd_ = -1;
};

// Hold a DiskCacheEntryLock on entry_path until the process exits, retrying if an earlier
// attempt failed. Returns whether the entry is now protected.
bool disk_cache_hold_entry_for_process(const std::filesystem::path& entry_path);

}  // namespace tt::tt_metal
