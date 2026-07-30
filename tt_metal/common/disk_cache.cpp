// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disk_cache.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <system_error>
#include <thread>
#include <utility>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/indestructible.hpp>

#include "common/filesystem_utils.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace tt::tt_metal {

namespace {

constexpr const char* kTrimLockName = ".trim.lock";
constexpr const char* kLastTrimName = ".last_trim";
constexpr const char* kTrashName = ".trash";
constexpr const char* kInUseName = ".inuse";
constexpr const char* kLastUsedName = ".last_used";

// A conflicting exclusive lock is held only across a rename, so a few short retries cover real
// contention while guaranteeing device init cannot stall on the filesystem.
constexpr int kEntryLockAttempts = 20;
constexpr std::chrono::milliseconds kEntryLockRetryDelay{25};

bool is_reserved_name(std::string_view name) { return name.empty() || name.front() == '.'; }

std::optional<struct ::stat> lstat_or_none(const fs::path& path) {
    struct ::stat st{};
    if (::lstat(path.c_str(), &st) != 0) {
        return std::nullopt;
    }
    return st;
}

std::chrono::system_clock::time_point mtime_of(const struct ::stat& st) {
    return std::chrono::system_clock::from_time_t(st.st_mtime);
}

// Compare recency in whole seconds. Subtracting two system_clock points yields nanoseconds, and
// a large duration promoted to nanoseconds overflows int64 and wraps negative.
std::chrono::seconds idle_for(std::chrono::system_clock::time_point now, std::chrono::system_clock::time_point last) {
    return std::chrono::duration_cast<std::chrono::seconds>(now - last);
}

// futimens with a null spec is what moves mtime without writing a byte; opening alone would
// leave an existing stamp untouched.
void stamp_now(const fs::path& path) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        log_debug(tt::LogMetal, "Disk cache: cannot stamp {}: {}", path.string(), ::strerror(errno));
        return;
    }
    if (::futimens(fd, nullptr) != 0) {
        log_debug(tt::LogMetal, "Disk cache: cannot stamp {}: {}", path.string(), ::strerror(errno));
    }
    ::close(fd);
}

// st_blocks is in 512-byte units by definition, and is what makes a sparse file cost less than
// its apparent length and a one-byte file cost a whole block -- the number `du` reports.
uint64_t allocated_bytes(const struct ::stat& st) { return static_cast<uint64_t>(st.st_blocks) * 512ULL; }

// Sum of allocated bytes under `root`, following no symlinks. The build hard-links artifacts,
// so a file reachable by several links inside one entry must be counted once; tracking only
// nlink>1 files keeps the dedup set empty in the common case.
uint64_t measure_tree_bytes(const fs::path& root) {
    uint64_t total = 0;
    std::set<std::pair<uint64_t, uint64_t>> multi_link_seen;

    if (auto st = lstat_or_none(root)) {
        total += allocated_bytes(*st);
        if (!S_ISDIR(st->st_mode)) {
            return total;
        }
    } else {
        return 0;
    }

    std::error_code ec;
    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        log_debug(tt::LogMetal, "Disk cache: cannot walk {}: {}", root.string(), ec.message());
        return total;
    }
    const fs::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) {
            // A concurrent writer or a vanishing entry is expected; keep the partial sum.
            log_debug(tt::LogMetal, "Disk cache: walk of {} interrupted: {}", root.string(), ec.message());
            break;
        }
        auto st = lstat_or_none(it->path());
        if (!st) {
            continue;
        }
        if (S_ISREG(st->st_mode) && st->st_nlink > 1) {
            const auto key = std::make_pair(static_cast<uint64_t>(st->st_dev), static_cast<uint64_t>(st->st_ino));
            if (!multi_link_seen.insert(key).second) {
                continue;
            }
        }
        total += allocated_bytes(*st);
    }
    return total;
}

// A .last_used stamp is authoritative. Without one, fall back to the newest mtime among the
// entry and its immediate children, which at least moves when kernels/ gains a subdirectory.
// Epoch rather than time_point::min() as the floor: min() overflows the nanosecond difference,
// and "unknown recency" should mean "oldest", which epoch already does.
std::chrono::system_clock::time_point entry_last_used(const fs::path& entry_path) {
    if (auto stamp = lstat_or_none(entry_path / kLastUsedName)) {
        return mtime_of(*stamp);
    }
    auto newest = std::chrono::system_clock::time_point{};
    if (auto st = lstat_or_none(entry_path)) {
        newest = mtime_of(*st);
    }
    for (const auto& child : tt::filesystem::safe_directory_entries(entry_path)) {
        if (auto st = lstat_or_none(child.path())) {
            newest = std::max(newest, mtime_of(*st));
        }
    }
    return newest;
}

// RAII flock, always non-blocking: every caller here would rather skip work, or retry on its
// own terms, than wait on another process.
class ScopedFlock {
public:
    ScopedFlock(const fs::path& path, bool shared, bool create) {
        const int flags = create ? (O_RDWR | O_CREAT | O_CLOEXEC) : (O_RDONLY | O_CLOEXEC);
        const int fd = ::open(path.c_str(), flags, 0644);
        if (fd < 0) {
            return;
        }
        opened_ = true;
        if (::flock(fd, (shared ? LOCK_SH : LOCK_EX) | LOCK_NB) != 0) {
            ::close(fd);
            return;
        }
        fd_ = fd;
    }

    ~ScopedFlock() { release(); }

    ScopedFlock(const ScopedFlock&) = delete;
    ScopedFlock& operator=(const ScopedFlock&) = delete;
    ScopedFlock(ScopedFlock&&) = delete;
    ScopedFlock& operator=(ScopedFlock&&) = delete;

    // Distinguishes "no such file, so nobody is announcing use" from "somebody holds it".
    bool opened() const { return opened_; }
    bool held() const { return fd_ >= 0; }

    // Drop early, to bound how long an evicting trimmer blocks a waiter: the lock covers the
    // rename into .trash, not the recursive delete that follows.
    void release() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    int fd_ = -1;
    bool opened_ = false;
};

// Entries this process has claimed. Consulted by this process's own trimmer, since flock would
// grant our own probe the exclusive lock. Never destroyed: the locks must outlive every static
// destructor that might still touch the cache.
struct HeldLocks {
    std::mutex mutex;
    std::map<std::string, std::unique_ptr<DiskCacheEntryLock>> by_path;
};

HeldLocks& held_locks() {
    static ttsl::Indestructible<HeldLocks> locks;
    return locks.get();
}

bool claimed_by_this_process(const fs::path& entry_path) {
    HeldLocks& locks = held_locks();
    const std::lock_guard lock(locks.mutex);
    const auto it = locks.by_path.find(entry_path.string());
    return it != locks.by_path.end() && it->second->held();
}

// A name no other process can pick: pid, a process-local counter, and the wall clock.
fs::path make_trash_path(const fs::path& trash_dir, const std::string& entry_name) {
    static std::atomic<uint64_t> counter{0};
    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    return trash_dir / fmt::format(
                           "{}-{}-{}-{}",
                           static_cast<long>(::getpid()),
                           static_cast<long long>(now),
                           counter.fetch_add(1, std::memory_order_relaxed),
                           entry_name);
}

// Move the entry out of the lookup namespace. Returns where it was staged, or nullopt if it
// could not be moved. Must hold the entry's exclusive lock. Uses a raw rename because a
// cross-device failure here is expected and handled below, not worth a warning.
std::optional<fs::path> stage_for_deletion(const fs::path& entry_path, const fs::path& trash_dir) {
    const fs::path staged = make_trash_path(trash_dir, entry_path.filename().string());
    std::error_code ec;
    fs::rename(entry_path, staged, ec);
    if (!ec) {
        return staged;
    }
    // Trash landed on another filesystem. Removing in place reintroduces the torn-entry window,
    // so it is a fallback and not the path.
    log_debug(tt::LogMetal, "Disk cache: cannot stage {} ({}), removing in place", entry_path.string(), ec.message());
    if (tt::filesystem::safe_remove_all(entry_path)) {
        return entry_path;  // already gone; nothing left to purge
    }
    return std::nullopt;
}

// Drop whatever a previous interrupted trim left staged. Returns bytes actually reclaimed.
uint64_t drain_trash(const fs::path& trash_dir) {
    uint64_t reclaimed = 0;
    for (const auto& child : tt::filesystem::safe_directory_entries(trash_dir)) {
        const uint64_t bytes = measure_tree_bytes(child.path());
        if (tt::filesystem::safe_remove_all(child.path())) {
            reclaimed += bytes;
        }
    }
    return reclaimed;
}

// Every direct child directory of the root, measured. `claimable` distinguishes the two kinds:
// a directory carrying .inuse has been used by a build that participates in the locking
// protocol and is therefore safe to evict, while one without it may be in use by a binary
// predating that protocol and cannot be told apart from an abandoned one.
//
// Both kinds are measured. Excluding the unclaimable ones from the total would make the bound
// ignore whatever the cache already held, which is exactly the case an operator sets a bound to
// fix.
DiskCacheStats scan_entries(const fs::path& root) {
    DiskCacheStats stats;
    if (!tt::filesystem::safe_is_directory(root).value_or(false)) {
        return stats;
    }

    for (const auto& child : tt::filesystem::safe_directory_entries(root)) {
        const std::string name = child.path().filename().string();
        if (is_reserved_name(name)) {
            continue;
        }
        std::error_code dir_ec;
        // directory_entry caches the type readdir reported, so this is usually free.
        if (!child.is_directory(dir_ec) || dir_ec) {
            continue;
        }

        DiskCacheEntry entry;
        entry.name = name;
        entry.path = child.path();

        {
            // Scoped deliberately tight. This is an *exclusive* probe, so holding it across the
            // recursive walk below would block a starting process for the whole walk -- long
            // enough to exhaust DiskCacheEntryLock's retry budget and leave that process
            // building into an unprotected tree. create=false so probing can never conjure an
            // .inuse file and thereby promote an unclaimable tree into a candidate.
            const ScopedFlock probe(child.path() / kInUseName, /*shared=*/false, /*create=*/false);
            entry.claimable = probe.opened();
            entry.in_use = entry.claimable && !probe.held();
        }
        // Our own claim counts as in use: flock would grant our own probe the exclusive lock, so
        // the file alone cannot tell us.
        entry.in_use = entry.in_use || claimed_by_this_process(child.path());

        entry.size_bytes = measure_tree_bytes(child.path());
        entry.last_used = entry_last_used(child.path());

        stats.total_size_bytes += entry.size_bytes;
        if (!entry.claimable) {
            stats.unclaimable_size_bytes += entry.size_bytes;
            stats.unclaimable_entries++;
        }
        stats.entries.push_back(std::move(entry));
    }

    // Least recently used first. Ties broken by name so the order is deterministic.
    std::sort(stats.entries.begin(), stats.entries.end(), [](const DiskCacheEntry& a, const DiskCacheEntry& b) {
        if (a.last_used != b.last_used) {
            return a.last_used < b.last_used;
        }
        return a.name < b.name;
    });
    return stats;
}

DiskCacheTrimResult skipped_result(std::string reason) {
    DiskCacheTrimResult result;
    result.skipped = true;
    result.skip_reason = std::move(reason);
    return result;
}

DiskCacheTrimResult run_trim(const DiskCacheConfig& config, bool honor_limits) {
    if (!tt::filesystem::safe_is_directory(config.root).value_or(false)) {
        return skipped_result("cache root does not exist");
    }

    // One trimmer per root. Losing the race is normal on a busy machine and not worth a
    // warning: the process holding the lock is doing the same work.
    const ScopedFlock trim_lock(config.root / kTrimLockName, /*shared=*/false, /*create=*/true);
    if (!trim_lock.held()) {
        return skipped_result("another process is trimming this cache");
    }

    DiskCacheTrimResult result;
    const fs::path trash_dir = config.root / kTrashName;
    result.bytes_reclaimed += drain_trash(trash_dir);

    const DiskCacheStats stats = scan_entries(config.root);
    result.bytes_before = stats.total_size_bytes;

    // Evicting what we may reclaim cannot get under the bound, so doing it would free space the
    // next build immediately regenerates and still leave the root over. Report instead.
    if (honor_limits && !config.reclaim_unclaimable && stats.unclaimable_size_bytes > config.max_size_bytes) {
        return skipped_result(fmt::format(
            "{} in {} directories predates in-use marking and cannot be evicted safely, which alone exceeds the {} "
            "limit; set TT_METAL_CACHE_RECLAIM_UNCLAIMED=1 once no older tt-metal build is running",
            format_disk_cache_size(stats.unclaimable_size_bytes),
            stats.unclaimable_entries,
            format_disk_cache_size(config.max_size_bytes)));
    }
    uint64_t live_bytes = stats.total_size_bytes;
    size_t entries_remaining = stats.entries.size();

    const auto now = std::chrono::system_clock::now();
    bool trash_dir_ready = false;

    for (const auto& entry : stats.entries) {
        const DiskCacheEviction decision =
            disk_cache_decide_eviction(entry, config, live_bytes, entries_remaining, now, honor_limits);
        if (decision == DiskCacheEviction::StopScan) {
            break;
        }
        if (decision == DiskCacheEviction::KeepWorkingSet) {
            result.entries_kept_as_working_set++;
            break;
        }
        if (decision == DiskCacheEviction::KeepInUse) {
            result.entries_skipped_in_use++;
            continue;
        }
        if (decision == DiskCacheEviction::KeepUnclaimable) {
            result.entries_skipped_unclaimable++;
            continue;
        }
        if (decision == DiskCacheEviction::KeepTooYoung) {
            result.entries_skipped_too_young++;
            continue;
        }

        std::optional<fs::path> staged;
        {
            // Hold the exclusive lock across the rename and nothing more. Without it a process
            // could take a shared lock between our probe and the rename and believe its tree is
            // protected; holding it across the delete instead would block that process for the
            // whole delete. create=false so an unclaimed tree stays unclaimed.
            ScopedFlock entry_lock(entry.path / kInUseName, /*shared=*/false, /*create=*/false);
            // An unclaimable entry has no .inuse to lock, so there is nothing to acquire and
            // nothing that could be holding it -- requiring the lock here would make
            // reclaim_unclaimable unable to evict anything at all.
            if (entry.claimable && !entry_lock.held()) {
                result.entries_skipped_in_use++;
                continue;
            }
            if (!trash_dir_ready) {
                tt::filesystem::safe_create_directories(trash_dir);
                trash_dir_ready = true;
            }
            staged = stage_for_deletion(entry.path, trash_dir);
        }
        if (!staged.has_value()) {
            log_warning(tt::LogMetal, "Disk cache: failed to evict {}", entry.path.string());
            continue;
        }

        // Gone from the lookup namespace either way; bytes count only if the removal completed.
        result.entries_removed++;
        entries_remaining--;
        live_bytes -= std::min(live_bytes, entry.size_bytes);
        if (*staged == entry.path || tt::filesystem::safe_remove_all(*staged)) {
            result.bytes_reclaimed += entry.size_bytes;
        }
    }

    if (honor_limits) {
        // Lets the automatic path skip the walk until the interval elapses. Written even when
        // nothing was evicted: the expensive part was the scan, and its answer stays good.
        stamp_now(config.root / kLastTrimName);
    }

    result.bytes_after = live_bytes;
    return result;
}

// Split "50G" into 50 and 'g'. nullopt unless `text` is digits followed by at most one unit
// character; one redundant trailing 'b' is dropped first, so "50GB" reads as "50G".
struct ScaledValue {
    uint64_t count = 0;
    char unit = '\0';
};

std::string_view trim_spaces(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.remove_suffix(1);
    }
    return text;
}

std::optional<bool> parse_truthy(std::string_view text) {
    std::string lowered;
    lowered.reserve(text.size());
    for (const char c : trim_spaces(text)) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        return false;
    }
    return std::nullopt;
}

std::optional<ScaledValue> split_scaled_value(std::string_view text) {
    text = trim_spaces(text);
    ScaledValue scaled;
    const char* const end = text.data() + text.size();
    // from_chars takes digits only -- no sign, whitespace or radix prefix -- and reports
    // overflow rather than wrapping.
    const auto parsed = std::from_chars(text.data(), end, scaled.count);
    if (parsed.ec != std::errc{}) {
        return std::nullopt;
    }

    std::string_view unit(parsed.ptr, static_cast<size_t>(end - parsed.ptr));
    // Accept the spellings format_disk_cache_size emits, so a value copied out of a log round
    // trips: "50GiB" -> "50Gi" -> "50G". The trailing 'b' and 'i' are both optional.
    if (unit.size() > 1 && std::tolower(static_cast<unsigned char>(unit.back())) == 'b') {
        unit.remove_suffix(1);
    }
    if (unit.size() > 1 && std::tolower(static_cast<unsigned char>(unit.back())) == 'i') {
        unit.remove_suffix(1);
    }
    if (unit.empty()) {
        return scaled;
    }
    if (unit.size() != 1) {
        return std::nullopt;
    }
    scaled.unit = static_cast<char>(std::tolower(static_cast<unsigned char>(unit.front())));
    return scaled;
}

}  // namespace

DiskCacheEviction disk_cache_decide_eviction(
    const DiskCacheEntry& entry,
    const DiskCacheConfig& config,
    uint64_t live_bytes,
    size_t entries_remaining,
    std::chrono::system_clock::time_point now,
    bool honor_limits) {
    if (honor_limits) {
        if (!(config.max_size_bytes > 0 && live_bytes > config.max_size_bytes)) {
            // Under budget. Must be checked before the working-set rule below, or a cache just
            // trimmed down to one entry would report itself as stuck over its bound.
            return DiskCacheEviction::StopScan;
        }
        if (entries_remaining <= 1) {
            return DiskCacheEviction::KeepWorkingSet;
        }
    }
    if (entry.in_use) {
        return DiskCacheEviction::KeepInUse;
    }
    if (!entry.claimable && !config.reclaim_unclaimable) {
        return DiskCacheEviction::KeepUnclaimable;
    }
    if (honor_limits && idle_for(now, entry.last_used) < config.min_eviction_age) {
        return DiskCacheEviction::KeepTooYoung;
    }
    return DiskCacheEviction::Evict;
}

fs::path default_kernel_cache_root() {
    // Must stay in step with RunTimeOptions::HandleEnvVar(TT_METAL_CACHE). The trailing
    // separator matters: the build appends "<build_key>/..." directly to this string.
    if (const char* configured = std::getenv("TT_METAL_CACHE"); configured != nullptr && *configured != '\0') {
        return (fs::path(configured) / "tt-metal-cache" / "").lexically_normal();
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0' && fs::exists(home)) {
        return fs::path(home) / ".cache" / "tt-metal-cache" / "";
    }
    // Per-uid, because /tmp is shared and a common root would have one user's trimmer sizing
    // and trying to evict another user's entries.
    return fs::path("/tmp") / fmt::format("tt-metal-cache-{}", static_cast<unsigned>(::getuid())) / "";
}

void disk_cache_apply_env(DiskCacheConfig& config, const char* max_size, const char* reclaim) {
    const auto given = [](const char* value) -> std::optional<std::string_view> {
        if (value == nullptr) {
            return std::nullopt;
        }
        const std::string_view text = trim_spaces(value);
        return text.empty() ? std::nullopt : std::optional(text);  // exported empty means unset
    };

    if (auto text = given(max_size)) {
        if (auto parsed = parse_disk_cache_size(*text)) {
            config.max_size_bytes = *parsed;
        } else {
            // Never throw over a tuning knob, but do not let the operator believe a bound is in
            // force when the value did not parse and the default is "no bound".
            log_warning(
                tt::LogMetal,
                "Disk cache: TT_METAL_CACHE_MAX_SIZE='{}' is not a byte count (try 50G, 512M, 20GiB); the kernel "
                "cache is NOT bounded for this run",
                std::string(*text));
        }
    }
    if (auto text = given(reclaim)) {
        if (auto parsed = parse_truthy(*text)) {
            config.reclaim_unclaimable = *parsed;
        } else {
            log_warning(
                tt::LogMetal,
                "Disk cache: ignoring TT_METAL_CACHE_RECLAIM_UNCLAIMED='{}' (expected 0/1, true/false, yes/no or "
                "on/off)",
                std::string(*text));
        }
    }
}

std::optional<uint64_t> parse_disk_cache_size(std::string_view text) {
    const auto scaled = split_scaled_value(text);
    if (!scaled.has_value()) {
        return std::nullopt;
    }
    uint64_t multiplier = 0;
    switch (scaled->unit) {
        case '\0':
        case 'b': multiplier = 1ULL; break;
        case 'k': multiplier = 1ULL << 10; break;
        case 'm': multiplier = 1ULL << 20; break;
        case 'g': multiplier = 1ULL << 30; break;
        case 't': multiplier = 1ULL << 40; break;
        default: return std::nullopt;
    }
    if (scaled->count > std::numeric_limits<uint64_t>::max() / multiplier) {
        return std::nullopt;
    }
    return scaled->count * multiplier;
}

std::string format_disk_cache_size(uint64_t bytes) {
    static constexpr std::array<const char*, 5> kUnits = {"B", "KiB", "MiB", "GiB", "TiB"};
    if (bytes < 1024) {
        return fmt::format("{} B", bytes);
    }
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < kUnits.size()) {
        value /= 1024.0;
        unit++;
    }
    return fmt::format("{:.1f} {}", value, kUnits[unit]);
}

DiskCacheStats disk_cache_stat(const DiskCacheConfig& config) { return scan_entries(config.root); }

DiskCacheTrimResult disk_cache_trim(const DiskCacheConfig& config) {
    if (config.max_size_bytes == 0) {
        // Still drain .trash. It is debris this component staged, and a trim interrupted while
        // a bound was set would otherwise leak it forever once the bound is removed -- invisible
        // to disk_cache_stat, since dot-names are not entries.
        DiskCacheTrimResult result = skipped_result("no size limit configured (set TT_METAL_CACHE_MAX_SIZE to enable)");
        result.bytes_reclaimed = drain_trash(config.root / kTrashName);
        return result;
    }
    return run_trim(config, /*honor_limits=*/true);
}

DiskCacheTrimResult disk_cache_clear(const DiskCacheConfig& config) { return run_trim(config, /*honor_limits=*/false); }

void disk_cache_trim_in_background(const DiskCacheConfig& config) {
    if (config.max_size_bytes == 0) {
        return;
    }

    // One stat standing in for a walk of the whole cache. Racing processes may both get past
    // here; only one wins the trim lock. Checked *before* claiming the root below: claiming
    // first would mean a process that starts inside the throttle window marks the root done and
    // then never trims for the rest of its life, however long that is.
    if (auto stamp = lstat_or_none(config.root / kLastTrimName)) {
        if (idle_for(std::chrono::system_clock::now(), mtime_of(*stamp)) < kDiskCacheTrimInterval) {
            return;
        }
    }

    // One attempt per root per process, so a multi-device bring-up spawns one thread, not one
    // per device.
    {
        static std::mutex mutex;
        static std::set<std::string> trimmed_roots;
        const std::lock_guard lock(mutex);
        if (!trimmed_roots.insert(config.root.string()).second) {
            return;
        }
    }

    // Detached on purpose. Trimming must never sit on the compile path, and a joinable thread
    // would either delay teardown behind a large remove_all or need an owner with a defined
    // lifetime relative to static destruction. Everything it touches is copied in, and eviction
    // renames before removing, so an abrupt exit leaves only .trash debris.
    try {
        std::thread worker([config]() {
            try {
                const DiskCacheTrimResult result = disk_cache_trim(config);
                if (result.entries_removed > 0 || result.bytes_reclaimed > 0) {
                    log_info(
                        tt::LogMetal,
                        "Disk cache: trimmed {} to {} (limit {}), evicted {} entries, reclaimed {}",
                        config.root.string(),
                        format_disk_cache_size(result.bytes_after),
                        format_disk_cache_size(config.max_size_bytes),
                        result.entries_removed,
                        format_disk_cache_size(result.bytes_reclaimed));
                }
            } catch (const std::exception& e) {
                log_debug(tt::LogMetal, "Disk cache: background trim failed: {}", e.what());
            } catch (...) {
                log_debug(tt::LogMetal, "Disk cache: background trim failed");
            }
        });
        worker.detach();
    } catch (const std::system_error& e) {
        // Out of threads. The cache stays unbounded this run, which is the old behaviour.
        log_debug(tt::LogMetal, "Disk cache: could not start background trim: {}", e.what());
    }
}

void disk_cache_touch(const fs::path& entry_path) {
    const fs::path stamp = entry_path / kLastUsedName;
    if (auto st = lstat_or_none(stamp)) {
        if (idle_for(std::chrono::system_clock::now(), mtime_of(*st)) < kDiskCacheTouchInterval) {
            return;
        }
    }
    stamp_now(stamp);
}

DiskCacheEntryLock::DiskCacheEntryLock(const fs::path& entry_path) {
    tt::filesystem::safe_create_directories(entry_path);
    const fs::path lock_path = entry_path / kInUseName;

    for (int attempt = 0; attempt < kEntryLockAttempts; attempt++) {
        if (attempt > 0) {
            std::this_thread::sleep_for(kEntryLockRetryDelay);
            // A trimmer may have moved the tree away between attempts.
            tt::filesystem::safe_create_directories(entry_path);
        }

        const int fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) {
            const int err = errno;
            log_debug(tt::LogMetal, "Disk cache: cannot open {}: {}", lock_path.string(), ::strerror(err));
            // Retrying a permanent condition -- someone else owns the root, or it is read-only --
            // just burns the whole sleep budget per device inside the caller's build mutex.
            if (err != ENOENT && err != EINTR) {
                return;
            }
            continue;
        }
        if (::flock(fd, LOCK_SH | LOCK_NB) != 0) {
            ::close(fd);
            continue;
        }

        // Confirm the file we locked is still the one at lock_path: a trimmer that renamed the
        // entry between our open and our lock would otherwise leave us holding a lock on a file
        // inside .trash while believing the tree is protected.
        struct ::stat locked{};
        struct ::stat current{};
        if (::fstat(fd, &locked) == 0 && ::stat(lock_path.c_str(), &current) == 0 && locked.st_dev == current.st_dev &&
            locked.st_ino == current.st_ino) {
            fd_ = fd;
            return;
        }
        ::flock(fd, LOCK_UN);
        ::close(fd);
    }
    log_debug(tt::LogMetal, "Disk cache: gave up acquiring {}", lock_path.string());
}

DiskCacheEntryLock::~DiskCacheEntryLock() {
    if (fd_ >= 0) {
        ::flock(fd_, LOCK_UN);
        ::close(fd_);
        fd_ = -1;
    }
}

bool disk_cache_hold_entry_for_process(const fs::path& entry_path) {
    HeldLocks& locks = held_locks();
    const std::lock_guard lock(locks.mutex);

    const std::string key = entry_path.string();
    const auto it = locks.by_path.find(key);
    if (it != locks.by_path.end() && it->second->held()) {
        return true;
    }

    // Replaces any earlier failed attempt, so a caller that retries gets a real try.
    auto claim = std::make_unique<DiskCacheEntryLock>(entry_path);
    const bool held = claim->held();
    locks.by_path[key] = std::move(claim);
    if (!held) {
        // The tree will be built but not announced, so another process's trimmer may remove it.
        // The one case where this component can lose work, so say it out loud.
        log_warning(
            tt::LogMetal,
            "Disk cache: could not claim {}; it is not protected from eviction by other processes",
            entry_path.string());
    }
    return held;
}

}  // namespace tt::tt_metal
