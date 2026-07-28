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

#ifdef __linux__
#include <sys/vfs.h>
#else
#include <sys/mount.h>
#include <sys/param.h>
#endif

namespace fs = std::filesystem;

namespace tt::tt_metal {

namespace {

constexpr const char* kMarkerName = ".tt_metal_cache";
constexpr const char* kTrimLockName = ".trim.lock";
constexpr const char* kLastTrimName = ".last_trim";
constexpr const char* kTrashName = ".trash";
constexpr const char* kInUseName = ".inuse";
constexpr const char* kLastUsedName = ".last_used";

// Bounded, non-blocking acquisition of the shared entry lock. A conflicting exclusive lock is
// held only across a rename, so a handful of short retries covers real contention while
// guaranteeing device init cannot stall on the filesystem.
constexpr int kEntryLockAttempts = 20;
constexpr std::chrono::milliseconds kEntryLockRetryDelay{25};

// Cache bookkeeping lives in dot-prefixed names so it can never collide with a build_key.
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

// Compare recency in whole seconds. Subtracting two system_clock points yields nanoseconds,
// and promoting a configured age of, say, a million days into nanoseconds overflows int64 and
// wraps negative -- which would make every entry look over-age and wipe the cache.
std::chrono::seconds idle_for(std::chrono::system_clock::time_point now, std::chrono::system_clock::time_point last) {
    return std::chrono::duration_cast<std::chrono::seconds>(now - last);
}

// Set a file's mtime to now, creating it if absent. futimens with a null spec is what moves
// mtime without writing a byte; opening alone would leave an existing stamp untouched.
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

// Disk usage as the filesystem accounts for it. st_blocks is in 512-byte units by
// definition, independent of the filesystem's own block size, and it is what makes a
// sparse file cost less than its apparent length and a one-byte file cost a whole block.
// Since the point of this cache is to stop filling disks, that is the number that matters.
uint64_t allocated_bytes(const struct ::stat& st) { return static_cast<uint64_t>(st.st_blocks) * 512ULL; }

// Sum of allocated bytes under `root`, following no symlinks. A file reachable by several
// hard links inside one tree must be counted once; tracking only nlink>1 files keeps the
// dedup set empty in the common case, since the build unlinks its temporaries after linking
// them and almost every file at rest has a single link.
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
            // A concurrent writer or a vanishing entry is expected here; keep the partial sum.
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

struct LastUsed {
    std::chrono::system_clock::time_point when;
    // False when there was no .last_used stamp and the time came from mtime instead.
    bool from_stamp = false;
};

// Recency for one entry. A .last_used stamp is authoritative. Without one, fall back to the
// newest mtime among the entry directory and its immediate children, which at least moves
// when firmware/ or kernels/ gains a subdirectory. A couple of extra stats, not a walk.
LastUsed entry_last_used(const fs::path& entry_path) {
    if (auto stamp = lstat_or_none(entry_path / kLastUsedName)) {
        return {mtime_of(*stamp), true};
    }

    // Epoch, not time_point::min(): subtracting min() from now overflows the nanosecond
    // difference, and "unknown recency" should mean "oldest", which epoch already does.
    auto newest = std::chrono::system_clock::time_point{};
    if (auto st = lstat_or_none(entry_path)) {
        newest = mtime_of(*st);
    }
    for (const auto& child : tt::filesystem::safe_directory_entries(entry_path)) {
        if (auto st = lstat_or_none(child.path())) {
            newest = std::max(newest, mtime_of(*st));
        }
    }
    return {newest, false};
}

// RAII flock. `shared` picks LOCK_SH over LOCK_EX. Non-blocking: every caller here would
// rather skip work, or retry on its own terms, than wait on another process.
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

    // The file exists and could be opened, whether or not the lock was granted. Separate
    // from held() so callers can tell "nobody is announcing use" from "somebody holds it".
    bool opened() const { return opened_; }
    bool held() const { return fd_ >= 0; }

    // Drop the lock early. Used to bound how long an evicting trimmer blocks a waiter: the
    // lock covers the rename into .trash, not the recursive delete that follows.
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

// Entries this process has claimed. Consulted by this process's own trimmer so it cannot
// evict a tree the process is using, and by disk_cache_hold_entry_for_process to retry a
// claim that failed earlier. Intentionally never destroyed: the locks must outlive every
// static destructor that might still touch the cache.
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

// A trash name no other process can pick: pid plus a process-local counter plus the wall
// clock. Collisions would only cost a retry, but a stable-looking name would invite one.
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

// Move `entry_path` out of the lookup namespace. Returns where it was staged, or nullopt if
// it could not be moved at all. Must be called while holding the entry's exclusive lock.
//
// Uses a raw rename rather than tt::filesystem::safe_rename because a cross-device failure
// here is expected and handled below, not worth a warning.
std::optional<fs::path> stage_for_deletion(const fs::path& entry_path, const fs::path& trash_dir) {
    const fs::path staged = make_trash_path(trash_dir, entry_path.filename().string());
    std::error_code ec;
    fs::rename(entry_path, staged, ec);
    if (!ec) {
        return staged;
    }

    // The trash landed on a different filesystem. Removing in place reintroduces the
    // torn-entry window, so it is a fallback and not the path.
    log_debug(
        tt::LogMetal,
        "Disk cache: cannot stage {} for deletion ({}), removing in place",
        entry_path.string(),
        ec.message());
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

// Collect entries from `root`. `want_managed` selects between entries that carry an .inuse
// file (eviction candidates) and those that do not (trees only ever written by builds
// predating the locking protocol, which no automatic path may touch).
DiskCacheStats scan_entries(const fs::path& root, bool want_managed) {
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
        // directory_entry caches the type readdir already reported, so this is usually free.
        if (!child.is_directory(dir_ec) || dir_ec) {
            continue;
        }

        // create=false is load bearing twice over: it must not conjure an .inuse file, which
        // would silently promote an unmanaged tree into an eviction candidate, and its
        // opened() result is exactly the managed/unmanaged test.
        const ScopedFlock probe(child.path() / kInUseName, /*shared=*/false, /*create=*/false);
        const bool managed = probe.opened();
        if (managed != want_managed) {
            continue;
        }

        DiskCacheEntry entry;
        entry.name = name;
        entry.path = child.path();
        entry.managed = managed;
        entry.size_bytes = measure_tree_bytes(child.path());

        const LastUsed last_used = entry_last_used(child.path());
        entry.last_used = last_used.when;
        entry.has_stamp = last_used.from_stamp;

        // A claim this process holds counts too: flock would grant our own probe the exclusive
        // lock, so the file alone cannot tell us.
        entry.in_use = (managed && !probe.held()) || claimed_by_this_process(child.path());

        stats.total_size_bytes += entry.size_bytes;
        stats.entries.push_back(std::move(entry));
    }

    // Least recently used first, which is eviction order. Ties broken by name so the order
    // is deterministic and the CLI output is stable.
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

bool has_size_limit(const DiskCacheConfig& config) { return config.max_size_bytes > 0; }

// Shared body of trim, clear and prune-unmanaged. `want_managed` selects which half of the
// root's entries are candidates; `honor_limits` false means "evict everything not in use".
DiskCacheTrimResult run_trim(const DiskCacheConfig& config, bool want_managed, bool honor_limits) {
    if (!tt::filesystem::safe_is_directory(config.root).value_or(false)) {
        return skipped_result("cache root does not exist");
    }

    // One trimmer per root. Losing the race is the normal case on a busy machine and is not
    // worth a warning: the process that holds the lock is doing the same work.
    const ScopedFlock trim_lock(config.root / kTrimLockName, /*shared=*/false, /*create=*/true);
    if (!trim_lock.held()) {
        return skipped_result("another process is trimming this cache");
    }

    DiskCacheTrimResult result;
    const fs::path trash_dir = config.root / kTrashName;
    result.bytes_reclaimed += drain_trash(trash_dir);

    const DiskCacheStats stats = scan_entries(config.root, want_managed);
    result.bytes_before = stats.total_size_bytes;
    uint64_t live_bytes = stats.total_size_bytes;

    const auto now = std::chrono::system_clock::now();
    bool trash_dir_ready = false;

    for (const auto& entry : stats.entries) {
        const DiskCacheEviction decision = disk_cache_decide_eviction(entry, config, live_bytes, now, honor_limits);
        if (decision == DiskCacheEviction::StopScan) {
            break;
        }
        if (decision == DiskCacheEviction::KeepInUse) {
            result.entries_skipped_in_use++;
            continue;
        }
        if (decision == DiskCacheEviction::KeepTooYoung) {
            result.entries_skipped_too_young++;
            continue;
        }

        std::optional<fs::path> staged;
        {
            // Hold the entry's exclusive lock across the rename and nothing more. Without it
            // a process could take a shared lock between our in-use probe and the rename and
            // believe its tree is protected while we move it away. Holding it across the
            // recursive delete instead would block that process for the whole delete.
            ScopedFlock entry_lock(entry.path / kInUseName, /*shared=*/false, /*create=*/false);
            if (entry.managed && !entry_lock.held()) {
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

        // Gone from the lookup namespace either way; the bytes are only reclaimed if the
        // recursive removal completes.
        result.entries_removed++;
        live_bytes -= std::min(live_bytes, entry.size_bytes);
        if (*staged == entry.path || tt::filesystem::safe_remove_all(*staged)) {
            result.bytes_reclaimed += entry.size_bytes;
        } else {
            result.entries_pending_deletion++;
        }
    }

    if (honor_limits) {
        // Records that this root has been swept, so the automatic path can skip the walk
        // until the interval elapses. Written on every completed pass, including one that
        // evicted nothing: the expensive part was the scan, and its answer stays good.
        stamp_now(config.root / kLastTrimName);
    }

    result.bytes_after = live_bytes;
    return result;
}

// Split "50G" into 50 and 'g'. Returns nullopt unless `text` is a run of digits followed by
// at most one unit character. One redundant trailing 'b' is dropped first, so "50GB" reads as
// "50G". `unit` is '\0' for a bare count, lowercase otherwise.
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

std::optional<ScaledValue> split_scaled_value(std::string_view text) {
    text = trim_spaces(text);

    ScaledValue scaled;
    const char* const end = text.data() + text.size();
    // from_chars takes digits only: no sign, no whitespace, no radix prefix, and it reports
    // overflow rather than wrapping.
    const auto parsed = std::from_chars(text.data(), end, scaled.count);
    if (parsed.ec != std::errc{}) {
        return std::nullopt;
    }

    std::string_view unit(parsed.ptr, static_cast<size_t>(end - parsed.ptr));
    if (unit.size() > 1 && std::tolower(static_cast<unsigned char>(unit.back())) == 'b') {
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

// True on filesystems where flock may be advisory to one client only, so a lock cannot be
// relied on to exclude a process on another host.
bool filesystem_locks_are_local_only(const fs::path& path) {
#ifdef __linux__
    // Linux does not expose "are locks local" directly. NFS is the case that matters: with
    // local_lock or nolock, flock never reaches the server's lock manager, and even without
    // them the guarantee depends on a working NLM. Treat NFS as untrustworthy.
    struct ::statfs sfs{};
    if (::statfs(path.c_str(), &sfs) != 0) {
        return false;  // cannot tell; do not block on a guess
    }
    // NFS_SUPER_MAGIC, spelled out because <linux/magic.h> is not present in every toolchain.
    // f_type's type varies across libc and architecture (__fsword_t, long, unsigned), so take
    // it from the object -- `decltype(::statfs{}...)` does not parse, since on Linux `statfs`
    // names both a struct and a function and the unqualified name resolves to the function.
    constexpr decltype(sfs.f_type) kNfsSuperMagic = 0x6969;
    return sfs.f_type == kNfsSuperMagic;
#else
    struct ::statfs sfs{};
    if (::statfs(path.c_str(), &sfs) != 0) {
        return false;
    }
    return std::strncmp(sfs.f_fstypename, "nfs", 3) == 0;
#endif
}

}  // namespace

DiskCacheEviction disk_cache_decide_eviction(
    const DiskCacheEntry& entry,
    const DiskCacheConfig& config,
    uint64_t live_bytes,
    std::chrono::system_clock::time_point now,
    bool honor_limits) {
    const std::chrono::seconds idle = idle_for(now, entry.last_used);
    if (honor_limits && !(config.max_size_bytes > 0 && live_bytes > config.max_size_bytes)) {
        // Under budget. Entries arrive least recently used first, and evicting only shrinks
        // live_bytes, so nothing after this one can be over budget either.
        return DiskCacheEviction::StopScan;
    }
    if (entry.in_use) {
        return DiskCacheEviction::KeepInUse;
    }
    if (honor_limits && idle < config.min_eviction_age) {
        return DiskCacheEviction::KeepTooYoung;
    }
    return DiskCacheEviction::Evict;
}

void disk_cache_initialize_root(const fs::path& root) {
    if (!tt::filesystem::safe_create_directories(root)) {
        return;
    }
    const fs::path marker = root / kMarkerName;
    if (!tt::filesystem::safe_exists(marker).value_or(false)) {
        stamp_now(marker);
    }
}

bool disk_cache_root_is_initialized(const fs::path& root) {
    return tt::filesystem::safe_exists(root / kMarkerName).value_or(false);
}

bool disk_cache_locking_is_trustworthy(const fs::path& path) { return !filesystem_locks_are_local_only(path); }

std::optional<std::string> disk_cache_automatic_trim_blocker(const DiskCacheConfig& config) {
    if (!config.trim_enabled) {
        return "trimming disabled by TT_METAL_CACHE_TRIM";
    }
    if (!has_size_limit(config)) {
        return "no size limit configured (set TT_METAL_CACHE_MAX_SIZE to enable)";
    }
    if (config.require_trusted_locks && !disk_cache_locking_is_trustworthy(config.root)) {
        // Evicting here could delete a tree another host is compiling into, because its
        // .inuse lock would be invisible to us.
        return "file locking on this filesystem cannot exclude other hosts";
    }
    return std::nullopt;
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
    // Per-uid, because /tmp is shared: a common root would have one user's trimmer sizing and
    // trying to evict another user's entries, which it cannot even read.
    return fs::path("/tmp") / fmt::format("tt-metal-cache-{}", static_cast<unsigned>(::getuid())) / "";
}

void disk_cache_apply_env(DiskCacheConfig& config, const char* max_size, const char* trim) {
    // An unset variable and one exported empty are the same thing. Anything unparseable warns
    // and leaves the default in place: a cache tuning knob must not stop a process running.
    const auto given = [](const char* value) -> std::optional<std::string_view> {
        if (value == nullptr) {
            return std::nullopt;
        }
        const std::string_view text = trim_spaces(value);
        if (text.empty()) {
            return std::nullopt;
        }
        return text;
    };

    if (auto text = given(max_size)) {
        if (auto parsed = parse_disk_cache_size(*text)) {
            config.max_size_bytes = *parsed;
        } else {
            log_warning(
                tt::LogMetal,
                "Disk cache: ignoring TT_METAL_CACHE_MAX_SIZE='{}' (expected a byte count, optionally suffixed "
                "K/M/G/T)",
                std::string(*text));
        }
    }
    if (auto text = given(trim)) {
        if (auto parsed = parse_disk_cache_bool(*text)) {
            config.trim_enabled = *parsed;
        } else {
            log_warning(
                tt::LogMetal,
                "Disk cache: ignoring TT_METAL_CACHE_TRIM='{}' (expected 0/1, true/false, yes/no or on/off)",
                std::string(*text));
        }
    }
}

DiskCacheConfig kernel_disk_cache_config_from_env() {
    DiskCacheConfig config;
    config.root = default_kernel_cache_root();
    disk_cache_apply_env(config, std::getenv("TT_METAL_CACHE_MAX_SIZE"), std::getenv("TT_METAL_CACHE_TRIM"));
    return config;
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

std::optional<bool> parse_disk_cache_bool(std::string_view text) {
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

DiskCacheStats disk_cache_stat(const DiskCacheConfig& config) {
    DiskCacheStats stats = scan_entries(config.root, /*want_managed=*/true);
    // Only the human-facing report cares about debris; a trim already knows what it drained.
    stats.trash_size_bytes = measure_tree_bytes(config.root / kTrashName);
    return stats;
}

DiskCacheStats disk_cache_stat_unmanaged(const DiskCacheConfig& config) {
    return scan_entries(config.root, /*want_managed=*/false);
}

DiskCacheTrimResult disk_cache_trim(const DiskCacheConfig& config) {
    if (auto blocker = disk_cache_automatic_trim_blocker(config)) {
        return skipped_result(*blocker);
    }
    return run_trim(config, /*want_managed=*/true, /*honor_limits=*/true);
}

DiskCacheTrimResult disk_cache_clear(const DiskCacheConfig& config) {
    return run_trim(config, /*want_managed=*/true, /*honor_limits=*/false);
}

DiskCacheTrimResult disk_cache_prune_unmanaged(const DiskCacheConfig& config) {
    // Every other path is guarded by requiring an .inuse file on the entry. This one's
    // candidates are precisely the entries without it, so it needs a different guard or a
    // mistyped --root would put arbitrary directories in scope.
    if (!disk_cache_root_is_initialized(config.root)) {
        return skipped_result("not a tt-metal cache root (no " + std::string(kMarkerName) + " marker)");
    }
    return run_trim(config, /*want_managed=*/false, /*honor_limits=*/false);
}

void disk_cache_trim_in_background(const DiskCacheConfig& config) {
    if (auto blocker = disk_cache_automatic_trim_blocker(config)) {
        log_debug(tt::LogMetal, "Disk cache: not trimming {} ({})", config.root.string(), *blocker);
        return;
    }

    // Cheapest gate first: after the first device, later ones cost a mutex and a string
    // compare instead of a stat.
    {
        static std::mutex mutex;
        static std::set<std::string> trimmed_roots;
        const std::lock_guard lock(mutex);
        if (!trimmed_roots.insert(config.root.string()).second) {
            return;
        }
    }

    // One stat, on the hot startup path, standing in for a walk of the whole cache. Without
    // this every process start would re-size a cache that may hold hundreds of thousands of
    // files. Racing processes may both get past here; only one wins the trim lock.
    if (auto stamp = lstat_or_none(config.root / kLastTrimName)) {
        if (idle_for(std::chrono::system_clock::now(), mtime_of(*stamp)) < kDiskCacheTrimInterval) {
            return;
        }
    }

    // Detached on purpose. Trimming must never sit on the compile path, and a joinable
    // thread would either delay process teardown behind a large remove_all or need an owner
    // with a well-defined lifetime relative to static destruction. Everything the thread
    // touches is copied in, and eviction renames before removing, so an abrupt exit leaves
    // only .trash debris for the next process to drain.
    try {
        std::thread worker([config]() {
            try {
                const DiskCacheTrimResult result = disk_cache_trim(config);
                if (result.skipped) {
                    log_debug(
                        tt::LogMetal, "Disk cache: skipped trim of {} ({})", config.root.string(), result.skip_reason);
                } else if (result.entries_removed > 0 || result.bytes_reclaimed > 0) {
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
            log_debug(tt::LogMetal, "Disk cache: cannot open {}: {}", lock_path.string(), ::strerror(errno));
            continue;
        }
        // Non-blocking: a conflicting exclusive lock is held only across a rename, so
        // retrying beats waiting, and device init can never hang on the filesystem.
        if (::flock(fd, LOCK_SH | LOCK_NB) != 0) {
            ::close(fd);
            continue;
        }

        // Confirm the file we locked is still the one at lock_path. A trimmer that renamed
        // the entry away between our open and our lock would otherwise leave us holding a
        // lock on a file inside .trash while believing the tree is protected.
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

    // Replaces any earlier failed attempt, so a caller that retries gets a real try rather
    // than the memory of a failure.
    auto claim = std::make_unique<DiskCacheEntryLock>(entry_path);
    const bool held = claim->held();
    locks.by_path[key] = std::move(claim);
    if (!held) {
        // The tree will be built but not announced, so another process's trimmer may remove
        // it underneath us. Worth saying out loud: it is the one case where this component
        // can lose work.
        log_warning(
            tt::LogMetal,
            "Disk cache: could not claim {}; it is not protected from eviction by other processes",
            entry_path.string());
    }
    return held;
}

}  // namespace tt::tt_metal
