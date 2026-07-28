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

// Bounded on-disk cache
// =====================
//
// Metal writes compiled kernels to disk and, historically, never deleted them. On a shared
// machine that eventually fills the disk, and one user cannot delete another user's files.
// This component gives a cache root a finite size, the way ccache and sccache do.
//
// Layout
// ------
// A cache root holds one directory per cache entry:
//
//   <root>/<entry>/...          e.g. <tt-metal-cache>/<build_key>/{firmware,kernels}/
//   <root>/.tt_metal_cache      marker: this directory is a cache this code owns
//   <root>/.trim.lock           trim mutual exclusion between processes
//   <root>/.last_trim           when this root was last swept
//   <root>/.trash/<unique>      entries staged for deletion
//   <root>/<entry>/.inuse       created by any lock-aware build; locked while in use
//   <root>/<entry>/.last_used   recency stamp, rewritten at most once per kTouchInterval
//
// Names beginning with '.' are reserved for cache bookkeeping and are never treated as
// entries. Eviction operates on whole entries; nothing inside an entry is evicted
// individually (that is a later phase).
//
// Managed and unmanaged entries
// ----------------------------
// Only entries that carry an .inuse file are eviction candidates. That file is created by
// DiskCacheEntryLock the first time a lock-aware build claims the tree, so its presence is
// exactly the statement "a build that participates in the locking protocol has used this".
//
// This is what makes automatic eviction safe to deploy. A binary predating this component
// never creates .inuse, so a trimmer has no way to tell its live tree from an abandoned one
// -- and therefore never touches it. Existing caches are adopted gradually: the first time a
// lock-aware build uses a tree, it gains .inuse and becomes managed. There is no flag day, no
// path change and no forced recompile.
//
// The same rule is the guard against a mistyped root. Point this at an arbitrary directory
// and none of its children carry .inuse, so nothing is a candidate and nothing is deleted.
// disk_cache_prune_unmanaged() is the one operation that removes entries *without* .inuse, so
// it is the one that additionally demands the root marker.
//
// A consequence worth stating: an unmanaged tree nobody reuses is never reclaimed and never
// counted against the size bound, so it can occupy disk indefinitely. disk_cache_stat_unmanaged()
// exists so that space is reported rather than invisible.
//
// Recency
// -------
// atime is unusable here: shared machines mount with relatime or noatime, and on NFS atime
// is not reliable. Instead a live process rewrites <entry>/.last_used, rate limited to
// kTouchInterval so cache hits do not become a write storm. An entry with no stamp falls
// back to the newest mtime among the entry directory and its immediate children.
//
// Concurrency
// -----------
// This is the reason the logic belongs in Metal rather than in a cleanup script. Many
// processes share one root, and a trim must not delete a tree another process is linking
// against.
//
//   - trim() takes an exclusive flock on <root>/.trim.lock and gives up immediately if
//     another process holds it, so only one process trims at a time.
//   - Every live process holds a shared flock on <root>/<entry>/.inuse for its whole
//     lifetime (see DiskCacheEntryLock). trim() skips any entry it cannot lock exclusively.
//     flock is owned by the open file description, so the kernel releases it when the holder
//     dies -- including under SIGKILL. There are no stale claims to time out.
//   - Eviction renames the entry into <root>/.trash/<unique>, releases the entry lock, and
//     only then removes the staged tree. A partially deleted tree is therefore never visible
//     as a cache hit, and a waiting process blocks on a rename rather than on a recursive
//     delete that can run for minutes.
//
// Where flock cannot be trusted, none of the above holds. On NFS mounted with local_lock or
// nolock, flock is client-local: two hosts sharing one home directory will not see each
// other's claims, and a trimmer on one host would happily evict a tree the other is building
// into. Automatic trimming therefore refuses to run on such filesystems; see
// disk_cache_automatic_trim_blocker.
//
// Best effort
// -----------
// Nothing here throws on filesystem failure. A cache that cannot be trimmed is a disk-space
// problem, not a correctness problem, and it must never fail a compile. Callers get counts
// back and may log them.

namespace tt::tt_metal {

// A size worth suggesting to an operator, not a default: automatic eviction stays off until
// TT_METAL_CACHE_MAX_SIZE is set, because a process that deletes another process's files
// should be something a machine's owner turns on deliberately. Used by the CLI's help text
// and the documentation.
inline constexpr uint64_t kSuggestedDiskCacheMaxSizeBytes = 50ULL * 1024 * 1024 * 1024;

// How stale a .last_used stamp must be before a cache hit rewrites it.
inline constexpr std::chrono::seconds kDiskCacheTouchInterval{3600};

// Minimum spacing between automatic trims of one root. Sizing a cache means walking every
// file in it, which is far too expensive to repeat on every process start, so
// disk_cache_trim_in_background() skips out after a single stat when the last pass was
// recent. Explicit disk_cache_trim() calls, including the CLI's, ignore this.
inline constexpr std::chrono::seconds kDiskCacheTrimInterval{3600};

// Do not evict an entry that was used this recently. Not a compatibility measure -- the
// version directory handles that -- but a guard against racing a writer that has created its
// tree and stamped it and is now compiling into it, and against evicting a tree this same
// process is about to initialize for a second device configuration.
inline constexpr std::chrono::seconds kDiskCacheMinEvictionAge{300};

struct DiskCacheConfig {
    // Cache root, e.g. what TT_METAL_CACHE resolves to. Its direct child directories are
    // entries; only those carrying an .inuse file are automatic eviction candidates.
    std::filesystem::path root;

    // Byte cap on the sum of all managed entries. 0, the default, means no cap and so no
    // automatic eviction. This is the only thing that triggers eviction: an entry is removed
    // because the cache is over budget, never merely because it is old.
    uint64_t max_size_bytes = 0;

    // When false, disk_cache_trim() does nothing and reports skipped. This gates automatic
    // trimming only; disk_cache_clear() is always an explicit human action and ignores it.
    // CI wants this off, and so does anything that deliberately runs against an isolated
    // cache.
    bool trim_enabled = true;

    // Refuse to evict an entry used more recently than this. See kDiskCacheMinEvictionAge.
    std::chrono::seconds min_eviction_age = kDiskCacheMinEvictionAge;

    // Require that flock on this root actually excludes other hosts before trimming
    // automatically. Only an explicit human action should be able to clear this.
    bool require_trusted_locks = true;
};

struct DiskCacheEntry {
    // Directory name under the root, e.g. the build_key.
    std::string name;
    std::filesystem::path path;

    // Disk usage in bytes, from allocated blocks rather than apparent file size, so sparse
    // files and block rounding are counted the way the filesystem counts them. Files
    // reachable by more than one hard link inside this entry are counted once.
    uint64_t size_bytes = 0;

    // Last use, from .last_used when present and mtime otherwise.
    std::chrono::system_clock::time_point last_used;

    // True when .last_used supplied last_used, false when it was inferred from mtime.
    bool has_stamp = false;

    // True when another live process holds the shared .inuse lock, so this entry cannot be
    // evicted right now.
    bool in_use = false;

    // True when the entry carries an .inuse file, i.e. a lock-aware build has used it and it
    // is therefore an eviction candidate at all.
    bool managed = false;
};

struct DiskCacheStats {
    uint64_t total_size_bytes = 0;

    // Bytes sitting in .trash, i.e. staged for deletion but not yet removed. Only
    // disk_cache_stat() fills this in; measuring it costs a walk a trim does not need.
    uint64_t trash_size_bytes = 0;

    // Entries ordered least recently used first, which is also eviction order.
    std::vector<DiskCacheEntry> entries;
};

struct DiskCacheTrimResult {
    // Live entry bytes before and after this pass. Neither counts .trash.
    uint64_t bytes_before = 0;
    uint64_t bytes_after = 0;

    // Bytes this pass actually gave back to the filesystem: entries whose removal completed,
    // plus any leftover .trash it drained. Never credits an entry that was staged but could
    // not be removed, so this number can be trusted against du.
    uint64_t bytes_reclaimed = 0;

    // Entries no longer reachable as cache hits.
    size_t entries_removed = 0;

    // Of those, entries whose recursive removal did not complete. Their bytes are still on
    // disk under .trash and are not counted in bytes_reclaimed; the next trim retries.
    size_t entries_pending_deletion = 0;

    // Entries that were eviction candidates but held by another live process.
    size_t entries_skipped_in_use = 0;

    // Entries that were eviction candidates but used more recently than min_eviction_age.
    size_t entries_skipped_too_young = 0;

    // True when no scan happened at all: trimming is disabled or unsafe here, the root does
    // not exist, or another process already holds the trim lock.
    bool skipped = false;
    std::string skip_reason;
};

// What a trim does with one entry.
enum class DiskCacheEviction : uint8_t {
    Evict,
    // Another live process holds the entry's shared .inuse lock.
    KeepInUse,
    // Used more recently than min_eviction_age.
    KeepTooYoung,
    // This entry is within the limits, and because entries arrive least recently used
    // first, so is every entry after it. Stop scanning.
    StopScan,
};

// The single definition of eviction policy. Feed entries in DiskCacheStats order, passing
// the bytes still live ahead of each one; `honor_limits` false means "evict everything not
// in use", which is what disk_cache_clear() does.
//
// Exported so the CLI's --dry-run reports the decisions a real trim would make rather than
// reimplementing them and drifting.
DiskCacheEviction disk_cache_decide_eviction(
    const DiskCacheEntry& entry,
    const DiskCacheConfig& config,
    uint64_t live_bytes,
    std::chrono::system_clock::time_point now,
    bool honor_limits);

// Create the cache root and its marker file. Call once before writing into an entry.
void disk_cache_initialize_root(const std::filesystem::path& root);

// True when the managed root carries the marker disk_cache_initialize_root() writes, i.e.
// this directory really is a cache this code created. disk_cache_prune_unmanaged() checks it,
// because it is the one operation whose candidates are entries *without* .inuse and so cannot
// rely on that file as its guard.
bool disk_cache_root_is_initialized(const std::filesystem::path& root);

// Why automatic trimming must not run against this root, or nullopt when it may. Reasons
// include trimming being disabled, no limit being configured, and flock not being
// trustworthy on the underlying filesystem.
std::optional<std::string> disk_cache_automatic_trim_blocker(const DiskCacheConfig& config);

// Whether flock on this path excludes other hosts. False on filesystems where flock may be
// client-local, so cross-host claims are invisible.
bool disk_cache_locking_is_trustworthy(const std::filesystem::path& path);

// Resolve the kernel cache root exactly as the runtime does: TT_METAL_CACHE with a
// tt-metal-cache subdirectory appended if set, else $HOME/.cache/tt-metal-cache/, else
// /tmp/tt-metal-cache-<uid>/. Always ends in a separator.
std::filesystem::path default_kernel_cache_root();

// Kernel cache configuration read straight from TT_METAL_CACHE_MAX_SIZE and
// TT_METAL_CACHE_TRIM. For callers that cannot construct a
// RunTimeOptions -- notably the tt-metal-cache CLI, which must work on a machine with no
// resolvable TT_METAL_HOME. In-process callers should go through RunTimeOptions instead, so
// a single parse of the environment governs the run.
DiskCacheConfig kernel_disk_cache_config_from_env();

// Apply the cache environment variables to `config`, warning about and then ignoring
// anything unparseable. Shared by RunTimeOptions and the CLI so one spelling of each
// variable, and one error policy, governs both. A null argument means "not set", as does an
// empty string.
//
// Deliberately forgiving: a malformed cache tuning knob must never stop a process running.
void disk_cache_apply_env(DiskCacheConfig& config, const char* max_size, const char* trim);

// Parse a human size such as "50G", "512M", "1024K", "2T" or a bare byte count. Suffixes
// are binary (1G == 1024^3), case insensitive, and one redundant trailing 'B' is allowed so
// "50GB" reads as "50G". Returns nullopt when the whole string is not a valid size.
std::optional<uint64_t> parse_disk_cache_size(std::string_view text);

// Parse a boolean setting: 1/0, true/false, yes/no, on/off, any case. Returns nullopt for
// anything else, so a caller can keep its default rather than guess.
std::optional<bool> parse_disk_cache_bool(std::string_view text);

// Render a byte count the way the CLI reports it, e.g. "3.4 GiB".
std::string format_disk_cache_size(uint64_t bytes);

// Scan managed entries -- those carrying an .inuse file. Never throws; an unreadable root
// yields empty stats.
DiskCacheStats disk_cache_stat(const DiskCacheConfig& config);

// Scan entries with no .inuse file: trees only ever written by builds that predate the
// locking protocol. Reported so a human can see space no automatic trim will touch.
DiskCacheStats disk_cache_stat_unmanaged(const DiskCacheConfig& config);

// Evict least recently used managed entries until they fit under max_size_bytes. Skips
// entries that are in use or used more recently than min_eviction_age. Also drains any
// leftover .trash. Never throws.
DiskCacheTrimResult disk_cache_trim(const DiskCacheConfig& config);

// Evict every managed entry that is not in use, ignoring size and min_eviction_age. Never
// throws.
DiskCacheTrimResult disk_cache_clear(const DiskCacheConfig& config);

// Remove entries with no .inuse file. Because they carry no claim there is no way to tell a
// live one from an abandoned one, which is why no automatic path does this and why it refuses
// to run unless the root carries the cache marker. Never throws.
DiskCacheTrimResult disk_cache_prune_unmanaged(const DiskCacheConfig& config);

// Run disk_cache_trim() on a detached background thread, at most once per root per process
// and at most once per kDiskCacheTrimInterval per root. Does nothing when
// disk_cache_automatic_trim_blocker() reports a reason. Safe to call from device init: it
// takes no Metal locks and touches nothing but the filesystem. Because eviction renames
// before removing, a trim interrupted by process exit leaves only .trash entries for the
// next trim to drain.
void disk_cache_trim_in_background(const DiskCacheConfig& config);

// Rewrite <entry>/.last_used if the existing stamp is older than kDiskCacheTouchInterval.
// Cheap enough for a cache hit path; a single stat in the common case. Never throws.
void disk_cache_touch(const std::filesystem::path& entry_path);

// Process-lifetime shared lock on <entry>/.inuse, announcing to trimmers in other
// processes that this entry must not be evicted. Holding one is what makes concurrent trim
// safe, so acquire it before the first artifact is written into the entry.
//
// Acquisition is non-blocking with bounded retries: a trimmer holds the conflicting
// exclusive lock only across a rename, so a brief wait suffices, and device init must never
// stall on the filesystem. A failure to acquire is not fatal -- the entry simply goes
// unprotected, which is how things behaved before this existed -- but it is worth logging,
// so check held().
class DiskCacheEntryLock {
public:
    // Creates entry_path if needed and takes a shared flock on <entry_path>/.inuse.
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

// Take a DiskCacheEntryLock on entry_path that lives until the process exits. Retries on a
// later call if an earlier attempt failed to acquire, and returns whether the entry is now
// protected. Use this from build setup, where the natural owner of the lock is the process
// itself rather than any object with a shorter life.
bool disk_cache_hold_entry_for_process(const std::filesystem::path& entry_path);

}  // namespace tt::tt_metal
