// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace tt::tt_metal {

// Size cap for the on-disk JIT kernel cache.
//
// The cap is a setting of the cache itself, not of a process, so every process sharing a cache uses the
// same one: max_size in <root>/tt-metal-cache.conf, like ccache's ccache.conf. No file, no max_size, or
// max_size = 0 means no cap. TT_METAL_CACHE_MAX_SIZE only seeds a missing file; once the file exists it wins.
//
// Layout of the cache root: <root>/<build_key>/{firmware,kernels}/ and <root>/pch/<hash>/. Every compile-time
// variant of a kernel is its own directory <root>/<build_key>/kernels/<kernel_name>/<variant_hash>/, and every
// flag set has its own precompiled header directory <root>/pch/<hash>/. Only those directories are evicted;
// firmware and anything else under the root count toward the size but are never removed.
//
// Recency is the directory's mtime: a build writes into it, and while a cap is set every use stamps it with
// note_kernel_cache_dir_used(), so eviction is least recently used first, not oldest first.

inline constexpr std::string_view kKernelCacheConfigName = "tt-metal-cache.conf";

// Parses a size the way ccache reads max_size: a number, optionally with a fraction ("1.5"), then an optional
// unit: k/K, M, G, T are powers of 1000, Ki, Mi, Gi, Ti powers of 1024, and a trailing "B" is allowed
// ("20GB", "1GiB"). A bare number means GiB, as in ccache ("20" is 20 GiB). Units are case-sensitive.
// Returns nullopt for anything else.
std::optional<std::uint64_t> parse_cache_size(std::string_view value);

// Reads max_size from <root>/tt-metal-cache.conf. The file holds "key = value" lines; blank lines and
// anything after a '#' are ignored, and the last valid max_size line wins. A value that is not a size, or a
// cap below 100M (most likely a typo), is warned about and skipped. Returns 0 (no cap) when the file or a
// valid max_size is missing: a broken setting must not stop the process.
std::uint64_t read_kernel_cache_max_size(const std::filesystem::path& root);

// Returns the cap for the cache under root, given the value of TT_METAL_CACHE_MAX_SIZE (nullopt: not set).
// - Env var set, no config file: writes "max_size = <env value>" to the file (temp file, then a rename that
//   never replaces a file another process wrote first) and uses it.
// - Env var set, config file exists: uses the file, and warns when the two values differ.
// - Env var set, config file unreadable or without a valid max_size: warns and uses the env var.
// - No env var: the file as read_kernel_cache_max_size() reads it; no file means no cap.
// An env value that is not a size, or below 100M, is warned about and ignored.
std::uint64_t resolve_kernel_cache_max_size(
    const std::filesystem::path& root, const std::optional<std::string>& env_value);

// Fraction of the cap a trim reduces the cache to, so a cache at the cap is not trimmed on every start.
inline constexpr double kKernelCacheLowWaterFraction = 0.8;

// An entry used within this window is never evicted: another process may be building or loading it.
inline constexpr std::chrono::seconds kKernelCacheMinEvictAge{15 * 60};

// Sets the cache this process keeps under max_bytes: the JIT build calls it once at init. With max_bytes 0
// (no cap, or the cap turned off with max_size = 0) nothing is tracked: note_kernel_cache_dir_used() and
// track_kernel_cache_growth() then touch nothing on disk.
void set_kernel_cache_cap(
    const std::filesystem::path& root,
    std::uint64_t max_bytes,
    std::chrono::seconds min_evict_age = kKernelCacheMinEvictAge);

// Stamps dir (a kernel variant or PCH directory) as used now. A process re-stamps a directory only when its
// own last stamp is older than restamp_after, so an entry in use is never older than that and a long-running
// process keeps its entries out of the eviction window.
void note_kernel_cache_dir_used(
    const std::filesystem::path& dir, std::chrono::seconds restamp_after = kKernelCacheMinEvictAge / 2);

// Creates a kernel variant directory and its parents. A trim removes a kernel's directory once it has
// evicted the kernel's last variant; when that lands between making the parent and making the variant
// directory, the parent is made again. Once is enough: the new variant keeps the parent from being empty.
void create_kernel_cache_dir(const std::filesystem::path& dir);

// Test hook, called where a trim in another process can race this one, with the directory concerned:
// "variant dir" (the parent of a variant directory exists, the variant directory does not yet) and "load"
// (a cached kernel variant is about to be read). nullptr, the default, does nothing.
using KernelCacheRaceHook = void (*)(std::string_view point, const std::filesystem::path& dir);
void set_kernel_cache_race_hook(KernelCacheRaceHook hook);
void kernel_cache_race_point(std::string_view point, const std::filesystem::path& dir);

struct KernelCacheTrimResult {
    bool scanned = false;  // the cache tree was walked
    bool trimmed = false;  // false: nothing evicted (no cap, under it, or another process holds the trim lock)
    std::uint64_t size_before = 0;
    std::uint64_t size_after = 0;
    std::size_t variants_evicted = 0;  // kernel variant and PCH directories
};

// Scans the cache under root and evicts least recently used kernel variants and PCH directories until it is
// at most kKernelCacheLowWaterFraction * max_bytes, then rewrites the size counter (<root>/.stats) from the
// scan's real total minus what was freed. No-op when max_bytes is 0. When the files that are never evicted
// alone are over that level, warns and evicts nothing. Safe with several processes sharing root: one trimmer
// at a time (flock on <root>/.trim.lock, which also guards the counter), a victim is renamed into
// <root>/.trash/ before it is deleted, and entries removed by someone else are skipped.
KernelCacheTrimResult trim_kernel_cache(
    const std::filesystem::path& root,
    std::uint64_t max_bytes,
    std::chrono::seconds min_evict_age = kKernelCacheMinEvictAge);

// Start-up check, like ccache: reads only the size counter, and runs trim_kernel_cache() when the counter is
// missing or corrupt, over the cap (but not again within min_evict_age of a scan that could not get under
// it), or older than a day (to fix drift from deletes by hand or crashed writers). No-op for max_bytes 0.
KernelCacheTrimResult maintain_kernel_cache(
    const std::filesystem::path& root,
    std::uint64_t max_bytes,
    std::chrono::seconds min_evict_age = kKernelCacheMinEvictAge);

// The size counter of the cache under root; nullopt when it is missing or corrupt.
std::optional<std::uint64_t> read_kernel_cache_size_counter(const std::filesystem::path& root);

// Runs build, which writes into dir (a kernel variant or PCH directory), and adds what dir grew by to the
// size counter of the cache set_kernel_cache_cap() named. When that takes the counter over the cap, trims
// right away, so one long run cannot outgrow the cap. Without a cap it only runs build.
void track_kernel_cache_growth(const std::filesystem::path& dir, const std::function<void()>& build);

}  // namespace tt::tt_metal
