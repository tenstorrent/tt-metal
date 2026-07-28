// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-metal-cache: inspect and bound the JIT kernel cache.
//
// This is the supported entry point for humans and for machine provisioning. Point
// provisioning scripts here instead of `rm -rf` on a path that is an implementation detail,
// so the layout can change without breaking them.
//
// Deliberately independent of RunTimeOptions: that requires a resolvable TT_METAL_HOME and
// aborts without one, which is no way for a disk-cleanup tool to behave on a machine whose
// disk is already full.

#include <internal/disk_cache.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>

namespace {

using tt::tt_metal::DiskCacheConfig;
using tt::tt_metal::DiskCacheEviction;
using tt::tt_metal::DiskCacheStats;
using tt::tt_metal::DiskCacheTrimResult;

void print_usage() {
    std::puts(
        "Usage: tt-metal-cache <command> [options]\n"
        "\n"
        "Inspect and bound the tt-metal JIT kernel cache.\n"
        "\n"
        "Commands:\n"
        "  stat            Report cache size and per-entry usage, least recently used first\n"
        "  trim            Evict least recently used entries until the cache is under a size limit\n"
        "  clear           Evict every entry not in use by a live process\n"
        "  prune-unmanaged Remove trees that no lock-aware build has ever claimed\n"
        "\n"
        "Options:\n"
        "  --root <path>      Cache root. Defaults to $TT_METAL_CACHE/tt-metal-cache/ when\n"
        "                     TT_METAL_CACHE is set, else ~/.cache/tt-metal-cache/. Note the\n"
        "                     tt-metal-cache component: pass the path the runtime resolves, not\n"
        "                     the parent directory you set TT_METAL_CACHE to.\n"
        "  --max-size <size>  Size limit for trim, e.g. 50G, 512M (default: TT_METAL_CACHE_MAX_SIZE)\n"
        "  --dry-run          Report what would be evicted and stop\n"
        "  --force            Allow trimming on a filesystem where file locking cannot exclude\n"
        "                     other hosts. Only safe when no other host shares this root.\n"
        "  -h, --help         Show this help\n"
        "\n"
        "Entries held by a live process are never evicted, so trim and clear are safe to run\n"
        "while jobs are compiling, and safe to run concurrently with themselves.\n"
        "\n"
        "Only entries a lock-aware build has claimed are ever evicted automatically. Trees left\n"
        "by older builds carry no in-use marker, cannot be told apart from live ones, and are\n"
        "therefore never touched -- `prune-unmanaged` removes those, and only when you ask. An\n"
        "existing tree becomes managed the first time a current build uses it.\n"
        "\n"
        "Each user has their own cache root, so this only ever touches your own files.");
}

std::string entries(size_t count) { return std::to_string(count) + (count == 1 ? " entry" : " entries"); }

std::string format_age(std::chrono::system_clock::time_point when) {
    const auto age = std::chrono::system_clock::now() - when;
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(age).count();
    if (seconds < 0) {
        return "just now";
    }
    if (seconds < 60) {
        return std::to_string(seconds) + "s ago";
    }
    if (seconds < 3600) {
        return std::to_string(seconds / 60) + "m ago";
    }
    if (seconds < 86400) {
        return std::to_string(seconds / 3600) + "h ago";
    }
    return std::to_string(seconds / 86400) + "d ago";
}

std::string size_of(uint64_t bytes) { return tt::tt_metal::format_disk_cache_size(bytes); }

void print_stat(const DiskCacheConfig& config) {
    const DiskCacheStats stats = tt::tt_metal::disk_cache_stat(config);
    std::printf("Cache root: %s\n", config.root.c_str());
    if (stats.entries.empty()) {
        std::printf("Empty (no cache entries).\n");
    } else {
        std::printf("%-24s %12s %14s %s\n", "ENTRY", "SIZE", "LAST USED", "STATE");
        for (const auto& entry : stats.entries) {
            std::printf(
                "%-24s %12s %14s %s\n",
                entry.name.c_str(),
                size_of(entry.size_bytes).c_str(),
                format_age(entry.last_used).c_str(),
                entry.in_use ? "in use" : (entry.has_stamp ? "" : "no stamp"));
        }
    }
    std::printf("\nTotal: %s in %s", size_of(stats.total_size_bytes).c_str(), entries(stats.entries.size()).c_str());
    if (config.max_size_bytes > 0) {
        std::printf(" (limit %s)", size_of(config.max_size_bytes).c_str());
    } else {
        std::printf(" (no limit set, so nothing is evicted automatically)");
    }
    std::printf("\n");

    if (stats.trash_size_bytes > 0) {
        std::printf(
            "Pending deletion: %s (a previous trim was interrupted; the next one drains it)\n",
            size_of(stats.trash_size_bytes).c_str());
    }

    // Space no automatic trim will ever reclaim, so it has to be reported or it is invisible.
    const DiskCacheStats unmanaged = tt::tt_metal::disk_cache_stat_unmanaged(config);
    if (!unmanaged.entries.empty()) {
        std::printf(
            "\nUnmanaged: %s in %s that no lock-aware build has claimed.\n"
            "  Never evicted automatically, and not counted against the limit above. These\n"
            "  become managed as current builds reuse them; `tt-metal-cache prune-unmanaged`\n"
            "  removes the rest, once no job built from an older tt-metal is running.\n",
            size_of(unmanaged.total_size_bytes).c_str(),
            entries(unmanaged.entries.size()).c_str());
    }

    if (auto blocker = tt::tt_metal::disk_cache_automatic_trim_blocker(config)) {
        std::printf("\nAutomatic trimming is off here: %s.\n", blocker->c_str());
    }
}

void print_trim_result(const DiskCacheTrimResult& result) {
    if (result.skipped) {
        std::printf("Nothing done: %s.\n", result.skip_reason.c_str());
        return;
    }
    std::printf(
        "Evicted %s, reclaimed %s. Cache is now %s.\n",
        entries(result.entries_removed).c_str(),
        size_of(result.bytes_reclaimed).c_str(),
        size_of(result.bytes_after).c_str());
    if (result.entries_pending_deletion > 0) {
        std::printf(
            "%s could not be fully removed and still occupy disk; the next trim retries them.\n",
            entries(result.entries_pending_deletion).c_str());
    }
    if (result.entries_skipped_in_use > 0) {
        std::printf("Kept %s in use by a live process.\n", entries(result.entries_skipped_in_use).c_str());
    }
    if (result.entries_skipped_too_young > 0) {
        std::printf(
            "Kept %s used too recently to evict safely.\n"
            "  They become evictable once idle; nothing to do.\n",
            entries(result.entries_skipped_too_young).c_str());
    }
}

// What a trim would evict, without touching anything. Routes every entry through the same
// disk_cache_decide_eviction() a real trim uses, so a dry run cannot disagree with it.
void print_dry_run(const DiskCacheConfig& config, const DiskCacheStats& stats, bool honor_limits) {
    const auto now = std::chrono::system_clock::now();
    uint64_t live_bytes = stats.total_size_bytes;
    uint64_t would_reclaim = 0;
    size_t would_remove = 0;

    std::printf("Cache root: %s\n", config.root.c_str());
    for (const auto& entry : stats.entries) {
        const DiskCacheEviction decision =
            tt::tt_metal::disk_cache_decide_eviction(entry, config, live_bytes, now, honor_limits);
        if (decision == DiskCacheEviction::StopScan) {
            break;
        }
        if (decision != DiskCacheEviction::Evict) {
            std::printf(
                "  keep   %-24s %12s (%s)\n",
                entry.name.c_str(),
                size_of(entry.size_bytes).c_str(),
                decision == DiskCacheEviction::KeepInUse ? "in use by a live process" : "used too recently");
            continue;
        }
        std::printf(
            "  evict  %-24s %12s (last used %s)\n",
            entry.name.c_str(),
            size_of(entry.size_bytes).c_str(),
            format_age(entry.last_used).c_str());
        would_remove++;
        would_reclaim += entry.size_bytes;
        live_bytes -= std::min(live_bytes, entry.size_bytes);
    }

    std::printf(
        "\nWould evict %s and reclaim %s, leaving %s. Nothing was changed.\n",
        entries(would_remove).c_str(),
        size_of(would_reclaim).c_str(),
        size_of(live_bytes).c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }

    const std::string_view command = argv[1];
    if (command == "-h" || command == "--help" || command == "help") {
        print_usage();
        return 0;
    }
    if (command != "stat" && command != "trim" && command != "clear" && command != "prune-unmanaged") {
        std::fprintf(stderr, "tt-metal-cache: unknown command '%s'\n\n", argv[1]);
        print_usage();
        return 2;
    }

    DiskCacheConfig config = tt::tt_metal::kernel_disk_cache_config_from_env();
    // An explicit invocation is a human asking for this to happen now; TT_METAL_CACHE_TRIM
    // exists to keep the automatic startup trim out of the way, not to disarm the CLI.
    config.trim_enabled = true;
    bool dry_run = false;
    bool limits_given = false;

    for (int i = 2; i < argc; i++) {
        const std::string_view arg = argv[i];
        auto next_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "tt-metal-cache: %s requires a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--root") {
            config.root = next_value("--root");
        } else if (arg == "--max-size") {
            const char* value = next_value("--max-size");
            auto parsed = tt::tt_metal::parse_disk_cache_size(value);
            if (!parsed.has_value()) {
                std::fprintf(
                    stderr, "tt-metal-cache: cannot parse size '%s' (try 50G, 512M, or a byte count)\n", value);
                return 2;
            }
            config.max_size_bytes = *parsed;
            limits_given = true;
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--force") {
            config.require_trusted_locks = false;
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        } else {
            std::fprintf(stderr, "tt-metal-cache: unknown option '%s'\n\n", argv[i]);
            print_usage();
            return 2;
        }
    }

    // clear and prune-unmanaged remove everything not in use; a size limit does not narrow
    // them. Silently ignoring the flag would turn an intended partial cleanup into a full
    // wipe, so refuse instead.
    if (limits_given && command != "trim" && command != "stat") {
        std::fprintf(
            stderr,
            "tt-metal-cache: --max-size only applies to `trim`; `%s` removes every entry not in\n"
            "use by a live process. Did you mean `tt-metal-cache trim`?\n",
            argv[1]);
        return 2;
    }

    try {
        if (command == "stat") {
            print_stat(config);
            return 0;
        }
        if (command == "prune-unmanaged") {
            // Every other command is guarded by requiring an .inuse file on the entry. This
            // one's candidates are precisely the entries without it, so a mistyped --root
            // would otherwise put unrelated directories in scope.
            if (!tt::tt_metal::disk_cache_root_is_initialized(config.root)) {
                std::fprintf(
                    stderr,
                    "tt-metal-cache: %s does not look like a tt-metal cache root, so prune-unmanaged\n"
                    "refuses to delete anything in it. Pass the path the runtime resolves, which ends\n"
                    "in tt-metal-cache/ and was created by a current tt-metal build.\n",
                    config.root.c_str());
                return 2;
            }
            if (dry_run) {
                print_dry_run(config, tt::tt_metal::disk_cache_stat_unmanaged(config), /*honor_limits=*/false);
                return 0;
            }
            print_trim_result(tt::tt_metal::disk_cache_prune_unmanaged(config));
            return 0;
        }

        const bool honor_limits = command == "trim";
        if (dry_run) {
            print_dry_run(config, tt::tt_metal::disk_cache_stat(config), honor_limits);
            return 0;
        }
        if (honor_limits) {
            print_trim_result(tt::tt_metal::disk_cache_trim(config));
        } else {
            print_trim_result(tt::tt_metal::disk_cache_clear(config));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tt-metal-cache: %s\n", e.what());
        return 1;
    }
}
