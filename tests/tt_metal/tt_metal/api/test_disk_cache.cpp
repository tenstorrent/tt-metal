// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <internal/disk_cache.hpp>

namespace tt::tt_metal {
namespace {

namespace fs = std::filesystem;

constexpr uint64_t kEntryFileBytes = 64 * 1024;

class DiskCacheTest : public ::testing::Test {
protected:
    fs::path root_;

    void SetUp() override {
        root_ =
            fs::temp_directory_path() / ("tt_disk_cache_test_" + std::to_string(::getpid()) + "_" +
                                         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root_);
        disk_cache_initialize_root(root_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    // A managed entry: a build_key tree that carries .inuse, i.e. one a lock-aware build has
    // claimed at some point. Only these are eviction candidates.
    fs::path make_entry(const std::string& name, uint64_t bytes = kEntryFileBytes) {
        const fs::path entry = make_bare_entry(name, bytes);
        const std::ofstream inuse(entry / ".inuse");  // present but unlocked
        return entry;
    }

    // An unmanaged entry: written by a build that predates the locking protocol, so it has no
    // .inuse and no automatic path may touch it.
    fs::path make_unmanaged_entry(const std::string& name, uint64_t bytes = kEntryFileBytes) {
        return make_bare_entry(name, bytes);
    }

    fs::path make_bare_entry(const std::string& name, uint64_t bytes) {
        const fs::path entry = root_ / name;
        fs::create_directories(entry / "kernels");
        // Real bytes, not a sparse file: size accounting reads st_blocks, so a hole would
        // measure as zero.
        const std::vector<char> payload(bytes, 'x');
        std::ofstream out(entry / "kernels" / "blob", std::ios::binary);
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        out.close();
        return entry;
    }

    // Force a file's mtime, creating it if absent, so recency-driven policy can be
    // exercised without waiting.
    static void backdate(const fs::path& path, std::chrono::system_clock::time_point when) {
        {
            const std::ofstream create(path, std::ios::app);
        }
        const auto secs = std::chrono::duration_cast<std::chrono::seconds>(when.time_since_epoch()).count();
        struct ::timeval times[2];
        times[0].tv_sec = static_cast<time_t>(secs);
        times[0].tv_usec = 0;
        times[1] = times[0];
        ASSERT_EQ(::utimes(path.c_str(), times), 0) << "failed to backdate " << path;
    }

    static time_t mtime_of(const fs::path& path) {
        struct ::stat st{};
        EXPECT_EQ(::stat(path.c_str(), &st), 0) << "failed to stat " << path;
        return st.st_mtime;
    }

    // Backdate an entry's recency stamp. Writes .last_used explicitly so the mtime fallback
    // is not in play.
    void set_idle_for(const std::string& name, std::chrono::seconds age) {
        backdate(root_ / name / ".last_used", std::chrono::system_clock::now() - age);
    }

    DiskCacheConfig config(uint64_t max_size_bytes) const {
        DiskCacheConfig cfg;
        cfg.root = root_;
        cfg.max_size_bytes = max_size_bytes;
        // Tests drive recency explicitly, so the safety grace period would only mask the
        // policy under test. Its own behaviour is covered by TrimRespectsMinEvictionAge.
        cfg.min_eviction_age = std::chrono::seconds{0};
        // The temp directory is local, but say so explicitly so the suite does not depend on
        // where the machine puts /tmp.
        cfg.require_trusted_locks = false;
        return cfg;
    }

    bool entry_exists(const std::string& name) const { return fs::exists(root_ / name); }

    std::vector<std::string> entry_names(const DiskCacheStats& stats) const {
        std::vector<std::string> names;
        names.reserve(stats.entries.size());
        for (const auto& entry : stats.entries) {
            names.push_back(entry.name);
        }
        return names;
    }
};

TEST_F(DiskCacheTest, ParseSizeAcceptsBinarySuffixes) {
    EXPECT_EQ(parse_disk_cache_size("1024"), 1024u);
    EXPECT_EQ(parse_disk_cache_size("1K"), 1024u);
    EXPECT_EQ(parse_disk_cache_size("1k"), 1024u);
    EXPECT_EQ(parse_disk_cache_size("512M"), 512ull * 1024 * 1024);
    EXPECT_EQ(parse_disk_cache_size("50G"), 50ull * 1024 * 1024 * 1024);
    EXPECT_EQ(parse_disk_cache_size("50GB"), 50ull * 1024 * 1024 * 1024);
    EXPECT_EQ(parse_disk_cache_size("2T"), 2ull * 1024 * 1024 * 1024 * 1024);
    EXPECT_EQ(parse_disk_cache_size("  8G  "), 8ull * 1024 * 1024 * 1024);
    EXPECT_EQ(parse_disk_cache_size("0"), 0u);
}

TEST_F(DiskCacheTest, ParseSizeRejectsMalformedInput) {
    EXPECT_FALSE(parse_disk_cache_size("").has_value());
    EXPECT_FALSE(parse_disk_cache_size("G").has_value());
    EXPECT_FALSE(parse_disk_cache_size("-1G").has_value());
    EXPECT_FALSE(parse_disk_cache_size("1Q").has_value());
    EXPECT_FALSE(parse_disk_cache_size("1GG").has_value());
    EXPECT_FALSE(parse_disk_cache_size("1 G").has_value());
    EXPECT_FALSE(parse_disk_cache_size("1.5G").has_value());
    // 2^64 bytes in tebibytes overflows.
    EXPECT_FALSE(parse_disk_cache_size("99999999999999999999T").has_value());
}

TEST_F(DiskCacheTest, FormatSizeUsesBinaryUnits) {
    EXPECT_EQ(format_disk_cache_size(0), "0 B");
    EXPECT_EQ(format_disk_cache_size(512), "512 B");
    EXPECT_EQ(format_disk_cache_size(1024), "1.0 KiB");
    EXPECT_EQ(format_disk_cache_size(3ull * 1024 * 1024 * 1024), "3.0 GiB");
}

// disk_cache_decide_eviction is the single definition of eviction policy: run_trim executes
// its verdicts and the CLI's --dry-run prints them, so covering it directly covers both.
TEST_F(DiskCacheTest, DecideEvictionAppliesPolicyInPriorityOrder) {
    const auto now = std::chrono::system_clock::now();
    DiskCacheConfig cfg = config(1000);
    cfg.min_eviction_age = std::chrono::hours{1};

    DiskCacheEntry entry;
    entry.size_bytes = 500;
    entry.last_used = now - std::chrono::hours{50};

    // Within the limits: stop, because entries arrive least recently used first.
    EXPECT_EQ(
        disk_cache_decide_eviction(entry, cfg, /*live_bytes=*/900, /*entries_remaining=*/2, now, /*honor_limits=*/true),
        DiskCacheEviction::StopScan);
    // Over the size cap: evict.
    EXPECT_EQ(
        disk_cache_decide_eviction(
            entry, cfg, /*live_bytes=*/2000, /*entries_remaining=*/2, now, /*honor_limits=*/true),
        DiskCacheEviction::Evict);

    // In use outranks every other verdict except StopScan.
    entry.in_use = true;
    EXPECT_EQ(
        disk_cache_decide_eviction(
            entry, cfg, /*live_bytes=*/2000, /*entries_remaining=*/2, now, /*honor_limits=*/true),
        DiskCacheEviction::KeepInUse);
    EXPECT_EQ(
        disk_cache_decide_eviction(entry, cfg, /*live_bytes=*/0, /*entries_remaining=*/2, now, /*honor_limits=*/false),
        DiskCacheEviction::KeepInUse);
    entry.in_use = false;

    // Too recently used to evict safely, even while over the cap.
    entry.last_used = now - std::chrono::seconds{30};
    EXPECT_EQ(
        disk_cache_decide_eviction(
            entry, cfg, /*live_bytes=*/2000, /*entries_remaining=*/2, now, /*honor_limits=*/true),
        DiskCacheEviction::KeepTooYoung);

    // honor_limits=false is what clear does: no size cap and no grace period, so being under
    // budget does not stop it.
    EXPECT_EQ(
        disk_cache_decide_eviction(entry, cfg, /*live_bytes=*/0, /*entries_remaining=*/2, now, /*honor_limits=*/false),
        DiskCacheEviction::Evict);
}

TEST_F(DiskCacheTest, StatOnMissingRootIsEmpty) {
    DiskCacheConfig cfg = config(0);
    cfg.root = root_ / "does-not-exist";  // no v1/ under it
    const DiskCacheStats stats = disk_cache_stat(cfg);
    EXPECT_TRUE(stats.entries.empty());
    EXPECT_EQ(stats.total_size_bytes, 0u);
}

TEST_F(DiskCacheTest, StatCountsEntriesAndIgnoresBookkeeping) {
    make_entry("aaa");
    make_entry("bbb");
    fs::create_directories(root_ / ".trash" / "leftover");
    {
        const std::ofstream lock(root_ / ".trim.lock");
    }
    {
        const std::ofstream last_trim(root_ / ".last_trim");
    }
    {
        const std::ofstream stray(root_ / "loose-file");
    }

    const DiskCacheStats stats = disk_cache_stat(config(0));
    EXPECT_EQ(stats.entries.size(), 2u);
    EXPECT_GE(stats.total_size_bytes, 2 * kEntryFileBytes);
    for (const auto& entry : stats.entries) {
        EXPECT_GE(entry.size_bytes, kEntryFileBytes);
        EXPECT_FALSE(entry.in_use);
    }
}

TEST_F(DiskCacheTest, StatOrdersLeastRecentlyUsedFirst) {
    make_entry("newest");
    make_entry("middle");
    make_entry("oldest");
    set_idle_for("newest", std::chrono::seconds{10});
    set_idle_for("middle", std::chrono::hours{5});
    set_idle_for("oldest", std::chrono::hours{50});

    const DiskCacheStats stats = disk_cache_stat(config(0));
    EXPECT_EQ(entry_names(stats), (std::vector<std::string>{"oldest", "middle", "newest"}));
    for (const auto& entry : stats.entries) {
        EXPECT_TRUE(entry.has_stamp);
    }
}

TEST_F(DiskCacheTest, StatFallsBackToMtimeWithoutStamp) {
    make_entry("unstamped");
    const DiskCacheStats stats = disk_cache_stat(config(0));
    ASSERT_EQ(stats.entries.size(), 1u);
    EXPECT_FALSE(stats.entries[0].has_stamp);
    EXPECT_GT(stats.entries[0].last_used, std::chrono::system_clock::time_point{});
}

TEST_F(DiskCacheTest, StatCountsHardLinkedFileOnce) {
    const fs::path entry = make_entry("linked");
    const uint64_t single = disk_cache_stat(config(0)).total_size_bytes;

    std::error_code ec;
    fs::create_hard_link(entry / "kernels" / "blob", entry / "kernels" / "blob_link", ec);
    if (ec) {
        GTEST_SKIP() << "filesystem does not support hard links: " << ec.message();
    }

    const uint64_t with_link = disk_cache_stat(config(0)).total_size_bytes;
    // The extra directory entry may cost a little, but the payload must not be counted twice.
    EXPECT_LT(with_link, single + kEntryFileBytes);
}

TEST_F(DiskCacheTest, TrimEvictsLeastRecentlyUsedUntilUnderLimit) {
    make_entry("oldest");
    make_entry("middle");
    make_entry("newest");
    set_idle_for("oldest", std::chrono::hours{50});
    set_idle_for("middle", std::chrono::hours{5});
    set_idle_for("newest", std::chrono::seconds{10});

    // Room for roughly one entry, so the two least recently used must go.
    const DiskCacheTrimResult result = disk_cache_trim(config(kEntryFileBytes + kEntryFileBytes / 2));

    EXPECT_FALSE(result.skipped);
    EXPECT_EQ(result.entries_removed, 2u);
    EXPECT_FALSE(entry_exists("oldest"));
    EXPECT_FALSE(entry_exists("middle"));
    EXPECT_TRUE(entry_exists("newest"));
    EXPECT_GT(result.bytes_reclaimed, 0u);
    EXPECT_LT(result.bytes_after, result.bytes_before);
}

TEST_F(DiskCacheTest, TrimLeavesCacheAloneWhenUnderLimit) {
    make_entry("keep_a");
    make_entry("keep_b");
    set_idle_for("keep_a", std::chrono::hours{50});
    set_idle_for("keep_b", std::chrono::hours{50});

    const DiskCacheTrimResult result = disk_cache_trim(config(100 * kEntryFileBytes));

    EXPECT_FALSE(result.skipped);
    EXPECT_EQ(result.entries_removed, 0u);
    EXPECT_TRUE(entry_exists("keep_a"));
    EXPECT_TRUE(entry_exists("keep_b"));
}

TEST_F(DiskCacheTest, TrimSkipsEntryHeldByALiveProcess) {
    make_entry("locked");
    make_entry("free");
    set_idle_for("locked", std::chrono::hours{50});
    set_idle_for("free", std::chrono::hours{40});

    const DiskCacheEntryLock hold(root_ / "locked");
    ASSERT_TRUE(hold.held());

    // Limit of 1 byte: both entries are eviction candidates.
    const DiskCacheTrimResult result = disk_cache_trim(config(1));

    EXPECT_EQ(result.entries_skipped_in_use, 1u);
    EXPECT_EQ(result.entries_removed, 1u);
    EXPECT_TRUE(entry_exists("locked"));
    EXPECT_FALSE(entry_exists("free"));
}

TEST_F(DiskCacheTest, StatReportsHeldEntryAsInUse) {
    make_entry("locked");
    const DiskCacheEntryLock hold(root_ / "locked");
    ASSERT_TRUE(hold.held());

    const DiskCacheStats stats = disk_cache_stat(config(0));
    ASSERT_EQ(stats.entries.size(), 1u);
    EXPECT_TRUE(stats.entries[0].in_use);
}

TEST_F(DiskCacheTest, TrimRespectsMinEvictionAge) {
    // Two entries, both recent. One candidate alone would be kept by the working-set rule
    // regardless of age, which would mask what this test is about.
    make_entry("recent_a");
    make_entry("recent_b");
    set_idle_for("recent_a", std::chrono::seconds{30});
    set_idle_for("recent_b", std::chrono::seconds{20});

    DiskCacheConfig cfg = config(1);
    cfg.min_eviction_age = std::chrono::hours{1};

    const DiskCacheTrimResult result = disk_cache_trim(cfg);

    EXPECT_EQ(result.entries_skipped_too_young, 2u);
    EXPECT_EQ(result.entries_removed, 0u);
    EXPECT_TRUE(entry_exists("recent_a"));
    EXPECT_TRUE(entry_exists("recent_b"));
}

// A single entry bigger than the bound can never satisfy it, so evicting it would only have
// the next run regenerate it and the next trim remove it again -- unbounded churn, and a
// process that crashed mid-build would lose the tree it is about to reuse.
TEST_F(DiskCacheTest, TrimKeepsTheLastEntryRatherThanEmptyTheCache) {
    make_entry("only_one");
    set_idle_for("only_one", std::chrono::hours{50});

    const DiskCacheTrimResult result = disk_cache_trim(config(1));

    EXPECT_FALSE(result.skipped);
    EXPECT_EQ(result.entries_removed, 0u);
    EXPECT_EQ(result.entries_kept_as_working_set, 1u);
    EXPECT_TRUE(entry_exists("only_one"));

    // Repeating it must be equally inert -- that is the anti-churn property.
    const DiskCacheTrimResult again = disk_cache_trim(config(1));
    EXPECT_EQ(again.entries_removed, 0u);
    EXPECT_TRUE(entry_exists("only_one"));
}

// The under-budget check must be tested before the working-set rule. Getting that order wrong
// makes a cache that was just trimmed down to one entry report itself as stuck over its bound.
TEST_F(DiskCacheTest, ATrimThatReachesItsBoundDoesNotReportBeingStuck) {
    make_entry("old");
    make_entry("newer");
    set_idle_for("old", std::chrono::hours{50});
    set_idle_for("newer", std::chrono::hours{10});

    // Room for one entry: "old" goes, "newer" is then both the last entry AND under budget.
    const DiskCacheTrimResult result = disk_cache_trim(config(kEntryFileBytes + kEntryFileBytes / 2));

    EXPECT_EQ(result.entries_removed, 1u);
    EXPECT_EQ(result.entries_kept_as_working_set, 0u) << "under budget is not 'stuck'";
    EXPECT_TRUE(entry_exists("newer"));
}

// clear is an explicit human action and is not subject to the working-set rule.
TEST_F(DiskCacheTest, ClearStillEmptiesASingleEntryCache) {
    make_entry("only_one");
    const DiskCacheTrimResult result = disk_cache_clear(config(0));
    EXPECT_EQ(result.entries_removed, 1u);
    EXPECT_EQ(result.entries_kept_as_working_set, 0u);
    EXPECT_FALSE(entry_exists("only_one"));
}

TEST_F(DiskCacheTest, TrimIsSkippedWhenDisabled) {
    make_entry("keep");
    set_idle_for("keep", std::chrono::hours{50});

    DiskCacheConfig cfg = config(1);
    cfg.trim_enabled = false;

    const DiskCacheTrimResult result = disk_cache_trim(cfg);

    EXPECT_TRUE(result.skipped);
    EXPECT_EQ(result.entries_removed, 0u);
    EXPECT_TRUE(entry_exists("keep"));
}

TEST_F(DiskCacheTest, TrimIsSkippedWithoutAnyLimit) {
    make_entry("keep");
    set_idle_for("keep", std::chrono::hours{50});

    const DiskCacheTrimResult result = disk_cache_trim(config(0));

    EXPECT_TRUE(result.skipped);
    EXPECT_TRUE(entry_exists("keep"));
}

TEST_F(DiskCacheTest, TrimIsSkippedWhileAnotherProcessHoldsTheRootLock) {
    make_entry("evictable");
    set_idle_for("evictable", std::chrono::hours{50});

    // Stand in for a concurrent trimmer. flock is owned by the open file description, so a
    // second descriptor in this process contends exactly as another process would.
    const int fd = ::open((root_ / ".trim.lock").c_str(), O_RDWR | O_CREAT, 0644);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::flock(fd, LOCK_EX | LOCK_NB), 0);

    const DiskCacheTrimResult result = disk_cache_trim(config(1));

    EXPECT_TRUE(result.skipped);
    EXPECT_TRUE(entry_exists("evictable"));

    ::flock(fd, LOCK_UN);
    ::close(fd);
}

TEST_F(DiskCacheTest, TrimDrainsLeftoverTrash) {
    // What a trim interrupted by process exit leaves behind.
    const fs::path staged = root_ / ".trash" / "1234-stale-entry";
    fs::create_directories(staged);
    {
        const std::ofstream blob(staged / "blob");
    }
    make_entry("live");
    set_idle_for("live", std::chrono::hours{50});

    const DiskCacheTrimResult result = disk_cache_trim(config(100 * kEntryFileBytes));

    EXPECT_FALSE(result.skipped);
    EXPECT_FALSE(fs::exists(staged));
    EXPECT_TRUE(entry_exists("live"));
}

TEST_F(DiskCacheTest, TrimStagesEvictionsThroughTrashRatherThanDeletingInPlace) {
    make_entry("doomed");
    make_entry("survivor");  // so "doomed" is not the last entry, which would be kept
    set_idle_for("doomed", std::chrono::hours{50});
    set_idle_for("survivor", std::chrono::hours{10});

    disk_cache_trim(config(1));

    // The entry must be gone from the lookup namespace, and the trash must be drained too,
    // so no bytes are left stranded on a clean pass.
    EXPECT_FALSE(entry_exists("doomed"));
    std::error_code ec;
    EXPECT_TRUE(!fs::exists(root_ / ".trash", ec) || fs::is_empty(root_ / ".trash", ec));
}

TEST_F(DiskCacheTest, TrimRecordsWhenItLastRanSoStartupsCanSkipTheWalk) {
    make_entry("keep");
    set_idle_for("keep", std::chrono::hours{50});

    // A pass that evicts nothing still records itself: the cost being avoided is the scan.
    const DiskCacheTrimResult result = disk_cache_trim(config(100 * kEntryFileBytes));
    ASSERT_FALSE(result.skipped);

    const fs::path stamp = root_ / ".last_trim";
    ASSERT_TRUE(fs::exists(stamp));
    EXPECT_LT(
        std::chrono::system_clock::now() - std::chrono::system_clock::from_time_t(mtime_of(stamp)),
        std::chrono::minutes{5});

    // And it must not read back as a cache entry.
    EXPECT_EQ(disk_cache_stat(config(0)).entries.size(), 1u);
}

TEST_F(DiskCacheTest, ClearRemovesEverythingNotInUse) {
    make_entry("a");
    make_entry("b");
    make_entry("held");
    // Deliberately recent: clear is an explicit human action and ignores age policy.
    set_idle_for("a", std::chrono::seconds{1});
    set_idle_for("b", std::chrono::seconds{1});

    const DiskCacheEntryLock hold(root_ / "held");
    ASSERT_TRUE(hold.held());

    const DiskCacheTrimResult result = disk_cache_clear(config(0));

    EXPECT_FALSE(result.skipped);
    EXPECT_EQ(result.entries_removed, 2u);
    EXPECT_EQ(result.entries_skipped_in_use, 1u);
    EXPECT_FALSE(entry_exists("a"));
    EXPECT_FALSE(entry_exists("b"));
    EXPECT_TRUE(entry_exists("held"));
}

TEST_F(DiskCacheTest, ClearRunsEvenWhenTrimmingIsDisabled) {
    make_entry("a");
    DiskCacheConfig cfg = config(0);
    cfg.trim_enabled = false;

    const DiskCacheTrimResult result = disk_cache_clear(cfg);

    EXPECT_FALSE(result.skipped);
    EXPECT_FALSE(entry_exists("a"));
}

TEST_F(DiskCacheTest, TouchCreatesStampThenRateLimitsRewrites) {
    const fs::path entry = make_entry("stamped");
    const fs::path stamp = entry / ".last_used";
    ASSERT_FALSE(fs::exists(stamp));

    disk_cache_touch(entry);
    ASSERT_TRUE(fs::exists(stamp));

    // Backdate well inside the touch interval; a hit must not rewrite the stamp.
    backdate(stamp, std::chrono::system_clock::now() - std::chrono::minutes{5});
    const time_t before = mtime_of(stamp);
    disk_cache_touch(entry);
    EXPECT_EQ(before, mtime_of(stamp));
}

TEST_F(DiskCacheTest, TouchRewritesStampOlderThanTheTouchInterval) {
    const fs::path entry = make_entry("stale_stamp");
    const fs::path stamp = entry / ".last_used";
    backdate(stamp, std::chrono::system_clock::now() - (kDiskCacheTouchInterval + std::chrono::hours{1}));
    const time_t before = mtime_of(stamp);

    disk_cache_touch(entry);

    EXPECT_GT(mtime_of(stamp), before);
}

TEST_F(DiskCacheTest, EntryLockCreatesTheEntryDirectory) {
    const fs::path entry = root_ / "fresh_entry";
    ASSERT_FALSE(fs::exists(entry));

    const DiskCacheEntryLock hold(entry);

    EXPECT_TRUE(hold.held());
    EXPECT_TRUE(fs::is_directory(entry));
    EXPECT_TRUE(fs::exists(entry / ".inuse"));
}

TEST_F(DiskCacheTest, EntryLockIsSharedBetweenConcurrentUsers) {
    const fs::path entry = make_entry("shared");
    const DiskCacheEntryLock first(entry);
    const DiskCacheEntryLock second(entry);
    EXPECT_TRUE(first.held());
    EXPECT_TRUE(second.held());
}

// The .inuse-presence rule is what makes automatic eviction deployable: a binary predating
// the locking protocol never creates that file, so a trimmer cannot mistake its live tree for
// garbage -- and never considers it at all.
TEST_F(DiskCacheTest, UnmanagedTreesAreNeverTouchedAutomatically) {
    make_entry("managed_a");
    make_entry("managed_b");  // two, so trim can evict one without hitting the working-set rule
    make_unmanaged_entry("old_build_key");
    set_idle_for("managed_a", std::chrono::hours{50});
    set_idle_for("managed_b", std::chrono::hours{10});
    set_idle_for("old_build_key", std::chrono::hours{100});

    // A limit of 1 byte makes everything a candidate, and clear ignores limits entirely.
    disk_cache_trim(config(1));
    EXPECT_FALSE(entry_exists("managed_a"));
    EXPECT_TRUE(entry_exists("old_build_key"));

    disk_cache_clear(config(0));
    EXPECT_TRUE(entry_exists("old_build_key"));
}

// The same rule is the guard against a mistyped --root: an arbitrary directory's children
// carry no .inuse, so nothing is a candidate and nothing is deleted.
TEST_F(DiskCacheTest, TrimAndClearDeleteNothingInADirectoryThatIsNotACache) {
    const fs::path stranger = fs::temp_directory_path() / ("tt_disk_cache_stranger_" + std::to_string(::getpid()));
    fs::create_directories(stranger / "precious_dataset");
    {
        const std::ofstream data(stranger / "precious_dataset" / "weights.bin");
    }

    DiskCacheConfig cfg = config(1);
    cfg.root = stranger;
    disk_cache_trim(cfg);
    disk_cache_clear(cfg);

    EXPECT_TRUE(fs::exists(stranger / "precious_dataset" / "weights.bin"));
    std::error_code ec;
    fs::remove_all(stranger, ec);
}

// Existing caches are adopted by use rather than by a migration: claiming a tree creates
// .inuse, which is what promotes it into the managed set.
TEST_F(DiskCacheTest, ClaimingAnUnmanagedTreeMakesItManaged) {
    make_unmanaged_entry("adopt_me");
    EXPECT_EQ(disk_cache_stat(config(0)).entries.size(), 0u);
    EXPECT_EQ(disk_cache_stat_unmanaged(config(0)).entries.size(), 1u);

    ASSERT_TRUE(disk_cache_hold_entry_for_process(root_ / "adopt_me"));

    const DiskCacheStats managed = disk_cache_stat(config(0));
    ASSERT_EQ(managed.entries.size(), 1u);
    EXPECT_EQ(managed.entries[0].name, "adopt_me");
    EXPECT_TRUE(managed.entries[0].managed);
    EXPECT_EQ(disk_cache_stat_unmanaged(config(0)).entries.size(), 0u);
}

// Scanning must never create .inuse, or merely looking at a cache would promote every
// unmanaged tree into an eviction candidate.
TEST_F(DiskCacheTest, ScanningDoesNotPromoteUnmanagedTrees) {
    make_unmanaged_entry("untouched");
    set_idle_for("untouched", std::chrono::hours{100});

    disk_cache_stat(config(0));
    disk_cache_stat_unmanaged(config(0));
    disk_cache_trim(config(1));

    EXPECT_FALSE(fs::exists(root_ / "untouched" / ".inuse"));
    EXPECT_TRUE(entry_exists("untouched"));
}

TEST_F(DiskCacheTest, StatSeparatesManagedFromUnmanagedEntries) {
    make_entry("managed_a");
    make_entry("managed_b");
    make_unmanaged_entry("old_a");

    EXPECT_EQ(disk_cache_stat(config(0)).entries.size(), 2u);

    const DiskCacheStats unmanaged = disk_cache_stat_unmanaged(config(0));
    ASSERT_EQ(unmanaged.entries.size(), 1u);
    EXPECT_EQ(unmanaged.entries[0].name, "old_a");
    EXPECT_FALSE(unmanaged.entries[0].managed);
    EXPECT_GE(unmanaged.total_size_bytes, kEntryFileBytes);
    EXPECT_LT(unmanaged.total_size_bytes, 2 * kEntryFileBytes);
}

TEST_F(DiskCacheTest, PruneUnmanagedRemovesOnlyUnclaimedTrees) {
    make_entry("managed");
    make_unmanaged_entry("old_a");
    make_unmanaged_entry("old_b");
    set_idle_for("managed", std::chrono::hours{50});

    const DiskCacheTrimResult result = disk_cache_prune_unmanaged(config(0));

    EXPECT_FALSE(result.skipped);
    EXPECT_EQ(result.entries_removed, 2u);
    EXPECT_FALSE(entry_exists("old_a"));
    EXPECT_FALSE(entry_exists("old_b"));
    EXPECT_TRUE(entry_exists("managed"));
}

// prune-unmanaged's candidates are the entries WITHOUT .inuse, so it cannot use that file as
// its guard and needs the root marker instead.
TEST_F(DiskCacheTest, PruneUnmanagedRefusesADirectoryThatIsNotACacheRoot) {
    const fs::path stranger = root_ / "not_a_cache";
    fs::create_directories(stranger / "precious_data");

    DiskCacheConfig cfg = config(0);
    cfg.root = stranger;
    ASSERT_FALSE(disk_cache_root_is_initialized(stranger));

    const DiskCacheTrimResult result = disk_cache_prune_unmanaged(cfg);

    EXPECT_TRUE(result.skipped);
    EXPECT_EQ(result.entries_removed, 0u);
    EXPECT_TRUE(fs::exists(stranger / "precious_data"));
}

TEST_F(DiskCacheTest, InitializeRootIsIdempotentAndMarksTheRoot) {
    EXPECT_TRUE(disk_cache_root_is_initialized(root_));
    disk_cache_initialize_root(root_);
    EXPECT_TRUE(disk_cache_root_is_initialized(root_));

    const fs::path fresh = root_ / "fresh_root";
    EXPECT_FALSE(disk_cache_root_is_initialized(fresh));
    disk_cache_initialize_root(fresh);
    EXPECT_TRUE(disk_cache_root_is_initialized(fresh));
}

TEST_F(DiskCacheTest, ParseBoolAcceptsTheUsualSpellings) {
    for (const char* yes : {"1", "true", "TRUE", "yes", "On", " on "}) {
        EXPECT_EQ(parse_disk_cache_bool(yes), true) << yes;
    }
    for (const char* no : {"0", "false", "FALSE", "no", "Off"}) {
        EXPECT_EQ(parse_disk_cache_bool(no), false) << no;
    }
    EXPECT_FALSE(parse_disk_cache_bool("").has_value());
    EXPECT_FALSE(parse_disk_cache_bool("maybe").has_value());
}

// One parser, one error policy: a malformed cache knob must never stop a process running, and
// the spellings must not mean the opposite of what a reader expects.
TEST_F(DiskCacheTest, ApplyEnvIsForgivingAndConsistent) {
    {  // Unset and exported-empty are the same thing, and neither disturbs the defaults.
        DiskCacheConfig cfg;
        disk_cache_apply_env(cfg, nullptr, nullptr);
        EXPECT_EQ(cfg.max_size_bytes, 0u);
        EXPECT_TRUE(cfg.trim_enabled);

        disk_cache_apply_env(cfg, "", "  ");
        EXPECT_EQ(cfg.max_size_bytes, 0u);
        EXPECT_TRUE(cfg.trim_enabled);
    }
    {  // Garbage keeps the default rather than aborting the process.
        DiskCacheConfig cfg;
        cfg.max_size_bytes = 123;
        disk_cache_apply_env(cfg, "not-a-size", "not-a-bool");
        EXPECT_EQ(cfg.max_size_bytes, 123u);
        EXPECT_TRUE(cfg.trim_enabled);
    }
    {  // A truthy spelling must not silently disable trimming.
        DiskCacheConfig cfg;
        disk_cache_apply_env(cfg, nullptr, "true");
        EXPECT_TRUE(cfg.trim_enabled);
        disk_cache_apply_env(cfg, nullptr, "0");
        EXPECT_FALSE(cfg.trim_enabled);
    }
    {  // Real values still land.
        DiskCacheConfig cfg;
        disk_cache_apply_env(cfg, "2G", "1");
        EXPECT_EQ(cfg.max_size_bytes, 2ull * 1024 * 1024 * 1024);
        EXPECT_TRUE(cfg.trim_enabled);
    }
}

TEST_F(DiskCacheTest, AutomaticTrimIsOffUntilALimitIsConfigured) {
    DiskCacheConfig cfg = config(0);
    auto blocker = disk_cache_automatic_trim_blocker(cfg);
    ASSERT_TRUE(blocker.has_value());
    EXPECT_NE(blocker->find("no size limit"), std::string::npos);

    cfg.max_size_bytes = 1024;
    EXPECT_FALSE(disk_cache_automatic_trim_blocker(cfg).has_value());

    cfg.trim_enabled = false;
    ASSERT_TRUE(disk_cache_automatic_trim_blocker(cfg).has_value());
    EXPECT_NE(disk_cache_automatic_trim_blocker(cfg)->find("disabled"), std::string::npos);
}

// Device init calls this while holding a build mutex, so it must never wait on the filesystem.
TEST_F(DiskCacheTest, EntryLockGivesUpInsteadOfBlockingOnAnExclusiveHolder) {
    const fs::path entry = make_entry("contended");
    const fs::path inuse = entry / ".inuse";
    {
        const std::ofstream create(inuse);
    }

    const int fd = ::open(inuse.c_str(), O_RDWR);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::flock(fd, LOCK_EX | LOCK_NB), 0);

    const auto start = std::chrono::steady_clock::now();
    const DiskCacheEntryLock blocked(entry);
    const auto waited = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(blocked.held());
    EXPECT_LT(waited, std::chrono::seconds{5}) << "constructor blocked instead of retrying";

    ::flock(fd, LOCK_UN);
    ::close(fd);
}

TEST_F(DiskCacheTest, HoldEntryForProcessReportsSuccessAndRetriesAfterFailure) {
    const fs::path entry = make_entry("claimed");
    const fs::path inuse = entry / ".inuse";
    {
        const std::ofstream create(inuse);
    }

    const int fd = ::open(inuse.c_str(), O_RDWR);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::flock(fd, LOCK_EX | LOCK_NB), 0);
    EXPECT_FALSE(disk_cache_hold_entry_for_process(entry)) << "should not claim while exclusively locked";

    // A failed claim must not be remembered as success, or the entry stays unprotected for
    // the life of the process.
    ::flock(fd, LOCK_UN);
    ::close(fd);
    EXPECT_TRUE(disk_cache_hold_entry_for_process(entry));
    EXPECT_TRUE(disk_cache_hold_entry_for_process(entry)) << "second call should be a no-op returning success";
}

// A tree this process holds must survive its own background trimmer, which runs in-process
// and would otherwise be granted the exclusive lock by flock's own-process semantics.
TEST_F(DiskCacheTest, TrimNeverEvictsAnEntryThisProcessClaimed) {
    make_entry("mine");
    make_entry("theirs");
    set_idle_for("mine", std::chrono::hours{50});
    set_idle_for("theirs", std::chrono::hours{50});
    ASSERT_TRUE(disk_cache_hold_entry_for_process(root_ / "mine"));

    const DiskCacheTrimResult result = disk_cache_trim(config(1));

    EXPECT_EQ(result.entries_skipped_in_use, 1u);
    EXPECT_TRUE(entry_exists("mine"));
    EXPECT_FALSE(entry_exists("theirs"));
}

}  // namespace
}  // namespace tt::tt_metal
