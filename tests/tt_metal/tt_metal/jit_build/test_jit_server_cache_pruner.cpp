// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "impl/jit_server/server_cache_pruner.hpp"

namespace tt::tt_metal::jit_server {
namespace cache_pruner_test {

namespace fs = std::filesystem;

constexpr std::uint64_t kCapacity = 1ull << 40;
constexpr std::time_t kHour = 3600;

void write_file(const fs::path& path, std::size_t bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << std::string(bytes, 'x');
}

// Sets the atime and mtime of everything under path, children first so setting them does not
// disturb the parent's times.
void set_times(const fs::path& path, std::time_t when) {
    if (fs::is_directory(path) && !fs::is_symlink(path)) {
        for (const auto& entry : fs::directory_iterator(path)) {
            set_times(entry.path(), when);
        }
    }
    const struct timeval times[2] = {{when, 0}, {when, 0}};
    ASSERT_EQ(::lutimes(path.c_str(), times), 0) << path;
}

std::uint64_t allocated_bytes(const fs::path& path) {
    struct stat info{};
    if (::lstat(path.c_str(), &info) != 0) {
        return 0;
    }
    std::uint64_t bytes = static_cast<std::uint64_t>(info.st_blocks) * 512;
    if (S_ISDIR(info.st_mode)) {
        for (const auto& entry : fs::directory_iterator(path)) {
            bytes += allocated_bytes(entry.path());
        }
    }
    return bytes;
}

class ServerCachePrunerTest : public ::testing::Test {
protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / ("jit-cache-pruner-" + std::to_string(::getpid()) + "-" +
                                             ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::remove_all(root_);
        fs::create_directories(root_);
        config_.start_below_free_pct = 20.0;
        config_.stop_at_free_pct = 30.0;
        config_.min_idle = std::chrono::seconds(kHour);
    }

    void TearDown() override { fs::remove_all(root_); }

    // Creates <root>/<build_key>/kernels/<name>/<hash>/ with one compiled target, last used
    // hours_ago hours ago.
    fs::path make_kernel(
        const std::string& build_key, const std::string& name, const std::string& hash, int hours_ago) {
        const fs::path dir = root_ / build_key / "kernels" / name / hash;
        write_file(dir / "brisc" / "brisc.elf", 64 * 1024);
        write_file(dir / "brisc" / "brisc.o", 32 * 1024);
        write_file(dir / "kernel_includes.hpp", 1024);
        set_times(dir, std::time(nullptr) - hours_ago * kHour);
        return dir;
    }

    fs::path make_firmware(const std::string& build_key, int hours_ago) {
        const fs::path dir = root_ / build_key / "firmware" / "brisc";
        write_file(dir / "brisc_weakened.elf", 64 * 1024);
        set_times(root_ / build_key / "firmware", std::time(nullptr) - hours_ago * kHour);
        return dir;
    }

    // A pruner whose filesystem reports `free_pct` free space and needs `needed` more bytes to
    // reach the 30% target. Exactly one of the two is used.
    ServerCachePruner make_pruner(double free_pct, std::uint64_t needed = 0) {
        DiskSpace space{kCapacity, 0};
        if (needed > 0) {
            // Just under the target is still above the usual start threshold, so start there.
            config_.start_below_free_pct = config_.stop_at_free_pct;
            space.available = static_cast<std::uint64_t>(kCapacity * (config_.stop_at_free_pct / 100.0)) - needed;
        } else {
            space.available = static_cast<std::uint64_t>(kCapacity * (free_pct / 100.0));
        }
        return ServerCachePruner(
            root_,
            config_,
            [this](const std::string& message) { logs_.push_back(message); },
            [space](const fs::path&) { return space; });
    }

    fs::path root_;
    CachePruneConfig config_;
    std::vector<std::string> logs_;
};

TEST_F(ServerCachePrunerTest, DoesNothingWhileFreeSpaceIsAboveTheThreshold) {
    const fs::path old_kernel = make_kernel("1", "reader", "11", 1000);
    auto pruner = make_pruner(25.0);
    const CachePruneResult result = pruner.prune_once();
    EXPECT_FALSE(result.ran);
    EXPECT_TRUE(fs::exists(old_kernel));
    EXPECT_TRUE(logs_.empty());
}

TEST_F(ServerCachePrunerTest, DeletesLeastRecentlyUsedFirstAcrossBuildKeysUntilEnoughIsFree) {
    const fs::path oldest = make_kernel("2", "writer", "21", 50);
    const fs::path second = make_kernel("1", "reader", "11", 40);
    const fs::path third = make_kernel("1", "reader", "12", 30);
    const fs::path newest = make_kernel("2", "compute", "22", 20);

    auto pruner = make_pruner(0.0, allocated_bytes(oldest) + 1);
    const CachePruneResult result = pruner.prune_once();

    EXPECT_TRUE(result.ran);
    EXPECT_EQ(result.deleted, 2u);
    EXPECT_TRUE(result.reached_target);
    EXPECT_FALSE(fs::exists(oldest));
    EXPECT_FALSE(fs::exists(second));
    EXPECT_TRUE(fs::exists(third));
    EXPECT_TRUE(fs::exists(newest));
    // The writer kernel had only one variant, so its directory goes too.
    EXPECT_FALSE(fs::exists(oldest.parent_path()));
    EXPECT_TRUE(fs::exists(second.parent_path()));
    ASSERT_EQ(logs_.size(), 1u);
    EXPECT_NE(logs_[0].find("deleted 2 kernel directories"), std::string::npos) << logs_[0];
}

TEST_F(ServerCachePrunerTest, NeverDeletesRecentlyUsedKernelsFirmwareOrPch) {
    const fs::path recent = make_kernel("1", "reader", "11", 0);
    const fs::path firmware = make_firmware("1", 5000);
    write_file(root_ / "pch" / "umbrella.gch", 4096);
    set_times(root_ / "pch", std::time(nullptr) - 5000 * kHour);

    auto pruner = make_pruner(1.0);
    const CachePruneResult result = pruner.prune_once();

    EXPECT_TRUE(result.ran);
    EXPECT_EQ(result.deleted, 0u);
    EXPECT_FALSE(result.reached_target);
    EXPECT_TRUE(fs::exists(recent));
    EXPECT_TRUE(fs::exists(firmware / "brisc_weakened.elf"));
    EXPECT_TRUE(fs::exists(root_ / "pch" / "umbrella.gch"));
    ASSERT_EQ(logs_.size(), 1u);
    EXPECT_NE(logs_[0].find("Could not reach 30.0%"), std::string::npos) << logs_[0];
}

TEST_F(ServerCachePrunerTest, RecentFileAccessKeepsAnOldDirectory) {
    // A kernel last used by an older server, which did not stamp the directory: only a file's
    // atime shows the recent hit.
    const fs::path kernel = make_kernel("1", "reader", "11", 1000);
    const std::time_t now = std::time(nullptr);
    const struct timeval times[2] = {{now, 0}, {now - 1000 * kHour, 0}};
    ASSERT_EQ(::lutimes((kernel / "brisc" / "brisc.elf").c_str(), times), 0);

    auto pruner = make_pruner(1.0);
    EXPECT_EQ(pruner.prune_once().deleted, 0u);
    EXPECT_TRUE(fs::exists(kernel));
}

TEST_F(ServerCachePrunerTest, LeasedKernelIsSkippedAndStampedAsUsed) {
    const fs::path leased = make_kernel("1", "reader", "11", 100);
    const fs::path idle = make_kernel("1", "reader", "12", 50);

    auto pruner = make_pruner(1.0);
    {
        // kernel_cache_dir() adds a trailing slash and kernel names end in one too.
        const auto lease = pruner.acquire(leased.string() + "//");
        const CachePruneResult result = pruner.prune_once();
        EXPECT_EQ(result.deleted, 1u);
        EXPECT_TRUE(fs::exists(leased));
        EXPECT_FALSE(fs::exists(idle));
    }
    // The lease stamped the directory, so it now counts as just used.
    EXPECT_EQ(pruner.prune_once().deleted, 0u);
    EXPECT_TRUE(fs::exists(leased));
}

TEST_F(ServerCachePrunerTest, ReleasedLeaseNoLongerProtectsAnIdleKernel) {
    const fs::path kernel = make_kernel("1", "reader", "11", 100);
    config_.min_idle = std::chrono::seconds(0);
    auto pruner = make_pruner(1.0);
    {
        const auto lease = pruner.acquire(kernel);
    }
    EXPECT_EQ(pruner.prune_once().deleted, 1u);
    EXPECT_FALSE(fs::exists(kernel));
}

TEST_F(ServerCachePrunerTest, RemovesTrashLeftByAnInterruptedPrune) {
    const fs::path kernel = make_kernel("1", "reader", "11", 0);
    const fs::path trash = kernel.parent_path() / ".jit-prune-trash.12.999.1";
    write_file(trash / "brisc" / "brisc.elf", 64 * 1024);

    auto pruner = make_pruner(50.0);
    EXPECT_GT(pruner.remove_leftover_trash(), 0u);
    EXPECT_FALSE(fs::exists(trash));
    EXPECT_TRUE(fs::exists(kernel));
}

TEST_F(ServerCachePrunerTest, IgnoresNonBuildKeyEntriesAndSymlinks) {
    const fs::path outside = root_ / "outside";
    write_file(outside / "keep.txt", 1024);
    set_times(outside, std::time(nullptr) - 1000 * kHour);
    fs::create_directories(root_ / "1" / "kernels" / "reader");
    fs::create_directory_symlink(outside, root_ / "1" / "kernels" / "reader" / "11");
    write_file(root_ / "notakey" / "kernels" / "reader" / "11" / "x.elf", 1024);
    set_times(root_ / "notakey", std::time(nullptr) - 1000 * kHour);

    auto pruner = make_pruner(1.0);
    EXPECT_EQ(pruner.prune_once().deleted, 0u);
    EXPECT_TRUE(fs::exists(outside / "keep.txt"));
    EXPECT_TRUE(fs::exists(root_ / "notakey" / "kernels" / "reader" / "11" / "x.elf"));
}

TEST_F(ServerCachePrunerTest, DisabledPrunerTracksNothing) {
    const fs::path kernel = make_kernel("1", "reader", "11", 100);
    config_ = CachePruneConfig{};
    auto pruner = make_pruner(1.0);
    {
        const auto lease = pruner.acquire(kernel);
    }
    EXPECT_FALSE(pruner.prune_once().ran);
    EXPECT_TRUE(fs::exists(kernel));
}

TEST_F(ServerCachePrunerTest, BackgroundThreadPrunesAndStops) {
    const fs::path kernel = make_kernel("1", "reader", "11", 100);
    config_.check_interval = std::chrono::seconds(3600);
    auto pruner = make_pruner(1.0);
    pruner.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (fs::exists(kernel) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_FALSE(fs::exists(kernel));
    // stop() must not wait out the hour-long interval.
    const auto stop_start = std::chrono::steady_clock::now();
    pruner.stop();
    EXPECT_LT(std::chrono::steady_clock::now() - stop_start, std::chrono::seconds(5));
}

CachePruneConfig config_from(const std::map<std::string, std::string>& values) {
    return CachePruneConfig::from_env([&](const char* name) -> const char* {
        auto it = values.find(name);
        return it == values.end() ? nullptr : it->second.c_str();
    });
}

TEST(CachePruneConfigTest, IsOptIn) {
    EXPECT_FALSE(config_from({}).enabled());
    EXPECT_FALSE(config_from({{kCacheMinFreePctEnv, "0"}}).enabled());
    EXPECT_FALSE(config_from({{kCacheMinIdleEnv, "60"}}).enabled());
}

TEST(CachePruneConfigTest, ParsesValuesAndDefaultsTheTarget) {
    const CachePruneConfig defaults = config_from({{kCacheMinFreePctEnv, "15"}});
    EXPECT_TRUE(defaults.enabled());
    EXPECT_DOUBLE_EQ(defaults.start_below_free_pct, 15.0);
    EXPECT_DOUBLE_EQ(defaults.stop_at_free_pct, 25.0);
    EXPECT_EQ(defaults.min_idle.count(), 3600);
    EXPECT_EQ(defaults.check_interval.count(), 300);
    EXPECT_DOUBLE_EQ(config_from({{kCacheMinFreePctEnv, "95"}}).stop_at_free_pct, 100.0);

    const CachePruneConfig custom = config_from({
        {kCacheMinFreePctEnv, "10"},
        {kCacheTargetFreePctEnv, "12.5"},
        {kCacheMinIdleEnv, "0"},
        {kCacheCheckIntervalEnv, "60"},
    });
    EXPECT_DOUBLE_EQ(custom.stop_at_free_pct, 12.5);
    EXPECT_EQ(custom.min_idle.count(), 0);
    EXPECT_EQ(custom.check_interval.count(), 60);
}

TEST(CachePruneConfigTest, RejectsInvalidValues) {
    EXPECT_THROW(config_from({{kCacheMinFreePctEnv, "20%"}}), std::invalid_argument);
    EXPECT_THROW(config_from({{kCacheMinFreePctEnv, "-5"}}), std::invalid_argument);
    EXPECT_THROW(config_from({{kCacheMinFreePctEnv, "101"}}), std::invalid_argument);
    EXPECT_THROW(config_from({{kCacheMinFreePctEnv, "20"}, {kCacheTargetFreePctEnv, "10"}}), std::invalid_argument);
    EXPECT_THROW(config_from({{kCacheMinFreePctEnv, "20"}, {kCacheCheckIntervalEnv, "0"}}), std::invalid_argument);
    EXPECT_THROW(config_from({{kCacheMinFreePctEnv, "20"}, {kCacheMinIdleEnv, "soon"}}), std::invalid_argument);
}

}  // namespace cache_pruner_test
}  // namespace tt::tt_metal::jit_server
