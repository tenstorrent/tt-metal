// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/jit_server/jit_compile_server_controller.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "jit_build/kernel_cache_eviction.hpp"
#include "llrt/rtoptions.hpp"
#include "mock_blackhole_fixture.hpp"

namespace tt::tt_metal {
namespace {

namespace fs = std::filesystem;

constexpr std::uint64_t kFileBytes = 64 * 1024;

std::uint64_t disk_usage(const fs::path& path) {
    std::uint64_t size = 0;
    struct stat st{};
    if (::lstat(path.c_str(), &st) == 0) {
        size += static_cast<std::uint64_t>(st.st_blocks) * 512;
    }
    std::error_code ec;
    if (fs::is_directory(fs::symlink_status(path, ec))) {
        for (const auto& entry : fs::recursive_directory_iterator(path, ec)) {
            if (::lstat(entry.path().c_str(), &st) == 0) {
                size += static_cast<std::uint64_t>(st.st_blocks) * 512;
            }
        }
    }
    return size;
}

void write_file(const fs::path& path, std::uint64_t bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    const std::string chunk(4096, 'x');
    for (std::uint64_t written = 0; written < bytes; written += chunk.size()) {
        out.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    }
}

void set_mtime(const fs::path& path, std::time_t when) {
    const struct timespec times[2] = {{when, 0}, {when, 0}};
    ASSERT_EQ(::utimensat(AT_FDCWD, path.c_str(), times, 0), 0);
}

std::time_t mtime_of(const fs::path& path) {
    struct stat st{};
    return ::lstat(path.c_str(), &st) == 0 ? st.st_mtime : 0;
}

// Sets an environment variable for one scope and restores what was there before.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            old_ = old;
        }
        set(value);
    }
    ~ScopedEnv() { set(old_ ? old_->c_str() : nullptr); }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    void set(const char* value) {
        if (value) {
            ::setenv(name_, value, 1);
        } else {
            ::unsetenv(name_);
        }
    }
    const char* name_;
    std::optional<std::string> old_;
};

class KernelCacheEvictionTest : public ::testing::Test {
protected:
    void SetUp() override {
        static std::atomic<int> counter{0};
        root_ = fs::temp_directory_path() /
                ("kernel-cache-eviction-" + std::to_string(::getpid()) + "-" + std::to_string(counter++));
        fs::remove_all(root_);
        fs::create_directories(root_);
    }
    void TearDown() override { fs::remove_all(root_); }

    // <root>/<build_key>/kernels/<kernel>/<variant>/ laid out like the JIT build, one ELF per variant.
    // Variant i gets an mtime i minutes after an hour ago, so variant 0 is the least recently used.
    std::vector<fs::path> make_variants(int count, const std::string& build_key = "1234") {
        std::vector<fs::path> variants;
        const std::time_t base = std::time(nullptr) - 3600;
        for (int i = 0; i < count; i++) {
            const fs::path dir = root_ / build_key / "kernels" / ("kernel" + std::to_string(i % 3)) / std::to_string(i);
            write_file(dir / "trisc0" / "trisc0.elf", kFileBytes);
            set_mtime(dir, base + i * 60);
            variants.push_back(dir);
        }
        return variants;
    }

    void write_config(const std::string& text) { std::ofstream(root_ / "tt-metal-cache.conf") << text; }

    fs::path root_;
};

TEST(KernelCacheSizeTest, ParsesSizes) {
    EXPECT_EQ(parse_cache_size("0"), 0u);
    EXPECT_EQ(parse_cache_size("20"), 20ull << 30);  // a bare number is GiB, as in ccache
    EXPECT_EQ(parse_cache_size("1.5"), 3ull << 29);
    EXPECT_EQ(parse_cache_size("2k"), 2000u);
    EXPECT_EQ(parse_cache_size("2K"), 2000u);
    EXPECT_EQ(parse_cache_size("500M"), 500'000'000u);
    EXPECT_EQ(parse_cache_size("20G"), 20'000'000'000u);
    EXPECT_EQ(parse_cache_size("20GB"), 20'000'000'000u);
    EXPECT_EQ(parse_cache_size("2 G"), 2'000'000'000u);
    EXPECT_EQ(parse_cache_size("1.5G"), 1'500'000'000u);
    EXPECT_EQ(parse_cache_size(".5G"), 500'000'000u);
    EXPECT_EQ(parse_cache_size("3T"), 3'000'000'000'000u);
    EXPECT_EQ(parse_cache_size("1Ki"), 1024u);
    EXPECT_EQ(parse_cache_size("500Mi"), 500ull << 20);
    EXPECT_EQ(parse_cache_size("1GiB"), 1ull << 30);
    EXPECT_EQ(parse_cache_size("1.5Ti"), 3ull << 39);
}

TEST(KernelCacheSizeTest, RejectsBadSizes) {
    for (const char* bad :
         {"",
          "G",
          ".",
          "-1",
          " 1G",
          "10X",
          "1Gx",
          "1g",
          "1m",
          "1gi",
          "100B",
          "1iB",
          "1GiBB",
          "1.5.5G",
          "20 x",
          "99999999999999999999",
          "20000000T"}) {
        EXPECT_FALSE(parse_cache_size(bad).has_value()) << "'" << bad << "'";
    }
}

// The JIT build appends "<build_key>/" to the cache directory, and the trim scans that same directory.
TEST(KernelCacheSizeTest, CacheDirEndsInSeparator) {
    ScopedEnv env("TT_METAL_CACHE", "/tmp/some-cache");
    llrt::RunTimeOptions rtoptions;
    EXPECT_EQ(rtoptions.get_cache_dir(), "/tmp/some-cache/tt-metal-cache/");
}

TEST_F(KernelCacheEvictionTest, MissingConfigMeansNoCap) { EXPECT_EQ(read_kernel_cache_max_size(root_), 0u); }

TEST_F(KernelCacheEvictionTest, ReadsMaxSizeFromConfig) {
    write_config("# shared by every process using this cache\n\n  max_size =  20G  \r\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 20'000'000'000u);
    write_config("max_size=1Gi\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 1ull << 30);
    write_config("max_size = 0\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 0u);
    write_config("max_size = 20G  # team cap\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 20'000'000'000u);
    write_config("max_size = 20\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 20ull << 30);
}

TEST_F(KernelCacheEvictionTest, LastValidMaxSizeWins) {
    write_config("max_size = 1G\nmax_size = lots\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 1'000'000'000u);
    write_config("max_size = lots\nmax_size = 2G\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 2'000'000'000u);
    write_config("max_size = 1G\nmax_size = 3G\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 3'000'000'000u);
}

TEST_F(KernelCacheEvictionTest, ConfigIgnoresUnknownLines) {
    write_config("compression = true\nnot a setting\nmax_size = 500M\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 500'000'000u);
    write_config("compression = true\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 0u);
}

TEST_F(KernelCacheEvictionTest, InvalidMaxSizeMeansNoCap) {
    for (const char* bad : {"max_size = 20 x\n", "max_size = lots\n", "max_size =\n", "max_size = -1\n"}) {
        write_config(bad);
        EXPECT_EQ(read_kernel_cache_max_size(root_), 0u) << bad;
    }
}

// A cap below 100M is most likely a typo ("max_size = 20" written as bytes): warned about and ignored.
TEST_F(KernelCacheEvictionTest, TinyCapIsIgnored) {
    write_config("max_size = 50M\n");
    ::testing::internal::CaptureStdout();
    EXPECT_EQ(read_kernel_cache_max_size(root_), 0u);
    const std::string log = ::testing::internal::GetCapturedStdout();
    EXPECT_NE(log.find("below the smallest kernel cache cap of 100M"), std::string::npos) << log;
    write_config("max_size = 100M\n");
    EXPECT_EQ(read_kernel_cache_max_size(root_), 100'000'000u);
}

TEST_F(KernelCacheEvictionTest, NoCapIsNoop) {
    const auto variants = make_variants(5);
    const auto result = trim_kernel_cache(root_, 0, std::chrono::seconds(0));
    EXPECT_FALSE(result.trimmed);
    for (const auto& v : variants) {
        EXPECT_TRUE(fs::exists(v));
    }
    EXPECT_FALSE(fs::exists(root_ / ".trim.lock"));
}

TEST_F(KernelCacheEvictionTest, UnderCapIsNoop) {
    const auto variants = make_variants(5);
    const auto result = trim_kernel_cache(root_, disk_usage(root_) + 4096, std::chrono::seconds(0));
    EXPECT_FALSE(result.trimmed);
    EXPECT_EQ(result.variants_evicted, 0u);
    for (const auto& v : variants) {
        EXPECT_TRUE(fs::exists(v));
    }
}

TEST_F(KernelCacheEvictionTest, EvictsLeastRecentlyUsedToLowWater) {
    const auto variants = make_variants(10);
    const std::uint64_t cap = disk_usage(root_) * 7 / 10;
    const auto result = trim_kernel_cache(root_, cap, std::chrono::seconds(0));

    EXPECT_TRUE(result.trimmed);
    const auto low_water = static_cast<std::uint64_t>(static_cast<double>(cap) * kKernelCacheLowWaterFraction);
    EXPECT_LE(result.size_after, low_water);
    EXPECT_LE(disk_usage(root_), low_water);
    // Evicted variants are exactly the oldest ones, and no more than needed: the next one would have fit.
    const std::size_t evicted = result.variants_evicted;
    ASSERT_GT(evicted, 0u);
    ASSERT_LT(evicted, variants.size());
    for (std::size_t i = 0; i < variants.size(); i++) {
        EXPECT_EQ(fs::exists(variants[i]), i >= evicted) << variants[i];
    }
    // All variants are the same size, so the first kept one stands in for the last evicted one.
    EXPECT_GT(result.size_after + disk_usage(variants[evicted]), low_water);
    EXPECT_FALSE(fs::exists(root_ / ".trash"));
}

TEST_F(KernelCacheEvictionTest, OrderIsLastUseNotCreation) {
    auto variants = make_variants(4);
    // The first-created variant was used most recently.
    set_mtime(variants[0], std::time(nullptr) - 60);
    const auto result = trim_kernel_cache(root_, disk_usage(root_) / 2, std::chrono::seconds(0));
    ASSERT_TRUE(result.trimmed);
    EXPECT_TRUE(fs::exists(variants[0]));
    EXPECT_FALSE(fs::exists(variants[1]));
}

// Stamps only while a cap is set; a process re-stamps a directory once its own last stamp is old enough.
TEST_F(KernelCacheEvictionTest, NoteStampsOnlyWithCapAndRestamps) {
    const auto variants = make_variants(1);
    const std::time_t old = mtime_of(variants[0]);
    set_kernel_cache_cap(root_, 0);
    note_kernel_cache_dir_used(variants[0].string() + "/");
    EXPECT_EQ(mtime_of(variants[0]), old);

    set_kernel_cache_cap(root_, 1ull << 40);
    note_kernel_cache_dir_used(variants[0].string() + "/");
    EXPECT_GT(mtime_of(variants[0]), old);
    set_mtime(variants[0], old);
    note_kernel_cache_dir_used(variants[0]);  // stamped by this process just now: no new stamp yet
    EXPECT_EQ(mtime_of(variants[0]), old);
    note_kernel_cache_dir_used(variants[0], std::chrono::seconds(0));
    EXPECT_GT(mtime_of(variants[0]), old);
    set_kernel_cache_cap(root_, 0);
}

TEST_F(KernelCacheEvictionTest, RecentlyUsedVariantsAreKept) {
    auto variants = make_variants(4);
    set_mtime(variants[0], std::time(nullptr));  // another process just used it
    const auto result = trim_kernel_cache(root_, disk_usage(root_) / 4, std::chrono::minutes(15));
    EXPECT_EQ(result.variants_evicted, 3u);
    EXPECT_TRUE(fs::exists(variants[0]));
    for (std::size_t i = 1; i < variants.size(); i++) {
        EXPECT_FALSE(fs::exists(variants[i])) << variants[i];
    }
}

// PCH directories are evicted by the same LRU as variants; firmware is not.
TEST_F(KernelCacheEvictionTest, PchDirsAreEvictedLikeVariants) {
    const auto variants = make_variants(2);
    const fs::path old_pch = root_ / "pch" / "0123456789abcdef";
    const fs::path new_pch = root_ / "pch" / "fedcba9876543210";
    write_file(old_pch / "pch.h.gch", 4 * kFileBytes);
    write_file(new_pch / "pch.h.gch", 4 * kFileBytes);
    set_mtime(old_pch, std::time(nullptr) - 7200);  // older than every variant
    const fs::path firmware = root_ / "1234" / "firmware" / "brisc" / "brisc.elf";
    write_file(firmware, kFileBytes);
    set_mtime(firmware.parent_path(), std::time(nullptr) - 86400);

    const auto result = trim_kernel_cache(root_, disk_usage(root_) - kFileBytes, std::chrono::seconds(0));
    EXPECT_EQ(result.variants_evicted, 1u);
    EXPECT_FALSE(fs::exists(old_pch));
    EXPECT_TRUE(fs::exists(new_pch));
    EXPECT_TRUE(fs::exists(firmware));
    EXPECT_TRUE(fs::exists(root_ / "pch"));
}

// Files that are never evicted are over the low-water level on their own: evicting every kernel would not
// get under the cap, so nothing is evicted and the user is told.
TEST_F(KernelCacheEvictionTest, NonEvictableOverCapEvictsNothing) {
    const auto variants = make_variants(3);
    const fs::path firmware = root_ / "1234" / "firmware" / "brisc" / "brisc.elf";
    write_file(firmware, 8 * kFileBytes);
    ::testing::internal::CaptureStdout();
    const auto result = trim_kernel_cache(root_, 4 * kFileBytes, std::chrono::seconds(0));
    const std::string log = ::testing::internal::GetCapturedStdout();
    EXPECT_FALSE(result.trimmed);
    EXPECT_EQ(result.variants_evicted, 0u);
    for (const auto& v : variants) {
        EXPECT_TRUE(fs::exists(v));
    }
    EXPECT_NE(log.find("is below the non-evictable"), std::string::npos) << log;
}

// A trim writes the size counter from its scan, minus what it freed.
TEST_F(KernelCacheEvictionTest, TrimWritesTheCounter) {
    make_variants(10);
    EXPECT_FALSE(read_kernel_cache_size_counter(root_).has_value());
    const auto result = trim_kernel_cache(root_, disk_usage(root_) / 2, std::chrono::seconds(0));
    ASSERT_TRUE(result.trimmed);
    EXPECT_EQ(read_kernel_cache_size_counter(root_), result.size_after);
}

// Start-up reads only the counter: no walk of the cache while the counter is valid and under the cap.
TEST_F(KernelCacheEvictionTest, StartupWithCounterDoesNotScan) {
    make_variants(5);
    const auto first = maintain_kernel_cache(root_, 1ull << 40);
    EXPECT_TRUE(first.scanned);  // no counter yet
    // Hide the tree: a walk would now see an empty cache and rewrite the counter to almost nothing.
    fs::rename(root_ / "1234", root_.string() + "-hidden");
    const auto second = maintain_kernel_cache(root_, 1ull << 40);
    fs::rename(root_.string() + "-hidden", root_ / "1234");
    EXPECT_FALSE(second.scanned);
    EXPECT_EQ(read_kernel_cache_size_counter(root_), first.size_after);
}

// A missing or corrupt counter triggers one rescan, which writes a valid counter again.
TEST_F(KernelCacheEvictionTest, MissingOrCorruptCounterRescansOnce) {
    make_variants(5);
    EXPECT_TRUE(maintain_kernel_cache(root_, 1ull << 40).scanned);
    EXPECT_FALSE(maintain_kernel_cache(root_, 1ull << 40).scanned);
    for (const char* bad : {"garbage\n", "size = 12\n", "size = -1\nscan_time = 1\nscan_size = 0\n", ""}) {
        std::ofstream(root_ / ".stats") << bad;
        EXPECT_TRUE(maintain_kernel_cache(root_, 1ull << 40).scanned) << bad;
        EXPECT_FALSE(maintain_kernel_cache(root_, 1ull << 40).scanned) << bad;
    }
    fs::remove(root_ / ".stats");
    EXPECT_TRUE(maintain_kernel_cache(root_, 1ull << 40).scanned);
    EXPECT_TRUE(read_kernel_cache_size_counter(root_).has_value());
}

// A counter last resynced over a day ago is resynced from a scan.
TEST_F(KernelCacheEvictionTest, DailyResync) {
    make_variants(3);
    const std::time_t two_days_ago = std::time(nullptr) - 2 * 86400;
    std::ofstream(root_ / ".stats") << "size = 1\nscan_time = " << two_days_ago << "\nscan_size = 1\n";
    const auto result = maintain_kernel_cache(root_, 1ull << 40);
    EXPECT_TRUE(result.scanned);
    EXPECT_GT(read_kernel_cache_size_counter(root_).value_or(0), 1u);
}

// Over the cap at start-up: the counter alone triggers the trim.
TEST_F(KernelCacheEvictionTest, CounterOverCapTriggersTrim) {
    const auto variants = make_variants(10);
    const std::uint64_t size = disk_usage(root_);
    std::ofstream(root_ / ".stats") << "size = " << size << "\nscan_time = " << std::time(nullptr)
                                    << "\nscan_size = " << size / 4 << "\n";
    const auto result = maintain_kernel_cache(root_, size / 2, std::chrono::seconds(0));
    EXPECT_TRUE(result.scanned);
    EXPECT_TRUE(result.trimmed);
    EXPECT_FALSE(fs::exists(variants[0]));
}

// A scan that could not get under the cap is not repeated on every start within min_evict_age.
TEST_F(KernelCacheEvictionTest, StuckCacheIsNotRescannedRightAway) {
    make_variants(3);
    write_file(root_ / "1234" / "firmware" / "brisc" / "brisc.elf", 8 * kFileBytes);
    ::testing::internal::CaptureStdout();
    EXPECT_TRUE(maintain_kernel_cache(root_, 4 * kFileBytes, std::chrono::minutes(15)).scanned);
    EXPECT_FALSE(maintain_kernel_cache(root_, 4 * kFileBytes, std::chrono::minutes(15)).scanned);
    ::testing::internal::GetCapturedStdout();
}

// A build adds what it wrote to the counter; a cache hit adds nothing.
TEST_F(KernelCacheEvictionTest, CounterGrowsOnBuild) {
    make_variants(2);
    trim_kernel_cache(root_, 1ull << 40, std::chrono::seconds(0));
    const std::uint64_t before = read_kernel_cache_size_counter(root_).value();
    set_kernel_cache_cap(root_, 1ull << 40);
    const fs::path dir = root_ / "1234" / "kernels" / "new_kernel" / "42";
    track_kernel_cache_growth(dir, [&] { write_file(dir / "trisc0" / "trisc0.elf", 4 * kFileBytes); });
    const std::uint64_t after = read_kernel_cache_size_counter(root_).value();
    EXPECT_GE(after - before, 4 * kFileBytes);
    EXPECT_LE(after - before, 4 * kFileBytes + 4 * 4096);
    track_kernel_cache_growth(dir, [] {});  // a hit
    EXPECT_EQ(read_kernel_cache_size_counter(root_), after);
    set_kernel_cache_cap(root_, 0);
}

// Builds on many threads at once: the flock serializes the counter updates, so none is lost.
TEST_F(KernelCacheEvictionTest, CounterAddsFromThreadsAreNotLost) {
    trim_kernel_cache(root_, 1ull << 40, std::chrono::seconds(0));
    const std::uint64_t before = read_kernel_cache_size_counter(root_).value();
    set_kernel_cache_cap(root_, 1ull << 40);
    constexpr int kThreads = 8;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back([&, i] {
            const fs::path dir = root_ / "1234" / "kernels" / "k" / std::to_string(i);
            track_kernel_cache_growth(dir, [&] { write_file(dir / "brisc" / "brisc.elf", kFileBytes); });
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    set_kernel_cache_cap(root_, 0);
    std::uint64_t added = 0;
    for (int i = 0; i < kThreads; i++) {
        added += disk_usage(root_ / "1234" / "kernels" / "k" / std::to_string(i));
    }
    EXPECT_EQ(read_kernel_cache_size_counter(root_).value() - before, added);
}

// A build that takes the counter over the cap trims right away, mid-run.
TEST_F(KernelCacheEvictionTest, TrimFiresMidRunWhenOverCap) {
    const auto variants = make_variants(10);
    trim_kernel_cache(root_, 1ull << 40, std::chrono::seconds(0));
    const std::uint64_t cap = disk_usage(root_) + 2 * kFileBytes;
    set_kernel_cache_cap(root_, cap, std::chrono::seconds(0));
    const fs::path dir = root_ / "1234" / "kernels" / "new_kernel" / "42";
    track_kernel_cache_growth(dir, [&] { write_file(dir / "trisc0" / "trisc0.elf", kFileBytes); });
    EXPECT_TRUE(fs::exists(variants[0]));  // still under the cap
    track_kernel_cache_growth(dir, [&] { write_file(dir / "brisc" / "brisc.elf", 4 * kFileBytes); });
    set_kernel_cache_cap(root_, 0);
    EXPECT_FALSE(fs::exists(variants[0]));
    const auto low_water = static_cast<std::uint64_t>(static_cast<double>(cap) * kKernelCacheLowWaterFraction);
    EXPECT_LE(read_kernel_cache_size_counter(root_).value(), low_water);
    EXPECT_LE(disk_usage(root_), low_water + 4096);
}

// max_size = 0 turns the cap off: no stamps, counter, scan, lock or trash.
TEST_F(KernelCacheEvictionTest, ZeroCapTouchesNothing) {
    const auto variants = make_variants(3);
    write_config("max_size = 0\n");
    const std::uint64_t cap = resolve_kernel_cache_max_size(root_, std::nullopt);
    EXPECT_EQ(cap, 0u);
    const std::time_t old = mtime_of(variants[0]);
    set_kernel_cache_cap(root_, cap);
    EXPECT_FALSE(maintain_kernel_cache(root_, cap).scanned);
    EXPECT_FALSE(trim_kernel_cache(root_, cap, std::chrono::seconds(0)).scanned);
    note_kernel_cache_dir_used(variants[0], std::chrono::seconds(0));
    const fs::path dir = root_ / "1234" / "kernels" / "new_kernel" / "42";
    track_kernel_cache_growth(dir, [&] { write_file(dir / "trisc0" / "trisc0.elf", kFileBytes); });
    EXPECT_EQ(mtime_of(variants[0]), old);
    for (const char* name : {".stats", ".trim.lock", ".trash"}) {
        EXPECT_FALSE(fs::exists(root_ / name)) << name;
    }
}

// The root may be a symlink, with or without a trailing slash.
TEST_F(KernelCacheEvictionTest, SymlinkedRootIsTrimmed) {
    const auto variants = make_variants(4);
    const fs::path link = root_.string() + "-link";
    fs::remove(link);
    fs::create_directory_symlink(root_, link);
    const auto result = trim_kernel_cache(link, disk_usage(root_) / 2, std::chrono::seconds(0));
    fs::remove(link);
    EXPECT_TRUE(result.trimmed);
    EXPECT_FALSE(fs::exists(variants[0]));
}

// A kernel's directory goes with its last variant.
TEST_F(KernelCacheEvictionTest, EmptyKernelDirsAreRemoved) {
    const auto variants = make_variants(6);  // kernel0: 0, 3; kernel1: 1, 4; kernel2: 2, 5
    set_mtime(variants[3], std::time(nullptr) - 7200);
    const auto result = trim_kernel_cache(root_, disk_usage(root_) * 3 / 4, std::chrono::seconds(0));
    EXPECT_GE(result.variants_evicted, 2u);
    EXPECT_FALSE(fs::exists(variants[0].parent_path()));
    EXPECT_TRUE(fs::exists(variants[5].parent_path()));
}

// A trim that evicts a kernel's last variant, and so removes the kernel's directory, right between a builder
// making that directory and making its new variant directory in it: the builder still gets its directory.
fs::path race_root;
int race_hits = 0;
bool race_parent_was_removed = false;

TEST_F(KernelCacheEvictionTest, TrimRemovingTheKernelDirDoesNotFailANewVariant) {
    const auto variants = make_variants(1);  // kernel0/0, an hour old: the only entry a trim can evict
    race_root = root_;
    race_hits = 0;
    race_parent_was_removed = false;
    set_kernel_cache_race_hook([](std::string_view point, const fs::path& dir) {
        if (point != "variant dir" || race_hits++ != 0) {
            return;
        }
        trim_kernel_cache(race_root, disk_usage(race_root) / 2, std::chrono::seconds(0));
        race_parent_was_removed = !fs::exists(dir.parent_path());
    });
    const fs::path fresh = variants[0].parent_path() / "fresh";
    create_kernel_cache_dir(fresh.string() + "/");  // the JIT passes the variant dir with a trailing '/'
    set_kernel_cache_race_hook(nullptr);
    EXPECT_EQ(race_hits, 1);
    EXPECT_TRUE(race_parent_was_removed);  // the race really happened: the kernel's directory was gone
    EXPECT_FALSE(fs::exists(variants[0]));
    EXPECT_TRUE(fs::is_directory(fresh));
}

// A victim that cannot be deleted stays in .trash, still counts toward the size, and is reported.
TEST_F(KernelCacheEvictionTest, UndeletableVictimStaysCounted) {
    if (::geteuid() == 0) {
        GTEST_SKIP() << "root can delete anything";
    }
    const auto variants = make_variants(2);
    const fs::path locked = variants[0] / "locked";
    write_file(locked / "obj.o", 4 * kFileBytes);
    fs::permissions(locked, fs::perms::owner_read | fs::perms::owner_exec);
    const std::uint64_t before = disk_usage(root_);
    ::testing::internal::CaptureStdout();
    const auto result = trim_kernel_cache(root_, before * 9 / 10, std::chrono::seconds(0));
    const std::string log = ::testing::internal::GetCapturedStdout();
    // Only part of the first victim was freed, so the second one goes too.
    EXPECT_EQ(result.variants_evicted, 2u);
    EXPECT_NE(log.find("could not delete 1 evicted entries"), std::string::npos) << log;
    const std::uint64_t stuck = disk_usage(root_ / ".trash");
    EXPECT_GE(stuck, 4 * kFileBytes);
    EXPECT_GE(result.size_after, stuck);  // what stayed in .trash is still counted

    const auto again = trim_kernel_cache(root_, 1ull << 40, std::chrono::seconds(0));
    EXPECT_GE(again.size_before, stuck);
    for (const auto& entry : fs::recursive_directory_iterator(root_ / ".trash")) {
        fs::permissions(entry.path(), fs::perms::owner_all);
    }
}

TEST_F(KernelCacheEvictionTest, SkipsWhileAnotherProcessTrims) {
    const auto variants = make_variants(3);
    const int fd = ::open((root_ / ".trim.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::flock(fd, LOCK_EX), 0);
    const auto result = trim_kernel_cache(root_, 1, std::chrono::seconds(0));
    ::close(fd);
    EXPECT_FALSE(result.trimmed);
    for (const auto& v : variants) {
        EXPECT_TRUE(fs::exists(v));
    }
}

TEST_F(KernelCacheEvictionTest, ClearsTrashLeftByDeadTrimmer) {
    make_variants(2);
    write_file(root_ / ".trash" / "4242-0" / "trisc0" / "trisc0.elf", kFileBytes);
    trim_kernel_cache(root_, 1ull << 40, std::chrono::seconds(0));
    EXPECT_FALSE(fs::exists(root_ / ".trash"));
}

TEST_F(KernelCacheEvictionTest, ToleratesConcurrentRemoval) {
    const auto variants = make_variants(200);
    const std::uint64_t cap = disk_usage(root_) / 4;
    std::atomic<bool> done{false};
    // Someone else (a user's rm, another tool) removes variants while the trim runs.
    std::thread remover([&] {
        for (std::size_t i = 0; i < variants.size() && !done; i += 2) {
            std::error_code ec;
            fs::remove_all(variants[i], ec);
        }
    });
    KernelCacheTrimResult result;
    EXPECT_NO_THROW(result = trim_kernel_cache(root_, cap, std::chrono::seconds(0)));
    done = true;
    remover.join();
    EXPECT_LE(result.variants_evicted, variants.size());
    EXPECT_FALSE(fs::exists(root_ / ".trash"));
}

TEST_F(KernelCacheEvictionTest, ConcurrentTrimmersEvictOnce) {
    const auto variants = make_variants(40);
    const std::uint64_t cap = disk_usage(root_) / 2;
    std::vector<KernelCacheTrimResult> results(4);
    std::vector<std::thread> threads;
    threads.reserve(results.size());
    for (auto& r : results) {
        threads.emplace_back([&] { r = trim_kernel_cache(root_, cap, std::chrono::seconds(0)); });
    }
    for (auto& t : threads) {
        t.join();
    }
    std::size_t evicted = 0;
    for (const auto& r : results) {
        evicted += r.variants_evicted;
    }
    std::size_t remaining = 0;
    for (const auto& v : variants) {
        remaining += fs::exists(v) ? 1 : 0;
    }
    EXPECT_EQ(evicted + remaining, variants.size());
    EXPECT_LE(disk_usage(root_), static_cast<std::uint64_t>(static_cast<double>(cap) * kKernelCacheLowWaterFraction));
}

// TT_METAL_CACHE_MAX_SIZE through RunTimeOptions: the raw string, parsed later by the JIT build.
TEST(KernelCacheSizeTest, RunTimeOptionsReadsMaxSizeEnv) {
    {
        ScopedEnv env("TT_METAL_CACHE_MAX_SIZE", "20G");
        EXPECT_EQ(llrt::RunTimeOptions().get_cache_max_size(), std::optional<std::string>("20G"));
    }
    ScopedEnv env("TT_METAL_CACHE_MAX_SIZE", nullptr);
    EXPECT_FALSE(llrt::RunTimeOptions().get_cache_max_size().has_value());
}

// Runs resolve_kernel_cache_max_size() and returns what it logged.
std::string resolve_and_capture(const fs::path& root, const std::optional<std::string>& env, std::uint64_t& cap) {
    ::testing::internal::CaptureStdout();
    cap = resolve_kernel_cache_max_size(root, env);
    return ::testing::internal::GetCapturedStdout();
}

std::vector<std::string> leftover_temp_files(const fs::path& root) {
    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(root)) {
        const auto name = entry.path().filename().string();
        if (name != "tt-metal-cache.conf") {
            names.push_back(name);
        }
    }
    return names;
}

// Rule 1: env var set, no file -> the file is written from the env var and used.
TEST_F(KernelCacheEvictionTest, EnvSeedsMissingConfig) {
    const fs::path root = root_ / "not-yet-created";
    std::uint64_t cap = 0;
    const std::string log = resolve_and_capture(root, "300M", cap);
    EXPECT_EQ(cap, 300'000'000u);
    EXPECT_EQ(log.find("differs"), std::string::npos) << log;
    EXPECT_NE(log.find("wrote max_size = 300M from TT_METAL_CACHE_MAX_SIZE"), std::string::npos) << log;
    ASSERT_TRUE(fs::exists(root / "tt-metal-cache.conf"));
    EXPECT_EQ(read_kernel_cache_max_size(root), 300'000'000u);
    EXPECT_TRUE(leftover_temp_files(root).empty());
    // A later process without the env var uses the seeded file.
    EXPECT_EQ(resolve_kernel_cache_max_size(root, std::nullopt), 300'000'000u);
}

// Rule 2: env var and file differ -> warning naming both values and the file; the file wins.
TEST_F(KernelCacheEvictionTest, ConfigWinsOverDifferentEnv) {
    write_config("max_size = 1G\n");
    std::uint64_t cap = 0;
    const std::string log = resolve_and_capture(root_, "2G", cap);
    EXPECT_EQ(cap, 1'000'000'000u);
    EXPECT_NE(log.find("TT_METAL_CACHE_MAX_SIZE=2G"), std::string::npos) << log;
    EXPECT_NE(log.find("max_size = 1G"), std::string::npos) << log;
    EXPECT_NE(log.find((root_ / "tt-metal-cache.conf").string()), std::string::npos) << log;
    EXPECT_EQ(read_kernel_cache_max_size(root_), 1'000'000'000u);  // file untouched

    write_config("# no cap set here\n");
    EXPECT_EQ(resolve_kernel_cache_max_size(root_, "2G"), 0u);
}

// Rule 2: equal values (compared in bytes, not as text) -> no warning.
TEST_F(KernelCacheEvictionTest, EqualEnvAndConfigDoNotWarn) {
    write_config("max_size = 1G\n");
    std::uint64_t cap = 0;
    const std::string log = resolve_and_capture(root_, "1000M", cap);
    EXPECT_EQ(cap, 1'000'000'000u);
    EXPECT_EQ(log.find("TT_METAL_CACHE_MAX_SIZE"), std::string::npos) << log;
}

// Rules 3 and 4: no env var -> the file if there is one, else no cap and no file is written.
TEST_F(KernelCacheEvictionTest, WithoutEnvTheFileDecides) {
    EXPECT_EQ(resolve_kernel_cache_max_size(root_, std::nullopt), 0u);
    EXPECT_FALSE(fs::exists(root_ / "tt-metal-cache.conf"));
    write_config("max_size = 5Gi\n");
    EXPECT_EQ(resolve_kernel_cache_max_size(root_, std::nullopt), 5ull << 30);
}

// Invalid env value: warned about and ignored; it is never written to the file.
TEST_F(KernelCacheEvictionTest, InvalidEnvIsIgnored) {
    std::uint64_t cap = 1;
    const std::string log = resolve_and_capture(root_, "lots", cap);
    EXPECT_EQ(cap, 0u);
    EXPECT_NE(log.find("TT_METAL_CACHE_MAX_SIZE: 'lots' is not a size"), std::string::npos) << log;
    EXPECT_FALSE(fs::exists(root_ / "tt-metal-cache.conf"));
    EXPECT_EQ(resolve_kernel_cache_max_size(root_, "50M"), 0u);  // below 100M: a typo, never written
    EXPECT_FALSE(fs::exists(root_ / "tt-metal-cache.conf"));
    write_config("max_size = 1G\n");
    EXPECT_EQ(resolve_kernel_cache_max_size(root_, "20 x"), 1'000'000'000u);
}

// A file without a valid max_size falls back to the env var, with a warning, and is left alone.
TEST_F(KernelCacheEvictionTest, InvalidConfigFallsBackToEnv) {
    write_config("max_size = lots\n");
    std::uint64_t cap = 0;
    const std::string log = resolve_and_capture(root_, "300M", cap);
    EXPECT_EQ(cap, 300'000'000u);
    EXPECT_NE(log.find("has no valid max_size; using TT_METAL_CACHE_MAX_SIZE=300M"), std::string::npos) << log;
    EXPECT_EQ(resolve_kernel_cache_max_size(root_, std::nullopt), 0u);
}

// An unreadable file is reported, not mistaken for a missing one, and the env var is used.
TEST_F(KernelCacheEvictionTest, UnreadableConfigWarnsAndUsesEnv) {
    if (::geteuid() == 0) {
        GTEST_SKIP() << "root can read anything";
    }
    write_config("max_size = 1G\n");
    fs::permissions(root_ / "tt-metal-cache.conf", fs::perms::none);
    std::uint64_t cap = 0;
    const std::string log = resolve_and_capture(root_, "300M", cap);
    fs::permissions(root_ / "tt-metal-cache.conf", fs::perms::owner_read | fs::perms::owner_write);
    EXPECT_EQ(cap, 300'000'000u);
    EXPECT_NE(log.find("cannot read"), std::string::npos) << log;
    EXPECT_EQ(read_kernel_cache_max_size(root_), 1'000'000'000u);  // not replaced
}

// TT_METAL_CACHE_MAX_SIZE=0 seeds "max_size = 0" (exempt from the 100M floor); a file 0 wins over a
// non-zero env value, with a warning.
TEST_F(KernelCacheEvictionTest, ZeroCapFromEnvAndFile) {
    EXPECT_EQ(resolve_kernel_cache_max_size(root_, "0"), 0u);
    EXPECT_TRUE(fs::exists(root_ / "tt-metal-cache.conf"));
    std::uint64_t cap = 1;
    const std::string log = resolve_and_capture(root_, "20G", cap);
    EXPECT_EQ(cap, 0u);
    EXPECT_NE(log.find("differs from max_size = 0"), std::string::npos) << log;
}

// Many seeders at once with different values: exactly one value lands in the file, intact, and every
// seeder ends up using that value.
TEST_F(KernelCacheEvictionTest, ConcurrentSeedersAgreeOnOneValue) {
    constexpr int kThreads = 8;
    std::vector<std::uint64_t> caps(kThreads);
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back([&, i] {
            ready++;
            while (ready < kThreads) {
            }
            caps[i] = resolve_kernel_cache_max_size(root_, std::to_string(i + 1) + "G");
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    const std::uint64_t file_cap = read_kernel_cache_max_size(root_);
    EXPECT_GE(file_cap, 1'000'000'000u);
    EXPECT_LE(file_cap, kThreads * 1'000'000'000ull);
    for (auto cap : caps) {
        EXPECT_EQ(cap, file_cap);
    }
    EXPECT_TRUE(leftover_temp_files(root_).empty());
}

// The same race between two real processes.
TEST_F(KernelCacheEvictionTest, TwoProcessesSeedAtOnce) {
    int go[2];
    ASSERT_EQ(::pipe(go), 0);
    std::vector<pid_t> children;
    for (const char* value : {"3G", "4G"}) {
        const pid_t pid = ::fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            ::close(go[1]);
            char c;
            (void)::read(go[0], &c, 1);  // returns 0 once the parent closes its end
            const std::uint64_t cap = resolve_kernel_cache_max_size(root_, std::string(value));
            ::_exit(cap == read_kernel_cache_max_size(root_) ? 0 : 1);
        }
        children.push_back(pid);
    }
    ::close(go[0]);
    ::close(go[1]);
    for (pid_t pid : children) {
        int status = 0;
        ASSERT_EQ(::waitpid(pid, &status, 0), pid);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    const std::uint64_t file_cap = read_kernel_cache_max_size(root_);
    EXPECT_TRUE(file_cap == 3'000'000'000u || file_cap == 4'000'000'000u) << file_cap;
    EXPECT_TRUE(leftover_temp_files(root_).empty());
}

}  // namespace

// The JIT build on a mock Blackhole: a compile stamps the variant it uses, and a variant removed from the
// cache after this process built it is built again at load instead of failing.
class KernelCacheMockBlackholeFixture : public MockBlackholeMeshDispatchFixture {
protected:
    struct Compiled {
        std::shared_ptr<Kernel> kernel;
        std::string cache_root;   // the cache's root
        std::string kernel_root;  // the cache's <build_key>/kernels/
        fs::path dir;             // the kernel's variant directory
        std::uint64_t build_key = 0;
    };

    // Compiles a one-kernel program.
    Compiled compile(const std::string& source) {
        auto* device = devices_.at(0).get();
        KernelDescriptor kernel_descriptor = {
            .kernel_source = source,
            .source_type = KernelDescriptor::SourceType::SOURCE_CODE,
            .core_ranges = CoreRange(CoreCoord{0, 0}),
            .config = DataMovementConfigDescriptor{},
        };
        Program program(*device, ProgramDescriptor{.kernels = {kernel_descriptor}});
        auto kernel = program.impl().get_kernel(0);
        program.impl().compile(device);
        const auto& build_env = MetalContext::instance(kernel->get_context_id())
                                    .build_env_manager()
                                    .get_device_build_env(device->build_id());
        const std::string root = build_env.build_env.get_out_kernel_root_path();
        return {
            kernel,
            build_env.build_env.get_out_root_path(),
            root,
            root + kernel->get_full_kernel_name(),
            build_env.build_key()};
    }
    static std::string unique_source() {
        return fmt::format(
            "#include \"api/dataflow/dataflow_api.h\"\n// {}-{}\nvoid kernel_main() {{}}\n",
            ::getpid(),
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
};

TEST_F(KernelCacheMockBlackholeFixture, CompileStampsTheVariant) {
    const std::string source = unique_source();
    set_kernel_cache_cap({}, 0);
    const Compiled first = compile(source);
    ASSERT_TRUE(fs::is_directory(first.dir)) << first.dir;
    const std::time_t old = std::time(nullptr) - 3600;
    set_mtime(first.dir, old);
    set_kernel_cache_cap(first.cache_root, 1ull << 50);
    compile(source);  // a cache hit: nothing is written, only the stamp moves the mtime
    set_kernel_cache_cap({}, 0);
    EXPECT_GT(mtime_of(first.dir), old);
}

TEST_F(KernelCacheMockBlackholeFixture, CompileGrowsTheCounter) {
    const Compiled warm = compile(unique_source());  // only to learn the cache root
    trim_kernel_cache(warm.cache_root, 1ull << 50, std::chrono::seconds(0));
    const std::uint64_t before = read_kernel_cache_size_counter(warm.cache_root).value();
    set_kernel_cache_cap(warm.cache_root, 1ull << 50);
    const Compiled built = compile(unique_source());
    set_kernel_cache_cap({}, 0);
    const std::uint64_t after = read_kernel_cache_size_counter(warm.cache_root).value();
    const std::uint64_t size = disk_usage(built.dir);
    EXPECT_EQ(after - before, size) << "counter " << before << " -> " << after;
}

// Deletes the variant directory the first time a variant is about to be read: a trim in another process
// between this process's cache check (here: its own build) and its read of the ELFs.
std::atomic<int> load_deletes{0};
void delete_variant_before_first_load(std::string_view point, const fs::path& dir) {
    if (point == "load" && load_deletes.fetch_add(1) == 0) {
        fs::remove_all(dir);
    }
}

TEST_F(KernelCacheMockBlackholeFixture, VariantEvictedBeforeItsReadIsRebuilt) {
    load_deletes = 0;
    set_kernel_cache_race_hook(delete_variant_before_first_load);
    const Compiled built = compile(unique_source());
    set_kernel_cache_race_hook(nullptr);
    EXPECT_EQ(load_deletes.load(), 2);  // the read that found nothing, and the one after the rebuild
    EXPECT_TRUE(built.kernel->binaries_exist_on_disk(devices_.at(0).get(), built.kernel_root));
    EXPECT_EQ(built.kernel->binaries(built.build_key).size(), 1u);
}

// The same on the remote compile path, against an in-process compile server that hands back the ELF of a
// kernel built locally first (the two sources differ only in a comment).
std::vector<std::vector<std::uint8_t>> remote_elfs;
std::atomic<int> remote_compiles{0};

std::string free_local_endpoint() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        ADD_FAILURE() << "socket: " << std::strerror(errno);
        return {};
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof(addr);
    EXPECT_EQ(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    EXPECT_EQ(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    ::close(fd);
    return fmt::format("127.0.0.1:{}", ntohs(addr.sin_port));
}

TEST_F(KernelCacheMockBlackholeFixture, RemoteVariantEvictedBeforeItsReadIsRebuilt) {
    const Compiled local = compile(unique_source());
    remote_elfs.clear();
    for (const auto& entry : fs::recursive_directory_iterator(local.dir)) {
        const std::string name = entry.path().filename().string();
        if (name.ends_with(".elf") && !name.ends_with(".xip.elf")) {
            remote_elfs.push_back(tt::jit_build::utils::read_file_bytes(entry.path().string()));
        }
    }
    ASSERT_EQ(remote_elfs.size(), 1u);  // a data movement kernel: one ELF
    remote_compiles = 0;
    jit_server::JitCompileServerController server(
        [](const jit_server::CompileRequest& request) {
            remote_compiles++;
            jit_server::CompileResponse response;
            response.success = request.targets.size() == remote_elfs.size();
            for (std::size_t i = 0; i < request.targets.size() && response.success; i++) {
                response.elf_blobs.push_back({.name = request.targets[i].target_name, .data = remote_elfs[i]});
            }
            return response;
        },
        [](const jit_server::UploadFirmwareRequest&) { return jit_server::UploadFirmwareResponse{.success = true}; });
    const std::string endpoint = free_local_endpoint();
    server.start(endpoint);
    load_deletes = 0;
    {
        ScopedEnv enable("TT_METAL_JIT_SERVER_ENABLE", "1");
        ScopedEnv endpoints("TT_METAL_JIT_SERVER_ENDPOINTS", endpoint.c_str());
        set_kernel_cache_race_hook(delete_variant_before_first_load);
        const Compiled remote = compile(unique_source());
        set_kernel_cache_race_hook(nullptr);
        EXPECT_EQ(load_deletes.load(), 2);
        EXPECT_EQ(remote_compiles.load(), 2);  // the first compile, and the one after the ELF was gone
        EXPECT_TRUE(remote.kernel->binaries_exist_on_disk(devices_.at(0).get(), remote.kernel_root));
        EXPECT_EQ(remote.kernel->binaries(remote.build_key).size(), 1u);
    }
    server.stop();
}

TEST_F(KernelCacheMockBlackholeFixture, VariantEvictedAfterBuildIsRebuiltAtLoad) {
    const std::string source = unique_source();
    const Compiled first = compile(source);
    ASSERT_TRUE(fs::is_directory(first.dir)) << first.dir;
    fs::remove_all(first.dir);  // another process evicted it; this process's build cache still says it is built
    const Compiled again = compile(source);
    EXPECT_EQ(again.dir, first.dir);
    EXPECT_TRUE(again.kernel->binaries_exist_on_disk(devices_.at(0).get(), again.kernel_root));
}

}  // namespace tt::tt_metal
