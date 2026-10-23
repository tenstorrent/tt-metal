// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "impl/context/metal_context.hpp"
#include "internal/disaggregation/kv_chunk_table_cache.hpp"

namespace tt::tt_metal::internal::disaggregation {
namespace {

namespace fs = std::filesystem;

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream contents;
    contents << in.rdbuf();
    return contents.str();
}

class KvChunkTableCache : public ::testing::Test {
protected:
    void SetUp() override {
        if (const char* arch = std::getenv("ARCH_NAME"); arch == nullptr || std::string_view(arch) != "blackhole") {
            GTEST_SKIP() << "Blackhole-only";
        }
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        root_ = fs::temp_directory_path() / "kv-chunk-table-cache-test" / info->name();
        fs::remove_all(root_);
        fs::create_directories(root_ / "run");
        cache_ = (root_ / "cache").string();
    }

    void TearDown() override {
        if (!root_.empty()) {
            fs::remove_all(root_);
        }
    }

    std::function<void(const std::string&)> writer(const std::string& contents) {
        return [this, contents](const std::string& path) {
            ++builds_;
            std::ofstream(path, std::ios::binary) << contents;
        };
    }

    bool get_or_build(
        const std::string& seed, const std::string& key, const std::string& contents, const fs::path& out) {
        return get_or_build_kv_chunk_table(cache_, seed, key, writer(contents), out.string());
    }

    std::string cache_path(const std::string& seed, const std::string& key) const {
        return kv_chunk_table_cache_path(cache_, seed, key);
    }

    fs::path out(const std::string& name) const { return root_ / "run" / name; }

    fs::path root_;
    std::string cache_;
    int builds_ = 0;
};

TEST_F(KvChunkTableCache, CPU_DefaultDirFollowsTtMetalCache) {
    if (MetalContext::instance_exists()) {
        GTEST_SKIP() << "A live MetalContext fixes the cache root";
    }
    std::optional<std::string> saved;
    if (const char* old = std::getenv("TT_METAL_CACHE")) {
        saved = old;
    }
    setenv("TT_METAL_CACHE", (root_ / "env/").c_str(), 1);

    const fs::path dir = default_kv_chunk_table_cache_dir();

    if (saved) {
        setenv("TT_METAL_CACHE", saved->c_str(), 1);
    } else {
        unsetenv("TT_METAL_CACHE");
    }
    EXPECT_EQ(dir, root_ / "env" / "tt-metal-cache" / "kv-chunk-tables");
}

TEST_F(KvChunkTableCache, CPU_PathLivesUnderTheCacheDir) {
    const fs::path path = cache_path("abc1234", "{\"mesh\":[4,2]}");
    EXPECT_EQ(path.parent_path(), fs::path(cache_));
    EXPECT_EQ(path.extension(), ".pb");
    EXPECT_EQ(path, fs::path(cache_path("abc1234", "{\"mesh\":[4,2]}")));
}

TEST_F(KvChunkTableCache, CPU_SeedAndKeyEachChangeThePath) {
    const std::string base = cache_path("abc1234", "key");
    EXPECT_NE(base, cache_path("abc1235", "key"));
    EXPECT_NE(base, cache_path("abc1234", "key2"));
    EXPECT_NE(cache_path("ab", "ckey"), cache_path("abc", "key"));
}

TEST_F(KvChunkTableCache, CPU_MissBuildsWritesOutputAndCachesIt) {
    const bool hit = get_or_build("abc1234", "key", "table-0", out("table.pb"));

    EXPECT_FALSE(hit);
    EXPECT_EQ(builds_, 1);
    EXPECT_EQ(read_file(out("table.pb")), "table-0");
    EXPECT_EQ(read_file(cache_path("abc1234", "key")), "table-0");
}

TEST_F(KvChunkTableCache, CPU_RelaunchHitCopiesWithoutBuilding) {
    get_or_build("abc1234", "key", "table-0", out("launch0.pb"));

    const bool hit = get_or_build("abc1234", "key", "rebuilt", out("launch1.pb"));

    EXPECT_TRUE(hit);
    EXPECT_EQ(builds_, 1);
    EXPECT_EQ(read_file(out("launch1.pb")), "table-0");
}

TEST_F(KvChunkTableCache, CPU_HitReplacesAStaleOutputFile) {
    get_or_build("abc1234", "key", "table-0", out("table.pb"));
    std::ofstream(out("table.pb"), std::ios::trunc) << "stale";

    EXPECT_TRUE(get_or_build("abc1234", "key", "rebuilt", out("table.pb")));
    EXPECT_EQ(read_file(out("table.pb")), "table-0");
}

TEST_F(KvChunkTableCache, CPU_ChangedKeyRebuildsAndKeepsBothFiles) {
    get_or_build("abc1234", "bases=A", "table-A", out("a.pb"));

    EXPECT_FALSE(get_or_build("abc1234", "bases=B", "table-B", out("b.pb")));
    EXPECT_EQ(builds_, 2);
    EXPECT_EQ(read_file(out("b.pb")), "table-B");

    EXPECT_TRUE(get_or_build("abc1234", "bases=A", "rebuilt", out("a2.pb")));
    EXPECT_EQ(read_file(out("a2.pb")), "table-A");
}

TEST_F(KvChunkTableCache, CPU_EmptySeedBypassesTheCache) {
    EXPECT_FALSE(get_or_build("", "key", "table-0", out("a.pb")));
    EXPECT_FALSE(get_or_build("", "key", "table-1", out("b.pb")));

    EXPECT_EQ(builds_, 2);
    EXPECT_EQ(read_file(out("b.pb")), "table-1");
    EXPECT_FALSE(fs::exists(cache_));
}

TEST_F(KvChunkTableCache, CPU_UnwritableCacheStillProducesTheOutput) {
    std::ofstream(root_ / "blocked") << "x";
    cache_ = (root_ / "blocked" / "kv-chunk-tables").string();

    EXPECT_FALSE(get_or_build("abc1234", "key", "table-0", out("table.pb")));
    EXPECT_EQ(read_file(out("table.pb")), "table-0");
}

TEST_F(KvChunkTableCache, CPU_FailedBuildLeavesNoFiles) {
    auto failing = [](const std::string& path) {
        std::ofstream(path) << "partial";
        throw std::runtime_error("build failed");
    };

    EXPECT_THROW(
        get_or_build_kv_chunk_table(cache_, "abc1234", "key", failing, out("table.pb").string()), std::runtime_error);
    EXPECT_FALSE(fs::exists(out("table.pb")));
    EXPECT_FALSE(fs::exists(cache_path("abc1234", "key")));
    EXPECT_TRUE(fs::is_empty(root_ / "run"));
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation
