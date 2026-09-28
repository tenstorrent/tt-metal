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

// Points TT_METAL_CACHE at a fresh per-test directory and restores it afterwards.
class KvChunkTableCache : public ::testing::Test {
protected:
    void SetUp() override {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        root_ = fs::temp_directory_path() / "kv-chunk-table-cache-test" / info->name();
        fs::remove_all(root_);
        fs::create_directories(root_ / "run");
        if (const char* old = std::getenv("TT_METAL_CACHE")) {
            saved_env_ = old;
        }
        setenv("TT_METAL_CACHE", (root_ / "cache").c_str(), 1);
    }

    void TearDown() override {
        if (saved_env_) {
            setenv("TT_METAL_CACHE", saved_env_->c_str(), 1);
        } else {
            unsetenv("TT_METAL_CACHE");
        }
        fs::remove_all(root_);
    }

    std::function<void(const std::string&)> writer(const std::string& contents) {
        return [this, contents](const std::string& path) {
            ++builds_;
            std::ofstream(path, std::ios::binary) << contents;
        };
    }

    fs::path out(const std::string& name) const { return root_ / "run" / name; }

    fs::path root_;
    std::optional<std::string> saved_env_;
    int builds_ = 0;
};

TEST_F(KvChunkTableCache, CPU_PathLivesUnderTheTtMetalCache) {
    const fs::path path = kv_chunk_table_cache_path("abc1234", "{\"mesh\":[4,2]}");
    EXPECT_EQ(path.parent_path(), root_ / "cache" / "tt-metal-cache" / "kv-chunk-tables");
    EXPECT_EQ(path.extension(), ".pb");
    EXPECT_EQ(path, fs::path(kv_chunk_table_cache_path("abc1234", "{\"mesh\":[4,2]}")));
}

TEST_F(KvChunkTableCache, CPU_SeedAndKeyEachChangeThePath) {
    const std::string base = kv_chunk_table_cache_path("abc1234", "key");
    EXPECT_NE(base, kv_chunk_table_cache_path("abc1235", "key"));
    EXPECT_NE(base, kv_chunk_table_cache_path("abc1234", "key2"));
    EXPECT_NE(kv_chunk_table_cache_path("ab", "ckey"), kv_chunk_table_cache_path("abc", "key"));
}

TEST_F(KvChunkTableCache, CPU_MissBuildsWritesOutputAndCachesIt) {
    const bool hit = get_or_build_kv_chunk_table("abc1234", "key", writer("table-0"), out("table.pb"));

    EXPECT_FALSE(hit);
    EXPECT_EQ(builds_, 1);
    EXPECT_EQ(read_file(out("table.pb")), "table-0");
    EXPECT_EQ(read_file(kv_chunk_table_cache_path("abc1234", "key")), "table-0");
}

TEST_F(KvChunkTableCache, CPU_RelaunchHitCopiesWithoutBuilding) {
    get_or_build_kv_chunk_table("abc1234", "key", writer("table-0"), out("launch0.pb"));

    const bool hit = get_or_build_kv_chunk_table("abc1234", "key", writer("rebuilt"), out("launch1.pb"));

    EXPECT_TRUE(hit);
    EXPECT_EQ(builds_, 1);
    EXPECT_EQ(read_file(out("launch1.pb")), "table-0");
}

TEST_F(KvChunkTableCache, CPU_HitReplacesAStaleOutputFile) {
    get_or_build_kv_chunk_table("abc1234", "key", writer("table-0"), out("table.pb"));
    std::ofstream(out("table.pb"), std::ios::trunc) << "stale";

    EXPECT_TRUE(get_or_build_kv_chunk_table("abc1234", "key", writer("rebuilt"), out("table.pb")));
    EXPECT_EQ(read_file(out("table.pb")), "table-0");
}

TEST_F(KvChunkTableCache, CPU_ChangedKeyRebuildsAndKeepsBothFiles) {
    get_or_build_kv_chunk_table("abc1234", "bases=A", writer("table-A"), out("a.pb"));

    EXPECT_FALSE(get_or_build_kv_chunk_table("abc1234", "bases=B", writer("table-B"), out("b.pb")));
    EXPECT_EQ(builds_, 2);
    EXPECT_EQ(read_file(out("b.pb")), "table-B");

    EXPECT_TRUE(get_or_build_kv_chunk_table("abc1234", "bases=A", writer("rebuilt"), out("a2.pb")));
    EXPECT_EQ(read_file(out("a2.pb")), "table-A");
}

TEST_F(KvChunkTableCache, CPU_EmptySeedBypassesTheCache) {
    EXPECT_FALSE(get_or_build_kv_chunk_table("", "key", writer("table-0"), out("a.pb")));
    EXPECT_FALSE(get_or_build_kv_chunk_table("", "key", writer("table-1"), out("b.pb")));

    EXPECT_EQ(builds_, 2);
    EXPECT_EQ(read_file(out("b.pb")), "table-1");
    EXPECT_FALSE(fs::exists(root_ / "cache"));
}

TEST_F(KvChunkTableCache, CPU_UnwritableCacheStillProducesTheOutput) {
    std::ofstream(root_ / "blocked") << "x";
    setenv("TT_METAL_CACHE", (root_ / "blocked").c_str(), 1);

    EXPECT_FALSE(get_or_build_kv_chunk_table("abc1234", "key", writer("table-0"), out("table.pb")));
    EXPECT_EQ(read_file(out("table.pb")), "table-0");
}

TEST_F(KvChunkTableCache, CPU_FailedBuildLeavesNoFiles) {
    auto failing = [](const std::string& path) {
        std::ofstream(path) << "partial";
        throw std::runtime_error("build failed");
    };

    EXPECT_THROW(get_or_build_kv_chunk_table("abc1234", "key", failing, out("table.pb")), std::runtime_error);
    EXPECT_FALSE(fs::exists(out("table.pb")));
    EXPECT_FALSE(fs::exists(kv_chunk_table_cache_path("abc1234", "key")));
    EXPECT_TRUE(fs::is_empty(root_ / "run"));
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation
