// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "jit_build/build_cache_telemetry.hpp"
#include "jit_build/jit_build_cache.hpp"

namespace tt::tt_metal {

TEST(JitBuildCacheTests, BuildOnceRunsFnOnce) {
    auto& cache = JitBuildCache::inst();
    cache.clear();

    int runs = 0;
    EXPECT_TRUE(cache.build_once(0xabc, [&] { ++runs; }));
    EXPECT_EQ(runs, 1);

    EXPECT_FALSE(cache.build_once(0xabc, [&] { ++runs; }));
    EXPECT_EQ(runs, 1);
}

TEST(JitBuildCacheTests, DistinctHashesRunSeparately) {
    auto& cache = JitBuildCache::inst();
    cache.clear();

    int runs = 0;
    EXPECT_TRUE(cache.build_once(1, [&] { ++runs; }));
    EXPECT_TRUE(cache.build_once(2, [&] { ++runs; }));
    EXPECT_EQ(runs, 2);
}

class BuildCacheTelemetryTest : public ::testing::Test {
protected:
    void TearDown() override {
        auto& tel = BuildCacheTelemetry::inst();
        if (!tel.is_enabled()) {
            tel.enable();
        }
    }
};

TEST_F(BuildCacheTelemetryTest, DoubleEnablePreservesCounters) {
    auto& tel = BuildCacheTelemetry::inst();
    tel.disable();
    tel.enable();
    ASSERT_TRUE(tel.is_enabled());

    tel.record_jit_once_dedup();
    EXPECT_EQ(tel.get_jit_once_dedup_count(), 1u);

    tel.enable();
    tel.record_jit_once_dedup();
    EXPECT_EQ(tel.get_jit_once_dedup_count(), 2u);
}

}  // namespace tt::tt_metal
