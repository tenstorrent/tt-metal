// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "llrt/rtoptions.hpp"

namespace tt::llrt {
namespace {

constexpr const char* kNocEnv = "TT_METAL_PROFILE_NOC_COUNTERS";

class NocCountersOptionTest : public ::testing::Test {
protected:
    void SetUp() override { unsetenv(kNocEnv); }
    void TearDown() override { unsetenv(kNocEnv); }

    static RunTimeOptions with(const char* value) {
        setenv(kNocEnv, value, 1);
        return RunTimeOptions{};
    }
};

TEST_F(NocCountersOptionTest, DisabledByDefault) {
    RunTimeOptions opts;
    EXPECT_FALSE(opts.get_profiler_noc_counters_enabled());
    EXPECT_EQ(opts.get_profiler_noc_counters_groups(), 0u);
    EXPECT_EQ(opts.get_compile_hash_string().find("_noc"), std::string::npos);
}

TEST_F(NocCountersOptionTest, ZeroLeavesItOff) {
    RunTimeOptions opts = with("0");
    EXPECT_FALSE(opts.get_profiler_noc_counters_enabled());
    EXPECT_EQ(opts.get_compile_hash_string().find("_noc"), std::string::npos);
}

TEST_F(NocCountersOptionTest, OnSelectsMasterAndSlaveAndForcesProfiler) {
    RunTimeOptions opts = with("on");
    EXPECT_TRUE(opts.get_profiler_noc_counters_enabled());
    EXPECT_TRUE(opts.get_profiler_enabled());
    EXPECT_EQ(opts.get_profiler_noc_counters_groups(), 3u);
}

TEST_F(NocCountersOptionTest, MaskParsedInHexAndDecimal) {
    EXPECT_EQ(with("0x5").get_profiler_noc_counters_groups(), 5u);
    EXPECT_EQ(with("7").get_profiler_noc_counters_groups(), 7u);
    EXPECT_EQ(with("1").get_profiler_noc_counters_groups(), 1u);
}

TEST_F(NocCountersOptionTest, MaskIsInCompileHash) {
    const std::string a = with("1").get_compile_hash_string();
    const std::string b = with("3").get_compile_hash_string();
    EXPECT_NE(a, b);
    EXPECT_NE(a.find("_noc1"), std::string::npos);
}

TEST_F(NocCountersOptionTest, InvalidValuesRejected) {
    EXPECT_ANY_THROW(with("banana"));
    EXPECT_ANY_THROW(with("8"));
    EXPECT_ANY_THROW(with("3,1"));
    EXPECT_ANY_THROW(with(""));
}

}  // namespace
}  // namespace tt::llrt
