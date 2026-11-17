// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "jit_build/build.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"
#include "perf_counters/router_counters_types.h"

namespace tt::llrt {
namespace {

constexpr const char* kRouterEnv = "TT_METAL_PROFILE_ROUTER_COUNTERS";
constexpr uint32_t kOn = llk::perf::router::OPTION_ALL;

class RouterCountersOptionTest : public ::testing::Test {
protected:
    void SetUp() override { unsetenv(kRouterEnv); }
    void TearDown() override { unsetenv(kRouterEnv); }

    static RunTimeOptions with(const char* value) {
        setenv(kRouterEnv, value, 1);
        return RunTimeOptions{};
    }
};

TEST_F(RouterCountersOptionTest, DisabledByDefault) {
    RunTimeOptions opts;
    EXPECT_FALSE(opts.get_profiler_router_counters_enabled());
    EXPECT_EQ(opts.get_profiler_router_counters_mask(), 0u);
    EXPECT_EQ(opts.get_compile_hash_string().find("_rtr"), std::string::npos);
}

TEST_F(RouterCountersOptionTest, ZeroLeavesItOff) {
    RunTimeOptions opts = with("0");
    EXPECT_FALSE(opts.get_profiler_router_counters_enabled());
    EXPECT_EQ(opts.get_compile_hash_string().find("_rtr"), std::string::npos);
}

TEST_F(RouterCountersOptionTest, OnSelectsEveryChannelAndPortAndForcesProfiler) {
    RunTimeOptions opts = with("on");
    EXPECT_TRUE(opts.get_profiler_router_counters_enabled());
    EXPECT_TRUE(opts.get_profiler_enabled());
    EXPECT_EQ(opts.get_profiler_router_counters_mask(), kOn);
    EXPECT_EQ(opts.get_profiler_router_counters_mask() & llk::perf::router::PER_VC, 0u);
}

TEST_F(RouterCountersOptionTest, MaskParsedInHexAndDecimal) {
    EXPECT_EQ(with("0x30002").get_profiler_router_counters_mask(), 0x30002u);
    EXPECT_EQ(with("65538").get_profiler_router_counters_mask(), 0x10002u);
    EXPECT_EQ(with("0x1030012").get_profiler_router_counters_mask(), 0x1030012u);
}

TEST_F(RouterCountersOptionTest, MaskIsInCompileHash) {
    const std::string a = with("0x10002").get_compile_hash_string();
    const std::string b = with("0x10004").get_compile_hash_string();
    EXPECT_NE(a, b);
    EXPECT_NE(a.find("_rtr10002"), std::string::npos);
}

TEST_F(RouterCountersOptionTest, OptionBitsMatchTheTableHeader) {
    EXPECT_EQ(kOn, 0x00FFFFFFu);
    EXPECT_EQ(llk::perf::router::PER_VC, 0x01000000u);
    EXPECT_EQ(llk::perf::router::BANK_SHIFT, 16u);
}

TEST_F(RouterCountersOptionTest, InvalidValuesRejected) {
    EXPECT_ANY_THROW(with("banana"));
    EXPECT_ANY_THROW(with("0x2000000"));
    EXPECT_ANY_THROW(with("3,1"));
    EXPECT_ANY_THROW(with(""));
}

TEST_F(RouterCountersOptionTest, MaskWithoutChannelsOrPortsRejected) {
    EXPECT_ANY_THROW(with("0x30000"));
    EXPECT_ANY_THROW(with("0x2"));
    EXPECT_ANY_THROW(with("0x1000000"));
}

TEST_F(RouterCountersOptionTest, WormholeBuildIsRefusedAndBlackholeIsNot) {
    const RunTimeOptions opts = with("on");
    const tt::tt_metal::Hal wh_hal(tt::ARCH::WORMHOLE_B0, false, false, 0, false);
    const tt::tt_metal::Hal bh_hal(tt::ARCH::BLACKHOLE, false, false, 0, false);
    const tt::tt_metal::JitDeviceConfig wh_config{.hal = &wh_hal, .arch = tt::ARCH::WORMHOLE_B0, .max_dfbs = 32};
    const tt::tt_metal::JitDeviceConfig bh_config{.hal = &bh_hal, .arch = tt::ARCH::BLACKHOLE, .max_dfbs = 32};
    tt::tt_metal::JitBuildEnv wormhole;
    tt::tt_metal::JitBuildEnv blackhole;
    EXPECT_ANY_THROW(wormhole.init(1, wh_config, opts, {}));
    EXPECT_NO_THROW(blackhole.init(2, bh_config, opts, {}));
}

}  // namespace
}  // namespace tt::llrt
