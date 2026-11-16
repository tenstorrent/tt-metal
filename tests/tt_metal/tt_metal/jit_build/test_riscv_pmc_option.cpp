// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "llrt/rtoptions.hpp"

namespace tt::llrt {
namespace {

constexpr const char* kEnv = "TT_METAL_PROFILE_RISCV_PMC";
constexpr const char* kEnvDm = "TT_METAL_PROFILE_RISCV_PMC_DM";

class RiscvPmcOptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        unsetenv(kEnv);
        unsetenv(kEnvDm);
    }
    void TearDown() override {
        unsetenv(kEnv);
        unsetenv(kEnvDm);
    }

    static RunTimeOptions with(const char* value) {
        setenv(kEnv, value, 1);
        return RunTimeOptions{};
    }
};

TEST_F(RiscvPmcOptionTest, DisabledByDefault) {
    RunTimeOptions opts;
    EXPECT_FALSE(opts.get_profiler_riscv_pmc_enabled());
    EXPECT_EQ(opts.get_compile_hash_string().find("_pmc"), std::string::npos);
}

TEST_F(RiscvPmcOptionTest, HexPairParsedAndForcesProfiler) {
    RunTimeOptions opts = with("0x10110111,0xA0000000");
    EXPECT_TRUE(opts.get_profiler_riscv_pmc_enabled());
    EXPECT_TRUE(opts.get_profiler_enabled());
    EXPECT_EQ(opts.get_profiler_riscv_pmc_event3(), 0x10110111u);
    EXPECT_EQ(opts.get_profiler_riscv_pmc_event4(), 0xA0000000u);
}

TEST_F(RiscvPmcOptionTest, SingleSelectLeavesCounterFourIdle) {
    RunTimeOptions opts = with("0xF0000000");
    EXPECT_EQ(opts.get_profiler_riscv_pmc_event3(), 0xF0000000u);
    EXPECT_EQ(opts.get_profiler_riscv_pmc_event4(), 0u);
}

TEST_F(RiscvPmcOptionTest, DecimalAccepted) {
    RunTimeOptions opts = with("53248,1");
    EXPECT_EQ(opts.get_profiler_riscv_pmc_event3(), 0xD000u);
    EXPECT_EQ(opts.get_profiler_riscv_pmc_event4(), 1u);
}

TEST_F(RiscvPmcOptionTest, SelectsAreInCompileHash) {
    const std::string a = with("0x1,0x2").get_compile_hash_string();
    const std::string b = with("0x1,0x3").get_compile_hash_string();
    EXPECT_NE(a, b);
    EXPECT_NE(a.find("_pmc1_2"), std::string::npos);
}

TEST_F(RiscvPmcOptionTest, InvalidValuesRejected) {
    EXPECT_ANY_THROW(with("banana"));
    EXPECT_ANY_THROW(with("0x1,zz"));
    EXPECT_ANY_THROW(with("0x1ffffffff"));
    EXPECT_ANY_THROW(with("0,0"));
    EXPECT_ANY_THROW(with("0x1,0x2,0x3"));
}

TEST_F(RiscvPmcOptionTest, DataMovementSelectsParsedAndHashed) {
    setenv(kEnvDm, "0x201,0x801", 1);
    RunTimeOptions opts = with("0x10110111");
    EXPECT_EQ(opts.get_profiler_riscv_pmc_dm_event3(), 0x201u);
    EXPECT_EQ(opts.get_profiler_riscv_pmc_dm_event4(), 0x801u);
    EXPECT_NE(opts.get_compile_hash_string().find("_dm201_801"), std::string::npos);
}

TEST_F(RiscvPmcOptionTest, DataMovementSelectsAlone) {
    setenv(kEnvDm, "0x201", 1);
    RunTimeOptions opts;
    EXPECT_FALSE(opts.get_profiler_riscv_pmc_enabled());
    EXPECT_EQ(opts.get_profiler_riscv_pmc_dm_event3(), 0x201u);
    EXPECT_EQ(opts.get_compile_hash_string().find("_dm"), std::string::npos);
}

TEST_F(RiscvPmcOptionTest, DataMovementInvalidRejected) {
    setenv(kEnvDm, "nope", 1);
    EXPECT_ANY_THROW(with("0x1"));
}

}  // namespace
}  // namespace tt::llrt
