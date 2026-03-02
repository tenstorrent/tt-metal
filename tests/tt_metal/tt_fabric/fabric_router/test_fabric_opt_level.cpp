// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <gtest/gtest.h>
#include <tt-metalium/kernel_types.hpp>
#include "rtoptions.hpp"

namespace tt::llrt {

class FabricOptLevelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Save original env var state
        const char* existing = std::getenv("TT_METAL_FABRIC_OPT_LEVEL");
        had_env_var_ = existing != nullptr;
        if (had_env_var_) {
            original_value_ = existing;
        }
    }

    void TearDown() override {
        // Restore original env var state
        if (had_env_var_) {
            setenv("TT_METAL_FABRIC_OPT_LEVEL", original_value_.c_str(), 1);
        } else {
            unsetenv("TT_METAL_FABRIC_OPT_LEVEL");
        }
    }

    RunTimeOptions create_rtoptions() { return RunTimeOptions(); }

private:
    bool had_env_var_ = false;
    std::string original_value_;
};

TEST_F(FabricOptLevelTest, DefaultIsNullopt) {
    unsetenv("TT_METAL_FABRIC_OPT_LEVEL");
    auto opts = create_rtoptions();
    EXPECT_FALSE(opts.get_fabric_kernel_opt_level().has_value());
}

TEST_F(FabricOptLevelTest, ParseOs) {
    setenv("TT_METAL_FABRIC_OPT_LEVEL", "Os", 1);
    auto opts = create_rtoptions();
    ASSERT_TRUE(opts.get_fabric_kernel_opt_level().has_value());
    EXPECT_EQ(opts.get_fabric_kernel_opt_level().value(), tt::tt_metal::KernelBuildOptLevel::Os);
}

TEST_F(FabricOptLevelTest, ParseO3) {
    setenv("TT_METAL_FABRIC_OPT_LEVEL", "O3", 1);
    auto opts = create_rtoptions();
    ASSERT_TRUE(opts.get_fabric_kernel_opt_level().has_value());
    EXPECT_EQ(opts.get_fabric_kernel_opt_level().value(), tt::tt_metal::KernelBuildOptLevel::O3);
}

TEST_F(FabricOptLevelTest, ParseO0) {
    setenv("TT_METAL_FABRIC_OPT_LEVEL", "O0", 1);
    auto opts = create_rtoptions();
    ASSERT_TRUE(opts.get_fabric_kernel_opt_level().has_value());
    EXPECT_EQ(opts.get_fabric_kernel_opt_level().value(), tt::tt_metal::KernelBuildOptLevel::O0);
}

TEST_F(FabricOptLevelTest, ParseO1) {
    setenv("TT_METAL_FABRIC_OPT_LEVEL", "O1", 1);
    auto opts = create_rtoptions();
    ASSERT_TRUE(opts.get_fabric_kernel_opt_level().has_value());
    EXPECT_EQ(opts.get_fabric_kernel_opt_level().value(), tt::tt_metal::KernelBuildOptLevel::O1);
}

TEST_F(FabricOptLevelTest, ParseO2) {
    setenv("TT_METAL_FABRIC_OPT_LEVEL", "O2", 1);
    auto opts = create_rtoptions();
    ASSERT_TRUE(opts.get_fabric_kernel_opt_level().has_value());
    EXPECT_EQ(opts.get_fabric_kernel_opt_level().value(), tt::tt_metal::KernelBuildOptLevel::O2);
}

TEST_F(FabricOptLevelTest, ParseOz) {
    setenv("TT_METAL_FABRIC_OPT_LEVEL", "Oz", 1);
    auto opts = create_rtoptions();
    ASSERT_TRUE(opts.get_fabric_kernel_opt_level().has_value());
    EXPECT_EQ(opts.get_fabric_kernel_opt_level().value(), tt::tt_metal::KernelBuildOptLevel::Oz);
}

TEST_F(FabricOptLevelTest, ParseOfast) {
    setenv("TT_METAL_FABRIC_OPT_LEVEL", "Ofast", 1);
    auto opts = create_rtoptions();
    ASSERT_TRUE(opts.get_fabric_kernel_opt_level().has_value());
    EXPECT_EQ(opts.get_fabric_kernel_opt_level().value(), tt::tt_metal::KernelBuildOptLevel::Ofast);
}

TEST_F(FabricOptLevelTest, InvalidValueThrows) {
    setenv("TT_METAL_FABRIC_OPT_LEVEL", "invalid", 1);
    EXPECT_ANY_THROW(create_rtoptions());
}

}  // namespace tt::llrt
