// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Whether kernels and firmware are built with -g. TT_METAL_RISCV_DEBUG_INFO decides when it is set.
// Unset, it follows the Inspector, except on a simulator, where it is off unless DPRINT is on or the
// simulated chips are served over sockets for a debug tool to attach.

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

#include "llrt/rtoptions.hpp"

namespace {

// Set (or clear) one environment variable for the test's life and put the old value back.
class ScopedDebugInfoEnv {
public:
    ScopedDebugInfoEnv(const char* name, const std::optional<std::string>& value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            old_ = old;
        }
        apply(value);
    }
    ~ScopedDebugInfoEnv() { apply(old_); }

private:
    void apply(const std::optional<std::string>& value) {
        if (value.has_value()) {
            setenv(name_, value->c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }
    const char* name_;
    std::optional<std::string> old_;
};

using Env = std::optional<std::string>;

bool debug_info(
    const Env& simulator,
    const Env& debug_info_env,
    const Env& inspector = std::nullopt,
    const Env& serve_over_sockets = std::nullopt,
    bool dprint = false) {
    ScopedDebugInfoEnv simulator_env("TT_METAL_SIMULATOR", simulator);
    ScopedDebugInfoEnv debug_info_env_var("TT_METAL_RISCV_DEBUG_INFO", debug_info_env);
    ScopedDebugInfoEnv inspector_env("TT_METAL_INSPECTOR", inspector);
    ScopedDebugInfoEnv sockets_env("TT_METAL_SIMULATOR_SERVE_OVER_SOCKETS", serve_over_sockets);
    ScopedDebugInfoEnv mock_env("TT_METAL_MOCK_CLUSTER_DESC_PATH", std::nullopt);
    ScopedDebugInfoEnv emule_env("TT_METAL_EMULE_MODE", std::nullopt);
    ScopedDebugInfoEnv dprint_env("TT_METAL_DPRINT_CORES", std::nullopt);
    tt::llrt::RunTimeOptions opts;
    if (dprint) {
        opts.set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, true);
    }
    return opts.get_riscv_debug_info_enabled();
}

const std::string SIM = "/sim/libttsim.so";

}  // namespace

TEST(RunTimeOptionsRiscvDebugInfo, CPU_WithoutASimulatorItFollowsTheInspector) {
    EXPECT_TRUE(debug_info(std::nullopt, std::nullopt));
    EXPECT_TRUE(debug_info(std::nullopt, std::nullopt, "1"));
    EXPECT_FALSE(debug_info(std::nullopt, std::nullopt, "0"));
}

TEST(RunTimeOptionsRiscvDebugInfo, CPU_OnASimulatorItIsOffByDefault) {
    EXPECT_FALSE(debug_info(SIM, std::nullopt));
    EXPECT_FALSE(debug_info(SIM, std::nullopt, "1"));
}

TEST(RunTimeOptionsRiscvDebugInfo, CPU_OnASimulatorDprintOrADebugToolKeepsIt) {
    EXPECT_TRUE(debug_info(SIM, std::nullopt, std::nullopt, std::nullopt, /*dprint=*/true));
    EXPECT_TRUE(debug_info(SIM, std::nullopt, std::nullopt, "1"));
    // Both still follow the Inspector.
    EXPECT_FALSE(debug_info(SIM, std::nullopt, "0", "1", /*dprint=*/true));
}

TEST(RunTimeOptionsRiscvDebugInfo, CPU_TheVariableWinsEitherWay) {
    EXPECT_TRUE(debug_info(SIM, "1"));
    EXPECT_FALSE(debug_info(SIM, "0", std::nullopt, "1", /*dprint=*/true));
    EXPECT_TRUE(debug_info(std::nullopt, "1", "0"));
    EXPECT_FALSE(debug_info(std::nullopt, "0"));
}
