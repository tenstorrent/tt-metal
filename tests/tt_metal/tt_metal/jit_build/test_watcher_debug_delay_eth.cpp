// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>

#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {
namespace {

using llrt::RunTimeDebugFeatureWriteDebugDelay;

// Ethernet kernels are only built with the delay instrumentation when an ethernet core is delayed, so moving the
// delay between worker and ethernet cores at the same delay value must change the build key.
TEST(WatcherDebugDelayEth, BuildKeyChangesWhenEthCoresAreDelayed) {
    llrt::RunTimeOptions rtoptions;
    rtoptions.set_disable_precompiled_fw(true);
    rtoptions.set_watcher_debug_delay(10);
    rtoptions.set_feature_enabled(RunTimeDebugFeatureWriteDebugDelay, true);

    bool checked = false;
    enumerate_offline_compile_device_configs(rtoptions, [&](const JitDeviceConfig& config) {
        if (checked) {
            return;
        }
        checked = true;
        const auto build_key = [&] {
            BuildEnvManager build_env_manager(*config.hal);
            build_env_manager.add_build_env(0, config, rtoptions);
            return build_env_manager.get_device_build_env(0).build_key();
        };

        rtoptions.set_feature_cores(RunTimeDebugFeatureWriteDebugDelay, {{CoreType::WORKER, {CoreCoord{0, 0}}}});
        const uint64_t worker_only_key = build_key();

        rtoptions.set_feature_cores(
            RunTimeDebugFeatureWriteDebugDelay,
            {{CoreType::WORKER, {CoreCoord{0, 0}}}, {CoreType::ETH, {CoreCoord{0, 0}}}});
        EXPECT_NE(build_key(), worker_only_key);

        rtoptions.set_feature_cores(RunTimeDebugFeatureWriteDebugDelay, {{CoreType::WORKER, {CoreCoord{0, 0}}}});
        EXPECT_EQ(build_key(), worker_only_key);
    });
    EXPECT_TRUE(checked);
}

}  // namespace
}  // namespace tt::tt_metal
