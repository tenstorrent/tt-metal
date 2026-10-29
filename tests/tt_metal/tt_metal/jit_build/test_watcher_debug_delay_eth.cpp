// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "llrt/rtoptions.hpp"

namespace tt::tt_metal {
namespace {

using llrt::RunTimeDebugFeatureWriteDebugDelay;

// Ethernet kernels are only built with the delay instrumentation when an ethernet core is delayed, so moving the
// delay between worker and ethernet cores at the same delay value must change the compile hash.
TEST(WatcherDebugDelayEth, CompileHashChangesWhenEthCoresAreDelayed) {
    llrt::RunTimeOptions rtoptions;
    rtoptions.set_watcher_debug_delay(10);
    rtoptions.set_feature_enabled(RunTimeDebugFeatureWriteDebugDelay, true);

    rtoptions.set_feature_cores(RunTimeDebugFeatureWriteDebugDelay, {{CoreType::WORKER, {CoreCoord{0, 0}}}});
    EXPECT_FALSE(rtoptions.get_watcher_debug_delay_targets_eth());
    const auto worker_only_hash = rtoptions.get_compile_hash_string();

    rtoptions.set_feature_cores(
        RunTimeDebugFeatureWriteDebugDelay,
        {{CoreType::WORKER, {CoreCoord{0, 0}}}, {CoreType::ETH, {CoreCoord{0, 0}}}});
    EXPECT_TRUE(rtoptions.get_watcher_debug_delay_targets_eth());
    EXPECT_NE(rtoptions.get_compile_hash_string(), worker_only_hash);

    rtoptions.set_feature_cores(RunTimeDebugFeatureWriteDebugDelay, {{CoreType::WORKER, {CoreCoord{0, 0}}}});
    EXPECT_EQ(rtoptions.get_compile_hash_string(), worker_only_hash);
}

}  // namespace
}  // namespace tt::tt_metal
