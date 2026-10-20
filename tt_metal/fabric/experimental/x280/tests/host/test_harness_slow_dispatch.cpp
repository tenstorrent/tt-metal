// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host tests for the guard of executables that link x280_slow_dispatch.

#include "harness/slow_dispatch.hpp"
#include "tests/host/scoped_env.hpp"

#include <gtest/gtest.h>

#include <string>

// x280_bench_host_tests does not link x280_slow_dispatch, whose initialiser would set
// the variable for the whole process; this stands in for its definition.
namespace x280::harness {
extern const bool kSlowDispatchLinked = true;
}

TEST(HarnessSlowDispatch, GuardPassesOnlyOnOneAndNamesTheLinkOtherwise) {
    using x280::testing::ScopedEnv;
    {
        ScopedEnv env("TT_METAL_SLOW_DISPATCH_MODE", "1");
        EXPECT_TRUE(x280::harness::slow_dispatch_linked());
    }
    for (const char* other : {static_cast<const char*>(nullptr), "0", "true"}) {
        SCOPED_TRACE(other == nullptr ? "unset" : other);
        ScopedEnv env("TT_METAL_SLOW_DISPATCH_MODE", other);
        const auto result = x280::harness::slow_dispatch_linked();
        EXPECT_FALSE(result);
        EXPECT_NE(std::string(result.message()).find("x280_slow_dispatch"), std::string::npos);
    }
}
