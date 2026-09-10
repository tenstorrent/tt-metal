// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Force gtest's "threadsafe" death-test style for the emule api test binary.
//
// Under the emule cooperative fiber engine the test process runs a shared
// worker-thread pool (tens of threads — gtest reports ~35-99). gtest's default
// "fast" death-test style fork()s the process at the EXPECT_DEATH/ASSERT_DEATH
// site; fork() in a multi-threaded program only carries the calling thread, so
// the child inherits a fiber scheduler whose worker threads no longer exist. Any
// death test whose statement drives the scheduler (e.g. the CB Boundary /
// Reservation kernels call cb_reserve_back, which parks on the pool) then hangs
// in the child until the fiber hang-detector trips (~121 s) — a "failed to die".
// gtest itself warns about exactly this ("Death tests use fork() ... detected N
// threads ... especially if this is the last message before your test times out").
//
// "threadsafe" instead re-exec's the test binary for each death test, so the
// child starts single-threaded and builds its OWN fiber pool from scratch — the
// death statement then runs like a normal launch, the ASAN check fires, and the
// abort is detected. Slower per death test (a re-exec + device init), but correct.
//
// Registered as a global test environment (SetUp runs after InitGoogleTest, so it
// can't be clobbered by flag parsing) and compiled ONLY into the emule build (listed
// under the TT_METAL_USE_EMULE block in sources.cmake), so non-emule builds are
// unaffected. See docs/ASAN.md "Death tests under the fiber engine".

#include <gtest/gtest.h>

namespace {

class EmuleThreadsafeDeathStyle : public ::testing::Environment {
public:
    void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

const ::testing::Environment* const kEmuleThreadsafeDeathStyle =
    ::testing::AddGlobalTestEnvironment(new EmuleThreadsafeDeathStyle);

}  // namespace
