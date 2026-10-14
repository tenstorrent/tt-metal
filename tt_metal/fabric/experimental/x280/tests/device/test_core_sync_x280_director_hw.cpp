// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// An X280 hart directs a core_sync_barrier over a grid of Tensix workers.
//
// The whole implementation lives under suites/core_sync/x280_director.hpp.
// This file wires the fixture, the parameterization axis and the RunSpec.

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "harness/slow_dispatch.hpp"
#include "suites/core_sync/x280_director.hpp"

#include <gtest/gtest.h>

namespace {

namespace suite = x280::suites::core_sync::x280_director;

class X280CoreSyncDirector : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_CORE_SYNC_DIRECTOR_FW_IMAGE",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "core_sync_x280_director",
        };
    }

    void SetUp() override {
        ASSERT_TRUE(x280::harness::slow_dispatch_linked());
        Runner::SetUp();
    }
};

}  // namespace

TEST_P(X280CoreSyncDirector, MulticastReleaseCompletes) { X280_HARNESS_DISPATCH(suite); }

TEST_P(X280CoreSyncDirector, WithheldParticipantUnwindsOnStop) {
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
        return;
    }
    suite::run_abandonment(fixture(), spec());
}

TEST_P(X280CoreSyncDirector, MixedTierArrivalIntoATensixDirector) {
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
        return;
    }
    suite::run_mixed_tier(fixture(), spec());
}

TEST_P(X280CoreSyncDirector, SingletonAndOrderedMultiGroupReleaseComplete) {
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
        return;
    }
    suite::run_release_groups(fixture());
}

TEST_P(X280CoreSyncDirector, ArrivalAndReleaseReachARenumberedColumn) {
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
        return;
    }
    suite::run_renumbered_column(fixture(), spec());
}

TEST_P(X280CoreSyncDirector, RequiredMulticastReleasesExactlyItsSelection) {
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
        return;
    }
    suite::run_guarded_multicast(fixture(), spec());
}

TEST_P(X280CoreSyncDirector, ActiveEriscWordWorkersOnBothNocs) {
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
        return;
    }
    suite::run_active_erisc_workers(fixture(), spec());
}

INSTANTIATE_TEST_SUITE_P(CoreSync, X280CoreSyncDirector, ::testing::Values(0));
