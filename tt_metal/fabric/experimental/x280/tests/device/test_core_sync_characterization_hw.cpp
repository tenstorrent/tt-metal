// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "harness/artifacts/artifacts.hpp"
#include "harness/runner.hpp"
#include "suites/core_sync/x280_director.hpp"

#include <gtest/gtest.h>

namespace {

namespace suite = x280::suites::core_sync::x280_director;

class X280CoreSyncCharacterization : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_CORE_SYNC_DIRECTOR_FW_IMAGE",
                    .image_file = "firmware-core-sync-director.bin",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "core_sync_characterization",
        };
    }
};

}  // namespace

TEST_P(X280CoreSyncCharacterization, ReleaseSkewDistribution) {
    if (::testing::Test::IsSkipped() || ::testing::Test::HasFatalFailure()) {
        return;
    }
    suite::run_characterization(fixture(), spec());
}

INSTANTIATE_TEST_SUITE_P(CoreSync, X280CoreSyncCharacterization, ::testing::Values(0));
