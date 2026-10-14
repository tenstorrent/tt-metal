// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pipeline_stage ID 7: router-side forwarding.

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "suites/pipeline_stage/route_forward.hpp"

#include "pipeline_stage_common.h"

#include <gtest/gtest.h>

namespace {

namespace suite = x280::suites::pipeline_stage::route_forward;

class PipelineStageRouteForward : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_PIPELINE_STAGE_ROUTE_FORWARD_FW_IMAGE",
                    .image_file = "firmware-pipeline-stage-route-forward.bin",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                    .prepared_suite_requested = true,
                },
            .suite_label = "pipeline_stage_route_forward",
        };
    }
};

}  // namespace

TEST_P(PipelineStageRouteForward, BatchedForwardDecisionCostPerItem) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    Sweep,
    PipelineStageRouteForward,
    ::testing::Values(suite::SweepPoint{"decide_plain"}),
    [](const ::testing::TestParamInfo<suite::SweepPoint>& info) { return std::string(info.param.label); });
