// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// transport_bandwidth suite -- X280 -> Tensix addressable member.
//
// The whole implementation lives under
// suites/transport_bandwidth/x280_to_tensix.hpp. This file wires the
// fixture, the parameterization axis (transfer size + boundary), and the
// RunSpec (image env var + kernel dir).

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "suites/transport_bandwidth/x280_to_tensix.hpp"

#include "transport_bandwidth_common.h"

#include <gtest/gtest.h>

namespace {

namespace suite = x280::suites::transport_bandwidth::x280_to_tensix;

class TransportBandwidthX280ToTensix : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_TRANSPORT_BW_X280_TO_TENSIX_FW_IMAGE",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "transport_bandwidth_x280_to_tensix",
        };
    }
};

}  // namespace

TEST_P(TransportBandwidthX280ToTensix, RepeatedIssueMeasuresSustainedAndInjection) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    Sweep,
    TransportBandwidthX280ToTensix,
    ::testing::Values(
        suite::SweepPoint{64, TRANSPORT_BW_BOUNDARY_DRAINED, "size64_drained"},
        suite::SweepPoint{64, TRANSPORT_BW_BOUNDARY_INJECTION, "size64_injection"},
        suite::SweepPoint{256, TRANSPORT_BW_BOUNDARY_DRAINED, "size256_drained"},
        suite::SweepPoint{256, TRANSPORT_BW_BOUNDARY_INJECTION, "size256_injection"}),
    [](const ::testing::TestParamInfo<suite::SweepPoint>& info) { return std::string(info.param.label); });
