// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// transport_bandwidth suite -- Tensix -> X280 addressable member.
//
// The whole implementation lives under
// suites/transport_bandwidth/tensix_to_x280.hpp. This file wires the
// fixture, the parameterization axis (transfer size + boundary), and the
// RunSpec (no X280 image, kernel dir).

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "suites/transport_bandwidth/tensix_to_x280.hpp"

#include "transport_bandwidth_common.h"

#include <gtest/gtest.h>

namespace {

namespace suite = x280::suites::transport_bandwidth::tensix_to_x280;

class TransportBandwidthTensixToX280 : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "",  // Tensix-only; no X280 image is submitted.
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "transport_bandwidth_tensix_to_x280",
        };
    }
};

}  // namespace

TEST_P(TransportBandwidthTensixToX280, RepeatedIssueMeasuresSustainedAndInjection) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    Sweep,
    TransportBandwidthTensixToX280,
    ::testing::Values(
        suite::SweepPoint{64, TRANSPORT_BW_BOUNDARY_DRAINED, "size64_drained"},
        suite::SweepPoint{64, TRANSPORT_BW_BOUNDARY_INJECTION, "size64_injection"},
        suite::SweepPoint{128, TRANSPORT_BW_BOUNDARY_DRAINED, "size128_drained"},
        suite::SweepPoint{128, TRANSPORT_BW_BOUNDARY_INJECTION, "size128_injection"},
        suite::SweepPoint{256, TRANSPORT_BW_BOUNDARY_DRAINED, "size256_drained"},
        suite::SweepPoint{256, TRANSPORT_BW_BOUNDARY_INJECTION, "size256_injection"},
        suite::SweepPoint{320, TRANSPORT_BW_BOUNDARY_DRAINED, "size320_drained"},
        suite::SweepPoint{320, TRANSPORT_BW_BOUNDARY_INJECTION, "size320_injection"},
        suite::SweepPoint{384, TRANSPORT_BW_BOUNDARY_DRAINED, "size384_drained"},
        suite::SweepPoint{384, TRANSPORT_BW_BOUNDARY_INJECTION, "size384_injection"},
        suite::SweepPoint{512, TRANSPORT_BW_BOUNDARY_DRAINED, "size512_drained"},
        suite::SweepPoint{512, TRANSPORT_BW_BOUNDARY_INJECTION, "size512_injection"},
        suite::SweepPoint{1024, TRANSPORT_BW_BOUNDARY_DRAINED, "size1024_drained"},
        suite::SweepPoint{1024, TRANSPORT_BW_BOUNDARY_INJECTION, "size1024_injection"},
        suite::SweepPoint{2048, TRANSPORT_BW_BOUNDARY_DRAINED, "size2048_drained"},
        suite::SweepPoint{2048, TRANSPORT_BW_BOUNDARY_INJECTION, "size2048_injection"},
        suite::SweepPoint{4096, TRANSPORT_BW_BOUNDARY_DRAINED, "size4096_drained"},
        suite::SweepPoint{4096, TRANSPORT_BW_BOUNDARY_INJECTION, "size4096_injection"},
        suite::SweepPoint{8192, TRANSPORT_BW_BOUNDARY_DRAINED, "size8192_drained"},
        suite::SweepPoint{8192, TRANSPORT_BW_BOUNDARY_INJECTION, "size8192_injection"},
        suite::SweepPoint{16384, TRANSPORT_BW_BOUNDARY_DRAINED, "size16384_drained"},
        suite::SweepPoint{16384, TRANSPORT_BW_BOUNDARY_INJECTION, "size16384_injection"}),
    [](const ::testing::TestParamInfo<suite::SweepPoint>& info) { return std::string(info.param.label); });
