// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// transport_bandwidth suite -- X280 -> Tensix addressable member.
//
// The whole implementation lives under
// suites/transport_bandwidth/x280_to_tensix.hpp. This file wires the
// fixture, the parameterization axis (transfer size + boundary + store
// width + harts), and the RunSpec (image env var + kernel dir).

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
                    .image_file = "firmware-transport-bandwidth-x280-to-tensix.bin",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "transport_bandwidth_x280_to_tensix",
        };
    }
};

}  // namespace

TEST_P(TransportBandwidthX280ToTensix, RepeatedIssueMeasuresSustainedAndInjection) { X280_HARNESS_DISPATCH(suite); }

// The sw (4 B) points keep their unprefixed labels; sd and vse32.v points are
// prefixed by the store, and points run by more than one hart by hN.
constexpr suite::SweepPoint kSweep[] = {
    {64, TRANSPORT_BW_BOUNDARY_DRAINED, 4, "size64_drained"},
    {64, TRANSPORT_BW_BOUNDARY_INJECTION, 4, "size64_injection"},
    {256, TRANSPORT_BW_BOUNDARY_DRAINED, 4, "size256_drained"},
    {256, TRANSPORT_BW_BOUNDARY_INJECTION, 4, "size256_injection"},
    {4096, TRANSPORT_BW_BOUNDARY_DRAINED, 4, "size4096_drained"},
    {4096, TRANSPORT_BW_BOUNDARY_INJECTION, 4, "size4096_injection"},
    {65536, TRANSPORT_BW_BOUNDARY_DRAINED, 4, "size65536_drained"},
    {65536, TRANSPORT_BW_BOUNDARY_INJECTION, 4, "size65536_injection"},
    {64, TRANSPORT_BW_BOUNDARY_DRAINED, 8, "sd_size64_drained"},
    {64, TRANSPORT_BW_BOUNDARY_INJECTION, 8, "sd_size64_injection"},
    {256, TRANSPORT_BW_BOUNDARY_DRAINED, 8, "sd_size256_drained"},
    {256, TRANSPORT_BW_BOUNDARY_INJECTION, 8, "sd_size256_injection"},
    {4096, TRANSPORT_BW_BOUNDARY_DRAINED, 8, "sd_size4096_drained"},
    {4096, TRANSPORT_BW_BOUNDARY_INJECTION, 8, "sd_size4096_injection"},
    {65536, TRANSPORT_BW_BOUNDARY_DRAINED, 8, "sd_size65536_drained"},
    {65536, TRANSPORT_BW_BOUNDARY_INJECTION, 8, "sd_size65536_injection"},
    {64, TRANSPORT_BW_BOUNDARY_DRAINED, 64, "vse32_size64_drained"},
    {64, TRANSPORT_BW_BOUNDARY_INJECTION, 64, "vse32_size64_injection"},
    {256, TRANSPORT_BW_BOUNDARY_DRAINED, 64, "vse32_size256_drained"},
    {256, TRANSPORT_BW_BOUNDARY_INJECTION, 64, "vse32_size256_injection"},
    {4096, TRANSPORT_BW_BOUNDARY_DRAINED, 64, "vse32_size4096_drained"},
    {4096, TRANSPORT_BW_BOUNDARY_INJECTION, 64, "vse32_size4096_injection"},
    {65536, TRANSPORT_BW_BOUNDARY_DRAINED, 64, "vse32_size65536_drained"},
    {65536, TRANSPORT_BW_BOUNDARY_INJECTION, 64, "vse32_size65536_injection"},
    {64, TRANSPORT_BW_BOUNDARY_DRAINED, 4, "h2_size64_drained", 2},
    {64, TRANSPORT_BW_BOUNDARY_INJECTION, 4, "h2_size64_injection", 2},
    {65536, TRANSPORT_BW_BOUNDARY_DRAINED, 4, "h2_size65536_drained", 2},
    {65536, TRANSPORT_BW_BOUNDARY_INJECTION, 4, "h2_size65536_injection", 2},
    {64, TRANSPORT_BW_BOUNDARY_DRAINED, 8, "h2_sd_size64_drained", 2},
    {64, TRANSPORT_BW_BOUNDARY_INJECTION, 8, "h2_sd_size64_injection", 2},
    {65536, TRANSPORT_BW_BOUNDARY_DRAINED, 8, "h2_sd_size65536_drained", 2},
    {65536, TRANSPORT_BW_BOUNDARY_INJECTION, 8, "h2_sd_size65536_injection", 2},
    {64, TRANSPORT_BW_BOUNDARY_DRAINED, 64, "h2_vse32_size64_drained", 2},
    {64, TRANSPORT_BW_BOUNDARY_INJECTION, 64, "h2_vse32_size64_injection", 2},
    {65536, TRANSPORT_BW_BOUNDARY_DRAINED, 64, "h2_vse32_size65536_drained", 2},
    {65536, TRANSPORT_BW_BOUNDARY_INJECTION, 64, "h2_vse32_size65536_injection", 2},
    {64, TRANSPORT_BW_BOUNDARY_DRAINED, 4, "h4_size64_drained", 4},
    {64, TRANSPORT_BW_BOUNDARY_INJECTION, 4, "h4_size64_injection", 4},
    {65536, TRANSPORT_BW_BOUNDARY_DRAINED, 4, "h4_size65536_drained", 4},
    {65536, TRANSPORT_BW_BOUNDARY_INJECTION, 4, "h4_size65536_injection", 4},
    {64, TRANSPORT_BW_BOUNDARY_DRAINED, 8, "h4_sd_size64_drained", 4},
    {64, TRANSPORT_BW_BOUNDARY_INJECTION, 8, "h4_sd_size64_injection", 4},
    {65536, TRANSPORT_BW_BOUNDARY_DRAINED, 8, "h4_sd_size65536_drained", 4},
    {65536, TRANSPORT_BW_BOUNDARY_INJECTION, 8, "h4_sd_size65536_injection", 4},
    {64, TRANSPORT_BW_BOUNDARY_DRAINED, 64, "h4_vse32_size64_drained", 4},
    {64, TRANSPORT_BW_BOUNDARY_INJECTION, 64, "h4_vse32_size64_injection", 4},
    {65536, TRANSPORT_BW_BOUNDARY_DRAINED, 64, "h4_vse32_size65536_drained", 4},
    {65536, TRANSPORT_BW_BOUNDARY_INJECTION, 64, "h4_vse32_size65536_injection", 4},
};

INSTANTIATE_TEST_SUITE_P(
    Sweep,
    TransportBandwidthX280ToTensix,
    ::testing::ValuesIn(kSweep),
    [](const ::testing::TestParamInfo<suite::SweepPoint>& info) { return std::string(info.param.label); });
