// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// x280_same_tile member of the p2p_latency suite: X280 hart <-> X280 hart, same tile, no NOC.
//
// The whole implementation lives under suites/p2p_latency/x280_same_tile.hpp. This
// file only wires the fixture, the parameterization axis (first direction),
// and the RunSpec that names the case's env var.

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "suites/p2p_latency/x280_same_tile.hpp"

#include <gtest/gtest.h>

namespace {

namespace suite = x280::suites::p2p_latency::x280_same_tile;

class X280P2PLatencyX280SameTile : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_P2P_LATENCY_X280_SAME_TILE_FW_IMAGE",
                    .image_file = "firmware-p2p-latency-x280-same-tile.bin",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "p2p_latency_x280_same_tile",
        };
    }
};

}  // namespace

TEST_P(X280P2PLatencyX280SameTile, TwoHartsOnOneTilePingPongThroughLim) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    FirstDirection,
    X280P2PLatencyX280SameTile,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "AInitiatesFirst" : "BInitiatesFirst"; });
