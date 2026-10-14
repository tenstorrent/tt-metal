// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// x280_peer_tile member of the p2p_latency suite: two X280 harts on different L2CPU tiles exchange over the NoC.
// Liveness is verified by the member's rendezvous sentinel.

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "suites/p2p_latency/x280_peer_tile.hpp"

#include <gtest/gtest.h>

namespace {

namespace suite = x280::suites::p2p_latency::x280_peer_tile;

class X280P2PLatencyX280PeerTile : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_P2P_LATENCY_X280_PEER_TILE_FW_IMAGE",
                    .image_file = "firmware-p2p-latency-x280-peer-tile.bin",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "p2p_latency_x280_peer_tile",
        };
    }
};

}  // namespace

TEST_P(X280P2PLatencyX280PeerTile, TwoComplexesPingPongOverNoc) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    FirstDirection,
    X280P2PLatencyX280PeerTile,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "AInitiatesFirst" : "BInitiatesFirst"; });
