// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// tensix_peer_tile member of the p2p_latency suite: X280 <-> Tensix, one chip.
//
// The host lifecycle lives under suites/p2p_latency/tensix_peer_tile.hpp. This
// file wires the fixture, parameterization axis (first direction), and
// the RunSpec (image env var + kernel dir).

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "suites/p2p_latency/tensix_peer_tile.hpp"

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>

#include "impl/program/program_impl.hpp"

namespace {

namespace suite = x280::suites::p2p_latency::tensix_peer_tile;

class X280P2PLatencyTensixPeerTile : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_P2P_LATENCY_METAL_PEER_TILE_FW_IMAGE",
                    .image_file = "firmware-p2p-latency-metal-peer-tile.bin",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "p2p_latency_tensix_peer_tile",
        };
    }
};

}  // namespace

TEST(P2PLatencyTensixPeerTileJit, CompilesExtractedMetalNocBackend) {
    namespace ttm = tt::tt_metal;
    namespace ttmd = tt::tt_metal::distributed;

    auto mesh = ttmd::MeshDevice::create_unit_mesh(0);
    ASSERT_FALSE(mesh->get_devices().empty());
    const ttm::CoreCoord worker{0, 0};
    const ttm::CoreRangeSet worker_set(ttm::CoreRange(worker, worker));
    const auto kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__);

    ttm::Program program = ttm::CreateProgram();
    (void)ttm::CreateKernel(
        program,
        (x280::harness::artifacts::shared_kernel_dir(kernel_dir, "p2p_latency") / "metal_peer_entry.cpp").string(),
        worker_set,
        ttm::DataMovementConfig{
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .noc = ttm::NOC::RISCV_0_default,
            .defines = {
                {"P2P_LATENCY_TARGET_TENSIX", "1"},
                x280::suites::p2p_latency::metal_peer::tx_mode_define(P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION)}});
    program.impl().compile(mesh->get_devices()[0]);
}

TEST_P(X280P2PLatencyTensixPeerTile, PairedPingPongMeasuresRttAndTurnPerSide) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    FirstDirection,
    X280P2PLatencyTensixPeerTile,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "X280InitiatesFirst" : "TensixInitiatesFirst";
    });
