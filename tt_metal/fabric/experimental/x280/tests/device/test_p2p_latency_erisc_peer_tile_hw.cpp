// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Idle-ERISC peer smoke runner for the p2p_latency suite.

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "harness/slow_dispatch.hpp"
#include "suites/p2p_latency/erisc_peer_tile.hpp"

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

namespace {

namespace suite = x280::suites::p2p_latency::erisc_peer_tile;

class X280P2PLatencyEriscPeerTile : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_P2P_LATENCY_METAL_PEER_TILE_FW_IMAGE",
                    .image_file = "firmware-p2p-latency-metal-peer-tile.bin",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "p2p_latency_erisc_peer_tile",
        };
    }

    void SetUp() override {
        ASSERT_TRUE(x280::harness::slow_dispatch_linked());
        Runner::SetUp();
    }
};

}  // namespace

TEST(P2PLatencyEriscPeerTileJit, CompilesIdleEriscKernelWithoutLaunchingIt) {
    namespace ttm = tt::tt_metal;
    namespace ttmd = tt::tt_metal::distributed;

    auto mesh = ttmd::MeshDevice::create_unit_mesh(0);
    const auto& hal = ttm::MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(ttm::HalProgrammableCoreType::IDLE_ETH)) {
        GTEST_SKIP() << "idle Ethernet programmable cores not enabled";
    }
    ASSERT_FALSE(mesh->get_devices().empty());
    auto* device = mesh->get_devices()[0];
    const auto inactive_cores = device->get_inactive_ethernet_cores();
    if (inactive_cores.empty()) {
        GTEST_SKIP() << "no inactive Ethernet core available";
    }

    ttm::Program program = ttm::CreateProgram();
    (void)ttm::CreateKernel(
        program,
        (x280::harness::artifacts::shared_kernel_dir(x280::harness::artifacts::kernel_dir_of(__FILE__), "p2p_latency") /
         "metal_peer_entry.cpp")
            .string(),
        *inactive_cores.begin(),
        ttm::EthernetConfig{
            .eth_mode = ttm::Eth::IDLE,
            .noc = ttm::NOC::NOC_0,
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .defines =
                {{"P2P_LATENCY_TARGET_IDLE_ERISC", "1"},
                 x280::suites::p2p_latency::metal_peer::tx_mode_define(P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION)},
            .opt_level = ttm::KernelBuildOptLevel::Os,
        });
    program.impl().compile(device);
}

TEST_P(X280P2PLatencyEriscPeerTile, PairedPingPongMeasuresRawLocalDeltas) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    FirstDirection,
    X280P2PLatencyEriscPeerTile,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "X280InitiatesFirst" : "EriscInitiatesFirst";
    });
