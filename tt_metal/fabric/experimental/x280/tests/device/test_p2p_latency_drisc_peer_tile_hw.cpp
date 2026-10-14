// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Free-DRISC peer smoke runner for the p2p_latency suite.

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "harness/slow_dispatch.hpp"
#include "suites/p2p_latency/drisc_peer_tile.hpp"

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"

#include <exception>

namespace {

namespace suite = x280::suites::p2p_latency::drisc_peer_tile;

class X280P2PLatencyDriscPeerTile : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_P2P_LATENCY_METAL_PEER_TILE_FW_IMAGE",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "p2p_latency_drisc_peer_tile",
        };
    }

    void SetUp() override {
        ASSERT_TRUE(x280::harness::slow_dispatch_linked());
        Runner::SetUp();
    }
};

}  // namespace

TEST(P2PLatencyDriscPeerTileJit, CompilesDriscKernelWithoutLaunchingIt) {
    namespace ttm = tt::tt_metal;
    namespace ttmd = tt::tt_metal::distributed;

    auto mesh = ttmd::MeshDevice::create_unit_mesh(0);
    const auto& hal = ttm::MetalContext::instance().hal();
    if (!hal.has_programmable_core_type(ttm::HalProgrammableCoreType::DRAM)) {
        GTEST_SKIP() << "DRAM programmable cores require Blackhole firmware 19.12 or newer";
    }
    ASSERT_FALSE(mesh->get_devices().empty());
    auto* device = mesh->get_devices()[0];
    ttm::CoreCoord drisc_logical{};
    try {
        drisc_logical = mesh->impl().pick_unused_dram_logical_core(0);
    } catch (const std::exception& error) {
        GTEST_SKIP() << "bank 0 has no free non-endpoint DRISC: " << error.what();
    }

    ttm::Program program = ttm::CreateProgram();
    (void)ttm::CreateKernel(
        program,
        (x280::harness::artifacts::shared_kernel_dir(x280::harness::artifacts::kernel_dir_of(__FILE__), "p2p_latency") /
         "metal_peer_entry.cpp")
            .string(),
        drisc_logical,
        ttm::DramConfig{
            .noc = ttm::NOC::NOC_0,
            .defines = {{"P2P_LATENCY_TARGET_DRISC", "1"}},
            .opt_level = ttm::KernelBuildOptLevel::O2,
        });
    ttm::detail::CompileProgram(device, program, /*force_slow_dispatch=*/true);
}

TEST_P(X280P2PLatencyDriscPeerTile, PairedPingPongMeasuresRawLocalDeltas) { X280_HARNESS_DISPATCH(suite); }

TEST_P(X280P2PLatencyDriscPeerTile, SharedX280ImageRejectsInvalidCommonArguments) {
    namespace peer = x280::suites::p2p_latency::metal_peer;
    namespace hreg = x280::harness::regions;
    namespace hexec = x280::harness::execution;
    auto& f = fixture();
    const auto regions = peer::X280Regions::compute(*f.tile);
    for (unsigned invalid = 0; invalid < 9; ++invalid) {
        SCOPED_TRACE(invalid);
        auto block = peer::make_x280_args(
            {
                .self = P2P_LATENCY_PARTICIPANT_A,
                .first_direction = GetParam() ? P2P_LATENCY_DIRECTION_A_TO_B : P2P_LATENCY_DIRECTION_B_TO_A,
                .payload_size = 4,
                .warmup_count = 1,
                .sample_count = 1,
                .run_nonce = 1,
                .warmup_max_poll_cycles = 1,
            },
            {.own_result_addr = regions.result.addr});
        // Zero backend addresses guarantee rejection without traffic if validation regresses.
        switch (invalid) {
            case 0: block.common.warmup_count = 0; break;
            case 1: block.common.sample_count = 0; break;
            case 2: block.common.sample_count = P2P_LATENCY_MAX_SAMPLES + 1; break;
            case 3: block.common.run_nonce = 0; break;
            case 4: block.common.warmup_max_poll_cycles = 0; break;
            case 5: block.common.payload_size = 8; break;
            case 6: block.common.self = P2P_LATENCY_PARTICIPANT_B; break;
            case 7: block.common.first_direction = 2; break;
            case 8: block.common.run_nonce = UINT32_MAX; break;
        }
        hreg::zero_region(*f.tile, regions.result);
        hreg::write_args(*f.tile, regions.args, block);
        const auto outcome = hexec::run_x280(
            f.handle(),
            {
                .image_path = x280::harness::artifacts::image_path(spec().open.image_env_var),
                .run_value = 0xDED000u + invalid,
                .args_addr = regions.args.addr,
                .args_bytes = regions.args.bytes,
                .cleanup_timeout = std::chrono::seconds{5},
                .wait_timeout = std::chrono::seconds{5},
            });
        ASSERT_TRUE(outcome.enqueued());
        ASSERT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady);
        ASSERT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);
        const auto result = hreg::read_struct<p2p_latency_result_t>(*f.tile, regions.result);
        EXPECT_EQ(result.terminal_state, P2P_LATENCY_STATE_INIT);
        EXPECT_EQ(result.backend_id, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FirstDirection,
    X280P2PLatencyDriscPeerTile,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "X280InitiatesFirst" : "DriscInitiatesFirst";
    });
