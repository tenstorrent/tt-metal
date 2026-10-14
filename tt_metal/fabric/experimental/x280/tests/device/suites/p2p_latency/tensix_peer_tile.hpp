// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/p2p_latency/tensix_peer_tile.hpp -- tensix_peer_tile member of the p2p_latency suite.
//
// One X280 hart + one Tensix data-movement core exchange tokens over the
// NOC through paired mailboxes. The descriptor owns the member's LIM+L1
// layouts, args pack, harvest, and records; the host test wires the
// fixture and parameterization.

#ifndef X280_SUITES_P2P_LATENCY_TENSIX_PEER_TILE_HPP
#define X280_SUITES_P2P_LATENCY_TENSIX_PEER_TILE_HPP

#include "harness/artifacts/artifacts.hpp"
#include "metal_peer.hpp"
#include "tests/device/suites/p2p_latency/metal_peer_arguments.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/regions/regions.hpp"
#include "harness/regions/tensix_l1.hpp"
#include "harness/runner.hpp"
#include "harness/metal_context.hpp"

#include "p2p_latency_frontend.h"
#include "p2p_latency_layout.h"
#include "metal_peer_args.h"
#include "p2p_latency_x280_noc_args.h"
#include "rt/x280_noc_map.h"

#include <gtest/gtest.h>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace x280::suites::p2p_latency::tensix_peer_tile {

namespace ttm = tt::tt_metal;
namespace ttmd = tt::tt_metal::distributed;
namespace hart_ = x280::harness::artifacts;
namespace hargs = x280::harness::arguments;
namespace hexec = x280::harness::execution;
namespace hmeas = x280::harness::measurement;
namespace hreg = x280::harness::regions;

constexpr const char* kWarmupEnv = "P2P_LATENCY_TENSIX_PEER_TILE_W";
constexpr uint32_t kDefaultWarmup = 8u;
constexpr uint32_t kDefaultSamples = 64u;
constexpr uint32_t kPayloadSize = 4u;
constexpr uint32_t kTlbIndex = X280_TLB_PAIRED_BASE;
constexpr uint64_t kWarmupBudgetX280Cycles = 500u * 1000000ull;
constexpr uint64_t kWarmupBudgetTensixTicks = 900u * 1000000ull;

using Axis = bool;  // x280_initiates_first

struct TensixL1Layout {
    uint32_t block_bytes;
    uint32_t mailbox_offset;
    uint32_t staging_offset;
    uint32_t result_offset;

    static TensixL1Layout compute() {
        TensixL1Layout L{};
        auto ru = [](uint64_t n, uint64_t to) { return static_cast<uint32_t>((n + to - 1u) & ~(to - 1u)); };
        uint32_t cursor = 0;
        L.mailbox_offset = cursor;
        cursor += ru(P2P_LATENCY_MAILBOX_BYTES, hreg::kCacheLineBytes);
        L.staging_offset = cursor;
        cursor += ru(P2P_LATENCY_STAGING_BYTES, hreg::kCacheLineBytes);
        L.result_offset = cursor;
        cursor += ru(sizeof(p2p_latency_result_t), hreg::kCacheLineBytes);
        L.block_bytes = cursor;
        return L;
    }
};

inline uint64_t mint_run_value(uint32_t& run_nonce) {
    static uint32_t sequence = 0;
    ++sequence;
    run_nonce = 0x200u + sequence;
    return (static_cast<uint64_t>(0xE11Cu) << 48) | (static_cast<uint64_t>(0xB3u) << 40) | run_nonce;
}

// One paired launch's arguments and results. clock is set only when the launch
// assessed AICLK around the workload.
struct Launch {
    p2p_latency_x280_noc_section_t block;
    p2p_latency_result_t result_x;
    p2p_latency_result_t result_t;
    uint64_t expected_aiclk_hz;
    std::optional<hmeas::AiclkRunEvidence> clock;
};

// Launches both participants once with W warm-up and N timed exchanges per
// phase under a minted run_nonce/run_value and validates both results; fatal
// failures leave `out` partial.
inline void launch(
    x280::harness::session::Fixture& f,
    const x280::harness::RunSpec& spec,
    p2p_latency_direction_t first_direction,
    uint32_t W,
    uint32_t N,
    uint32_t run_nonce,
    uint64_t run_value,
    bool assess_clock,
    Launch& out) {
    const uint64_t max_token = static_cast<uint64_t>(run_nonce) + 2ull * (static_cast<uint64_t>(W) + N) - 1ull;
    ASSERT_LE(max_token, 0xFFFFFFFFull) << "run_nonce + 2*(W+N)-1 exceeds u32";

    const TensixL1Layout TL = TensixL1Layout::compute();
    const ttm::CoreCoord worker{0, 0};
    auto l1 = x280::harness::regions::allocate_tile_l1_buffer(
        f.mesh.get(), ttm::CoreRangeSet(ttm::CoreRange(worker, worker)), TL.block_bytes);
    const ttm::CoreRangeSet& worker_set = l1.worker_set;
    const uint32_t l1_base = l1.l1_tile_base;

    ASSERT_FALSE(f.mesh->get_devices().empty()) << "unit mesh has no devices";
    const auto* device = f.mesh->get_devices()[0];
    const auto worker_virtual =
        x280::X280Runtime::virtual_core_from_logical_core(*f.mesh, worker, tt::CoreType::TENSIX, ttm::NOC::NOC_0);
    const auto l2cpu_translated = f.runtime->virtual_core_from_tile(f.tile_id, ttm::NOC::RISCV_0_default);

    const metal_peer::X280Regions XR = metal_peer::X280Regions::compute(*f.tile);
    hreg::zero_region(*f.tile, XR.mailbox);
    hreg::zero_region(*f.tile, XR.result);

    out.block = metal_peer::make_x280_args(
        {
            .self = P2P_LATENCY_PARTICIPANT_A,
            .first_direction = static_cast<uint32_t>(first_direction),
            .payload_size = kPayloadSize,
            .warmup_count = W,
            .sample_count = N,
            .run_nonce = run_nonce,
            .warmup_max_poll_cycles = kWarmupBudgetX280Cycles,
        },
        {
            .own_inbound_addr = XR.mailbox.addr,
            .own_result_addr = XR.result.addr,
            .peer_inbound_addr = static_cast<uint64_t>(l1_base + TL.mailbox_offset),
            .peer_noc = worker_virtual.noc(),
            .peer_tlb_index = kTlbIndex,
            .reserved = 0u,
        });
    hreg::write_args(*f.tile, XR.args, out.block);

    const auto tensix_args = hargs::pack_metal_peer_runtime_args(hargs::MetalPeerRuntimeArgs{
        .layout_cookie = P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE,
        .common =
            {
                .self = P2P_LATENCY_PARTICIPANT_B,
                .first_direction = static_cast<uint32_t>(first_direction),
                .payload_size = kPayloadSize,
                .warmup_count = W,
                .sample_count = N,
                .run_nonce = run_nonce,
                .warmup_max_poll_cycles = kWarmupBudgetTensixTicks,
            },
        .own_inbound_addr = l1_base + TL.mailbox_offset,
        .own_staging_addr = l1_base + TL.staging_offset,
        .own_result_addr = l1_base + TL.result_offset,
        .peer_inbound_addr = XR.mailbox.addr,
        .peer_noc = l2cpu_translated,
    });

    ttm::Program program = ttm::CreateProgram();
    const ttm::KernelHandle kernel = ttm::CreateKernel(
        program,
        (hart_::shared_kernel_dir(spec.open.kernel_dir, "p2p_latency") / "metal_peer_entry.cpp").string(),
        worker_set,
        ttm::DataMovementConfig{
            .processor = ttm::DataMovementProcessor::RISCV_0,
            .noc = ttm::NOC::RISCV_0_default,
            .defines = {{"P2P_LATENCY_TARGET_TENSIX", "1"}}});
    ttm::SetRuntimeArgs(program, kernel, worker_set, tensix_args);

    ttmd::MeshWorkload workload;
    workload.add_program(ttmd::MeshCoordinateRange(f.mesh->shape()), std::move(program));

    const hexec::X280Launch x280_launch{
        .image_path =
            hart_::image_path("X280_P2P_LATENCY_METAL_PEER_TILE_FW_IMAGE", "firmware-p2p-latency-metal-peer-tile.bin"),
        .run_value = run_value,
        .args_addr = XR.args.addr,
        .args_bytes = XR.args.bytes,
        .cleanup_timeout = std::chrono::seconds{5},
        .wait_timeout = metal_peer::kX280WaitTimeout,
    };

    out.expected_aiclk_hz = hexec::programmed_aiclk_hz(x280::harness::cluster(), *device);
    const ttm::CoreCoord clock_core{worker_virtual.x(), worker_virtual.y()};
    hmeas::AiclkEndpoint clock_start;
    if (assess_clock) {
        clock_start = hexec::read_aiclk_start(x280::harness::cluster(), *device, clock_core);
        ASSERT_TRUE(clock_start.errors.empty()) << "pre-run AICLK evidence is unavailable";
    }
    hexec::enqueue_tensix(*f.mesh, workload);
    const auto outcome = hexec::run_x280(f.handle(), x280_launch);
    hexec::finish_tensix(*f.mesh);
    if (assess_clock) {
        const auto clock_end = hexec::read_aiclk_end(x280::harness::cluster(), *device, clock_core);
        out.clock = hmeas::assess_aiclk_run(out.expected_aiclk_hz, clock_start, clock_end);
    }

    ASSERT_TRUE(outcome.enqueued()) << "enqueue rejected";
    ASSERT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);

    hexec::expect_window_cleared(*f.tile, kTlbIndex);

    out.result_x = hreg::read_struct<p2p_latency_result_t>(*f.tile, XR.result);

    std::vector<uint32_t> l1_readback;
    ttmd::EnqueueReadMeshBuffer(f.mesh->mesh_command_queue(), l1_readback, l1.buffer, /*blocking=*/true);
    ASSERT_EQ(l1_readback.size(), TL.block_bytes / sizeof(uint32_t));
    std::memcpy(&out.result_t, l1_readback.data() + TL.result_offset / sizeof(uint32_t), sizeof(out.result_t));

    EXPECT_EQ(out.result_x.backend_id, P2P_LATENCY_BACKEND_ID_X280_METAL_PEER_TILE);
    EXPECT_EQ(out.result_t.backend_id, P2P_LATENCY_BACKEND_ID_TENSIX_TENSIX_PEER_TILE);
    ASSERT_EQ(out.result_x.terminal_state, static_cast<uint32_t>(P2P_LATENCY_STATE_DONE))
        << "X280 terminal state: " << metal_peer::state_name(out.result_x.terminal_state);
    ASSERT_EQ(out.result_t.terminal_state, static_cast<uint32_t>(P2P_LATENCY_STATE_DONE))
        << "Tensix terminal state: " << metal_peer::state_name(out.result_t.terminal_state);
    EXPECT_EQ(out.result_x.path_sentinel, run_nonce);
    EXPECT_EQ(out.result_t.path_sentinel, run_nonce);
    ASSERT_NO_FATAL_FAILURE(metal_peer::validate_phases(out.result_x, P2P_LATENCY_PARTICIPANT_A, first_direction, N));
    ASSERT_NO_FATAL_FAILURE(metal_peer::validate_phases(out.result_t, P2P_LATENCY_PARTICIPANT_B, first_direction, N));
}

inline void run_axis(
    x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, bool x280_initiates_first) {
    const p2p_latency_direction_t first_direction =
        x280_initiates_first ? P2P_LATENCY_DIRECTION_A_TO_B : P2P_LATENCY_DIRECTION_B_TO_A;

    // An explicit W is used as given; otherwise a default-W pilot sizes it.
    const auto explicit_warmup = hart_::env_u32(kWarmupEnv);
    const uint32_t N = hart_::env_u32("P2P_LATENCY_TENSIX_PEER_TILE_N", kDefaultSamples);
    ASSERT_GT(N, 0u);
    ASSERT_LE(N, P2P_LATENCY_MAX_SAMPLES);
    uint32_t run_nonce = 0;
    const uint64_t run_value = mint_run_value(run_nonce);
    hmeas::WarmupSelection warmup{.warmup = explicit_warmup.value_or(kDefaultWarmup)};
    ASSERT_GT(warmup.warmup, 0u);
    if (!explicit_warmup) {
        uint32_t pilot_nonce = 0;
        const uint64_t pilot_value = mint_run_value(pilot_nonce);
        Launch pilot{};
        ASSERT_NO_FATAL_FAILURE(
            launch(f, spec, first_direction, kDefaultWarmup, N, pilot_nonce, pilot_value, false, pilot));
        warmup = hmeas::calibrate_warmup(metal_peer::warmup_calibration_request(
            pilot.result_x, pilot.result_t, pilot.expected_aiclk_hz, kDefaultWarmup, N, run_nonce));
        ASSERT_FALSE(warmup.refusal.has_value())
            << "warm-up calibration refused: " << hmeas::warmup_refusal_name(*warmup.refusal);
    }
    const uint32_t W = static_cast<uint32_t>(warmup.warmup);

    Launch run{};
    ASSERT_NO_FATAL_FAILURE(launch(f, spec, first_direction, W, N, run_nonce, run_value, true, run));
    const auto& block = run.block;
    const auto& result_x = run.result_x;
    const auto& result_t = run.result_t;
    ASSERT_TRUE(run.clock.has_value()) << "measured launch assessed no AICLK";
    const auto& clock = *run.clock;

    auto report_phase = [&](const p2p_latency_result_t& r, const char* who, uint32_t p) {
        const bool tensix = r.backend_id == P2P_LATENCY_BACKEND_ID_TENSIX_TENSIX_PEER_TILE;
        const auto body = x280::harness::output::record(metal_peer::phase_record(
            "tensix_peer_tile",
            {{"participant", who}},
            r,
            p,
            block.common,
            block.backend.peer_tlb_index,
            tensix ? std::nullopt : std::optional{P2P_LATENCY_X280_NOC_ORDERING},
            tensix ? std::optional{metal_peer::MetalClock{hmeas::Counter::TensixLiveTimer, clock}} : std::nullopt,
            warmup));
        x280::harness::output::write_jsonl(spec.suite_label, body);
        std::printf("%s\n", body.c_str());
    };
    report_phase(result_x, "X280", 0);
    report_phase(result_t, "Tensix", 0);
    report_phase(result_x, "X280", 1);
    report_phase(result_t, "Tensix", 1);

    for (uint32_t p = 0; p < 2; ++p) {
        EXPECT_TRUE(metal_peer::phase_samples_nonzero(result_x.phases[p])) << "X280 phase " << p;
        EXPECT_TRUE(metal_peer::phase_samples_nonzero(result_t.phases[p])) << "Tensix phase " << p;
    }
    EXPECT_TRUE(clock.errors.empty()) << x280::harness::output::clock_json(clock);
}

}  // namespace x280::suites::p2p_latency::tensix_peer_tile

#endif  // X280_SUITES_P2P_LATENCY_TENSIX_PEER_TILE_HPP
