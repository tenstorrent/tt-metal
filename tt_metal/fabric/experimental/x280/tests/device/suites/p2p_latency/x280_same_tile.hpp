// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/p2p_latency/x280_same_tile.hpp -- x280_same_tile member of the p2p_latency suite.
//
// Two X280 harts of one L2CPU exchange tokens through shared LIM: no NOC,
// no Tensix. This descriptor carries the member's axes, LIM layout, args
// build, harvest, and per-phase records; the host test wires the fixture
// and the parameterization.

#ifndef X280_SUITES_P2P_LATENCY_X280_SAME_TILE_HPP
#define X280_SUITES_P2P_LATENCY_X280_SAME_TILE_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "tests/device/suites/p2p_latency/metal_peer.hpp"
#include "harness/regions/regions.hpp"
#include "harness/runner.hpp"

#include "p2p_latency_frontend.h"
#include "p2p_latency_layout.h"
#include "x280_same_tile_args.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace x280::suites::p2p_latency::x280_same_tile {

namespace hout = x280::harness::output;
namespace hreg = x280::harness::regions;
namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;

constexpr uint32_t kDefaultWarmup = 8u;
constexpr uint32_t kDefaultSamples = 64u;
constexpr uint32_t kPayloadSize = 4u;
constexpr uint64_t kWarmupBudgetCycles = 500u * 1000000ull;

using Axis = bool;  // a_initiates_first

struct ParticipantRegions {
    hreg::Region mailbox;
    hreg::Region staging;
    hreg::Region result;
};

struct LimLayout {
    hreg::Region args;
    ParticipantRegions a;
    ParticipantRegions b;

    static LimLayout compute(x280::X280Tile& tile) {
        const auto avail = tile.available_lim();
        const hreg::Region w{avail.begin, avail.size()};
        hreg::Arena arena(w);
        LimLayout L{};
        L.args = arena.allocate(sizeof(p2p_latency_x280_same_tile_args_block_t));
        auto place = [&](ParticipantRegions& p) {
            p.mailbox = arena.allocate(P2P_LATENCY_MAILBOX_BYTES);
            p.staging = arena.allocate(P2P_LATENCY_STAGING_BYTES);
            p.result = arena.allocate(sizeof(p2p_latency_result_t));
        };
        place(L.a);
        place(L.b);
        return L;
    }
};

inline uint64_t mint_run_value(uint32_t& run_nonce) {
    static uint32_t sequence = 0;
    ++sequence;
    run_nonce = 0x100u + sequence;
    return (static_cast<uint64_t>(0xE11Cu) << 48) | (static_cast<uint64_t>(0xB1u) << 40) | run_nonce;
}

inline void fill_section(
    p2p_latency_x280_same_tile_section_t& s,
    p2p_latency_participant_t self,
    p2p_latency_direction_t first_direction,
    uint32_t W,
    uint32_t N,
    uint32_t run_nonce,
    const ParticipantRegions& own,
    const ParticipantRegions& peer) {
    s.cookie_common = p2p_latency_cookie_for(UBENCH_SECTION_COMMON);
    s.pad0 = 0u;
    s.common.self = static_cast<uint32_t>(self);
    s.common.first_direction = static_cast<uint32_t>(first_direction);
    s.common.payload_size = kPayloadSize;
    s.common.warmup_count = W;
    s.common.sample_count = N;
    s.common.run_nonce = run_nonce;
    s.common.warmup_max_poll_cycles = kWarmupBudgetCycles;
    s.cookie_backend = p2p_latency_cookie_for(UBENCH_SECTION_BACKEND);
    s.pad1 = 0u;
    s.backend.own_inbound_addr = own.mailbox.addr;
    s.backend.peer_inbound_addr = peer.mailbox.addr;
    s.backend.own_staging_addr = own.staging.addr;
    s.backend.own_result_addr = own.result.addr;
    s.backend.medium = P2P_LATENCY_X280_SAME_TILE_MEDIUM_COHERENT;
    s.backend.reserved = 0u;
}

inline void run_axis(
    x280::harness::session::Fixture& f, const x280::harness::RunSpec& /*spec*/, bool a_initiates_first) {
    const p2p_latency_direction_t first_direction =
        a_initiates_first ? P2P_LATENCY_DIRECTION_A_TO_B : P2P_LATENCY_DIRECTION_B_TO_A;

    const uint32_t W = hart_::env_u32("P2P_LATENCY_X280_SAME_TILE_W", kDefaultWarmup);
    const uint32_t N = hart_::env_u32("P2P_LATENCY_X280_SAME_TILE_N", kDefaultSamples);
    ASSERT_GT(W, 0u);
    ASSERT_GT(N, 0u);
    ASSERT_LE(N, P2P_LATENCY_MAX_SAMPLES);

    uint32_t run_nonce = 0;
    const uint64_t run_value = mint_run_value(run_nonce);
    const uint64_t max_token = static_cast<uint64_t>(run_nonce) + 2ull * (static_cast<uint64_t>(W) + N) - 1ull;
    ASSERT_LE(max_token, 0xFFFFFFFFull) << "run_nonce + 2*(W+N)-1 exceeds u32";

    const LimLayout L = LimLayout::compute(*f.tile);

    hreg::zero_region(*f.tile, L.a.mailbox);
    hreg::zero_region(*f.tile, L.a.staging);
    hreg::zero_region(*f.tile, L.a.result);
    hreg::zero_region(*f.tile, L.b.mailbox);
    hreg::zero_region(*f.tile, L.b.staging);
    hreg::zero_region(*f.tile, L.b.result);

    p2p_latency_x280_same_tile_args_block_t block{};
    fill_section(block.sections[0], P2P_LATENCY_PARTICIPANT_A, first_direction, W, N, run_nonce, L.a, L.b);
    fill_section(block.sections[1], P2P_LATENCY_PARTICIPANT_B, first_direction, W, N, run_nonce, L.b, L.a);
    hreg::write_args(*f.tile, L.args, block);

    hexec::X280Launch launch{
        .image_path = hart_::image_path("X280_P2P_LATENCY_X280_SAME_TILE_FW_IMAGE"),
        .run_value = run_value,
        .args_addr = L.args.addr,
        .args_bytes = L.args.bytes,
    };
    const auto outcome = hexec::run_x280(f.handle(), launch);
    ASSERT_TRUE(outcome.enqueued()) << "enqueue rejected";
    ASSERT_EQ(outcome.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome.wait);
    EXPECT_TRUE(outcome.cleaned_up()) << x280::describe(outcome.cleanup);

    const auto result_a = hreg::read_struct<p2p_latency_result_t>(*f.tile, L.a.result);
    const auto result_b = hreg::read_struct<p2p_latency_result_t>(*f.tile, L.b.result);

    EXPECT_EQ(result_a.backend_id, P2P_LATENCY_BACKEND_ID_X280_X280_SAME_TILE)
        << "participant A did not compile the X280 x280_same_tile backend";
    EXPECT_EQ(result_b.backend_id, P2P_LATENCY_BACKEND_ID_X280_X280_SAME_TILE);
    EXPECT_EQ(result_a.terminal_state, static_cast<uint32_t>(P2P_LATENCY_STATE_DONE))
        << "participant A terminal state: " << metal_peer::state_name(result_a.terminal_state);
    EXPECT_EQ(result_b.terminal_state, static_cast<uint32_t>(P2P_LATENCY_STATE_DONE))
        << "participant B terminal state: " << metal_peer::state_name(result_b.terminal_state);

    EXPECT_EQ(result_a.path_sentinel, run_nonce)
        << "participant A observed " << result_a.path_sentinel << " at the rendezvous, expected " << run_nonce;
    EXPECT_EQ(result_b.path_sentinel, run_nonce)
        << "participant B observed " << result_b.path_sentinel << " at the rendezvous, expected " << run_nonce;
    ASSERT_NO_FATAL_FAILURE(metal_peer::validate_phases(result_a, P2P_LATENCY_PARTICIPANT_A, first_direction, N));
    ASSERT_NO_FATAL_FAILURE(metal_peer::validate_phases(result_b, P2P_LATENCY_PARTICIPANT_B, first_direction, N));

    auto report_phase = [&](const p2p_latency_result_t& r, const char* who, uint32_t p) {
        const auto body = x280::harness::output::record(metal_peer::phase_record(
            "x280_same_tile",
            {{"participant", who}},
            r,
            p,
            block.sections[0].common,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            x280::harness::measurement::WarmupSelection{.warmup = block.sections[0].common.warmup_count}));
        x280::harness::output::write_jsonl("p2p_latency_x280_same_tile", body);
        std::printf("%s\n", body.c_str());
    };
    report_phase(result_a, "A", 0);
    report_phase(result_b, "B", 0);
    report_phase(result_a, "A", 1);
    report_phase(result_b, "B", 1);

    for (uint32_t p = 0; p < 2; ++p) {
        for (const auto& [who, r] : {std::pair{"A", &result_a}, std::pair{"B", &result_b}}) {
            EXPECT_TRUE(metal_peer::phase_samples_nonzero(r->phases[p])) << who << " phase " << p;
        }
    }
}

}  // namespace x280::suites::p2p_latency::x280_same_tile

#endif  // X280_SUITES_P2P_LATENCY_X280_SAME_TILE_HPP
