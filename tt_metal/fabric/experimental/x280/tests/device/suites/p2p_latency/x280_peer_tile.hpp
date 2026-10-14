// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/p2p_latency/x280_peer_tile.hpp -- x280_peer_tile member of the p2p_latency suite.
//
// Two X280 harts on different L2CPU tiles exchange tokens through paired-range
// NoC windows. Their sessions run concurrently and use per-tile LIM.

#ifndef X280_SUITES_P2P_LATENCY_X280_PEER_TILE_HPP
#define X280_SUITES_P2P_LATENCY_X280_PEER_TILE_HPP

#include "harness/artifacts/artifacts.hpp"
#include "harness/execution/execution.hpp"
#include "harness/inventory/inventory.hpp"
#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "harness/regions/regions.hpp"
#include "harness/runner.hpp"
#include "tests/device/suites/p2p_latency/metal_peer.hpp"

#include "p2p_latency_frontend.h"
#include "p2p_latency_layout.h"
#include "p2p_latency_x280_noc_args.h"
#include "rt/x280_noc_map.h"

#include <gtest/gtest.h>
#include <tt-metalium/kernel_types.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace x280::suites::p2p_latency::x280_peer_tile {

namespace hart_ = x280::harness::artifacts;
namespace hexec = x280::harness::execution;
namespace hinv = x280::harness::inventory;
namespace hmeas = x280::harness::measurement;
namespace hout = x280::harness::output;
namespace hreg = x280::harness::regions;

constexpr uint32_t kDefaultWarmup = 8u;
constexpr uint32_t kDefaultSamples = 64u;
constexpr uint32_t kPayloadSize = 4u;
constexpr uint32_t kTlbIndex = X280_TLB_PAIRED_BASE;

constexpr uint64_t kWarmupBudgetCycles = 500u * 1000000ull;

using Axis = bool;  // a_initiates_first

// Per-tile LIM regions; each active-run request has its own args address.
struct TileLayout {
    hreg::Region args;
    hreg::Region mailbox;
    hreg::Region result;

    static TileLayout compute(x280::X280Tile& tile) {
        const auto avail = tile.available_lim();
        const hreg::Region w{avail.begin, avail.size()};
        hreg::Arena arena(w);
        TileLayout L{};
        L.args = arena.allocate(sizeof(p2p_latency_x280_noc_section_t));
        L.mailbox = arena.allocate(P2P_LATENCY_MAILBOX_BYTES);
        L.result = arena.allocate(sizeof(p2p_latency_result_t));
        return L;
    }
};

inline uint64_t mint_run_value(uint32_t& run_nonce) {
    static uint32_t sequence = 0;
    ++sequence;
    run_nonce = 0x200u + sequence;
    return (static_cast<uint64_t>(0xE11Cu) << 48) | (static_cast<uint64_t>(0xB2u) << 40) | run_nonce;
}

// Peer LIM addresses are already NoC-visible; peer_virtual names the destination
// in UMD TRANSLATED space.
inline p2p_latency_x280_noc_section_t make_section(
    p2p_latency_participant_t self,
    p2p_latency_direction_t first_direction,
    uint32_t W,
    uint32_t N,
    uint32_t run_nonce,
    const TileLayout& own,
    const TileLayout& peer,
    const x280::TranslatedCoord& peer_virtual) {
    return metal_peer::make_x280_args(
        {
            .self = static_cast<uint32_t>(self),
            .first_direction = static_cast<uint32_t>(first_direction),
            .payload_size = kPayloadSize,
            .warmup_count = W,
            .sample_count = N,
            .run_nonce = run_nonce,
            .warmup_max_poll_cycles = kWarmupBudgetCycles,
        },
        {
            .own_inbound_addr = own.mailbox.addr,
            .own_result_addr = own.result.addr,
            .peer_inbound_addr = peer.mailbox.addr,
            .peer_noc = peer_virtual.noc(),
            .peer_tlb_index = kTlbIndex,
        });
}

inline void run_axis(x280::harness::session::Fixture& f, const x280::harness::RunSpec& spec, bool a_initiates_first) {
    const auto pair = hinv::selected_tile_pair(*f.runtime, hart_::env_str("P2P_LATENCY_X280_PEER_TILE_PAIR"));
    if (!pair.has_value()) {
        GTEST_SKIP() << "x280_peer_tile needs two enabled L2CPU tiles on one chip";
    }
    const x280::TileId id_a = pair->first;
    const x280::TileId id_b = pair->second;
    if (id_a.chip_id != id_b.chip_id) {
        GTEST_SKIP() << "x280_peer_tile is one chip, but L2CPU " << id_a.hw_index << " is on chip " << id_a.chip_id
                     << " and L2CPU " << id_b.hw_index << " is on chip " << id_b.chip_id;
    }

    auto& tile_a = f.runtime->tile(id_a);
    auto& tile_b = f.runtime->tile(id_b);

    // This case does not release reset-held tiles.
    for (const auto& id : {id_a, id_b}) {
        const auto state = f.runtime->tile(id).inspect();
        ASSERT_TRUE(state.observation_valid) << state.invalid_reason;
        if (!state.resident_idle_live) {
            GTEST_SKIP() << "L2CPU " << id.hw_index
                         << " has no live resident idle; boot both tiles with "
                            "x280_runtime_cli --boot --tile "
                         << id_a.hw_index << ',' << id_b.hw_index;
        }
    }

    const p2p_latency_direction_t first_direction =
        a_initiates_first ? P2P_LATENCY_DIRECTION_A_TO_B : P2P_LATENCY_DIRECTION_B_TO_A;

    const uint32_t W = hart_::env_u32("P2P_LATENCY_X280_PEER_TILE_W", kDefaultWarmup);
    const uint32_t N = hart_::env_u32("P2P_LATENCY_X280_PEER_TILE_N", kDefaultSamples);
    ASSERT_GT(W, 0u);
    ASSERT_GT(N, 0u);
    ASSERT_LE(N, P2P_LATENCY_MAX_SAMPLES);

    uint32_t run_nonce = 0;
    const uint64_t run_value = mint_run_value(run_nonce);
    const uint64_t max_token = static_cast<uint64_t>(run_nonce) + 2ull * (static_cast<uint64_t>(W) + N) - 1ull;
    ASSERT_LE(max_token, 0xFFFFFFFFull) << "run_nonce + 2*(W+N)-1 exceeds u32";

    const TileLayout la = TileLayout::compute(tile_a);
    const TileLayout lb = TileLayout::compute(tile_b);

    // Zero whole lines before either peer performs partial-line NoC writes.
    hreg::zero_region(tile_a, la.mailbox);
    hreg::zero_region(tile_a, la.result);
    hreg::zero_region(tile_b, lb.mailbox);
    hreg::zero_region(tile_b, lb.result);

    const auto block_a = make_section(
        P2P_LATENCY_PARTICIPANT_A,
        first_direction,
        W,
        N,
        run_nonce,
        la,
        lb,
        f.runtime->virtual_core_from_tile(id_b, tt::tt_metal::NOC::NOC_0));
    const auto block_b = make_section(
        P2P_LATENCY_PARTICIPANT_B,
        first_direction,
        W,
        N,
        run_nonce,
        lb,
        la,
        f.runtime->virtual_core_from_tile(id_a, tt::tt_metal::NOC::NOC_0));
    hreg::write_args(tile_a, la.args, block_a);
    hreg::write_args(tile_b, lb.args, block_b);

    const std::string image = hart_::image_path("X280_P2P_LATENCY_X280_PEER_TILE_FW_IMAGE");
    const hexec::X280Launch launch_a{
        .image_path = image,
        .run_value = run_value,
        .args_addr = la.args.addr,
        .args_bytes = la.args.bytes,
    };
    const hexec::X280Launch launch_b{
        .image_path = image,
        .run_value = run_value,
        .args_addr = lb.args.addr,
        .args_bytes = lb.args.bytes,
    };

    // Both participants have to be running for either to make progress, so the
    // two sessions are enqueued before either is waited on.
    const auto handle_a = f.handle_for(id_a);
    const auto handle_b = f.handle_for(id_b);
    auto outcome_a = hexec::enqueue_x280(handle_a, launch_a);
    auto outcome_b = hexec::enqueue_x280(handle_b, launch_b);

    if (outcome_a.enqueued() && outcome_b.enqueued()) {
        hexec::wait_x280(handle_a, launch_a, outcome_a);
        hexec::wait_x280(handle_b, launch_b, outcome_b);
    }
    if (outcome_a.enqueued()) {
        hexec::cleanup_x280(handle_a, launch_a, outcome_a);
    }
    if (outcome_b.enqueued()) {
        hexec::cleanup_x280(handle_b, launch_b, outcome_b);
    }

    ASSERT_TRUE(outcome_a.enqueued()) << "enqueue rejected on L2CPU " << id_a.hw_index;
    ASSERT_TRUE(outcome_b.enqueued()) << "enqueue rejected on L2CPU " << id_b.hw_index;
    ASSERT_EQ(outcome_a.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome_a.wait);
    ASSERT_EQ(outcome_b.wait.outcome, x280::WaitOutcome::ResultReady) << x280::describe(outcome_b.wait);
    EXPECT_TRUE(outcome_a.cleaned_up()) << x280::describe(outcome_a.cleanup);
    EXPECT_TRUE(outcome_b.cleaned_up()) << x280::describe(outcome_b.cleanup);
    hexec::expect_window_cleared(tile_a, kTlbIndex);
    hexec::expect_window_cleared(tile_b, kTlbIndex);

    const auto result_a = hreg::read_struct<p2p_latency_result_t>(tile_a, la.result);
    const auto result_b = hreg::read_struct<p2p_latency_result_t>(tile_b, lb.result);

    EXPECT_EQ(result_a.backend_id, P2P_LATENCY_BACKEND_ID_X280_X280_PEER_TILE)
        << "participant A did not compile the X280 x280_peer_tile backend";
    EXPECT_EQ(result_b.backend_id, P2P_LATENCY_BACKEND_ID_X280_X280_PEER_TILE);
    EXPECT_EQ(result_a.terminal_state, static_cast<uint32_t>(P2P_LATENCY_STATE_DONE))
        << "participant A terminal state: " << metal_peer::state_name(result_a.terminal_state);
    EXPECT_EQ(result_b.terminal_state, static_cast<uint32_t>(P2P_LATENCY_STATE_DONE))
        << "participant B terminal state: " << metal_peer::state_name(result_b.terminal_state);

    // Preserve the observed rendezvous token so wrong and missing replies
    // remain distinguishable.
    EXPECT_EQ(result_a.path_sentinel, run_nonce)
        << "participant A observed " << result_a.path_sentinel << " at the rendezvous, expected " << run_nonce;
    EXPECT_EQ(result_b.path_sentinel, run_nonce)
        << "participant B observed " << result_b.path_sentinel << " at the rendezvous, expected " << run_nonce;

    ASSERT_NO_FATAL_FAILURE(metal_peer::validate_phases(result_a, P2P_LATENCY_PARTICIPANT_A, first_direction, N));
    ASSERT_NO_FATAL_FAILURE(metal_peer::validate_phases(result_b, P2P_LATENCY_PARTICIPANT_B, first_direction, N));

    const uint32_t hops_ab =
        hinv::hops(x280::Noc0Coord{id_a.noc_x, id_a.noc_y}, x280::Noc0Coord{id_b.noc_x, id_b.noc_y}, hinv::NocId::Noc0);
    const uint32_t hops_ba =
        hinv::hops(x280::Noc0Coord{id_b.noc_x, id_b.noc_y}, x280::Noc0Coord{id_a.noc_x, id_a.noc_y}, hinv::NocId::Noc0);

    std::printf(
        "\n[p2p_latency x280_peer_tile] L2CPU %d (%u,%u) <-> L2CPU %d (%u,%u) "
        "first_direction=%s W=%u N=%u payload=%u hops=%u/%u\n",
        id_a.hw_index,
        id_a.noc_x,
        id_a.noc_y,
        id_b.hw_index,
        id_b.noc_x,
        id_b.noc_y,
        metal_peer::direction_name(static_cast<uint32_t>(first_direction)),
        W,
        N,
        kPayloadSize,
        hops_ab,
        hops_ba);

    // Do not subtract values from separate rdcycle domains.
    auto report =
        [&](const p2p_latency_result_t& r, const char* who, int hw_index, const p2p_latency_x280_noc_section_t& block) {
            for (uint32_t p = 0; p < 2; ++p) {
                hout::write_jsonl(
                    spec.suite_label,
                    hout::record(metal_peer::phase_record(
                        "x280_peer_tile",
                        {{"participant", who},
                         {"hw_index", hw_index},
                         {"peer_hw_index", hw_index == id_a.hw_index ? id_b.hw_index : id_a.hw_index},
                         {"hops_a_to_b", hops_ab},
                         {"hops_b_to_a", hops_ba}},
                        r,
                        p,
                        block.common,
                        block.backend.peer_tlb_index,
                        P2P_LATENCY_X280_NOC_ORDERING,
                        std::nullopt,
                        x280::harness::measurement::WarmupSelection{.warmup = block.common.warmup_count})));
            }
        };
    report(result_a, "A", id_a.hw_index, block_a);
    report(result_b, "B", id_b.hw_index, block_b);

    for (uint32_t p = 0; p < 2; ++p) {
        for (const auto& [who, r] : {std::pair{"A", &result_a}, std::pair{"B", &result_b}}) {
            EXPECT_TRUE(metal_peer::phase_samples_nonzero(r->phases[p])) << who << " phase " << p;
        }
    }
}

}  // namespace x280::suites::p2p_latency::x280_peer_tile

#endif  // X280_SUITES_P2P_LATENCY_X280_PEER_TILE_HPP
