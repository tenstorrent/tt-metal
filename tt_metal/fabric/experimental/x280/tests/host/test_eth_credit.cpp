// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// tests/host/test_eth_credit.cpp -- pure host tests for the eth_credit ABI,
// header use, 2N-wrap pointers, layout, admission and argument packing.
//
// Expected values come from the eth_credit design ("the plan"), cited by section in
// each test.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "kernels/eth_credit/eth_credit_abi.h"
#include "kernels/eth_credit/eth_credit_header.hpp"
#include "tests/device/suites/eth_credit/regions.hpp"
#include "include/rt/measurement.h"
#include "rt/x280_noc_map.h"
#include "rt/x280_pinned_input.h"
#include "x280_address_map.h"
#include "x280/device/x280_state.hpp"
#include "tests/host/translated_coord_factory.hpp"

using namespace x280::harness::regions::eth_credit;
using x280::harness::regions::Window;
using x280::testing::TranslatedCoordFactory;

namespace {

constexpr uint32_t kKiB = 1024u;

// Plan §ABI and layout / Memory map: "the 359 KiB window".
constexpr uint64_t kEthWindowBytes = 359u * kKiB;
constexpr uint64_t kEthWindowBase = 0x1A000u;

// Plan §Header: a slot is the 96-byte header followed by the payload.
constexpr uint32_t Slot(uint32_t payload) { return 96u + payload; }

Capacity RealisticCapacity() {
    Capacity cap;
    cap.eth = Window{kEthWindowBase, kEthWindowBytes};
    cap.lim = Window{0x08100000u, 512u * kKiB};
    cap.x280_hz = 50000000u;     // X280 mtime
    cap.erisc_hz = 1000000000u;  // ERISC nominal
    return cap;
}

// A valid Config: L2CPU at the synthetic translated (8, 26 + c), range-valid but
// no real L2CPU tile, Ethernet tiles at translated Y = 25, stream ids 1..P per
// tile.
Config MakeConfig(uint32_t P, uint32_t n_link, uint32_t n_deliver, uint32_t payload, uint32_t batch, Direction dir) {
    Config cfg;
    cfg.shape = ec_shape_t{P, n_link, n_deliver, Slot(payload), batch};
    cfg.direction = dir;
    for (uint32_t p = 0; p < EC_MAX_P; ++p) {
        cfg.packets[p] = p < P ? 1000u + p : 0u;
    }
    for (uint32_t c = 0; c < kChips; ++c) {
        ChipPlacement& pl = cfg.chips[c];
        pl.x280_noc0 = TranslatedCoordFactory::make(8u, 26u + c);
        pl.x280_noc1 = TranslatedCoordFactory::make(8u, 26u + c);
        pl.eth = TranslatedCoordFactory::make(20u + c, 25u);
        for (uint32_t p = 0; p < EC_MAX_P; ++p) {
            pl.stream_ids[p] = 1u + p;
        }
    }
    return cfg;
}

// Absolute addresses of one tile's 6P hot control units, in unit order.
std::vector<uint32_t> hot_unit_addresses(const ChipLayout& l) {
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < l.eth.hot_units; ++i) {
        out.push_back(l.eth_ctrl_base + i * EC_UNIT_BYTES);
    }
    return out;
}

// The 6P hot unit addresses of one tile, from the named layout fields, in
// layout order. Plan §Memory map: E0 {wr_sh[p], crd[p], rd_src[p]}, E1
// {out_rd_sh[p], crd_src[p], out_wr_src[p]}.
std::vector<uint32_t> NamedHotUnits(const ChipLayout& l) {
    const ec_eth_layout_t& e = l.eth;
    std::vector<uint32_t> out;
    for (uint32_t base : {e.e0_wr_sh, e.e0_crd, e.e0_rd_src, e.e1_out_rd_sh, e.e1_crd_src, e.e1_out_wr_src}) {
        for (uint32_t p = 0; p < e.P; ++p) {
            out.push_back(l.eth_ctrl_base + base + p * 16u);
        }
    }
    return out;
}

// Plan §Memory map: the inferred interleave, restated here rather than taken
// from ec_eth_bank() so the test does not grade the mapping by itself.
uint32_t InferredBank(uint32_t addr) { return (addr >> 4) & 31u; }

}  // namespace

// ---------------------------------------------------------------------------
// Header (plan §ABI and layout / Header; production PacketHeaderBase in
// tt_metal/fabric/fabric_edm_packet_header.hpp: NocCommandFields is 40 bytes,
// then payload_size_bytes (uint16), noc_send_type (uint8), src_ch_id (uint8)).
// ---------------------------------------------------------------------------

TEST(EthCreditHeader, EachAccessorTouchesOnlyItsOwnBytes) {
    alignas(16) fs_hybrid_mesh_packet_header_t h;
    uint8_t* b = reinterpret_cast<uint8_t*>(&h);

    std::memset(&h, 0, sizeof(h));
    ec_header_set_seq(&h, 0x11223344u);
    for (uint32_t i = 0; i < 96u; ++i) {
        const uint8_t want = i == 8u ? 0x44u : i == 9u ? 0x33u : i == 10u ? 0x22u : i == 11u ? 0x11u : 0u;
        EXPECT_EQ(b[i], want) << "seq write, byte " << i;
    }
    EXPECT_EQ(ec_header_seq(&h), 0x11223344u);

    std::memset(&h, 0, sizeof(h));
    fs_set_payload_size_bytes(&h, 0xBEEFu);
    for (uint32_t i = 0; i < 96u; ++i) {
        const uint8_t want = i == 40u ? 0xEFu : i == 41u ? 0xBEu : 0u;
        EXPECT_EQ(b[i], want) << "payload_size_bytes write, byte " << i;
    }
    EXPECT_EQ(fs_payload_size_bytes(&h), 0xBEEFu);

    std::memset(&h, 0, sizeof(h));
    fs_set_src_ch_id(&h, 0x5Au);
    for (uint32_t i = 0; i < 96u; ++i) {
        EXPECT_EQ(b[i], i == 43u ? 0x5Au : 0u) << "src_ch_id write, byte " << i;
    }
    EXPECT_EQ(fs_src_ch_id(&h), 0x5Au);
}

// Plan §Header: fs_seed_unicast_header rewrites all 96 bytes, so src_ch_id and
// seq are written after it; everything else stays as the unicast seed made it.
TEST(EthCreditHeader, SeedIsTheUnicastSeedPlusSeqSrcAndPayload) {
    alignas(16) fs_hybrid_mesh_packet_header_t plain;
    alignas(16) fs_hybrid_mesh_packet_header_t seeded;
    std::memset(&plain, 0xA5, sizeof(plain));
    std::memset(&seeded, 0x5A, sizeof(seeded));

    fs_seed_unicast_header(&plain, 1u, 7u);
    ec_header_seed(&seeded, 1u, 7u, 3u, 4096u, 0xFFFFFFFEu);

    EXPECT_EQ(ec_header_seq(&seeded), 0xFFFFFFFEu);
    EXPECT_EQ(fs_src_ch_id(&seeded), 3u);
    EXPECT_EQ(fs_payload_size_bytes(&seeded), 4096u);

    const uint8_t* a = reinterpret_cast<const uint8_t*>(&plain);
    const uint8_t* s = reinterpret_cast<const uint8_t*>(&seeded);
    for (uint32_t i = 0; i < 96u; ++i) {
        const bool owned = (i >= 8u && i < 12u) || i == 40u || i == 41u || i == 43u;
        if (!owned) {
            EXPECT_EQ(s[i], a[i]) << "byte " << i << " differs from the plain unicast seed";
        }
    }
}

// ---------------------------------------------------------------------------
// Status word, result block, kernel ids (plan §Status word, §Result block).
// ---------------------------------------------------------------------------

TEST(EthCreditAbi, StatusWordIsOneUnitWithThePlannedFields) {
    EXPECT_EQ(sizeof(ec_status_t), 16u);
    EXPECT_EQ(offsetof(ec_status_t, state), 0u);
    EXPECT_EQ(offsetof(ec_status_t, origin), 4u);
    EXPECT_EQ(offsetof(ec_status_t, code), 6u);
    EXPECT_EQ(offsetof(ec_status_t, detail), 8u);
    EXPECT_EQ(sizeof(ec_status_t::origin), 2u);
    EXPECT_EQ(sizeof(ec_status_t::code), 2u);
    EXPECT_EQ(static_cast<uint32_t>(EC_STATE_RUNNING), 0u);
    EXPECT_EQ(static_cast<uint32_t>(EC_STATE_DONE), 1u);
    EXPECT_EQ(static_cast<uint32_t>(EC_STATE_FAILED), 2u);
    EXPECT_EQ(static_cast<uint32_t>(EC_ERROR_NONE), 0u);

    // Value 2 is unused: no kernel reports another kernel's failure.
    const std::set<uint32_t> codes{
        EC_ERROR_NONE,
        EC_ERROR_STALL_TIMEOUT,
        EC_ERROR_BAD_SEQUENCE,
        EC_ERROR_COUNTER_DISTANCE,
        EC_ERROR_LINK_DOWN,
        EC_ERROR_BAD_ARGS};
    EXPECT_EQ(codes.size(), 6u);
    EXPECT_EQ(codes.count(2u), 0u);
}

// Only a kernel that finished admits a measurement, and a failed one never
// reads as still running, whatever its code.
TEST(EthCreditAbi, OnlyDoneIsAMeasurementAndEveryFailureIsTerminal) {
    for (const ec_state_t state : {EC_STATE_RUNNING, EC_STATE_DONE, EC_STATE_FAILED}) {
        for (const ec_error_t code :
             {EC_ERROR_NONE,
              EC_ERROR_STALL_TIMEOUT,
              EC_ERROR_BAD_SEQUENCE,
              EC_ERROR_COUNTER_DISTANCE,
              EC_ERROR_LINK_DOWN,
              EC_ERROR_BAD_ARGS}) {
            const ubench_outcome_t outcome = ec_outcome(state, code);
            EXPECT_EQ(ubench_outcome_is_measurement(outcome), state == EC_STATE_DONE)
                << "state " << state << " code " << code;
            EXPECT_EQ(outcome == UBENCH_OUTCOME_IN_PROGRESS, state == EC_STATE_RUNNING)
                << "state " << state << " code " << code;
        }
    }
}

// eth_credit ABI migration spec §1: no schema field; every record begins
// {magic, kernel_id}; the result block keeps its other offsets and 256 bytes.
TEST(EthCreditAbi, ResultBlockHasFixedSizeAndHeaderFirst) {
    EXPECT_EQ(sizeof(ec_result_t), EC_RESULT_BYTES);
    EXPECT_EQ(EC_RESULT_BYTES, 256u);
    EXPECT_EQ(offsetof(ec_result_t, magic), 0u);
    EXPECT_EQ(offsetof(ec_result_t, kernel_id), 4u);
    EXPECT_EQ(offsetof(ec_result_t, first_batch_packets), 8u);
    EXPECT_EQ(offsetof(ec_result_t, status), 16u);
    EXPECT_EQ(offsetof(ec_result_t, packets_in), 32u);
    EXPECT_EQ(offsetof(ec_result_t, packets_out), 48u);
    EXPECT_EQ(offsetof(ec_result_t, stalls), 64u);
    EXPECT_EQ(offsetof(ec_result_t, first_ts), 160u);
    EXPECT_EQ(offsetof(ec_result_t, failure), 176u);
    EXPECT_EQ(EC_RESULT_BYTES % 16u, 0u);
}

// Spec (eth_credit_abi.h, MEASUREMENT.md checked subtraction): a stall interval
// is added under measurement_delta's rule, so a wrapped counter still counts and a
// half-range interval, or a total reaching half-range, leaves the total
// invalid for every later addition: measurement_delta's rule, checked against it.
TEST(EthCreditAbi, StallTotalsAddCheckedIntervalsAndKeepARefusal) {
    constexpr uint64_t kHalf = UINT64_C(1) << 63;
    EXPECT_EQ(ec_stall_add(10, 100, 130), 40u);
    EXPECT_EQ(ec_stall_add(0, UINT64_MAX - 4, 5), 10u);
    EXPECT_EQ(ec_stall_add(10, 0, kHalf), EC_STALL_TICKS_INVALID);
    EXPECT_EQ(ec_stall_add(10, 130, 100), EC_STALL_TICKS_INVALID);
    EXPECT_EQ(ec_stall_add(EC_STALL_TICKS_INVALID, 100, 130), EC_STALL_TICKS_INVALID);
    EXPECT_EQ(ec_stall_sum(kHalf - 2, 1), kHalf - 1);
    EXPECT_EQ(ec_stall_sum(kHalf - 1, 1), EC_STALL_TICKS_INVALID);
    EXPECT_EQ(ec_stall_sum(5, EC_STALL_TICKS_INVALID), EC_STALL_TICKS_INVALID);
    for (const auto& [start, end] :
         {std::pair<uint64_t, uint64_t>{0, kHalf - 1},
          {0, kHalf},
          {100, 99},
          {UINT64_MAX, kHalf - 2},
          {UINT64_MAX, kHalf - 1},
          {7, 7}}) {
        uint64_t delta = 0;
        const bool accepted = measurement_delta(start, end, &delta) != 0;
        EXPECT_EQ(ec_stall_add(0, start, end), accepted ? delta : EC_STALL_TICKS_INVALID) << start << ".." << end;
    }
}

// Spec §1, the ec_x280_fwd_args_t offset table.
TEST(EthCreditAbi, ForwarderRecordOffsetsMatchTheTable) {
    EXPECT_EQ(sizeof(ec_x280_fwd_args_t), 128u);
    EXPECT_EQ(EC_X280_FWD_ARGS_BYTES, 128u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, magic), 0u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, kernel_id), 4u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, flags), 8u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, p), 12u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, shape), 16u);
    EXPECT_EQ(sizeof(ec_shape_t), 20u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, director_p), 36u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, run_nonce), 40u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, poll_cadence), 44u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, send_count), 48u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, recv_count), 52u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, dst_chip), 56u);
    EXPECT_EQ(sizeof(ec_x280_fwd_args_t::dst_chip), 2u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, dst_mesh), 58u);
    EXPECT_EQ(sizeof(ec_x280_fwd_args_t::dst_mesh), 2u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, tlb_base), 60u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, stall_budget_ticks), 64u);
    EXPECT_EQ(sizeof(ec_x280_fwd_args_t::stall_budget_ticks), 8u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, lim_base), 72u);
    EXPECT_EQ(sizeof(ec_x280_fwd_args_t::lim_base), 8u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, eth), 80u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, eth_ctrl_base), 88u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, lim_end), 96u);
    EXPECT_EQ(offsetof(ec_x280_fwd_args_t, reserved), 104u);
    EXPECT_EQ(sizeof(ec_x280_fwd_args_t::reserved), 6u * 4u);
}

// Plan §Status word: ids X280_FWD_p, E0, E1 plus a chip bit, so a
// failure's origin names which chip's core failed.
TEST(EthCreditAbi, KernelIdsRoundTripAndAreDistinctAcrossBothChips) {
    std::set<uint32_t> ids;
    uint32_t expected = 0;
    for (uint32_t chip = 0; chip < 2u; ++chip) {
        const auto add = [&](uint32_t role, uint32_t index) {
            const uint32_t id = EC_KERNEL_ID(role, index, chip);
            EXPECT_NE(id, 0u);
            EXPECT_LE(id, 0xFFFFu) << "origin is a uint16";
            EXPECT_EQ(ec_kernel_role(id), role);
            EXPECT_EQ(ec_kernel_index(id), index);
            EXPECT_EQ(ec_kernel_chip(id), chip);
            ids.insert(id);
            ++expected;
        };
        add(EC_ROLE_E0, 0u);
        add(EC_ROLE_E1, 0u);
        for (uint32_t p = 0; p < EC_MAX_P; ++p) {
            add(EC_ROLE_X280_FWD, p);
        }
    }
    EXPECT_EQ(ids.size(), expected);
}

// Cost sites (eth_credit_abi.h, ec_site_is_cost contract): exactly
// FWD_W1_STORE, E0_TXQ0_BUSY, E1_TXQ1_BUSY and E1_NOC_BARRIER run right after
// their own issue, so their count is batches, not stalls, and the host leaves
// them out of the "stalls least" comparison. Every enumerator is checked by
// name so a site dropped from, or added to, the predicate fails here. A value
// outside ec_site_t is not a cost site, including the first value past the enum
// and the ends of the byte and word ranges.
TEST(EthCreditSites, CostSitesAreExactlyTheFourPostIssueWaits) {
    const std::map<uint32_t, bool> expected{
        {EC_SITE_NONE, false},
        {EC_SITE_FWD_OUT_FULL, false},
        {EC_SITE_FWD_W1_STORE, true},
        {EC_SITE_FWD_IN_EMPTY, false},
        {EC_SITE_E0_INPUTS_EMPTY, false},
        {EC_SITE_E0_LINK_CREDITS, false},
        {EC_SITE_E0_TXQ0_BUSY, true},
        {EC_SITE_E1_LINK_EMPTY, false},
        {EC_SITE_E1_IN_FULL, false},
        {EC_SITE_E1_NOC_BARRIER, true},
        {EC_SITE_E1_TXQ1_BUSY, true},
        {EC_SITE_BARRIER, false},
        {EC_SITE_HANDSHAKE, false},
        {EC_SITE_DRAIN, false},
        {EC_SITE_E0_R3_SOURCE, false},
    };
    ASSERT_EQ(expected.size(), 15u) << "every ec_site_t enumerator, 0 through EC_SITE_E0_R3_SOURCE";
    ASSERT_EQ(static_cast<uint32_t>(EC_SITE_E0_R3_SOURCE), 14u);
    uint32_t cost = 0;
    for (const auto& [site, is_cost] : expected) {
        EXPECT_EQ(ec_site_is_cost(site) != 0, is_cost) << "site " << site;
        cost += is_cost ? 1u : 0u;
    }
    EXPECT_EQ(cost, 4u);
    for (const uint32_t site : {15u, 16u, 255u, 256u + EC_SITE_FWD_W1_STORE, 0x80000000u, 0xFFFFFFFFu}) {
        EXPECT_EQ(ec_site_is_cost(site), 0) << "site " << site;
    }
}

// ---------------------------------------------------------------------------
// 2N-wrap pointers (plan §Protocol / Pointers: slot = x & (N-1),
// used = (wr - rd) mod 2N, empty = rd == wr, full = used == N, and every reader
// checks (wr - rd) mod 2N <= depth).
// ---------------------------------------------------------------------------

// A producer and consumer driven at random against an unbounded 64-bit model.
// The inputs cross the 2N boundary many times (asserted), so a pointer that
// wraps at N, or at 2^32, would disagree with the model.
TEST(EthCreditPointers, AgreeWithAnUnboundedModelAcrossManyWraps) {
    std::mt19937 rng(0xEC0FFEEu);
    for (uint32_t n : {1u, 2u, 4u, 8u, 64u, 1024u}) {
        uint64_t W = 0;
        uint64_t R = 0;
        uint32_t wr = 0;
        uint32_t rd = 0;
        uint32_t fulls = 0;
        uint32_t empties = 0;
        for (uint32_t step = 0; step < 200000u; ++step) {
            const uint64_t used = W - R;
            ASSERT_EQ(ec_ptr_used(wr, rd, n), used) << "n " << n << " step " << step;
            ASSERT_EQ(ec_ptr_credit(wr, rd, n), n - used) << "n " << n << " step " << step;
            ASSERT_EQ(ec_ptr_empty(wr, rd) != 0, used == 0u) << "n " << n << " step " << step;
            ASSERT_EQ(ec_ptr_full(wr, rd, n) != 0, used == n) << "n " << n << " step " << step;
            ASSERT_TRUE(ec_ptr_distance_ok(wr, rd, n)) << "n " << n << " step " << step;
            ASSERT_LT(wr, 2u * n);
            ASSERT_LT(rd, 2u * n);
            ASSERT_EQ(ec_ptr_slot(wr, n), W % n);
            ASSERT_EQ(ec_ptr_slot(rd, n), R % n);
            fulls += used == n ? 1u : 0u;
            empties += used == 0u ? 1u : 0u;

            // Bias towards the extremes so full and empty are both visited.
            const uint32_t roll = rng() % 4u;
            if (roll < 2u) {
                const uint32_t credit = static_cast<uint32_t>(n - used);
                const uint32_t k = roll == 0u ? credit : (credit == 0u ? 0u : rng() % (credit + 1u));
                wr = ec_ptr_advance(wr, k, n);
                W += k;
            } else {
                const uint32_t avail = static_cast<uint32_t>(used);
                const uint32_t k = roll == 2u ? avail : (avail == 0u ? 0u : rng() % (avail + 1u));
                rd = ec_ptr_advance(rd, k, n);
                R += k;
            }
        }
        EXPECT_GE(W / (2u * n), 50u) << "n " << n << ": too few 2N wraps to test anything";
        EXPECT_GT(fulls, 0u) << "n " << n;
        EXPECT_GT(empties, 0u) << "n " << n;
    }
}

// Plan §Pointers: a violation of (wr - rd) mod 2N <= depth is COUNTER_DISTANCE.
TEST(EthCreditPointers, DistanceRejectsOverrunAndOutOfRangePointers) {
    for (uint32_t n : {1u, 2u, 8u, 1024u}) {
        for (uint32_t rd = 0; rd < 2u * n; ++rd) {
            for (uint32_t over = n + 1u; over < 2u * n; ++over) {
                EXPECT_FALSE(ec_ptr_distance_ok(ec_ptr_advance(rd, over, n), rd, n))
                    << "n " << n << " rd " << rd << " used " << over;
            }
        }
        EXPECT_FALSE(ec_ptr_distance_ok(2u * n, 0u, n)) << "n " << n;
        EXPECT_FALSE(ec_ptr_distance_ok(0u, 2u * n, n)) << "n " << n;
        EXPECT_FALSE(ec_ptr_distance_ok(0xFFFFFFFFu, 0xFFFFFFFFu, n)) << "n " << n;
    }
}

// ---------------------------------------------------------------------------
// E0 idle site (audit C5, ec_e0_idle_site contract): a pass that sends nothing
// is charged to EC_SITE_E0_LINK_CREDITS when its link credit is zero, and to
// EC_SITE_E0_INPUTS_EMPTY otherwise. Credit is tested first because the
// forwarder refills tx_fifo only as credit returns, so zero credit implies
// zero input; an input-first test never reaches LINK_CREDITS.
// ---------------------------------------------------------------------------

// C5 ordering rationale: in every consistent state (crd <= sent <= wr in
// unbounded order, used(wr, crd) <= N because the forwarder fills only slots
// credit has freed), credit == 0 implies avail == 0, and an idle pass is
// charged to LINK_CREDITS exactly when credit is 0. Enumerates every crd
// phase, outstanding a = used(sent, crd) in [0, N] and filled-unsent
// b = used(wr, sent) with a + b <= N.
TEST(EthCreditE0IdleSite, ZeroCreditImpliesEmptyInputAndIdleSiteFollowsCredit) {
    for (uint32_t n : {1u, 2u, 4u, 8u, 64u}) {
        uint32_t link_credit_states = 0;
        uint32_t input_empty_states = 0;
        for (uint32_t crd = 0; crd < 2u * n; ++crd) {
            for (uint32_t a = 0; a <= n; ++a) {
                for (uint32_t b = 0; a + b <= n; ++b) {
                    const uint32_t sent = ec_ptr_advance(crd, a, n);
                    const uint32_t wr = ec_ptr_advance(sent, b, n);
                    ASSERT_LE(ec_ptr_used(wr, crd, n), n);
                    const uint32_t avail = ec_ptr_used(wr, sent, n);
                    const uint32_t credit = ec_ptr_credit(sent, crd, n);
                    ASSERT_EQ(avail, b) << "n " << n << " crd " << crd << " a " << a << " b " << b;
                    ASSERT_EQ(credit, n - a) << "n " << n << " crd " << crd << " a " << a << " b " << b;
                    if (credit == 0u) {
                        ASSERT_EQ(avail, 0u) << "n " << n << " crd " << crd << ": zero credit with input";
                    }
                    if (std::min(avail, credit) == 0u) {
                        const uint32_t want = credit == 0u ? EC_SITE_E0_LINK_CREDITS : EC_SITE_E0_INPUTS_EMPTY;
                        ASSERT_EQ(ec_e0_idle_site(credit), want)
                            << "n " << n << " crd " << crd << " a " << a << " b " << b;
                        (credit == 0u ? link_credit_states : input_empty_states) += 1u;
                    }
                }
            }
        }
        EXPECT_EQ(link_credit_states, 2u * n) << "n " << n << ": one exhausted state per crd phase";
        EXPECT_EQ(input_empty_states, 2u * n * n) << "n " << n;
    }
}

// ---------------------------------------------------------------------------
// Shape (plan §Arguments: 1 <= BATCH <= min(N_LINK, N_DELIVER); §Pointers:
// power-of-two depths; §Header: SLOT_SIZE = 96 + 16k; §Forwarders: P <= 4).
// ---------------------------------------------------------------------------

TEST(EthCreditShape, BatchBoundsAreOneThroughMinOfTheDepths) {
    struct Case {
        uint32_t n_link, n_deliver, batch;
        bool ok;
    };
    const Case cases[] = {
        {8u, 8u, 0u, false},
        {8u, 8u, 1u, true},
        {8u, 8u, 8u, true},
        {8u, 8u, 9u, false},
        {8u, 4u, 4u, true},
        {8u, 4u, 5u, false},
        {4u, 16u, 4u, true},
        {4u, 16u, 5u, false},
        {1u, 1u, 1u, true},
        {1u, 1u, 2u, false},
    };
    for (const Case& c : cases) {
        const ec_shape_t s{2u, c.n_link, c.n_deliver, Slot(64u), c.batch};
        const ec_check_t k = ec_validate_shape(&s);
        if (c.ok) {
            EXPECT_EQ(k.code, EC_CHECK_OK) << c.n_link << "/" << c.n_deliver << " batch " << c.batch;
        } else {
            EXPECT_EQ(k.code, EC_CHECK_BAD_PARAM) << c.n_link << "/" << c.n_deliver << " batch " << c.batch;
            EXPECT_EQ(k.constraint, EC_CONSTRAINT_BATCH);
            EXPECT_EQ(k.value, c.batch);
            EXPECT_EQ(k.bound, std::min(c.n_link, c.n_deliver));
        }

        // Admission applies the same bound.
        const Config cfg = MakeConfig(2u, c.n_link, c.n_deliver, 64u, c.batch, Direction::AToB);
        const Admission adm = admit(cfg, RealisticCapacity());
        EXPECT_EQ(adm.ok(), c.ok) << adm.describe();
        if (!c.ok) {
            EXPECT_EQ(adm.check.constraint, EC_CONSTRAINT_BATCH);
        }
    }
}

TEST(EthCreditShape, EachDomainViolationNamesItsConstraint) {
    struct Case {
        const char* name;
        ec_shape_t s;
        uint32_t constraint;
    };
    const Case cases[] = {
        {"P = 0", {0u, 8u, 8u, 96u, 1u}, EC_CONSTRAINT_P},
        {"P = 5", {EC_MAX_P + 1u, 8u, 8u, 96u, 1u}, EC_CONSTRAINT_P},
        {"N_LINK = 0", {1u, 0u, 8u, 96u, 1u}, EC_CONSTRAINT_N_LINK},
        {"N_LINK = 3", {1u, 3u, 8u, 96u, 1u}, EC_CONSTRAINT_N_LINK},
        {"N_LINK = 12", {1u, 12u, 8u, 96u, 1u}, EC_CONSTRAINT_N_LINK},
        {"N_LINK > max", {1u, 2u * EC_MAX_DEPTH, 8u, 96u, 1u}, EC_CONSTRAINT_N_LINK},
        {"N_DELIVER = 6", {1u, 8u, 6u, 96u, 1u}, EC_CONSTRAINT_N_DELIVER},
        {"N_DELIVER > max", {1u, 8u, 2u * EC_MAX_DEPTH, 96u, 1u}, EC_CONSTRAINT_N_DELIVER},
        {"SLOT = 0", {1u, 8u, 8u, 0u, 1u}, EC_CONSTRAINT_SLOT_SIZE},
        {"SLOT = 95", {1u, 8u, 8u, 95u, 1u}, EC_CONSTRAINT_SLOT_SIZE},
        {"SLOT = 104", {1u, 8u, 8u, 104u, 1u}, EC_CONSTRAINT_SLOT_SIZE},
        {"SLOT = 97", {1u, 8u, 8u, 97u, 1u}, EC_CONSTRAINT_SLOT_SIZE},
        {"SLOT > max", {1u, 8u, 8u, EC_MAX_SLOT_BYTES + 16u, 1u}, EC_CONSTRAINT_SLOT_SIZE},
    };
    for (const Case& c : cases) {
        const ec_check_t k = ec_validate_shape(&c.s);
        EXPECT_EQ(k.code, EC_CHECK_BAD_PARAM) << c.name;
        EXPECT_EQ(k.constraint, c.constraint) << c.name;
    }

    // The valid side of each boundary.
    const ec_shape_t ok[] = {
        {1u, 1u, 1u, 96u, 1u},
        {EC_MAX_P, EC_MAX_DEPTH, EC_MAX_DEPTH, EC_MAX_SLOT_BYTES, EC_MAX_DEPTH},
    };
    for (const ec_shape_t& s : ok) {
        EXPECT_EQ(ec_validate_shape(&s).code, EC_CHECK_OK)
            << "P " << s.P << " N_LINK " << s.n_link << " N_DELIVER " << s.n_deliver << " slot " << s.slot_size;
    }
}

TEST(EthCreditShape, RejectedShapeLeavesLayoutOutputUntouched) {
    const ec_shape_t bad{1u, 8u, 8u, 96u, 0u};
    ec_eth_layout_t eth;
    ec_eth_layout_t poison;
    std::memset(&eth, 0xA5, sizeof(eth));
    std::memset(&poison, 0xA5, sizeof(poison));
    EXPECT_NE(ec_compute_eth_layout(&bad, 1, 1, &eth).code, EC_CHECK_OK);
    EXPECT_EQ(std::memcmp(&eth, &poison, sizeof(eth)), 0);

    ec_x280_layout_t lim;
    ec_x280_layout_t lpoison;
    std::memset(&lim, 0xA5, sizeof(lim));
    std::memset(&lpoison, 0xA5, sizeof(lpoison));
    EXPECT_NE(ec_compute_x280_layout(&bad, &lim).code, EC_CHECK_OK);
    EXPECT_EQ(std::memcmp(&lim, &lpoison, sizeof(lim)), 0);
}

// ---------------------------------------------------------------------------
// Layout (plan §ABI and layout / Memory map, §Protocol / Layout).
// ---------------------------------------------------------------------------

TEST(EthCreditLayout, ControlAreaHasHotUnitsFirstAndSixPOfThem) {
    for (uint32_t P = 1; P <= EC_MAX_P; ++P) {
        const ec_shape_t s{P, 8u, 8u, Slot(256u), 1u};
        ec_eth_layout_t e{};
        ASSERT_EQ(ec_compute_eth_layout(&s, 1, 1, &e).code, EC_CHECK_OK);
        EXPECT_EQ(e.hot_units, 6u * P);
        const uint32_t hot_end = e.hot_units * 16u;

        ChipLayout l;
        l.eth = e;
        l.eth_ctrl_base = 0;
        std::vector<uint32_t> hot = NamedHotUnits(l);
        ASSERT_EQ(hot.size(), 6u * P);
        // Hot units are packed in the order E0 {wr_sh, crd, rd_src}, E1
        // {out_rd_sh, crd_src, out_wr_src}, each P units wide.
        for (uint32_t i = 0; i < hot.size(); ++i) {
            EXPECT_EQ(hot[i], i * 16u) << "P " << P << ": hot unit " << i << " out of order";
        }
        std::set<uint32_t> unique(hot.begin(), hot.end());
        EXPECT_EQ(unique.size(), hot.size()) << "P " << P << ": two hot units share an address";
        for (uint32_t a : hot) {
            EXPECT_EQ(a % 16u, 0u);
            EXPECT_LT(a, hot_end) << "P " << P << ": named hot unit outside the hot prefix";
        }
        for (uint32_t cold : {e.e0_release, e.e1_release, e.handshake, e.e0_result, e.e1_result}) {
            EXPECT_GE(cold, hot_end) << "P " << P << ": cold unit inside the hot prefix";
            EXPECT_EQ(cold % 16u, 0u);
        }
        // Cold units follow in the order e0_release, e1_release, handshake,
        // e0_result, e1_result, then control_bytes.
        EXPECT_LT(e.e0_release, e.e1_release);
        EXPECT_LT(e.e1_release, e.handshake);
        EXPECT_LE(e.handshake + EC_HANDSHAKE_BYTES, e.e0_result);
        EXPECT_LE(e.e0_result + EC_RESULT_BYTES, e.e1_result);
        EXPECT_LE(e.e1_result + EC_RESULT_BYTES, e.control_bytes);
        EXPECT_GE(e.e0_result + EC_RESULT_BYTES, e.e0_result);
        EXPECT_LE(e.e0_result + EC_RESULT_BYTES, e.control_bytes);
        EXPECT_LE(e.e1_result + EC_RESULT_BYTES, e.control_bytes);
        EXPECT_TRUE(e.e1_result >= e.e0_result + EC_RESULT_BYTES || e.e0_result >= e.e1_result + EC_RESULT_BYTES);
        EXPECT_LE(e.handshake + EC_HANDSHAKE_BYTES, e.control_bytes);
        EXPECT_EQ(e.control_bytes % 16u, 0u);
        EXPECT_GE(e.tx_offset, e.control_bytes);
    }
}

TEST(EthCreditLayout, ControlBaseIs512AlignedAndSlotsAreSizedToNeed) {
    const uint32_t P = 3u;
    const uint32_t N = 8u;
    const uint32_t S = Slot(1024u);
    Capacity cap = RealisticCapacity();
    cap.eth.base = 0x1A010u;  // not 512-aligned

    for (Direction dir : {Direction::AToB, Direction::BToA, Direction::Both}) {
        const Config cfg = MakeConfig(P, N, 8u, 1024u, 1u, dir);
        const Admission adm = admit(cfg, cap);
        ASSERT_TRUE(adm.ok()) << adm.describe();
        for (uint32_t c = 0; c < kChips; ++c) {
            const ChipLayout& l = adm.layout.chips[c];
            EXPECT_EQ(l.outbound, outbound(dir, c));
            EXPECT_EQ(l.inbound, inbound(dir, c));
            EXPECT_EQ(EC_ETH_CONTROL_ALIGN, 512u);
            EXPECT_EQ(l.eth_ctrl_base % 512u, 0u) << "chip " << c;
            EXPECT_GE(l.eth_ctrl_base, cap.eth.base);
            EXPECT_LT(l.eth_ctrl_base - cap.eth.base, EC_ETH_CONTROL_ALIGN);
            EXPECT_EQ(l.eth.tx_bytes, l.outbound ? P * N * S : 0u) << "chip " << c;
            EXPECT_EQ(l.eth.rx_bytes, l.inbound ? P * N * S : 0u) << "chip " << c;
            EXPECT_LE(l.eth_ctrl_base + l.eth.total_bytes, cap.eth.base + cap.eth.size);
            EXPECT_GE(l.eth.total_bytes, l.eth.control_bytes + l.eth.tx_bytes + l.eth.rx_bytes);
            for (uint32_t p = 0; p < P; ++p) {
                for (uint32_t k = 0; k < N; ++k) {
                    EXPECT_EQ(l.tx_slot(p, k), l.eth_ctrl_base + l.eth.tx_offset + (p * N + k) * S);
                    EXPECT_EQ(l.rx_slot(p, k), l.eth_ctrl_base + l.eth.rx_offset + (p * N + k) * S);
                }
            }
            if (l.outbound && l.inbound) {
                // E0's and E1's slot regions follow one another without overlap.
                EXPECT_TRUE(
                    l.eth.rx_offset >= l.eth.tx_offset + l.eth.tx_bytes ||
                    l.eth.tx_offset >= l.eth.rx_offset + l.eth.rx_bytes);
            }
        }
        // Exactly one direction per chip in a one-way run.
        EXPECT_EQ(outbound(dir, 0u), inbound(dir, 1u));
        EXPECT_EQ(outbound(dir, 1u), inbound(dir, 0u));
    }
    EXPECT_TRUE(outbound(Direction::AToB, 0u));
    EXPECT_FALSE(outbound(Direction::AToB, 1u));
    EXPECT_TRUE(outbound(Direction::BToA, 1u));
    EXPECT_FALSE(outbound(Direction::BToA, 0u));
}

// Plan §Memory map: each forwarder has its own LIM area; X280 FIFOs live in LIM;
// in[p] is N_DELIVER x 96 B.
TEST(EthCreditLayout, ForwarderLimAreasAreDisjointAndInsideTheWindow) {
    const Capacity cap = RealisticCapacity();
    const Config cfg = MakeConfig(4u, 8u, 16u, 256u, 1u, Direction::Both);
    const Admission adm = admit(cfg, cap);
    ASSERT_TRUE(adm.ok()) << adm.describe();
    for (uint32_t c = 0; c < kChips; ++c) {
        const ChipLayout& l = adm.layout.chips[c];
        EXPECT_GE(l.lim_base, cap.lim.base);
        EXPECT_LE(l.lim_base + l.x280.total_bytes, cap.lim.base + cap.lim.size);
        EXPECT_GE(l.x280.args_bytes, EC_X280_HARTS * sizeof(ec_x280_fwd_args_t));
        EXPECT_GE(l.x280.area_bytes, l.x280.in_slots + 16u * EC_HEADER_BYTES);
        EXPECT_GE(l.x280.in_slots, l.x280.staging + EC_HEADER_BYTES);
        for (uint32_t p = 0; p + 1u < cfg.shape.P; ++p) {
            EXPECT_GE(l.fwd_area(p + 1u), l.fwd_area(p) + l.x280.area_bytes);
        }
        EXPECT_GE(l.fwd_area(0), l.lim_base + l.x280.ready_offset + l.x280.ready_bytes);
        EXPECT_GE(l.x280.ready_offset, l.x280.args_offset + l.x280.args_bytes);
        EXPECT_LE(l.fwd_area(cfg.shape.P - 1u) + l.x280.area_bytes, l.lim_base + l.x280.total_bytes);
    }
}

// Plan §Memory map: a forwarder area holds, in order, rd_sh, wr_src, wr_sh,
// rd_src, seq_expected, release, result, staging, in_slots; there are no
// status units from E0 or E1.
TEST(EthCreditLayout, ForwarderAreaUnitsAreInPlannedOrder) {
    for (uint32_t P = 1; P <= EC_MAX_P; ++P) {
        const Admission adm = admit(MakeConfig(P, 8u, 16u, 256u, 1u, Direction::Both), RealisticCapacity());
        ASSERT_TRUE(adm.ok()) << adm.describe();
        const ec_x280_layout_t& x = adm.layout.chips[0].x280;
        const uint32_t units[] = {x.rd_sh, x.wr_src, x.wr_sh, x.rd_src, x.seq_expected, x.release, x.result};
        for (uint32_t i = 0; i < std::size(units); ++i) {
            EXPECT_EQ(units[i] % 16u, 0u) << "P " << P << " unit " << i;
            if (i > 0u) {
                EXPECT_GE(units[i], units[i - 1u] + 16u) << "P " << P << " unit " << i << " out of order";
            }
        }
        EXPECT_GE(x.staging, x.result + EC_RESULT_BYTES) << "P " << P;
        EXPECT_GE(x.in_slots, x.staging + EC_HEADER_BYTES) << "P " << P;
        EXPECT_LE(x.in_slots + 16u * EC_HEADER_BYTES, x.area_bytes) << "P " << P;
    }
}

// ---------------------------------------------------------------------------
// Bank placement. Tied to the INFERRED interleave bank = (addr >> 4) & 31
// (plan §Memory map: inferred from the Wormhole Tensix L1 MAS and the Neo L1
// spec's low-order interleave, not documented for Blackhole Ethernet). If the
// real Blackhole Ethernet mapping differs, this test says nothing about it.
// ---------------------------------------------------------------------------

TEST(EthCreditBanks, EveryHotUnitHasItsOwnBankUpToPFour) {
    Capacity cap = RealisticCapacity();
    for (uint64_t base : {kEthWindowBase, uint64_t{0x1A010u}, uint64_t{0x1A1F0u}}) {
        cap.eth.base = base;
        for (uint32_t P = 1; P <= EC_MAX_P; ++P) {
            const Admission adm = admit(MakeConfig(P, 8u, 8u, 1024u, 1u, Direction::Both), cap);
            ASSERT_TRUE(adm.ok()) << adm.describe();
            for (uint32_t c = 0; c < kChips; ++c) {
                const ChipLayout& l = adm.layout.chips[c];
                const std::vector<uint32_t> hot = NamedHotUnits(l);
                ASSERT_EQ(hot.size(), 6u * P);
                std::set<uint32_t> banks;
                for (uint32_t a : hot) {
                    EXPECT_EQ(ec_eth_bank(a), InferredBank(a));
                    banks.insert(InferredBank(a));
                }
                EXPECT_EQ(banks.size(), hot.size()) << "P " << P << " base " << base << " chip " << c;

                // hot_unit_addresses() names the same set.
                std::vector<uint32_t> listed = hot_unit_addresses(l);
                std::vector<uint32_t> named = hot;
                std::sort(listed.begin(), listed.end());
                std::sort(named.begin(), named.end());
                EXPECT_EQ(listed, named);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fit (plan §Memory map: tile slots take 2 * P * N_LINK * SLOT_SIZE; at P = 4,
// N_LINK = 8 that is 64 slots, ~262 KiB at 4 KiB and ~134 KiB at 2 KiB, both
// within 359 KiB; N_LINK = 16 at 4 KiB needs ~524 KiB and does not fit).
// ---------------------------------------------------------------------------

TEST(EthCreditFit, PlanExampleNLinkSixteenAtFourKiBIsRefusedWithBytesOver) {
    const Config cfg = MakeConfig(4u, 16u, 8u, 4096u, 1u, Direction::Both);
    const Admission adm = admit(cfg, RealisticCapacity());
    ASSERT_FALSE(adm.ok());
    EXPECT_EQ(adm.check.code, EC_CHECK_CAPACITY);
    EXPECT_EQ(adm.check.constraint, EC_CONSTRAINT_ETH_WINDOW);
    EXPECT_EQ(adm.check.bound, kEthWindowBytes);

    ec_eth_layout_t e{};
    ASSERT_EQ(ec_compute_eth_layout(&cfg.shape, 1, 1, &e).code, EC_CHECK_OK);
    EXPECT_EQ(e.tx_bytes + e.rx_bytes, 128u * Slot(4096u));
    EXPECT_EQ((e.tx_bytes + e.rx_bytes) / kKiB, 524u);
    // Base is 512-aligned, so the requirement is exactly the tile's total.
    EXPECT_EQ(adm.check.value, e.total_bytes);
    EXPECT_GE(adm.check.value - adm.check.bound, 128u * Slot(4096u) - kEthWindowBytes);
    EXPECT_NE(adm.describe().find(std::to_string(adm.check.value - adm.check.bound) + " bytes over"), std::string::npos)
        << adm.describe();

    // The same shape one-way needs only one FIFO set per tile, ~262 KiB.
    const Admission one = admit(MakeConfig(4u, 16u, 8u, 4096u, 1u, Direction::AToB), RealisticCapacity());
    EXPECT_TRUE(one.ok()) << one.describe();
}

// Each window is checked on its own and names itself; at the exact requirement
// it fits and one byte short it is refused one byte over.
TEST(EthCreditFit, EachWindowReportsItselfAndExactBytesOver) {
    const Config cfg = MakeConfig(4u, 8u, 8u, 2048u, 2u, Direction::Both);
    const Capacity roomy = RealisticCapacity();
    const Admission ref = admit(cfg, roomy);
    ASSERT_TRUE(ref.ok()) << ref.describe();

    // Requirements from the ABI's own layout functions, at the aligned bases.
    ec_eth_layout_t e{};
    ec_x280_layout_t x{};
    ASSERT_EQ(ec_compute_eth_layout(&cfg.shape, 1, 1, &e).code, EC_CHECK_OK);
    ASSERT_EQ(ec_compute_x280_layout(&cfg.shape, &x).code, EC_CHECK_OK);

    struct Case {
        const char* name;
        Window Capacity::* window;
        uint64_t need;
        uint32_t constraint;
    };
    const Case cases[] = {
        {"eth", &Capacity::eth, e.total_bytes, EC_CONSTRAINT_ETH_WINDOW},
        {"lim", &Capacity::lim, x.total_bytes, EC_CONSTRAINT_LIM_WINDOW},
    };
    for (const Case& c : cases) {
        Capacity cap = roomy;
        (cap.*c.window).size = c.need;
        const Admission exact = admit(cfg, cap);
        EXPECT_TRUE(exact.ok()) << c.name << ": " << exact.describe();

        (cap.*c.window).size = c.need - 1u;
        const Admission shy = admit(cfg, cap);
        ASSERT_FALSE(shy.ok()) << c.name;
        EXPECT_EQ(shy.check.code, EC_CHECK_CAPACITY) << c.name;
        EXPECT_EQ(shy.check.constraint, c.constraint) << c.name;
        EXPECT_EQ(shy.check.value, c.need) << c.name;
        EXPECT_EQ(shy.check.bound, c.need - 1u) << c.name;
        EXPECT_NE(shy.describe().find("1 bytes over"), std::string::npos) << shy.describe();
    }

    // A base that needs rounding costs its padding against the window.
    Capacity cap = roomy;
    cap.eth.base = kEthWindowBase + 16u;
    cap.eth.size = e.total_bytes + (EC_ETH_CONTROL_ALIGN - 16u);
    EXPECT_TRUE(admit(cfg, cap).ok());
    cap.eth.size -= 1u;
    const Admission padded = admit(cfg, cap);
    ASSERT_FALSE(padded.ok());
    EXPECT_EQ(padded.check.constraint, EC_CONSTRAINT_ETH_WINDOW);
    EXPECT_EQ(padded.check.value - padded.check.bound, 1u);
}

// ---------------------------------------------------------------------------
// Window placement (ABI migration spec §3): compute_layout() and admit()
// refuse any window whose placement cannot end at or below 2^32, without
// wrapping, as EC_CHECK_CAPACITY on the window's base constraint
// (EC_CONSTRAINT_ETH_BASE for cap.eth, EC_CONSTRAINT_LIM_BASE for cap.lim).
// For a base above 2^32 the refusal is {value = the base, bound = 2^32}.
// ---------------------------------------------------------------------------

namespace {

constexpr uint64_t kFourGiB = 1ull << 32;

Config PlacementConfig() { return MakeConfig(4u, 8u, 8u, 1024u, 1u, Direction::Both); }

// Expects both compute_layout() and admit() to refuse `cap` on `constraint`.
// `base_above` also pins {value = window base, bound = 2^32}.
void ExpectPlacementRefused(
    const Capacity& cap, uint32_t constraint, uint64_t window_base, bool base_above, const std::string& why) {
    const Config cfg = PlacementConfig();
    const Layout l = compute_layout(cfg, cap);
    EXPECT_FALSE(l.ok()) << why;
    EXPECT_EQ(l.check.code, static_cast<uint32_t>(EC_CHECK_CAPACITY)) << "compute_layout, " << why;
    EXPECT_EQ(l.check.constraint, constraint) << "compute_layout, " << why << ": " << describe(l.check, l.chip);
    const Admission adm = admit(cfg, cap);
    EXPECT_FALSE(adm.ok()) << why;
    EXPECT_EQ(adm.check.code, static_cast<uint32_t>(EC_CHECK_CAPACITY)) << "admit, " << why;
    EXPECT_EQ(adm.check.constraint, constraint) << "admit, " << why << ": " << adm.describe();
    if (base_above) {
        EXPECT_EQ(l.check.value, window_base) << "compute_layout, " << why;
        EXPECT_EQ(l.check.bound, kFourGiB) << "compute_layout, " << why;
        EXPECT_EQ(adm.check.value, window_base) << "admit, " << why;
        EXPECT_EQ(adm.check.bound, kFourGiB) << "admit, " << why;
    }
}

}  // namespace

// Refused, one window moved at a time (359 KiB Ethernet, 512 KiB LIM):
// - near UINT64_MAX, where rounding the base up to its alignment wraps to a
//   small address a wrapping placement would accept;
// - at or just above 2^32, already aligned so no rounding wraps; truncated to
//   32 bits these bases are 0 and 512 / 64, which a truncating placement would
//   accept;
// - below 2^32 with a placed extent crossing it: the window is large enough,
//   so only the end-at-or-below-2^32 rule refuses it.
// Placed: a window whose whole extent ends at 2^32, at its own (aligned) base.
TEST(EthCreditPlacement, OnlyAPlacementEndingAtOrBelowFourGiBIsAccepted) {
    struct Case {
        const char* why;
        bool lim;
        uint64_t base;
        bool base_above;
    };
    const Case refused[] = {
        {"eth at UINT64_MAX - 255", false, UINT64_MAX - 255u, true},
        {"lim at UINT64_MAX - 63", true, UINT64_MAX - 63u, true},
        {"eth at 2^32", false, kFourGiB, false},
        {"eth at 2^32 + 512", false, kFourGiB + 512u, true},
        {"eth at 2^32 + 0x1A000", false, kFourGiB + 0x1A000u, true},
        {"lim at 2^32", true, kFourGiB, false},
        {"lim at 2^32 + 64", true, kFourGiB + 64u, true},
        {"lim at 2^32 + 0x08100000", true, kFourGiB + 0x08100000u, true},
        {"eth at 2^32 - 512", false, kFourGiB - 512u, false},
        {"lim at 2^32 - 64", true, kFourGiB - 64u, false},
    };
    for (const Case& c : refused) {
        Capacity cap = RealisticCapacity();
        (c.lim ? cap.lim : cap.eth) = Window{c.base, c.lim ? 512u * kKiB : kEthWindowBytes};
        ExpectPlacementRefused(
            cap, c.lim ? EC_CONSTRAINT_LIM_BASE : EC_CONSTRAINT_ETH_BASE, c.base, c.base_above, c.why);
    }

    ASSERT_EQ(kEthWindowBytes % EC_ETH_CONTROL_ALIGN, 0u);
    Capacity cap = RealisticCapacity();
    cap.eth = Window{kFourGiB - kEthWindowBytes, kEthWindowBytes};
    cap.lim = Window{kFourGiB - 512u * kKiB, 512u * kKiB};
    const Layout l = compute_layout(PlacementConfig(), cap);
    ASSERT_TRUE(l.ok()) << describe(l.check, l.chip);
    for (uint32_t c = 0; c < kChips; ++c) {
        EXPECT_EQ(l.chips[c].eth_ctrl_base, static_cast<uint32_t>(cap.eth.base)) << "chip " << c;
        EXPECT_EQ(l.chips[c].lim_base, cap.lim.base) << "chip " << c;
        EXPECT_LE(static_cast<uint64_t>(l.chips[c].eth_ctrl_base) + l.chips[c].eth.total_bytes, kFourGiB);
        EXPECT_LE(l.chips[c].lim_base + l.chips[c].x280.total_bytes, kFourGiB);
    }
}

// ---------------------------------------------------------------------------
// Mod-16 congruence (plan §Protocol / Layout: Blackhole requires a NoC write's
// source and destination to be congruent mod 16; the host checks that every
// base and stride is a multiple of 16).
// ---------------------------------------------------------------------------

namespace {

// The five values check_congruence() inspects per chip, in its documented
// order (regions.hpp, check_congruence: every base and stride).
struct CongruenceField {
    const char* name;
    uint32_t constraint;
    uint64_t (*get)(const ChipLayout&);
    void (*add)(ChipLayout&, uint32_t);
};

const std::vector<CongruenceField>& CongruenceFields() {
    static const std::vector<CongruenceField> fields = {
        {"eth_ctrl_base",
         EC_CONSTRAINT_ETH_BASE,
         [](const ChipLayout& l) -> uint64_t { return l.eth_ctrl_base; },
         [](ChipLayout& l, uint32_t d) { l.eth_ctrl_base += d; }},
        {"lim_base",
         EC_CONSTRAINT_LIM_BASE,
         [](const ChipLayout& l) -> uint64_t { return l.lim_base; },
         [](ChipLayout& l, uint32_t d) { l.lim_base += d; }},
        {"eth.fifo_stride",
         EC_CONSTRAINT_ETH_WINDOW,
         [](const ChipLayout& l) -> uint64_t { return l.eth.fifo_stride; },
         [](ChipLayout& l, uint32_t d) { l.eth.fifo_stride += d; }},
        {"x280.fwd_stride",
         EC_CONSTRAINT_LIM_WINDOW,
         [](const ChipLayout& l) -> uint64_t { return l.x280.fwd_stride; },
         [](ChipLayout& l, uint32_t d) { l.x280.fwd_stride += d; }},
        {"slot_size",
         EC_CONSTRAINT_SLOT_SIZE,
         [](const ChipLayout& l) -> uint64_t { return l.slot_size; },
         [](ChipLayout& l, uint32_t d) { l.slot_size += d; }},
    };
    return fields;
}

// Admitted layouts are the inputs every congruence test perturbs, so each
// starts from values the check must accept.
Admission AdmitForCongruence(uint32_t P) {
    const Admission adm = admit(MakeConfig(P, 8u, 8u, 256u, 1u, Direction::Both), RealisticCapacity());
    EXPECT_TRUE(adm.ok()) << adm.describe();
    return adm;
}

void ExpectCongruenceFailure(
    const ec_check_t& k, uint32_t chip, uint32_t want_constraint, uint64_t want_value, uint32_t want_chip) {
    EXPECT_EQ(k.code, EC_CHECK_CONGRUENCE);
    EXPECT_EQ(k.constraint, want_constraint) << constraint_name(k.constraint);
    EXPECT_EQ(k.value, want_value);
    EXPECT_EQ(k.bound, 16u);
    EXPECT_EQ(chip, want_chip);
}

}  // namespace

// Spec: the first value not a multiple of 16 returns EC_CHECK_CONGRUENCE with
// its constraint, value = that value, bound = 16 and *chip = its chip. Each
// residue 4, 8, 12 is tried so a check testing % 8 or % 4 misses one; +16
// keeps the value a multiple of 16 and must stay OK.
TEST(EthCreditCongruence, EachMisalignedFieldOnEachChipIsNamed) {
    for (uint32_t P : {2u, 4u}) {
        const Admission adm = AdmitForCongruence(P);
        ASSERT_TRUE(adm.ok());
        for (uint32_t c = 0; c < kChips; ++c) {
            for (const CongruenceField& f : CongruenceFields()) {
                for (uint32_t shift : {4u, 8u, 12u, 16u}) {
                    SCOPED_TRACE(testing::Message() << "P " << P << " chip " << c << " " << f.name << " +" << shift);
                    Layout l = adm.layout;
                    f.add(l.chips[c], shift);
                    const uint64_t shifted = f.get(l.chips[c]);
                    ASSERT_EQ(shifted, f.get(adm.layout.chips[c]) + shift);
                    uint32_t chip = 0xDEADu;
                    const ec_check_t k = check_congruence(l, &chip);
                    if (shift == 16u) {
                        EXPECT_EQ(k.code, EC_CHECK_OK);
                    } else {
                        ExpectCongruenceFailure(k, chip, f.constraint, shifted, c);
                    }
                }
            }
        }
    }
}

// Spec: chips are checked 0 then 1, and within a chip in field order, so a
// chip-0 fault is reported ahead of chip 1's, and the earlier field ahead of
// the later one on the same chip.
TEST(EthCreditCongruence, TheFirstFaultInChipThenFieldOrderIsReported) {
    const Admission adm = AdmitForCongruence(2u);
    ASSERT_TRUE(adm.ok());

    // Chip 1's two faults are its first fields; chip 0's is its last, so a
    // check scanning fields before chips, or chip 1 first, reports chip 1.
    Layout l = adm.layout;
    l.chips[1].eth_ctrl_base += 4u;
    l.chips[1].lim_base += 8u;
    l.chips[0].slot_size += 12u;
    uint32_t chip = 0xDEADu;
    ec_check_t k = check_congruence(l, &chip);
    ExpectCongruenceFailure(k, chip, EC_CONSTRAINT_SLOT_SIZE, l.chips[0].slot_size, 0u);

    // Chip 0 clean: chip 1's earlier field wins over its later one.
    Layout later = adm.layout;
    later.chips[1].x280.fwd_stride += 4u;
    later.chips[1].lim_base += 8u;
    chip = 0xDEADu;
    k = check_congruence(later, &chip);
    ExpectCongruenceFailure(k, chip, EC_CONSTRAINT_LIM_BASE, later.chips[1].lim_base, 1u);
}

// ---------------------------------------------------------------------------
// Per-role peer NoC coordinates (plan §Ethernet cores and queues: ERISC0 on
// NoC 0, ERISC1 on NoC 1; the host derives each role's X280 coordinates for
// that role's NoC).
// ---------------------------------------------------------------------------

// The X280 tile resolves to the same pair on both NoCs, so distinct test values
// are what separate "E1 takes the NoC 1 coordinate" from "E1 takes NoC 0's".
TEST(EthCreditNoc, EachRoleCarriesTheX280CoordinateResolvedForItsNoc) {
    Config cfg = MakeConfig(2u, 8u, 8u, 256u, 1u, Direction::Both);
    for (uint32_t c = 0; c < kChips; ++c) {
        cfg.chips[c].x280_noc1 = TranslatedCoordFactory::make(9u, 30u + c);
    }
    const Admission ok = admit(cfg, RealisticCapacity());
    ASSERT_TRUE(ok.ok()) << ok.describe();
    const Arguments args = derive_arguments(cfg, ok);
    for (uint32_t c = 0; c < kChips; ++c) {
        EXPECT_EQ(args.chips[c].e0.erisc.fwd.x, cfg.chips[c].x280_noc0->x());
        EXPECT_EQ(args.chips[c].e0.erisc.fwd.y, cfg.chips[c].x280_noc0->y());
        ASSERT_TRUE(args.chips[c].has_e1);
        EXPECT_EQ(args.chips[c].e1.erisc.fwd.x, 9u);
        EXPECT_EQ(args.chips[c].e1.erisc.fwd.y, 30u + c);
    }
}

TEST(EthCreditAdmission, PlacementMissingACoordinateIsRefused) {
    for (uint32_t bit = 0; bit < 3u; ++bit) {
        SCOPED_TRACE(bit);
        Config cfg = MakeConfig(2u, 8u, 8u, 256u, 1u, Direction::Both);
        ChipPlacement& pl = cfg.chips[1];
        (bit == 0u ? pl.x280_noc0 : bit == 1u ? pl.x280_noc1 : pl.eth).reset();
        const Admission adm = admit(cfg, RealisticCapacity());
        ASSERT_FALSE(adm.ok());
        EXPECT_EQ(adm.check.code, EC_CHECK_COORD);
        EXPECT_EQ(adm.check.value, 1u << bit);
        EXPECT_EQ(adm.chip, 1u);
    }
}

// ---------------------------------------------------------------------------
// Admission (plan §Stream-ID budget: IDs 1-31, P per tile; §Files).
// ---------------------------------------------------------------------------

TEST(EthCreditAdmission, StreamIdsAreOneThroughThirtyOneAndDistinctPerTile) {
    const Capacity cap = RealisticCapacity();
    struct Case {
        const char* name;
        std::array<uint32_t, EC_MAX_P> ids;
        bool ok;
    };
    const Case cases[] = {
        {"1..4", {1u, 2u, 3u, 4u}, true},
        {"28..31", {28u, 29u, 30u, 31u}, true},
        {"stream 0", {0u, 2u, 3u, 4u}, false},
        {"stream 32", {1u, 2u, 3u, 32u}, false},
        {"duplicate", {5u, 6u, 5u, 7u}, false},
    };
    for (const Case& c : cases) {
        Config cfg = MakeConfig(4u, 8u, 8u, 256u, 1u, Direction::Both);
        cfg.chips[1].stream_ids = c.ids;
        const Admission adm = admit(cfg, cap);
        EXPECT_EQ(adm.ok(), c.ok) << c.name << ": " << adm.describe();
        if (!c.ok) {
            EXPECT_EQ(adm.check.code, EC_CHECK_STREAM_ID) << c.name;
            EXPECT_EQ(adm.chip, 1u) << c.name;
        }
    }
    // Stream ids at p >= P are unused.
    Config cfg = MakeConfig(2u, 8u, 8u, 256u, 1u, Direction::Both);
    cfg.chips[0].stream_ids = {7u, 8u, 0u, 0u};
    EXPECT_TRUE(admit(cfg, cap).ok());
}

TEST(EthCreditAdmission, StallBudgetConvertsOnEachRolesClock) {
    Config cfg = MakeConfig(1u, 8u, 8u, 256u, 1u, Direction::AToB);
    cfg.stall_budget_ms = 1000u;
    const Admission adm = admit(cfg, RealisticCapacity());
    ASSERT_TRUE(adm.ok()) << adm.describe();
    EXPECT_EQ(adm.x280_stall_ticks, 50000000u);
    EXPECT_EQ(adm.erisc_stall_ticks, 1000000000u);
}

// ---------------------------------------------------------------------------
// Arguments (ABI migration spec §2: E0 and E1 share one 32-word prefix,
// `.erisc`, first in both records; runtime word i is the record's i-th uint32
// in declaration order; one host function derives every role from one layout).
// ---------------------------------------------------------------------------

namespace {

// Word indices of ec_erisc_args_t, restated from the declaration order the
// spec lists (§2) rather than taken from offsetof, so a reordered struct fails.
constexpr uint32_t kWMagic = 0, kWKernelId = 1, kWFlags = 2, kWShape = 3;  // shape: P, n_link, n_deliver, slot, batch
constexpr uint32_t kWRunNonce = 8, kWPollCadence = 9, kWBudgetLo = 10, kWBudgetHi = 11;
constexpr uint32_t kWCtrlBase = 12, kWFwdX = 13, kWFwdY = 14, kWReadyBase = 15;
constexpr uint32_t kWCapLo = 16, kWCapHi = 17;
constexpr uint32_t kWCount = 18, kWFwdArea = 22, kWStreamId = 26, kWLimEndLo = 30, kWLimEndHi = 31;
constexpr uint32_t kWE0PeerCtrl = 32, kWE0PeerRx = 33, kWE0LocalStream = 34;
constexpr uint32_t kWE1RxBase = 32, kWE1PeerCtrl = 33;

// Gives every prefix field a distinct value, 0xA0000000 + its word index,
// assigned by name.
void FillPrefix(ec_erisc_args_t& e) {
    const auto v = [](uint32_t i) { return 0xA0000000u + i; };
    e.magic = v(kWMagic);
    e.kernel_id = v(kWKernelId);
    e.flags = v(kWFlags);
    e.shape.P = v(kWShape + 0u);
    e.shape.n_link = v(kWShape + 1u);
    e.shape.n_deliver = v(kWShape + 2u);
    e.shape.slot_size = v(kWShape + 3u);
    e.shape.batch = v(kWShape + 4u);
    e.run_nonce = v(kWRunNonce);
    e.poll_cadence = v(kWPollCadence);
    e.stall_budget_lo = v(kWBudgetLo);
    e.stall_budget_hi = v(kWBudgetHi);
    e.ctrl_base = v(kWCtrlBase);
    e.fwd.x = v(kWFwdX);
    e.fwd.y = v(kWFwdY);
    e.barrier_ready_base = v(kWReadyBase);
    e.barrier_cap_lo = v(kWCapLo);
    e.barrier_cap_hi = v(kWCapHi);
    for (uint32_t p = 0; p < EC_MAX_P; ++p) {
        e.count[p] = v(kWCount + p);
        e.fwd_area[p] = v(kWFwdArea + p);
        e.stream_id[p] = v(kWStreamId + p);
    }
    e.lim_end_lo = v(kWLimEndLo);
    e.lim_end_hi = v(kWLimEndHi);
}

}  // namespace

// Spec §2, serialization: word i is the record's i-th uint32 in declaration
// order, 38 words for E0 and 34 for E1. Every field carries a distinct value,
// so a swapped, shifted or dropped field lands on a word the expected vector
// does not hold.
TEST(EthCreditArgs, EachRecordPacksInDeclarationOrder) {
    ec_e0_args_t e0{};
    FillPrefix(e0.erisc);
    e0.peer_ctrl_base = 0xA0000000u + kWE0PeerCtrl;
    e0.peer_rx_base = 0xA0000000u + kWE0PeerRx;
    for (uint32_t p = 0; p < EC_MAX_P; ++p) {
        e0.local_stream_id[p] = 0xA0000000u + kWE0LocalStream + p;
    }
    ec_e1_args_t e1{};
    FillPrefix(e1.erisc);
    e1.rx_base = 0xA0000000u + kWE1RxBase;
    e1.peer_ctrl_base = 0xA0000000u + kWE1PeerCtrl;

    const auto expect_words = [](const char* who, const std::vector<uint32_t>& w, size_t words) {
        ASSERT_EQ(w.size(), words) << who;
        for (uint32_t i = 0; i < w.size(); ++i) {
            EXPECT_EQ(w[i], 0xA0000000u + i) << who << " word " << i;
        }
    };
    expect_words("E0", pack_runtime_args(e0), 38u);
    expect_words("E1", pack_runtime_args(e1), 34u);
}

// Spec §3: each ERISC kernel's barrier slot is its role's fixed index: E0 0,
// E1 1, forwarder p 2 + p, all inside the ready table.
TEST(EthCreditArgs, BarrierWorkerIndexFollowsTheRole) {
    EXPECT_EQ(ec_barrier_index_e0(), 0u);
    EXPECT_EQ(ec_barrier_index_e1(), 1u);
    std::set<uint32_t> slots{ec_barrier_index_e0(), ec_barrier_index_e1()};
    for (uint32_t p = 0; p < EC_MAX_P; ++p) {
        EXPECT_EQ(ec_barrier_index_fwd(p), 2u + p);
        EXPECT_LT(ec_barrier_index_fwd(p), EC_BARRIER_MAX_WORKERS);
        slots.insert(ec_barrier_index_fwd(p));
    }
    EXPECT_EQ(slots.size(), EC_BARRIER_MAX_WORKERS);
}

TEST(EthCreditArgs, DerivedArgumentsAgreeWithTheLayoutTheyCameFrom) {
    const uint32_t P = 4u;
    Config cfg = MakeConfig(P, 8u, 8u, 1024u, 2u, Direction::Both);
    cfg.director_p = 2u;
    cfg.chips[0].stream_ids = {1u, 2u, 3u, 4u};
    cfg.chips[1].stream_ids = {11u, 12u, 13u, 14u};
    const Admission adm = admit(cfg, RealisticCapacity());
    ASSERT_TRUE(adm.ok()) << adm.describe();
    const Arguments args = derive_arguments(cfg, adm);

    std::set<uint32_t> ids;
    for (uint32_t c = 0; c < kChips; ++c) {
        const uint32_t o = 1u - c;
        const ChipLayout& l = adm.layout.chips[c];
        const ChipLayout& peer = adm.layout.chips[o];
        const ChipArguments& a = args.chips[c];

        // E0. The shape is read from the packed words 3..7 (spec §2), so the
        // serialized record, not just the struct, carries it.
        const std::vector<uint32_t> e0w = pack_runtime_args(a.e0);
        ASSERT_EQ(e0w.size(), static_cast<size_t>(EC_E0_ARG_WORDS));
        const ec_shape_t e0s{e0w[3], e0w[4], e0w[5], e0w[6], e0w[7]};
        EXPECT_TRUE(ec_shape_equal(&e0s, &cfg.shape));
        EXPECT_EQ(e0w[0], EC_ABI_MAGIC);
        EXPECT_EQ(a.e0.erisc.magic, EC_ABI_MAGIC);
        EXPECT_EQ(ec_kernel_role(a.e0.erisc.kernel_id), static_cast<uint32_t>(EC_ROLE_E0));
        EXPECT_EQ(ec_kernel_chip(a.e0.erisc.kernel_id), c);
        EXPECT_EQ(a.e0.erisc.ctrl_base, l.eth_ctrl_base);
        EXPECT_EQ(e0w[12], l.eth_ctrl_base);
        EXPECT_EQ(a.e0.peer_ctrl_base, peer.eth_ctrl_base);
        EXPECT_EQ(a.e0.peer_rx_base, peer.rx_slot(0u, 0u)) << "W3 lands in the peer's rx[0]";
        EXPECT_EQ(
            (static_cast<uint64_t>(a.e0.erisc.stall_budget_hi) << 32) | a.e0.erisc.stall_budget_lo,
            adm.erisc_stall_ticks);
        ids.insert(a.e0.erisc.kernel_id);

        // E1.
        ASSERT_TRUE(a.has_e1);
        const std::vector<uint32_t> e1w = pack_runtime_args(a.e1);
        ASSERT_EQ(e1w.size(), static_cast<size_t>(EC_E1_ARG_WORDS));
        const ec_shape_t e1s{e1w[3], e1w[4], e1w[5], e1w[6], e1w[7]};
        EXPECT_TRUE(ec_shape_equal(&e1s, &cfg.shape));
        EXPECT_EQ(e1w[0], EC_ABI_MAGIC);
        EXPECT_EQ(ec_kernel_role(a.e1.erisc.kernel_id), static_cast<uint32_t>(EC_ROLE_E1));
        EXPECT_EQ(ec_kernel_chip(a.e1.erisc.kernel_id), c);
        EXPECT_EQ(a.e1.erisc.ctrl_base, l.eth_ctrl_base);
        EXPECT_EQ(a.e1.rx_base, l.rx_slot(0u, 0u));
        EXPECT_EQ(e1w[32], l.rx_slot(0u, 0u));
        EXPECT_EQ(a.e1.peer_ctrl_base, peer.eth_ctrl_base) << "R2 lands in the peer's control area";
        EXPECT_EQ(e1w[33], peer.eth_ctrl_base);
        // E1's prefix scalars come from the same config as E0's.
        EXPECT_EQ(ec_u64(a.e1.erisc.stall_budget_lo, a.e1.erisc.stall_budget_hi), adm.erisc_stall_ticks);
        EXPECT_EQ(a.e1.erisc.poll_cadence, a.e0.erisc.poll_cadence);
        EXPECT_EQ(a.e1.erisc.run_nonce, a.e0.erisc.run_nonce);
        EXPECT_EQ(
            ec_u64(a.e1.erisc.barrier_cap_lo, a.e1.erisc.barrier_cap_hi),
            ec_u64(a.e0.erisc.barrier_cap_lo, a.e0.erisc.barrier_cap_hi));
        ids.insert(a.e1.erisc.kernel_id);

        EXPECT_EQ(a.e0.erisc.barrier_ready_base, a.e1.erisc.barrier_ready_base);
        // Every role bounds LIM by the end of the window the layout was placed in.
        const Capacity cap = RealisticCapacity();
        EXPECT_EQ(l.lim_end, cap.lim.base + cap.lim.size);
        EXPECT_EQ(ec_u64(e0w[30], e0w[31]), l.lim_end);
        EXPECT_EQ(ec_u64(e1w[30], e1w[31]), l.lim_end);

        for (uint32_t p = 0; p < P; ++p) {
            // W4 (plan §The nine writes): E0 on this chip increments pkts[p]
            // on the peer tile, and E1 reads this tile's own.
            EXPECT_EQ(a.e0.erisc.stream_id[p], cfg.chips[o].stream_ids[p]) << "chip " << c << " p " << p;
            // E0 resets its own tile's counters before the handshake.
            EXPECT_EQ(a.e0.local_stream_id[p], cfg.chips[c].stream_ids[p]) << "chip " << c << " p " << p;
            EXPECT_EQ(a.e1.erisc.stream_id[p], cfg.chips[c].stream_ids[p]) << "chip " << c << " p " << p;
            EXPECT_EQ(e0w[26 + p], cfg.chips[o].stream_ids[p]) << "chip " << c << " p " << p;
            EXPECT_EQ(e0w[34 + p], cfg.chips[c].stream_ids[p]) << "chip " << c << " p " << p;
            EXPECT_EQ(e1w[26 + p], cfg.chips[c].stream_ids[p]) << "chip " << c << " p " << p;
            EXPECT_EQ(a.e0.erisc.fwd_area[p], static_cast<uint32_t>(l.fwd_area(p)));
            EXPECT_EQ(a.e1.erisc.fwd_area[p], static_cast<uint32_t>(l.fwd_area(p)));
            EXPECT_EQ(a.e0.erisc.count[p], cfg.packets[p]);
            EXPECT_EQ(a.e1.erisc.count[p], cfg.packets[p]);

            const ec_x280_fwd_args_t& f = a.fwd[p];
            EXPECT_EQ(f.magic, EC_ABI_MAGIC);
            EXPECT_EQ(ec_kernel_role(f.kernel_id), static_cast<uint32_t>(EC_ROLE_X280_FWD));
            EXPECT_EQ(ec_kernel_index(f.kernel_id), p);
            EXPECT_EQ(ec_kernel_chip(f.kernel_id), c);
            EXPECT_TRUE(ec_shape_equal(&f.shape, &cfg.shape));
            EXPECT_EQ(f.p, p);
            EXPECT_EQ((f.flags & EC_FWD_FLAG_DIRECTOR) != 0u, p == cfg.director_p);
            EXPECT_EQ(f.lim_base, l.lim_base);
            EXPECT_EQ(f.lim_end, l.lim_end);
            EXPECT_EQ(f.eth_ctrl_base, l.eth_ctrl_base);
            EXPECT_EQ(f.eth.x, cfg.chips[c].eth->x());
            EXPECT_EQ(f.eth.y, cfg.chips[c].eth->y());
            EXPECT_EQ(f.send_count, cfg.packets[p]);
            EXPECT_EQ(f.recv_count, cfg.packets[p]);
            EXPECT_EQ(f.stall_budget_ticks, adm.x280_stall_ticks);
            EXPECT_EQ(f.tlb_base, fwd_tlb_base(p));
            ids.insert(f.kernel_id);
        }
    }
    EXPECT_EQ(ids.size(), kChips * (2u + P)) << "every kernel on both chips has its own id";
}

TEST(EthCreditArgs, OneWayRunIdlesTheSendersInboundSideAndTheReceiversOutbound) {
    const uint32_t P = 2u;
    const Config cfg = MakeConfig(P, 8u, 8u, 256u, 1u, Direction::AToB);
    const Admission adm = admit(cfg, RealisticCapacity());
    ASSERT_TRUE(adm.ok()) << adm.describe();
    const Arguments args = derive_arguments(cfg, adm);

    const ChipArguments& a = args.chips[0];
    const ChipArguments& b = args.chips[1];
    // Plan §Pre-run ordering: E0 runs on both tiles in every run.
    EXPECT_EQ(a.e0.erisc.magic, EC_ABI_MAGIC);
    EXPECT_EQ(b.e0.erisc.magic, EC_ABI_MAGIC);
    EXPECT_NE(a.e0.erisc.flags & EC_E0_FLAG_OUTBOUND_ACTIVE, 0u);
    EXPECT_EQ(b.e0.erisc.flags & EC_E0_FLAG_OUTBOUND_ACTIVE, 0u);
    EXPECT_NE(
        (a.e0.erisc.flags & EC_E0_FLAG_HANDSHAKE_SENDER) != 0u, (b.e0.erisc.flags & EC_E0_FLAG_HANDSHAKE_SENDER) != 0u)
        << "exactly one side of the handshake sends";
    EXPECT_TRUE(b.has_e1);
    EXPECT_FALSE(a.has_e1) << "nothing arrives on the sender";
    for (uint32_t p = 0; p < P; ++p) {
        EXPECT_EQ(a.e0.erisc.count[p], cfg.packets[p]);
        EXPECT_EQ(b.e0.erisc.count[p], 0u);
        EXPECT_EQ(a.fwd[p].send_count, cfg.packets[p]);
        EXPECT_EQ(a.fwd[p].recv_count, 0u);
        EXPECT_EQ(b.fwd[p].send_count, 0u);
        EXPECT_EQ(b.fwd[p].recv_count, cfg.packets[p]);
        EXPECT_EQ(b.e1.erisc.count[p], cfg.packets[p]);
    }
}

// X280_TLB_ETH_CREDIT_*: hart p owns one window onto its Ethernet tile, so the
// per-hart windows must not overlap.
TEST(EthCreditArgs, ForwarderTlbRangesAreDisjointPerHart) {
    for (uint32_t p = 0; p + 1u < EC_X280_HARTS; ++p) {
        EXPECT_GE(fwd_tlb_base(p + 1u), fwd_tlb_base(p) + 1u);
    }
    EXPECT_EQ(fwd_tlb_base(0u), X280_TLB_ETH_CREDIT_BASE);
    EXPECT_LE(fwd_tlb_base(EC_X280_HARTS - 1u) + 1u, X280_TLB_ETH_CREDIT_BASE + X280_TLB_ETH_CREDIT_COUNT);
}

// ---------------------------------------------------------------------------
// Zero-traffic admission (Config::packets contract in suites/eth_credit/
// regions.hpp: "zero is a run with no traffic, and entries at p >= P must
// be 0"). A zero-traffic run brings up the barrier and handshake alone.
// ---------------------------------------------------------------------------

namespace {

constexpr Direction kAllDirections[] = {Direction::AToB, Direction::BToA, Direction::Both};

Config ZeroTrafficConfig(uint32_t P, Direction dir) {
    Config cfg = MakeConfig(P, 8u, 8u, 256u, 1u, dir);
    cfg.packets.fill(0u);
    return cfg;
}

}  // namespace

// All counts zero at every P and direction, and each p < P zero on its own
// beside nonzero counts: the contract is per entry.
TEST(EthCreditZeroTraffic, ZeroCountsBelowPAreAdmitted) {
    for (uint32_t P = 1; P <= EC_MAX_P; ++P) {
        for (Direction dir : kAllDirections) {
            const Admission adm = admit(ZeroTrafficConfig(P, dir), RealisticCapacity());
            EXPECT_TRUE(adm.ok()) << "P " << P << " dir " << static_cast<uint32_t>(dir) << ": " << adm.describe();
        }
    }
    for (uint32_t zero_p = 0; zero_p < EC_MAX_P; ++zero_p) {
        Config cfg = MakeConfig(EC_MAX_P, 8u, 8u, 256u, 1u, Direction::Both);
        cfg.packets[zero_p] = 0u;
        const Admission adm = admit(cfg, RealisticCapacity());
        EXPECT_TRUE(adm.ok()) << "zero at p " << zero_p << ": " << adm.describe();
    }
}

TEST(EthCreditZeroTraffic, NonzeroCountAtOrPastPIsRejectedNamingThatP) {
    const uint32_t kStray = 7u;
    for (uint32_t P = 1; P < EC_MAX_P; ++P) {
        for (uint32_t p = P; p < EC_MAX_P; ++p) {
            // Both with traffic below P and with none: the p >= P rule is
            // independent of whether the active entries are zero.
            for (bool traffic : {true, false}) {
                Config cfg =
                    traffic ? MakeConfig(P, 8u, 8u, 256u, 1u, Direction::Both) : ZeroTrafficConfig(P, Direction::Both);
                cfg.packets[p] = kStray + p;
                const Admission adm = admit(cfg, RealisticCapacity());
                const std::string where = "P " + std::to_string(P) + " stray p " + std::to_string(p) +
                                          (traffic ? " with traffic" : " zero traffic");
                EXPECT_FALSE(adm.ok()) << where;
                EXPECT_EQ(adm.check.code, static_cast<uint32_t>(EC_CHECK_BAD_PARAM)) << where;
                EXPECT_EQ(adm.check.constraint, static_cast<uint32_t>(EC_CONSTRAINT_P)) << where;
                EXPECT_EQ(adm.check.value, p) << where;
                EXPECT_EQ(adm.check.bound, kStray + p) << where;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tail marker (eth_credit_abi.h, EC_TAIL_MARKER_*): the four words E0 stamps in
// a slot's last 16 bytes, which E1 checks once pkts[p] shows the packet.
// ---------------------------------------------------------------------------

TEST(EthCreditTailMarker, DistinctAcrossForwardersAndManyRingWrapsAndNeverAllZero) {
    ASSERT_EQ(EC_TAIL_MARKER_BYTES, 4u * sizeof(uint32_t)) << "the slot tail must hold exactly the four words";
    // A stale slot holds the marker of an earlier ordinal on the same p (one or
    // more N_LINK wraps back) or of another forwarder; neither may match. The
    // range covers several wraps of the deepest ring, plus the top of uint32.
    std::vector<uint32_t> ordinals;
    for (uint32_t o = 0; o < 4u * EC_MAX_DEPTH + 3u; ++o) {
        ordinals.push_back(o);
    }
    for (uint32_t o = 0xFFFFFFFFu - 64u; o != 0u; ++o) {
        ordinals.push_back(o);
    }
    std::set<std::vector<uint32_t>> seen;
    const std::vector<uint32_t> zero(4u, 0u);
    for (uint32_t p = 0; p < EC_MAX_P; ++p) {
        for (uint32_t ord : ordinals) {
            uint32_t out[4];
            ec_tail_marker(p, ord, out);
            const std::vector<uint32_t> m(out, out + 4);
            EXPECT_NE(m, zero) << "an all-zero rx slot would match p " << p << " ordinal " << ord;
            EXPECT_TRUE(seen.insert(m).second) << "duplicate marker at p " << p << " ordinal " << ord;
        }
    }
    EXPECT_EQ(seen.size(), EC_MAX_P * ordinals.size());
}

// ---------------------------------------------------------------------------
// Argument checks (eth_credit_abi.h §Argument checks). Expected outcomes come
// from that section's doc comments and the check contract the ERISC kernels
// rely on: each bad field is refused with its own (code, constraint), and the
// records derive_arguments() produces for an admitted Config are accepted
// against that admission's windows.
// ---------------------------------------------------------------------------

namespace {

// The windows an admission was made against: the Ethernet window as L1 and
// cap.lim as LIM.
ec_bounds_t AdmittedBounds(const Capacity& cap) {
    ec_bounds_t b;
    b.l1_lo = static_cast<uint32_t>(cap.eth.base);
    b.l1_hi = static_cast<uint32_t>(cap.eth.base + cap.eth.size);
    b.lim_lo = cap.lim.base;
    b.lim_hi = cap.lim.base + cap.lim.size;
    return b;
}

struct Admitted {
    Config cfg;
    Capacity cap;
    Admission adm;
    Arguments args;
    ec_bounds_t b{};
};

// P = 4 so p = 3 exists; Both so every E0 is outbound and every tile has an E1.
Admitted AdmitForChecks(uint32_t P = 4u, Direction dir = Direction::Both, uint32_t payload = 1024u) {
    Admitted f;
    f.cfg = MakeConfig(P, 8u, 8u, payload, 2u, dir);
    f.cap = RealisticCapacity();
    f.adm = admit(f.cfg, f.cap);
    EXPECT_TRUE(f.adm.ok()) << f.adm.describe();
    f.args = derive_arguments(f.cfg, f.adm);
    f.b = AdmittedBounds(f.cap);
    return f;
}

::testing::AssertionResult IsCheck(const ec_check_t& got, uint32_t code, uint32_t constraint) {
    if (got.code == code && got.constraint == constraint) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "got " << describe(got, 0u) << " (code " << got.code << ", constraint "
                                         << got.constraint << "), want code " << code << ", constraint " << constraint;
}

template <typename Mutate>
ec_check_t CheckE0(const Admitted& f, uint32_t c, Mutate mutate) {
    ec_e0_args_t a = f.args.chips[c].e0;
    mutate(a);
    return ec_check_e0_args(&a, &f.cfg.shape, &f.b);
}

template <typename Mutate>
ec_check_t CheckE1(const Admitted& f, uint32_t c, Mutate mutate) {
    EXPECT_TRUE(f.args.chips[c].has_e1) << "chip " << c;
    ec_e1_args_t a = f.args.chips[c].e1;
    mutate(a);
    return ec_check_e1_args(&a, &f.cfg.shape, &f.b);
}

// Bytes of one role's P x N_LINK slot region.
uint32_t SlotRegionBytes(const Admitted& f) { return f.cfg.shape.P * f.cfg.shape.n_link * f.cfg.shape.slot_size; }

}  // namespace

// §Argument checks, ec_range_ok: "[base, base + bytes) lies inside [lo, hi),
// without overflow". The wrapping cases sum to a small value that a naive
// `base + bytes <= hi` accepts; the whole-space cases check that the limits
// themselves still work.
TEST(EthCreditArgChecksRange, AcceptsExactlyTheRangesInsideTheWindowWithoutWrapping) {
    const uint64_t lo = 0x1000u, hi = 0x2000u, max = UINT64_MAX;
    const struct {
        uint64_t base, bytes, lo, hi;
        bool ok;
        const char* why;
    } cases[] = {
        {lo, hi - lo, lo, hi, true, "the whole window"},
        {lo, 0u, lo, hi, true, "empty at lo"},
        {hi - 1u, 1u, lo, hi, true, "last byte"},
        {lo + 16u, 32u, lo, hi, true, "inside"},
        {lo, hi - lo + 1u, lo, hi, false, "one byte past hi"},
        {hi - 1u, 2u, lo, hi, false, "straddles hi"},
        {hi, 1u, lo, hi, false, "starts at hi"},
        {lo - 1u, 1u, lo, hi, false, "one byte below lo"},
        {lo - 1u, 2u, lo, hi, false, "straddles lo"},
        {0u, 16u, lo, hi, false, "wholly below"},
        {lo, 1u, hi, lo, false, "inverted window holds nothing"},
        {max - 15u, 32u, 0u, max, false, "base + bytes wraps"},
        {lo, max, lo, hi, false, "bytes near 2^64"},
        {0x1800u, max - 0x7FFu, lo, hi, false, "0x1800 + bytes wraps to 0x1000"},
        {16u, max, 0u, max, false, "past the whole space"},
        {0u, max, 0u, max, true, "the whole space"},
        {max - 16u, 16u, 0u, max, true, "ends at the top"},
        {max - 16u, 17u, 0u, max, false, "one byte past the top"},
        {max - 1u, 1u, max - 1u, max, true, "a one-byte window at the top"},
    };
    for (const auto& c : cases) {
        EXPECT_EQ(ec_range_ok(c.base, c.bytes, c.lo, c.hi) != 0, c.ok) << c.why;
    }
}

// §Argument checks, ec_check_erisc_ctrl: "512-aligned and wholly inside L1",
// checked directly and, for each refusal, through both roles' ctrl_base.
TEST(EthCreditArgChecksCtrl, AlignmentAndFitAtTheEdgesOfL1) {
    const Admitted f = AdmitForChecks();
    ASSERT_EQ(f.b.l1_lo % EC_ETH_CONTROL_ALIGN, 0u);
    ASSERT_EQ(f.b.l1_hi % EC_ETH_CONTROL_ALIGN, 0u);
    for (uint32_t c = 0; c < kChips; ++c) {
        const ec_eth_layout_t& l = f.adm.layout.chips[c].eth;
        const uint32_t ctrl = f.adm.layout.chips[c].eth_ctrl_base;
        ASSERT_GT(l.control_bytes, 0u);
        const struct {
            uint32_t value;
            uint32_t code;
            uint32_t constraint;
            const char* why;
        } cases[] = {
            {ctrl, EC_CHECK_OK, EC_CONSTRAINT_NONE, "the admitted base"},
            {f.b.l1_lo, EC_CHECK_OK, EC_CONSTRAINT_NONE, "at l1_lo"},
            {ctrl + 16u, EC_CHECK_ALIGNMENT, EC_CONSTRAINT_ETH_BASE, "ctrl_base + 16"},
            {ctrl + 64u, EC_CHECK_ALIGNMENT, EC_CONSTRAINT_ETH_BASE, "ctrl_base + 64"},
            {ctrl + 256u, EC_CHECK_ALIGNMENT, EC_CONSTRAINT_ETH_BASE, "ctrl_base + 256"},
            {ctrl + 511u, EC_CHECK_ALIGNMENT, EC_CONSTRAINT_ETH_BASE, "ctrl_base + 511"},
            // Aligned but outside L1.
            {f.b.l1_lo - 512u, EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW, "below l1_lo"},
            {0u, EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW, "at 0"},
            {f.b.l1_hi, EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW, "at l1_hi"},
            {f.b.l1_hi - 512u,
             EC_CHECK_CAPACITY,
             EC_CONSTRAINT_ETH_WINDOW,
             "control_bytes does not fit in the last 512 bytes"},
        };
        for (const auto& k : cases) {
            EXPECT_TRUE(IsCheck(ec_check_erisc_ctrl(k.value, &l, &f.b), k.code, k.constraint))
                << k.why << ", chip " << c;
            if (k.code == EC_CHECK_OK) {
                continue;
            }
            EXPECT_TRUE(
                IsCheck(CheckE0(f, c, [&](ec_e0_args_t& a) { a.erisc.ctrl_base = k.value; }), k.code, k.constraint))
                << "E0 " << k.why << ", chip " << c;
            EXPECT_TRUE(
                IsCheck(CheckE1(f, c, [&](ec_e1_args_t& a) { a.erisc.ctrl_base = k.value; }), k.code, k.constraint))
                << "E1 " << k.why << ", chip " << c;
        }

        // The control area ends exactly at l1_hi, then one byte past it.
        ec_bounds_t tight = f.b;
        tight.l1_hi = ctrl + l.control_bytes;
        EXPECT_TRUE(IsCheck(ec_check_erisc_ctrl(ctrl, &l, &tight), EC_CHECK_OK, EC_CONSTRAINT_NONE));
        tight.l1_hi -= 1u;
        EXPECT_TRUE(IsCheck(ec_check_erisc_ctrl(ctrl, &l, &tight), EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW));
        // l1_lo above ctrl_base.
        ec_bounds_t raised = f.b;
        raised.l1_lo = ctrl + 16u;
        EXPECT_TRUE(IsCheck(ec_check_erisc_ctrl(ctrl, &l, &raised), EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW));
    }
}

// §Argument checks, ec_check_fwd_targets: "the X280 coordinate and every
// forwarder area, each 64-aligned and wholly inside LIM". A coordinate past 63
// is COORD; a bad area is CAPACITY/FWD_AREA with value = the area, bound = p.
// Each case is checked directly and through both roles.
TEST(EthCreditArgChecksFwdTargets, CoordinatesAndAreasAtTheirEdges) {
    const Admitted f = AdmitForChecks();
    const uint32_t P = f.cfg.shape.P;
    const uint32_t lim_lo = static_cast<uint32_t>(f.b.lim_lo);
    const uint32_t lim_hi = static_cast<uint32_t>(f.b.lim_hi);
    for (uint32_t c = 0; c < kChips; ++c) {
        const ChipLayout& l = f.adm.layout.chips[c];
        const ec_x280_layout_t* lim = &l.x280;
        std::array<uint32_t, EC_MAX_P> area{};
        for (uint32_t p = 0; p < P; ++p) {
            area[p] = static_cast<uint32_t>(l.fwd_area(p));
        }

        const struct {
            uint32_t x, y;
            bool ok;
        } coords[] = {
            {8u, 26u, true},
            {63u, 63u, true},
            {0u, 0u, true},
            {64u, 26u, false},
            {8u, 64u, false},
            {0xFFFFFFFFu, 0u, false}};
        for (const auto& k : coords) {
            const uint32_t want = k.ok ? EC_CHECK_OK : EC_CHECK_COORD;
            const std::string at =
                "(" + std::to_string(k.x) + ", " + std::to_string(k.y) + "), chip " + std::to_string(c);
            EXPECT_EQ(ec_check_fwd_targets(x280_noc_coord_t{k.x, k.y}, area.data(), P, lim, &f.b).code, want) << at;
            EXPECT_EQ(
                CheckE0(
                    f,
                    c,
                    [&](ec_e0_args_t& a) {
                        a.erisc.fwd.x = k.x;
                        a.erisc.fwd.y = k.y;
                    })
                    .code,
                want)
                << "E0 " << at;
            EXPECT_EQ(
                CheckE1(
                    f,
                    c,
                    [&](ec_e1_args_t& a) {
                        a.erisc.fwd.x = k.x;
                        a.erisc.fwd.y = k.y;
                    })
                    .code,
                want)
                << "E1 " << at;
        }

        for (uint32_t p = 0; p < P; ++p) {
            const struct {
                uint32_t value;
                const char* why;
            } bads[] = {
                {area[p] + 16u, "16-aligned only"},
                {area[p] + 32u, "32-aligned only"},
                {lim_lo - 64u, "below lim_lo"},
                {lim_hi, "at lim_hi"},
                {lim_hi - 64u, "area_bytes overruns lim_hi"},
            };
            for (const auto& k : bads) {
                std::array<uint32_t, EC_MAX_P> a = area;
                a[p] = k.value;
                const std::initializer_list<std::pair<const char*, ec_check_t>> got = {
                    {"direct", ec_check_fwd_targets(x280_noc_coord_t{8u, 26u}, a.data(), P, lim, &f.b)},
                    {"E0", CheckE0(f, c, [&](ec_e0_args_t& r) { r.erisc.fwd_area[p] = k.value; })},
                    {"E1", CheckE1(f, c, [&](ec_e1_args_t& r) { r.erisc.fwd_area[p] = k.value; })},
                };
                for (const auto& [who, check] : got) {
                    EXPECT_TRUE(IsCheck(check, EC_CHECK_CAPACITY, EC_CONSTRAINT_FWD_AREA))
                        << who << " " << k.why << ", p " << p << ", chip " << c;
                    EXPECT_EQ(check.value, k.value) << who << " " << k.why << ", p " << p << ", chip " << c;
                    EXPECT_EQ(check.bound, p) << who << " " << k.why << ", p " << p << ", chip " << c;
                }
            }
            // The area ends exactly at lim_hi, then one byte past it.
            ec_bounds_t tight = f.b;
            std::array<uint32_t, EC_MAX_P> a = area;
            for (uint32_t q = 0; q < P; ++q) {
                a[q] = area[0];  // every forwarder names area 0 so only one end matters
            }
            a[p] = area[p];
            tight.lim_hi = std::max(area[0], area[p]) + static_cast<uint64_t>(l.x280.area_bytes);
            EXPECT_TRUE(IsCheck(
                ec_check_fwd_targets(x280_noc_coord_t{8u, 26u}, a.data(), P, lim, &tight),
                EC_CHECK_OK,
                EC_CONSTRAINT_NONE))
                << "area ends at lim_hi, p " << p;
            tight.lim_hi -= 1u;
            EXPECT_TRUE(IsCheck(
                ec_check_fwd_targets(x280_noc_coord_t{8u, 26u}, a.data(), P, lim, &tight),
                EC_CHECK_CAPACITY,
                EC_CONSTRAINT_FWD_AREA))
                << "area overruns lim_hi by one byte, p " << p;
        }
    }
}

// §Argument checks, prefix: identity, the shape against the expected one, and
// the scalar controls, each BAD_PARAM/ARGS. Checked through both roles, since
// they share the prefix (ABI migration spec §2); each case mutates the shared
// ec_erisc_args_t field by name. The role follows from kernel_id (spec §3), so
// a record carrying E1's, E0's or any forwarder's id of either chip is refused
// by the other role, and a correct record is refused against another expected
// shape. The unmutated record is accepted, so each refusal is the case's doing.
TEST(EthCreditArgChecksPrefix, EachBadFieldWrongRoleOrShapeIsRefusedByBothRoles) {
    const Admitted f = AdmitForChecks();
    struct Case {
        const char* name;
        void (*mutate)(ec_erisc_args_t& e, const ChipArguments& a);
    };
    const Case cases[] = {
        {"magic", [](ec_erisc_args_t& e, const ChipArguments&) { e.magic = EC_RESULT_MAGIC; }},
        {"magic zero", [](ec_erisc_args_t& e, const ChipArguments&) { e.magic = 0u; }},
        {"magic ECB1", [](ec_erisc_args_t& e, const ChipArguments&) { e.magic = 0x45434231u; }},
        {"role fwd", [](ec_erisc_args_t& e, const ChipArguments& a) { e.kernel_id = a.fwd[0].kernel_id; }},
        {"role none", [](ec_erisc_args_t& e, const ChipArguments&) { e.kernel_id = 0u; }},
        {"P", [](ec_erisc_args_t& e, const ChipArguments&) { e.shape.P -= 1u; }},
        {"n_link", [](ec_erisc_args_t& e, const ChipArguments&) { e.shape.n_link *= 2u; }},
        {"n_deliver", [](ec_erisc_args_t& e, const ChipArguments&) { e.shape.n_deliver /= 2u; }},
        {"slot_size", [](ec_erisc_args_t& e, const ChipArguments&) { e.shape.slot_size += 16u; }},
        {"batch", [](ec_erisc_args_t& e, const ChipArguments&) { e.shape.batch = 1u; }},
        {"nonce 0", [](ec_erisc_args_t& e, const ChipArguments&) { e.run_nonce = 0u; }},
        {"cadence 0", [](ec_erisc_args_t& e, const ChipArguments&) { e.poll_cadence = 0u; }},
        {"cadence 1000", [](ec_erisc_args_t& e, const ChipArguments&) { e.poll_cadence = 1000u; }},
        {"cadence 3", [](ec_erisc_args_t& e, const ChipArguments&) { e.poll_cadence = 3u; }},
        {"budget 0",
         [](ec_erisc_args_t& e, const ChipArguments&) {
             e.stall_budget_lo = 0u;
             e.stall_budget_hi = 0u;
         }},
        {"barrier cap 0",
         [](ec_erisc_args_t& e, const ChipArguments&) {
             e.barrier_cap_lo = 0u;
             e.barrier_cap_hi = 0u;
         }},
    };
    for (uint32_t c = 0; c < kChips; ++c) {
        const ChipArguments& a = f.args.chips[c];
        ASSERT_EQ(a.e0.erisc.shape.batch, 2u) << "the batch case needs a batch other than 1";
        ASSERT_TRUE(a.has_e1);
        ASSERT_TRUE(IsCheck(CheckE0(f, c, [](ec_e0_args_t&) {}), EC_CHECK_OK, EC_CONSTRAINT_NONE));
        ASSERT_TRUE(IsCheck(CheckE1(f, c, [](ec_e1_args_t&) {}), EC_CHECK_OK, EC_CONSTRAINT_NONE));
        for (const Case& k : cases) {
            EXPECT_TRUE(IsCheck(
                CheckE0(f, c, [&](ec_e0_args_t& r) { k.mutate(r.erisc, a); }), EC_CHECK_BAD_PARAM, EC_CONSTRAINT_ARGS))
                << "E0 " << k.name << ", chip " << c;
            EXPECT_TRUE(IsCheck(
                CheckE1(f, c, [&](ec_e1_args_t& r) { k.mutate(r.erisc, a); }), EC_CHECK_BAD_PARAM, EC_CONSTRAINT_ARGS))
                << "E1 " << k.name << ", chip " << c;
        }

        // lim_end is checked after identity: outside (lim_lo, lim_hi] it is
        // CAPACITY/LIM_WINDOW, and with a bad magic as well it is still the
        // identity refusal.
        const uint64_t lim_hi = f.b.lim_hi;
        for (const uint64_t end : {uint64_t{0}, f.b.lim_lo, lim_hi + 1u}) {
            const auto set_end = [end](ec_erisc_args_t& e) {
                e.lim_end_lo = static_cast<uint32_t>(end);
                e.lim_end_hi = static_cast<uint32_t>(end >> 32);
            };
            EXPECT_TRUE(IsCheck(
                CheckE0(f, c, [&](ec_e0_args_t& r) { set_end(r.erisc); }), EC_CHECK_CAPACITY, EC_CONSTRAINT_LIM_WINDOW))
                << "E0 lim_end 0x" << std::hex << end << ", chip " << c;
            EXPECT_TRUE(IsCheck(
                CheckE1(f, c, [&](ec_e1_args_t& r) { set_end(r.erisc); }), EC_CHECK_CAPACITY, EC_CONSTRAINT_LIM_WINDOW))
                << "E1 lim_end 0x" << std::hex << end << ", chip " << c;
            EXPECT_TRUE(IsCheck(
                CheckE0(
                    f,
                    c,
                    [&](ec_e0_args_t& r) {
                        set_end(r.erisc);
                        r.erisc.magic = 0u;
                    }),
                EC_CHECK_BAD_PARAM,
                EC_CONSTRAINT_ARGS))
                << "E0 bad magic and lim_end 0x" << std::hex << end << ", chip " << c;
        }

        std::vector<std::pair<std::string, uint32_t>> not_e0{{"E1", a.e1.erisc.kernel_id}};
        std::vector<std::pair<std::string, uint32_t>> not_e1{{"E0", a.e0.erisc.kernel_id}};
        for (uint32_t p = 0; p < f.cfg.shape.P; ++p) {
            not_e0.emplace_back("fwd " + std::to_string(p), a.fwd[p].kernel_id);
            not_e1.emplace_back("fwd " + std::to_string(p), a.fwd[p].kernel_id);
        }
        // The other chip's record of the wrong role is still the wrong role.
        not_e0.emplace_back("other chip's E1", f.args.chips[1u - c].e1.erisc.kernel_id);
        not_e1.emplace_back("other chip's E0", f.args.chips[1u - c].e0.erisc.kernel_id);
        for (const auto& [who, id] : not_e0) {
            ASSERT_NE(ec_kernel_role(id), static_cast<uint32_t>(EC_ROLE_E0)) << who;
            EXPECT_TRUE(IsCheck(
                CheckE0(f, c, [&](ec_e0_args_t& r) { r.erisc.kernel_id = id; }),
                EC_CHECK_BAD_PARAM,
                EC_CONSTRAINT_ARGS))
                << "E0 record carrying " << who << "'s kernel id, chip " << c;
        }
        for (const auto& [who, id] : not_e1) {
            ASSERT_NE(ec_kernel_role(id), static_cast<uint32_t>(EC_ROLE_E1)) << who;
            EXPECT_TRUE(IsCheck(
                CheckE1(f, c, [&](ec_e1_args_t& r) { r.erisc.kernel_id = id; }),
                EC_CHECK_BAD_PARAM,
                EC_CONSTRAINT_ARGS))
                << "E1 record carrying " << who << "'s kernel id, chip " << c;
        }

        for (int field = 0; field < 5; ++field) {
            ec_shape_t other = f.cfg.shape;
            uint32_t* v[] = {&other.P, &other.n_link, &other.n_deliver, &other.slot_size, &other.batch};
            *v[field] = field == 0 ? 3u : field == 4 ? 1u : field == 3 ? other.slot_size + 16u : *v[field] * 2u;
            EXPECT_TRUE(IsCheck(ec_check_e0_args(&a.e0, &other, &f.b), EC_CHECK_BAD_PARAM, EC_CONSTRAINT_ARGS))
                << "E0 against expected shape field " << field << ", chip " << c;
            EXPECT_TRUE(IsCheck(ec_check_e1_args(&a.e1, &other, &f.b), EC_CHECK_BAD_PARAM, EC_CONSTRAINT_ARGS))
                << "E1 against expected shape field " << field << ", chip " << c;
        }
    }
}

// The stall budget and barrier cap are 64-bit values split over two words:
// "budget 0" and "barrier cap 0" refer to the value, so a zero low word with
// a nonzero high word is a nonzero value. Every power of two is a cadence.
TEST(EthCreditArgChecksPrefix, SplitWordsAndPowerOfTwoCadencesAreAccepted) {
    const Admitted f = AdmitForChecks();
    const auto ok_e0 = [&](auto mutate, const char* why) {
        for (uint32_t c = 0; c < kChips; ++c) {
            EXPECT_TRUE(IsCheck(CheckE0(f, c, mutate), EC_CHECK_OK, EC_CONSTRAINT_NONE)) << "E0 " << why;
            ec_e0_args_t e0 = f.args.chips[c].e0;
            mutate(e0);
            // Copy the mutated prefix scalars onto E1.
            ec_e1_args_t e1 = f.args.chips[c].e1;
            e1.erisc.poll_cadence = e0.erisc.poll_cadence;
            e1.erisc.stall_budget_lo = e0.erisc.stall_budget_lo;
            e1.erisc.stall_budget_hi = e0.erisc.stall_budget_hi;
            e1.erisc.barrier_cap_lo = e0.erisc.barrier_cap_lo;
            e1.erisc.barrier_cap_hi = e0.erisc.barrier_cap_hi;
            EXPECT_TRUE(IsCheck(ec_check_e1_args(&e1, &f.cfg.shape, &f.b), EC_CHECK_OK, EC_CONSTRAINT_NONE))
                << "E1 " << why;
        }
    };
    ok_e0(
        [](ec_e0_args_t& a) {
            a.erisc.stall_budget_lo = 0u;
            a.erisc.stall_budget_hi = 1u;
        },
        "budget high word only");
    ok_e0(
        [](ec_e0_args_t& a) {
            a.erisc.stall_budget_lo = 1u;
            a.erisc.stall_budget_hi = 0u;
        },
        "budget 1");
    ok_e0(
        [](ec_e0_args_t& a) {
            a.erisc.barrier_cap_lo = 0u;
            a.erisc.barrier_cap_hi = 1u;
        },
        "cap high word only");
    ok_e0(
        [](ec_e0_args_t& a) {
            a.erisc.barrier_cap_lo = 1u;
            a.erisc.barrier_cap_hi = 0u;
        },
        "cap 1");
    ok_e0([](ec_e0_args_t& a) { a.erisc.poll_cadence = 1u; }, "cadence 1");
    ok_e0([](ec_e0_args_t& a) { a.erisc.poll_cadence = 2u; }, "cadence 2");
    ok_e0([](ec_e0_args_t& a) { a.erisc.poll_cadence = EC_DEFAULT_POLL_CADENCE; }, "cadence default");
}

// §Argument checks, prefix: the barrier ready table (EC_BARRIER_MAX_WORKERS
// core_sync slots) is 8-aligned and inside LIM, else CAPACITY/READY.
TEST(EthCreditArgChecksPrefix, BarrierReadyBaseMisalignedOrOutsideLimIsRefused) {
    const Admitted f = AdmitForChecks();
    const uint32_t table = EC_BARRIER_MAX_WORKERS * CORE_SYNC_SLOT_BYTES;
    ASSERT_EQ(CORE_SYNC_SLOT_BYTES, 8u);
    const uint32_t lim_lo = static_cast<uint32_t>(f.b.lim_lo);
    const uint32_t lim_hi = static_cast<uint32_t>(f.b.lim_hi);
    for (uint32_t c = 0; c < kChips; ++c) {
        const uint32_t base = f.args.chips[c].e0.erisc.barrier_ready_base;
        for (uint32_t bad : {base + 4u, base + 1u, lim_lo - 8u, lim_hi, lim_hi - table + 8u, 0u}) {
            EXPECT_TRUE(IsCheck(
                CheckE0(f, c, [&](ec_e0_args_t& a) { a.erisc.barrier_ready_base = bad; }),
                EC_CHECK_CAPACITY,
                EC_CONSTRAINT_READY))
                << "E0 ready base " << std::hex << bad << ", chip " << c;
            EXPECT_TRUE(IsCheck(
                CheckE1(f, c, [&](ec_e1_args_t& a) { a.erisc.barrier_ready_base = bad; }),
                EC_CHECK_CAPACITY,
                EC_CONSTRAINT_READY))
                << "E1 ready base " << std::hex << bad << ", chip " << c;
        }
        // The table ending exactly at lim_hi, and starting exactly at lim_lo, fit.
        for (uint32_t good : {lim_hi - table, lim_lo, base + 8u}) {
            EXPECT_TRUE(IsCheck(
                CheckE0(f, c, [&](ec_e0_args_t& a) { a.erisc.barrier_ready_base = good; }),
                EC_CHECK_OK,
                EC_CONSTRAINT_NONE))
                << "E0 ready base " << std::hex << good << ", chip " << c;
            EXPECT_TRUE(IsCheck(
                CheckE1(f, c, [&](ec_e1_args_t& a) { a.erisc.barrier_ready_base = good; }),
                EC_CHECK_OK,
                EC_CONSTRAINT_NONE))
                << "E1 ready base " << std::hex << good << ", chip " << c;
        }
    }
}

// The prepared shape backs one LIM way fewer than the reset shape every kernel
// compiles in as its LIM bound. A forwarder area in that unbacked way passes
// the reset-shape bound, and both roles' validators refuse it once the record
// carries the backed end as lim_end.
TEST(EthCreditArgChecksPrefix, AreaInAnUnbackedWayIsRefusedAtTheRecordsLimEnd) {
    const Admitted f = AdmitForChecks();
    const ec_bounds_t map{f.b.l1_lo, f.b.l1_hi, f.b.lim_lo, X280_LIM_END};
    const uint64_t backed_end = X280_LIM_BASE + x280::lim_backed_bytes(X280_PINNED_INPUT_WAYENABLE);
    ec_x280_layout_t lim{};
    ASSERT_EQ(ec_compute_x280_layout(&f.cfg.shape, &lim).code, static_cast<uint32_t>(EC_CHECK_OK));
    const uint32_t unbacked = static_cast<uint32_t>((X280_LIM_END - lim.area_bytes) & ~uint64_t{EC_LIM_ALIGN - 1u});
    ASSERT_GT(unbacked + uint64_t{lim.area_bytes}, backed_end) << "the area must reach the unbacked way";
    for (uint32_t c = 0; c < kChips; ++c) {
        for (const auto& [end, want, constraint, what] :
             {std::tuple{uint64_t{X280_LIM_END}, EC_CHECK_OK, EC_CONSTRAINT_NONE, "reset-shape end"},
              std::tuple{backed_end, EC_CHECK_CAPACITY, EC_CONSTRAINT_FWD_AREA, "prepared backed end"}}) {
            ec_e0_args_t e0 = f.args.chips[c].e0;
            ec_e1_args_t e1 = f.args.chips[c].e1;
            for (ec_erisc_args_t* e : {&e0.erisc, &e1.erisc}) {
                e->fwd_area[0] = unbacked;
                e->lim_end_lo = static_cast<uint32_t>(end);
                e->lim_end_hi = static_cast<uint32_t>(end >> 32);
            }
            EXPECT_TRUE(IsCheck(ec_check_e0_args(&e0, &f.cfg.shape, &map), want, constraint))
                << "E0 at the " << what << ", chip " << c;
            EXPECT_TRUE(IsCheck(ec_check_e1_args(&e1, &f.cfg.shape, &map), want, constraint))
                << "E1 at the " << what << ", chip " << c;
        }
    }
}

// An E0 record of the previous layout ('ECB2', a 30-word prefix without
// lim_end), read into the current record as a kernel reads its runtime words,
// skews every later word into the wrong field. It is refused as the ABI
// mismatch it is, not by whichever field the skew corrupts first.
TEST(EthCreditArgChecksPrefix, PreviousLayoutRecordIsRefusedAsTheAbiMismatch) {
    const Admitted f = AdmitForChecks();
    for (uint32_t c = 0; c < kChips; ++c) {
        std::vector<uint32_t> words = pack_runtime_args(f.args.chips[c].e0);
        ASSERT_EQ(words.size(), static_cast<size_t>(EC_E0_ARG_WORDS));
        words.erase(words.begin() + 30, words.begin() + 32);  // drop lim_end_lo, lim_end_hi
        words[0] = 0x45434232u;                               // 'ECB2'
        ec_e0_args_t old{};
        std::memcpy(&old, words.data(), words.size() * sizeof(uint32_t));
        EXPECT_TRUE(IsCheck(ec_check_e0_args(&old, &f.cfg.shape, &f.b), EC_CHECK_BAD_PARAM, EC_CONSTRAINT_ARGS))
            << "chip " << c;
    }
}

// §Argument checks, prefix: stream ids 1..31, else STREAM_ID/STREAM with value
// = the id and bound = its index p (ABI migration spec §3). E0 also checks
// local_stream_id. A repeat within the first P entries of erisc.stream_id
// (both roles) or of E0's local_stream_id is STREAM_ID/STREAM with value = the
// repeated id and bound = the earlier index holding it. Every ordered pair
// q < p is tried, with the repeated id placed only at p, so the first
// duplicate the check meets is (q, p).
TEST(EthCreditArgChecksStream, IdsOutsideOneThroughThirtyOneOrRepeatedAreRefusedInEveryField) {
    const Admitted f = AdmitForChecks();
    const uint32_t P = f.cfg.shape.P;
    ASSERT_EQ(P, 4u);
    const auto expect_refused = [](const ec_check_t& k, uint32_t id, uint32_t p, const std::string& where) {
        EXPECT_TRUE(IsCheck(k, EC_CHECK_STREAM_ID, EC_CONSTRAINT_STREAM)) << where;
        EXPECT_EQ(k.value, id) << where;
        EXPECT_EQ(k.bound, p) << where;
    };
    for (uint32_t c = 0; c < kChips; ++c) {
        const ChipArguments& a = f.args.chips[c];
        for (uint32_t p = 0; p < P; ++p) {
            for (uint32_t bad : {0u, 32u, 0xFFFFFFFFu}) {
                const std::string at =
                    "[" + std::to_string(p) + "] = " + std::to_string(bad) + ", chip " + std::to_string(c);
                expect_refused(
                    CheckE0(f, c, [&](ec_e0_args_t& r) { r.erisc.stream_id[p] = bad; }), bad, p, "E0 stream_id" + at);
                expect_refused(
                    CheckE0(f, c, [&](ec_e0_args_t& r) { r.local_stream_id[p] = bad; }),
                    bad,
                    p,
                    "E0 local_stream_id" + at);
                expect_refused(
                    CheckE1(f, c, [&](ec_e1_args_t& r) { r.erisc.stream_id[p] = bad; }), bad, p, "E1 stream_id" + at);
            }
            // The fixture's ids are 1..P, so MIN at p = 0 and MAX anywhere keep
            // them distinct; MIN elsewhere would duplicate p = 0's.
            std::vector<uint32_t> goods{EC_STREAM_ID_MAX};
            if (p == 0u) {
                goods.push_back(EC_STREAM_ID_MIN);
            }
            for (uint32_t good : goods) {
                EXPECT_TRUE(IsCheck(
                    CheckE0(f, c, [&](ec_e0_args_t& r) { r.erisc.stream_id[p] = good; }),
                    EC_CHECK_OK,
                    EC_CONSTRAINT_NONE))
                    << "E0 stream_id[" << p << "] = " << good;
                EXPECT_TRUE(IsCheck(
                    CheckE0(f, c, [&](ec_e0_args_t& r) { r.local_stream_id[p] = good; }),
                    EC_CHECK_OK,
                    EC_CONSTRAINT_NONE))
                    << "E0 local_stream_id[" << p << "] = " << good;
                EXPECT_TRUE(IsCheck(
                    CheckE1(f, c, [&](ec_e1_args_t& r) { r.erisc.stream_id[p] = good; }),
                    EC_CHECK_OK,
                    EC_CONSTRAINT_NONE))
                    << "E1 stream_id[" << p << "] = " << good;
            }
        }

        // The fixture's ids are distinct, so each case holds exactly one repeat.
        for (const uint32_t* ids : {a.e0.erisc.stream_id, a.e0.local_stream_id, a.e1.erisc.stream_id}) {
            ASSERT_EQ(std::set<uint32_t>(ids, ids + P).size(), P) << "chip " << c;
        }
        for (uint32_t p = 1; p < P; ++p) {
            for (uint32_t q = 0; q < p; ++q) {
                const std::string at =
                    "[" + std::to_string(q) + "] repeated at [" + std::to_string(p) + "], chip " + std::to_string(c);
                const uint32_t e0_id = a.e0.erisc.stream_id[q];
                expect_refused(
                    CheckE0(f, c, [&](ec_e0_args_t& r) { r.erisc.stream_id[p] = e0_id; }),
                    e0_id,
                    q,
                    "E0 stream_id" + at);
                const uint32_t local_id = a.e0.local_stream_id[q];
                expect_refused(
                    CheckE0(f, c, [&](ec_e0_args_t& r) { r.local_stream_id[p] = local_id; }),
                    local_id,
                    q,
                    "E0 local_stream_id" + at);
                const uint32_t e1_id = a.e1.erisc.stream_id[q];
                expect_refused(
                    CheckE1(f, c, [&](ec_e1_args_t& r) { r.erisc.stream_id[p] = e1_id; }),
                    e1_id,
                    q,
                    "E1 stream_id" + at);
            }
        }
        // All four equal: the first repeat is at p = 1 against index 0.
        const uint32_t id0 = a.e0.erisc.stream_id[0];
        expect_refused(
            CheckE0(
                f,
                c,
                [&](ec_e0_args_t& r) {
                    for (uint32_t p = 0; p < P; ++p) {
                        r.erisc.stream_id[p] = id0;
                    }
                }),
            id0,
            0u,
            "E0 all equal, chip " + std::to_string(c));
    }
}

// §Argument checks, E0 when EC_E0_FLAG_OUTBOUND_ACTIVE: tx_fifo, at
// ctrl_base + tx_offset, fits in L1, else CAPACITY/ETH_WINDOW; peer_ctrl_base
// is 512-aligned and inside L1, and peer_rx_base 16-aligned, at or past
// peer_ctrl_base + control_bytes, with its P x N_LINK x SLOT region inside L1,
// else CAPACITY/PEER_BASE. A record with outbound clear skips the tx and peer
// checks.
TEST(EthCreditArgChecksE0, TxAndPeerRegionsAreCheckedOnlyWhenOutbound) {
    const Admitted f = AdmitForChecks();
    const uint32_t region = SlotRegionBytes(f);
    const auto garble = [](ec_e0_args_t& a) {
        a.peer_ctrl_base = 0xDEADBEE3u;
        a.peer_rx_base = 0xDEADBEE5u;
    };
    for (uint32_t c = 0; c < kChips; ++c) {
        const ChipLayout& l = f.adm.layout.chips[c];
        const ec_e0_args_t& e0 = f.args.chips[c].e0;
        const uint32_t control = l.eth.control_bytes;
        ASSERT_NE(e0.erisc.flags & EC_E0_FLAG_OUTBOUND_ACTIVE, 0u);

        // L1 ends inside tx_fifo: the control area still fits, P x N_LINK x
        // SLOT does not.
        ec_bounds_t tight = f.b;
        tight.l1_hi = e0.erisc.ctrl_base + l.eth.tx_offset + region - 16u;
        ASSERT_GE(tight.l1_hi, e0.erisc.ctrl_base + control);
        EXPECT_TRUE(IsCheck(ec_check_e0_args(&e0, &f.cfg.shape, &tight), EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW))
            << "tx_fifo overruns l1_hi, chip " << c;

        for (uint32_t bad :
             {e0.peer_ctrl_base + 16u, e0.peer_ctrl_base + 256u, f.b.l1_lo - 512u, f.b.l1_hi, f.b.l1_hi - 512u}) {
            EXPECT_TRUE(IsCheck(
                CheckE0(f, c, [&](ec_e0_args_t& a) { a.peer_ctrl_base = bad; }),
                EC_CHECK_CAPACITY,
                EC_CONSTRAINT_PEER_BASE))
                << "peer_ctrl_base " << std::hex << bad << ", chip " << c;
        }
        for (uint32_t bad :
             {e0.peer_rx_base + 8u,
              e0.peer_rx_base + 4u,
              e0.peer_ctrl_base,
              e0.peer_ctrl_base + control - 16u,
              f.b.l1_hi - region + 16u,
              f.b.l1_hi}) {
            EXPECT_TRUE(IsCheck(
                CheckE0(f, c, [&](ec_e0_args_t& a) { a.peer_rx_base = bad; }),
                EC_CHECK_CAPACITY,
                EC_CONSTRAINT_PEER_BASE))
                << "peer_rx_base " << std::hex << bad << ", chip " << c;
        }
        // Edges that fit: rx directly after the peer's control area, and the
        // region ending exactly at l1_hi.
        ASSERT_EQ(control % 16u, 0u);
        ASSERT_EQ(region % 16u, 0u);
        for (uint32_t good : {e0.peer_ctrl_base + control, f.b.l1_hi - region}) {
            EXPECT_TRUE(IsCheck(
                CheckE0(f, c, [&](ec_e0_args_t& a) { a.peer_rx_base = good; }), EC_CHECK_OK, EC_CONSTRAINT_NONE))
                << "peer_rx_base " << std::hex << good << ", chip " << c;
        }

        // The same garbled peer fields pass once outbound is cleared, and are
        // refused while it is set.
        EXPECT_TRUE(IsCheck(
            CheckE0(
                f,
                c,
                [&](ec_e0_args_t& a) {
                    garble(a);
                    a.erisc.flags &= ~EC_E0_FLAG_OUTBOUND_ACTIVE;
                }),
            EC_CHECK_OK,
            EC_CONSTRAINT_NONE))
            << "outbound cleared, chip " << c;
        EXPECT_EQ(CheckE0(f, c, garble).code, static_cast<uint32_t>(EC_CHECK_CAPACITY)) << "outbound set, chip " << c;
    }

    const Admitted one = AdmitForChecks(2u, Direction::AToB);
    ASSERT_EQ(one.args.chips[1].e0.erisc.flags & EC_E0_FLAG_OUTBOUND_ACTIVE, 0u) << "chip 1 does not send in AToB";
    EXPECT_TRUE(IsCheck(CheckE0(one, 1u, garble), EC_CHECK_OK, EC_CONSTRAINT_NONE)) << "derived idle E0";
}

// §Argument checks, E1: rx_base 16-aligned, at or past ctrl_base +
// control_bytes, and its P x N_LINK x SLOT region inside L1, else
// CAPACITY/ETH_WINDOW; a bad peer_ctrl_base is CAPACITY/PEER_BASE.
TEST(EthCreditArgChecksE1, BadRxRegionOrPeerControlBaseIsRefused) {
    const Admitted f = AdmitForChecks();
    const uint32_t region = SlotRegionBytes(f);
    for (uint32_t c = 0; c < kChips; ++c) {
        const ec_e1_args_t& e1 = f.args.chips[c].e1;
        const uint32_t control = f.adm.layout.chips[c].eth.control_bytes;
        for (uint32_t bad :
             {e1.rx_base + 8u,
              e1.rx_base + 1u,
              e1.erisc.ctrl_base,
              e1.erisc.ctrl_base + control - 16u,
              f.b.l1_hi - region + 16u,
              f.b.l1_hi,
              0u}) {
            EXPECT_TRUE(IsCheck(
                CheckE1(f, c, [&](ec_e1_args_t& a) { a.rx_base = bad; }), EC_CHECK_CAPACITY, EC_CONSTRAINT_ETH_WINDOW))
                << "rx_base " << std::hex << bad << ", chip " << c;
        }
        for (uint32_t good : {e1.erisc.ctrl_base + control, f.b.l1_hi - region}) {
            EXPECT_TRUE(
                IsCheck(CheckE1(f, c, [&](ec_e1_args_t& a) { a.rx_base = good; }), EC_CHECK_OK, EC_CONSTRAINT_NONE))
                << "rx_base " << std::hex << good << ", chip " << c;
        }
        for (uint32_t bad :
             {e1.peer_ctrl_base + 16u, e1.peer_ctrl_base + 256u, f.b.l1_lo - 512u, f.b.l1_hi, f.b.l1_hi - 512u}) {
            EXPECT_TRUE(IsCheck(
                CheckE1(f, c, [&](ec_e1_args_t& a) { a.peer_ctrl_base = bad; }),
                EC_CHECK_CAPACITY,
                EC_CONSTRAINT_PEER_BASE))
                << "peer_ctrl_base " << std::hex << bad << ", chip " << c;
        }
    }
}

// ---------------------------------------------------------------------------
// Host X280 wait (eth-credit host-wait spec: the floor is
// kHostWaitPhases x stall_budget_ms + kHostWaitSlackMs, false on overflow;
// Config::host_wait_ms 0 takes the floor, >= floor is kept, 0 < w < floor is
// refused as EC_CHECK_BAD_PARAM / EC_CONSTRAINT_HOST_WAIT {w, floor}).
// ---------------------------------------------------------------------------

namespace {

constexpr uint64_t kU64Max = ~0ull;
constexpr uint64_t kUntouched = 0xDEADBEEFCAFEF00Dull;

// The floor, restated from the spec rather than taken from host_wait_floor_ms().
constexpr uint64_t SpecFloor(uint64_t stall_budget_ms) { return 4u * stall_budget_ms + 5000u; }

}  // namespace

// Floor values at zero, the default budget, and the largest representable
// budget, where the result is within 3 of UINT64_MAX; then both overflow
// routes, the sum alone (the product still fits) and the product, which return
// false and leave `out` untouched.
TEST(EthCreditHostWait, FloorIsFourBudgetsPlusSlackOrFalseOnOverflow) {
    const uint64_t largest = (kU64Max - 5000u) / 4u;
    for (uint64_t budget : {uint64_t{0}, uint64_t{1}, uint64_t{1000}, uint64_t{123456789}, largest}) {
        uint64_t out = kUntouched;
        ASSERT_TRUE(x280::harness::measurement::host_wait_floor_ms(kHostWaitPhases, budget, kHostWaitSlackMs, &out))
            << "budget " << budget;
        EXPECT_EQ(out, SpecFloor(budget)) << "budget " << budget;
    }
    uint64_t out = 0;
    ASSERT_TRUE(x280::harness::measurement::host_wait_floor_ms(kHostWaitPhases, largest, kHostWaitSlackMs, &out));
    EXPECT_GE(out, kU64Max - 3u);

    const uint64_t sum_overflows = largest + 1u;
    const uint64_t product_overflows = kU64Max / 4u + 1u;
    ASSERT_LE(sum_overflows, kU64Max / 4u) << "the sum-only case must keep the product in range";
    for (uint64_t budget : {sum_overflows, kU64Max / 4u, product_overflows, kU64Max}) {
        uint64_t untouched = kUntouched;
        EXPECT_FALSE(
            x280::harness::measurement::host_wait_floor_ms(kHostWaitPhases, budget, kHostWaitSlackMs, &untouched))
            << "budget " << budget;
        EXPECT_EQ(untouched, kUntouched) << "budget " << budget;
    }
}

// Config::host_wait_ms 0 takes the floor for the budget, and a wait at or above
// the floor is kept exactly. The budget is 1234 ms rather than the default, so
// a floor computed from the default would not match.
TEST(EthCreditHostWait, ZeroTakesTheFloorAndAtOrAboveItIsKeptExactly) {
    EXPECT_EQ(Config{}.host_wait_ms, 0u);
    const uint64_t floor = SpecFloor(1234u);
    for (uint64_t w : {uint64_t{0}, floor, floor + 1u, 10u * floor}) {
        Config cfg = MakeConfig(2u, 8u, 8u, 256u, 1u, Direction::Both);
        cfg.stall_budget_ms = 1234u;
        cfg.host_wait_ms = w;
        const Admission adm = admit(cfg, RealisticCapacity());
        ASSERT_TRUE(adm.ok()) << "host_wait_ms " << w << ": " << adm.describe();
        EXPECT_EQ(adm.host_wait_ms, w == 0u ? floor : w) << "host_wait_ms " << w;
    }
}

TEST(EthCreditHostWait, BelowTheFloorIsRefusedWithValueAndBound) {
    const uint64_t floor = SpecFloor(1234u);
    for (uint64_t w : {floor - 1u, uint64_t{1}}) {
        Config cfg = MakeConfig(2u, 8u, 8u, 256u, 1u, Direction::Both);
        cfg.stall_budget_ms = 1234u;
        cfg.host_wait_ms = w;
        const Admission adm = admit(cfg, RealisticCapacity());
        ASSERT_FALSE(adm.ok()) << "host_wait_ms " << w;
        EXPECT_TRUE(IsCheck(adm.check, EC_CHECK_BAD_PARAM, EC_CONSTRAINT_HOST_WAIT)) << "host_wait_ms " << w;
        EXPECT_EQ(adm.check.value, w);
        EXPECT_EQ(adm.check.bound, floor);
        EXPECT_NE(adm.describe().find("host X280 wait"), std::string::npos) << adm.describe();
    }
}
