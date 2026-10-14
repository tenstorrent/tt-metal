// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// eth_credit across the direct Ethernet link between chips 0 and 1, in the
// plan's bring-up order: barrier and handshake with no traffic, the NoC hop
// alone, the Ethernet hop alone, the admission test, the full path, P
// forwarders, and both directions. Each step carries a run that must fail.
//
// Requires X280_ETH_CREDIT_FW_IMAGE and live resident idle on L2CPU 0 of
// both chips. Slow dispatch comes from the x280_slow_dispatch link.

#include "harness/artifacts/artifacts.hpp"
#include "harness/runner.hpp"
#include "harness/slow_dispatch.hpp"
#include "suites/eth_credit/run.hpp"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace suite = x280::suites::eth_credit;
namespace ec = x280::harness::regions::eth_credit;

/* The variable is read once, when the Metal context starts, so it is set
 * before any test body runs. It bounds each device's wait inside a
 * slow-dispatch Finish, from when Finish starts waiting, not from launch;
 * each E0/E1 wait carries its own budget. */
const bool kOperationTimeoutSet = [] {
    return setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "120", /*overwrite=*/1) == 0;
}();

class EthCreditHw : public x280::harness::Runner<int> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_ETH_CREDIT_FW_IMAGE",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                    .mesh_device_ids = suite::kChipPair,
                },
            .suite_label = "eth_credit",
        };
    }

    void SetUp() override {
        ASSERT_TRUE(kOperationTimeoutSet) << "could not set TT_METAL_OPERATION_TIMEOUT_SECONDS";
        ASSERT_TRUE(x280::harness::slow_dispatch_linked());
        Runner<int>::SetUp();
    }

    suite::RunResult run(const suite::Case& c) {
        auto r = suite::run_case(fixture(), spec(), c);
        suite::print_run(c, r);
        return r;
    }

    /** Runs `c` with every gtest failure from any thread captured instead of
     * reported, and returns the run's result; `failures` receives each
     * captured failure's message. */
    suite::RunResult run_capturing(const suite::Case& c, std::vector<std::string>& failures) {
        ::testing::TestPartResultArray captured;
        suite::RunResult r;
        {
            const ::testing::ScopedFakeTestPartResultReporter reporter(
                ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &captured);
            r = run(c);
        }
        for (int i = 0; i < captured.size(); ++i) {
            const auto& part = captured.GetTestPartResult(i);
            if (part.failed()) {
                failures.emplace_back(part.message());
            }
        }
        return r;
    }

    /** Runs `c`, which dirties `chip`'s read-back, and requires run_case to
     * report that chip's mismatch and stop before any enqueue. Captured
     * failures other than the mismatch are re-reported. */
    void expect_readback_mismatch_stops(const suite::Case& c, int chip) {
        std::vector<std::string> failures;
        const auto r = run_capturing(c, failures);
        bool reported = false;
        for (const auto& m : failures) {
            if (m.find("did not read back zero") != std::string::npos) {
                reported = true;
                EXPECT_NE(m.find(std::to_string(chip)), std::string::npos)
                    << "the mismatch failure must name chip " << chip << ": " << m;
            } else {
                ADD_FAILURE() << "unexpected failure during the run: " << m;
            }
        }
        EXPECT_TRUE(reported) << "no \"did not read back zero\" failure for dirtied chip " << chip;
        EXPECT_FALSE(r.metal_enqueued) << "EnqueueMeshWorkload called after a read-back mismatch";
        EXPECT_FALSE(r.launched);
        EXPECT_FALSE(r.chips[0].x280.enqueued()) << "chip 0 X280 session enqueued after a mismatch";
        EXPECT_FALSE(r.chips[1].x280.enqueued()) << "chip 1 X280 session enqueued after a mismatch";
    }
};

std::array<uint32_t, EC_MAX_P> each(uint32_t P, uint32_t n) {
    std::array<uint32_t, EC_MAX_P> a{};
    for (uint32_t p = 0; p < P; ++p) {
        a[p] = n;
    }
    return a;
}

constexpr uint32_t Slot(uint32_t payload) { return EC_HEADER_BYTES + payload; }

}  // namespace

// Steps 3 and 4: every participant of both chips' barriers, E0's packet mode
// and the handshake, with no traffic.
TEST_P(EthCreditHw, BarrierAndHandshakeWithNoTraffic) {
    const suite::Case c{
        .name = "barrier_handshake",
        .shape = {2u, 8u, 8u, Slot(64u), 1u},
        .direction = ec::Direction::Both,
        .packets = each(2u, 0u)};
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
}

// A per-run salt define gives E0 and E1 a JIT cache key no earlier build had,
// so both chips compile them at this launch. Slow dispatch compiles each
// chip's program inside its own LaunchProgram (distributed.cpp,
// sd_mesh_command_queue.cpp), so the handshake must survive the launch skew
// that costs.
TEST_P(EthCreditHw, BarrierAndHandshakeAtAFreshlyCompiledShape) {
    suite::Case c{
        .name = "barrier_handshake_fresh_jit",
        .shape = {3u, 4u, 4u, Slot(48u), 1u},
        .direction = ec::Direction::Both,
        .packets = each(3u, 0u)};
    c.stage.jit_salt = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 1u;
    std::printf("  JIT salt %u\n", c.stage.jit_salt);
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
}

// A refused argument is attributed to the kernel that refused it: chip A's E0
// rejects a misaligned peer_rx_base and records BAD_ARGS in its own result
// block. No kernel reads another's status, so its neighbours leave on their own
// stall budgets (or host STOP), and the host names E0 as the only origin
// (SPEC §Failure and termination).
TEST_P(EthCreditHw, RefusedE0ArgumentIsAttributedToE0) {
    suite::Case c{
        .name = "refused_e0_argument",
        .shape = {1u, 8u, 8u, Slot(64u), 1u},
        .direction = ec::Direction::AToB,
        .packets = each(1u, 100u),
        .stall_budget_ms = 500,
        .expect_x280_ready = false};
    c.stage.corrupt_e0_peer_rx = true;
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    const auto& e0 = r.chips[0].e0.block;
    EXPECT_EQ(e0.magic, EC_RESULT_MAGIC) << "E0 must leave a result for a refused argument";
    EXPECT_EQ(e0.status.state, static_cast<uint32_t>(EC_STATE_FAILED));
    EXPECT_EQ(e0.status.code, static_cast<uint32_t>(EC_ERROR_BAD_ARGS));
    EXPECT_EQ(e0.status.detail, static_cast<uint32_t>(EC_CONSTRAINT_PEER_BASE));

    const suite::Attribution& a = r.attribution;
    EXPECT_TRUE(a.failed());
    ASSERT_EQ(a.origins.size(), 1u) << a.describe();
    const suite::Culprit& origin = a.origins[0];
    EXPECT_EQ(origin.chip, r.chips[0].chip);
    EXPECT_EQ(origin.kernel_id, EC_KERNEL_ID(EC_ROLE_E0, 0u, 0u));
    EXPECT_EQ(origin.code, static_cast<uint32_t>(EC_ERROR_BAD_ARGS));
    EXPECT_EQ(origin.detail, static_cast<uint32_t>(EC_CONSTRAINT_PEER_BASE));
    const std::string text = a.describe();
    EXPECT_NE(text.find("origin"), std::string::npos) << text;
    EXPECT_NE(text.find("BAD_ARGS"), std::string::npos) << text;
}

namespace {

/* The withheld-peer negative control: chip B's E0 is withheld, so chip A's
 * E0 ends FAILED in the handshake and its forwarder leaves only on host STOP. */
suite::Case withheld_peer_case(const char* name, bool expect_x280_ready) {
    suite::Case c{
        .name = name,
        .shape = {1u, 8u, 8u, Slot(64u), 1u},
        .direction = ec::Direction::AToB,
        .packets = each(1u, 0u),
        .stall_budget_ms = 500,
        .expect_x280_ready = expect_x280_ready};
    c.stage.launch_e0_b = false;
    return c;
}

/* Step 8's first case, the smallest clean run: one forwarder and one slot. */
suite::Case small_clean_case(const char* name) {
    return suite::Case{
        .name = name, .shape = {1u, 1u, 1u, Slot(64u), 1u}, .direction = ec::Direction::AToB, .packets = each(1u, 1u)};
}

bool contains(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

}  // namespace

// Negative control for steps 3 and 4: chip B's E0 is withheld, so chip A's
// E0 must time out in the handshake, and the host attribute the run to that
// stall rather than to an origin. Undrained spec: that planned negative
// control (expect_x280_ready false, every kernel ending with a result) is
// Undrained::None, records nothing, and leaves the next run free to launch,
// here step 8's full path with one forwarder and one slot.
TEST_P(EthCreditHw, WithheldPeerE0TimesOutTheHandshakeAndLeavesTheProcessClean) {
    ASSERT_EQ(suite::g_undrained, suite::Undrained::None)
        << "an earlier run (" << suite::g_undrained_case << ") already ended undrained ("
        << suite::undrained_name(suite::g_undrained) << ")";
    const auto neg = run(withheld_peer_case("withheld_peer_e0", /*expect_x280_ready=*/false));
    ASSERT_TRUE(neg.launched);
    const auto& e0 = neg.chips[0].e0.block;
    EXPECT_EQ(e0.status.state, static_cast<uint32_t>(EC_STATE_FAILED));
    EXPECT_EQ(e0.status.code, static_cast<uint32_t>(EC_ERROR_STALL_TIMEOUT));
    EXPECT_EQ(e0.status.detail, static_cast<uint32_t>(EC_SITE_HANDSHAKE));

    // SPEC §Failure and termination: a STALL_TIMEOUT is never an origin; the
    // timed-out kernel is listed as stalled with its site as the detail.
    const suite::Attribution& a = neg.attribution;
    EXPECT_TRUE(a.origins.empty()) << a.describe();
    const uint32_t e0_id = EC_KERNEL_ID(EC_ROLE_E0, 0u, 0u);
    const bool e0_stalled = std::any_of(a.stalled.begin(), a.stalled.end(), [&](const suite::Culprit& k) {
        return k.chip == neg.chips[0].chip && k.kernel_id == e0_id &&
               k.code == static_cast<uint32_t>(EC_ERROR_STALL_TIMEOUT) &&
               k.detail == static_cast<uint32_t>(EC_SITE_HANDSHAKE);
    });
    EXPECT_TRUE(e0_stalled) << "chip 0's E0 not listed as stalled at EC_SITE_HANDSHAKE: " << a.describe();

    EXPECT_EQ(neg.undrained, suite::Undrained::None)
        << "planned negative control classified " << suite::undrained_name(neg.undrained);
    ASSERT_EQ(suite::g_undrained, suite::Undrained::None)
        << "planned negative control recorded as undrained (" << suite::undrained_name(suite::g_undrained) << ")";

    const suite::Case clean = small_clean_case("full_one_slot");
    const auto r = run(clean);
    EXPECT_TRUE(r.metal_enqueued) << "a clean setup read-back after a planned negative control must permit the enqueue";
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
    suite::expect_conserved(clean, r, 0u);
}

// A setup read-back that is not all zero stops run_case before anything is
// enqueued: each chip's ERISC control area, and each chip's LIM arena, is
// dirtied between its zero write and its read-back, so run_case must report
// "did not read back zero" for that chip and leave Metal and both X280
// sessions unenqueued.
TEST_P(EthCreditHw, ReadbackMismatchOnEitherChipStopsBeforeEnqueue) {
    const struct {
        const char* name;
        bool lim;
        int chip;
    } cases[] = {
        {"dirty_erisc_chip0", false, 0},
        {"dirty_erisc_chip1", false, 1},
        {"dirty_lim_chip0", true, 0},
        {"dirty_lim_chip1", true, 1},
    };
    for (const auto& k : cases) {
        SCOPED_TRACE(k.name);
        suite::Case c = small_clean_case(k.name);
        (k.lim ? c.stage.dirty_lim_chip : c.stage.dirty_erisc_chip) = k.chip;
        expect_readback_mismatch_stops(c, k.chip);
    }
}

// Undrained spec: a run whose X280 wait expires where the case expected
// ResultReady is UnplannedDeadline, reports "ended undrained", and is
// recorded process-wide; every later run_case then reports "refusing to
// launch" and returns before anything is enqueued. forget_undrained_run
// clears the record.
TEST_P(EthCreditHw, UndrainedRunRefusesTheNextLaunch) {
    ASSERT_EQ(suite::g_undrained, suite::Undrained::None)
        << "an earlier run (" << suite::g_undrained_case << ") already ended undrained ("
        << suite::undrained_name(suite::g_undrained) << ")";
    struct ForgetOnExit {
        ~ForgetOnExit() { suite::forget_undrained_run(); }
    } forget_on_exit;

    // The forwarders wait for host STOP, so an X280 wait expecting ResultReady
    // times out: unplanned.
    std::vector<std::string> undrained_failures;
    const auto bad =
        run_capturing(withheld_peer_case("withheld_peer_e0_unplanned", /*expect_x280_ready=*/true), undrained_failures);
    const suite::Undrained recorded = suite::g_undrained;

    std::vector<std::string> refused_failures;
    const auto refused = run_capturing(small_clean_case("full_one_slot_after_undrained"), refused_failures);
    const suite::Undrained recorded_after_refusal = suite::g_undrained;

    suite::forget_undrained_run();
    EXPECT_EQ(suite::g_undrained, suite::Undrained::None) << "forget_undrained_run left a record";
    EXPECT_TRUE(suite::g_undrained_case.empty()) << "forget_undrained_run left case " << suite::g_undrained_case;

    // Allow-list: the run's own X280 wait expiry on each chip, from run_case's
    // ResultReady expectation, and the undrained verdict. Anything else is a
    // real failure.
    bool ended_undrained = false;
    uint32_t wait_expiries = 0;
    for (const auto& m : undrained_failures) {
        if (contains(m, "ended undrained")) {
            ended_undrained = true;
        } else if (contains(m, "cr.x280.wait.outcome") && contains(m, "timed_out")) {
            ++wait_expiries;
        } else {
            ADD_FAILURE() << "unexpected failure during the undrained run: " << m;
        }
    }
    EXPECT_EQ(wait_expiries, 2u) << "expected one X280 wait expiry per chip";
    EXPECT_TRUE(bad.launched) << "the undrained run must launch for its deadline to be unplanned";
    EXPECT_TRUE(ended_undrained) << "no \"ended undrained\" failure for an unplanned X280 deadline";
    EXPECT_EQ(bad.undrained, suite::Undrained::UnplannedDeadline)
        << "classified " << suite::undrained_name(bad.undrained);
    EXPECT_EQ(recorded, suite::Undrained::UnplannedDeadline)
        << "g_undrained recorded " << suite::undrained_name(recorded);

    bool refusal = false;
    for (const auto& m : refused_failures) {
        if (contains(m, "refusing to launch")) {
            refusal = true;
        } else {
            ADD_FAILURE() << "unexpected failure during the refused run: " << m;
        }
    }
    EXPECT_TRUE(refusal) << "no \"refusing to launch\" failure after an undrained run";
    EXPECT_FALSE(refused.metal_enqueued) << "EnqueueMeshWorkload called after an undrained run";
    EXPECT_FALSE(refused.launched);
    EXPECT_FALSE(refused.chips[0].x280.enqueued()) << "chip 0 X280 session enqueued after an undrained run";
    EXPECT_FALSE(refused.chips[1].x280.enqueued()) << "chip 1 X280 session enqueued after an undrained run";
    EXPECT_EQ(recorded_after_refusal, suite::Undrained::UnplannedDeadline)
        << "a refused run changed the record to " << suite::undrained_name(recorded_after_refusal);
}

// Step 5: the NoC hop alone. Forwarder -> E0 with W1, W2, and R3 credits, E0
// checking each header's sequence number so a W2 ahead of its W1 fails.
TEST_P(EthCreditHw, NocHopAlone) {
    suite::Case c{
        .name = "noc_hop",
        .shape = {1u, 8u, 8u, Slot(64u), 2u},
        .direction = ec::Direction::AToB,
        .packets = each(1u, 5000u)};
    c.stage.e0_sink_local = true;
    c.stage.launch_e1 = false;
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
    EXPECT_EQ(r.chips[0].fwd[0].block.packets_out[0], 5000u);
    EXPECT_EQ(r.chips[0].e0.block.packets_out[0], 5000u);
}

// Step 6: the Ethernet hop alone. E0 fills tx_fifo itself; E1 checks each
// header and returns R2 without delivering.
TEST_P(EthCreditHw, EthernetHopAlone) {
    suite::Case c{
        .name = "ethernet_hop",
        .shape = {1u, 8u, 8u, Slot(4096u), 2u},
        .direction = ec::Direction::AToB,
        .packets = each(1u, 20000u)};
    c.stage.e0_source_self = true;
    c.stage.e1_sink_local = true;
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
    EXPECT_EQ(r.chips[0].e0.block.packets_out[0], 20000u);
    EXPECT_EQ(r.chips[1].e1.block.packets_in[0], 20000u);
}

// Step 6: single-slot ping-pong. With one link slot, E0 cannot send packet
// k + 1 until R2 has returned slot k, so E0's own window over n packets is n
// round trips on its clock.
TEST_P(EthCreditHw, SingleSlotPingPong) {
    suite::Case c{
        .name = "ping_pong",
        .shape = {1u, 1u, 1u, Slot(16u), 1u},
        .direction = ec::Direction::AToB,
        .packets = each(1u, 10000u)};
    c.stage.e0_source_self = true;
    c.stage.e1_sink_local = true;
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
    const auto& e0 = r.chips[0].e0.block;
    std::printf(
        "  ping-pong: %u packets, E0 window %llu ERISC ticks\n",
        e0.packets_out[0],
        static_cast<unsigned long long>(e0.last_ts - e0.first_ts));
}

// Step 7: the Ethernet admission test. E1 checks each slot's tail marker the
// moment pkts[p] shows the packet, at payloads up to the ABI maximum. A slot
// larger than ETH_TXQ_MAX_PKT_SIZE leaves as several TT-link packets, the
// case where the tail can lag the counter, so at least one payload must be.
TEST_P(EthCreditHw, AdmissionTailMarkerHoldsWhenCounterFollowsData) {
    uint32_t split_payloads = 0;
    for (uint32_t payload : {16u, 4096u, 16384u, 65520u}) {
        suite::Case c{
            .name = "admission",
            .shape = {1u, 2u, 2u, Slot(payload), 1u},
            .direction = ec::Direction::AToB,
            .packets = each(1u, 20000u)};
        c.stage.e0_source_self = true;
        c.stage.e1_sink_local = true;
        c.stage.tail_marker = true;
        const auto r = run(c);
        ASSERT_TRUE(r.launched);
        suite::expect_all_done(r);
        EXPECT_EQ(r.chips[1].e1.block.packets_in[0], 20000u) << "payload " << payload;
        ASSERT_NE(r.chips[0].txq_max_pkt_bytes, 0u) << "ETH_TXQ_MAX_PKT_SIZE read as zero";
        if (Slot(payload) > r.chips[0].txq_max_pkt_bytes) {
            ++split_payloads;
        }
    }
    EXPECT_GE(split_payloads, 1u) << "no payload's W3 left as more than one TT-link packet";
}

// Step 7 positive control: W4 is sent before its W3, so the counter is
// premature by construction, and E1 must see a stale tail.
TEST_P(EthCreditHw, AdmissionPositiveControlCatchesPrematureCounter) {
    suite::Case c{
        .name = "admission_positive_control",
        .shape = {1u, 2u, 2u, Slot(4096u), 1u},
        .direction = ec::Direction::AToB,
        .packets = each(1u, 1000u),
        .stall_budget_ms = 500,
        .expect_x280_ready = true};
    c.stage.e0_source_self = true;
    c.stage.e1_sink_local = true;
    c.stage.tail_marker = true;
    c.stage.counter_first_delay = 20000u;
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    const auto& e1 = r.chips[1].e1.block;
    EXPECT_EQ(e1.status.state, static_cast<uint32_t>(EC_STATE_FAILED));
    EXPECT_EQ(e1.status.code, static_cast<uint32_t>(EC_ERROR_BAD_SEQUENCE))
        << "E1 must see the stale marker of a packet whose counter arrived first";
}

// Step 8: many slots, with the packet count forcing both N_LINK and N_DELIVER
// round their 2N pointer range many times.
TEST_P(EthCreditHw, FullPathForcedWrap) {
    const suite::Case c{
        .name = "full_wrap",
        .shape = {1u, 4u, 2u, Slot(1024u), 2u},
        .direction = ec::Direction::AToB,
        .packets = each(1u, 10003u)};
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
    suite::expect_conserved(c, r, 0u);
    suite::record_throughput(spec(), fixture(), c, r, 0u);
}

// Step 9: P forwarders, one per hart.
TEST_P(EthCreditHw, FourForwarders) {
    const suite::Case c{
        .name = "four_forwarders",
        .shape = {4u, 8u, 8u, Slot(2048u), 4u},
        .direction = ec::Direction::AToB,
        .packets = each(4u, 20000u)};
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
    suite::expect_conserved(c, r, 0u);
    suite::record_throughput(spec(), fixture(), c, r, 0u);
}

// Step 10: both directions on the edge, validated by per-hop conservation
// and the forwarders' per-p sequence checks.
TEST_P(EthCreditHw, BothDirections) {
    const suite::Case c{
        .name = "both_directions",
        .shape = {2u, 8u, 8u, Slot(2048u), 4u},
        .direction = ec::Direction::Both,
        .packets = each(2u, 20000u)};
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
    suite::expect_conserved(c, r, 0u);
    suite::expect_conserved(c, r, 1u);
}

// The ERISC wall clock, which scales every ERISC stall budget and reported
// time, ticks at CMFW's programmed AICLK: host-bracketed samples of the
// counter on both tiles hold that rate across the run. The nominal
// ETH_CLOCK_CYCLE_1MS rate must fail the same assessment.
TEST_P(EthCreditHw, EriscWallClockTicksAtProgrammedAiclk) {
    const suite::Case c{
        .name = "erisc_clock",
        .shape = {1u, 8u, 8u, Slot(1024u), 4u},
        .direction = ec::Direction::AToB,
        .packets = each(1u, 100000u)};
    const auto r = run(c);
    ASSERT_TRUE(r.launched);
    suite::expect_all_done(r);
    suite::expect_conserved(c, r, 0u);
    constexpr uint64_t kNominalHz = 1000000000ull;  // ETH_CLOCK_CYCLE_1MS per ms
    for (uint32_t i = 0; i < ec::kChips; ++i) {
        const auto& cr = r.chips[i];
        const auto& e = cr.erisc_clock;
        EXPECT_TRUE(e.errors.empty()) << "chip " << cr.chip << ": " << suite::describe_clock(e);
        EXPECT_EQ(e.conversion_hz, cr.aiclk_hz) << "chip " << cr.chip;
        const auto nominal = x280::harness::measurement::assess_aiclk_run(kNominalHz, e.start, e.end);
        EXPECT_FALSE(nominal.errors.empty())
            << "chip " << cr.chip << ": the samples cannot tell AICLK from the nominal rate";
        std::printf(
            "  chip %d: AICLK %llu Hz, host-sampled %s\n",
            cr.chip,
            static_cast<unsigned long long>(cr.aiclk_hz),
            suite::describe_clock(e).c_str());
    }
}

// Which hop bounds throughput: packet rate on the receiving forwarder's
// clock across slot size, batch and P, beside the sending forwarder's share
// of its window spent in W1 stores. A rate flat in slot size with W1
// dominating the sender's window names the X280's W1 as the bound.
TEST_P(EthCreditHw, ThroughputBySlotBatchAndP) {
    struct Point {
        uint32_t P, slot, batch;
    };
    const Point points[] = {
        {1u, Slot(64u), 1u},
        {1u, Slot(64u), 4u},
        {1u, Slot(1024u), 1u},
        {1u, Slot(1024u), 4u},
        {1u, Slot(4096u), 1u},
        {1u, Slot(4096u), 4u},
        {1u, Slot(16384u), 4u},
        {2u, Slot(4096u), 4u},
        {4u, Slot(4096u), 4u}};
    for (const Point& pt : points) {
        const suite::Case c{
            .name = "throughput_sweep",
            .shape = {pt.P, 8u, 8u, pt.slot, pt.batch},
            .direction = ec::Direction::AToB,
            .packets = each(pt.P, 20000u)};
        const auto r = run(c);
        ASSERT_TRUE(r.launched);
        suite::expect_all_done(r);
        suite::expect_conserved(c, r, 0u);
        suite::record_throughput(spec(), fixture(), c, r, 0u);
        for (uint32_t p = 0; p < pt.P; ++p) {
            const auto& tx = r.chips[0].fwd[p].block;
            const ec_stall_t* w1 = suite::find_stall(tx, EC_SITE_FWD_W1_STORE);
            ASSERT_NE(w1, nullptr) << "sender p " << p << " keeps no FWD_W1_STORE entry";
            /* w1->ticks includes the first batch's W1 stores, which precede
             * first_ts, so it overstates the window's W1 time by about one batch. */
            std::printf(
                "  sender p %u: W1 store %llu of %llu mtime ticks\n",
                p,
                static_cast<unsigned long long>(w1->ticks),
                static_cast<unsigned long long>(tx.last_ts - tx.first_ts));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(EthCredit, EthCreditHw, ::testing::Values(0));
