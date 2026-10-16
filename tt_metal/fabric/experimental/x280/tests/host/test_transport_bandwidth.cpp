// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The transport_bandwidth frontend's publication order, driven through a fake
// backend: the footer runs before any terminal state is published, a footer
// that fails ends the run WINDOW_CLEANUP_FAILED, and a rendezvous that gives up
// ends it RENDEZVOUS_TIMEOUT; and its size dispatch,
// which runs only a backend's listed sizes.

#include "kernels/transport_bandwidth/transport_bandwidth_frontend_cxx.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

struct FakeState {
    transport_bw_protocol_args_t args{};
    transport_bw_result_t result{};
    bool path_ok = true;
    bool together = true;
    bool footer_ok = true;
    uint64_t now = 0;
    std::vector<uint32_t> issued;  // TransferSize of every issue
    // The terminal_state each footer call saw, and the one each barrier published.
    std::vector<uint32_t> footer_saw;
    std::vector<uint32_t> barrier_published;
};

struct FakeBackend {
    using state_t = FakeState;
    using timestamp_t = uint64_t;
    struct counter_delta_t {
        uint64_t primary;
    };

    // 320 stands in for a size only some member admits; 256 for one this
    // backend does not.
    using sizes = transport_bandwidth::size_list<64u, 320u>;

    static const transport_bw_protocol_args_t& args(state_t& s) { return s.args; }
    static transport_bw_result_t& result(state_t& s) { return s.result; }
    static bool verify_path(state_t& s) { return s.path_ok; }
    static bool rendezvous(state_t& s) { return s.together; }
    template <uint32_t TransferSize>
    static void issue(state_t& s) {
        s.issued.push_back(TransferSize);
    }
    static void drain(state_t&) {}
    static timestamp_t measurement_now(state_t& s) { return s.now += 10; }
    static counter_delta_t delta(state_t&, const timestamp_t& t0, const timestamp_t& t1) { return {t1 - t0}; }
    static void publish_delta(state_t& s, const counter_delta_t& d) { s.result.raw_batch_delta = d.primary; }
    static void publish_barrier(state_t& s) { s.barrier_published.push_back(s.result.terminal_state); }
    static bool publish_footer(state_t& s) {
        s.footer_saw.push_back(s.result.terminal_state);
        return s.footer_ok;
    }
};

FakeState run(
    bool path_ok,
    bool footer_ok,
    uint32_t transfer_size = 64,
    uint32_t boundary = TRANSPORT_BW_BOUNDARY_DRAINED,
    bool together = true) {
    FakeState s;
    s.together = together;
    s.args.transfer_size = transfer_size;
    s.args.transfer_count = 4;
    s.args.warmup_count = 2;
    s.args.boundary = boundary;
    s.path_ok = path_ok;
    s.footer_ok = footer_ok;
    transport_bandwidth::run_member<FakeBackend>(s);
    return s;
}

}  // namespace

// Spec (transport_bandwidth_frontend_cxx.hpp): each exit runs the
// footer once while the state is still non-terminal, then publishes one
// terminal state behind one barrier -- WINDOW_CLEANUP_FAILED when the footer
// failed, which the outcome map reads as a trap.
TEST(TransportBandwidthFrontend, FooterRunsBeforeTheTerminalStateAndItsFailureIsTheTerminalState) {
    struct Case {
        const char* name;
        bool path_ok;
        bool footer_ok;
        uint32_t transfer_size;
        transport_bw_state_t footer_saw;
        transport_bw_state_t terminal;
        bool together = true;
    } cases[] = {
        {"done", true, true, 64, TRANSPORT_BW_STATE_PUBLISH, TRANSPORT_BW_STATE_DONE},
        {"done, footer fails", true, false, 64, TRANSPORT_BW_STATE_PUBLISH, TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED},
        {"path mismatch", false, true, 64, TRANSPORT_BW_STATE_VERIFY_PATH, TRANSPORT_BW_STATE_PATH_MISMATCH},
        {"path mismatch, footer fails",
         false,
         false,
         64,
         TRANSPORT_BW_STATE_VERIFY_PATH,
         TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED},
        {"bad size", true, true, 256, TRANSPORT_BW_STATE_INIT, TRANSPORT_BW_STATE_ARG_LAYOUT_MISMATCH},
        {"bad size, footer fails", true, false, 256, TRANSPORT_BW_STATE_INIT, TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED},
        {"rendezvous times out",
         true,
         true,
         64,
         TRANSPORT_BW_STATE_WARMUP,
         TRANSPORT_BW_STATE_RENDEZVOUS_TIMEOUT,
         false},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        const auto s = run(c.path_ok, c.footer_ok, c.transfer_size, TRANSPORT_BW_BOUNDARY_DRAINED, c.together);
        EXPECT_EQ(s.footer_saw, std::vector<uint32_t>{static_cast<uint32_t>(c.footer_saw)});
        EXPECT_EQ(s.barrier_published, std::vector<uint32_t>{static_cast<uint32_t>(c.terminal)});
        EXPECT_EQ(s.result.terminal_state, static_cast<uint32_t>(c.terminal));
    }
    EXPECT_EQ(transport_bw_outcome(TRANSPORT_BW_STATE_WINDOW_CLEANUP_FAILED), UBENCH_OUTCOME_TRAP);
    EXPECT_EQ(transport_bw_outcome(TRANSPORT_BW_STATE_RENDEZVOUS_TIMEOUT), UBENCH_OUTCOME_TIMEOUT);
}

// Spec (transport_bandwidth_frontend_cxx.hpp, size_list): warmup and the timed
// loop issue the listed specialization equal to args.transfer_size, under
// both boundaries; an unlisted size issues nothing.
TEST(TransportBandwidthFrontend, IssuesOnlyTheListedSizeMatchingTheArgs) {
    for (const uint32_t boundary : {TRANSPORT_BW_BOUNDARY_DRAINED, TRANSPORT_BW_BOUNDARY_INJECTION}) {
        SCOPED_TRACE(boundary);
        const auto s = run(true, true, 320, boundary);
        EXPECT_EQ(s.result.terminal_state, static_cast<uint32_t>(TRANSPORT_BW_STATE_DONE));
        EXPECT_EQ(s.issued, std::vector<uint32_t>(2 + 4, 320u));
    }
    EXPECT_TRUE(run(true, true, 256).issued.empty());
}
