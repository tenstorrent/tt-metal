// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* p2p_latency_x280_peer_tile.cpp -- X280 image for x280_peer_tile member of the p2p_latency suite.
 *
 * x280_peer_tile token exchange. Only hart 0 participates; initialized TLB slots are
 * cleared on every exit path. */

extern "C" {
#include "x280_hart_protocol.h"
#include "rt/x280_idle.h"
}

#include "p2p_latency_entry_cxx.hpp"
#include "p2p_latency_frontend.h"
#include "p2p_latency_frontend_cxx.hpp"
#include "p2p_latency_layout.h"
#include "p2p_latency_x280_noc_args.h"
#include "x280_section_load.hpp"
#include "backends/x280_noc.hpp"

namespace {

constexpr uint64_t kParticipants = 1u;

struct X280PeerTile {
    using Backend = ::p2p_latency::backend_x280_noc::Backend;
    using Section = ::p2p_latency_x280_noc_section_t;

    static constexpr p2p_latency_transfer_shape_t SHAPE = P2P_LATENCY_TRANSFER_TOKEN_ONLY;
    static constexpr p2p_latency_poll_mode_t POLL_MODE = P2P_LATENCY_POLL_TIGHT;
    static constexpr p2p_latency_transmit_mode_t TX_MODE = P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION;

    static bool validate(const Section& s, uint64_t /*hartid*/) {
        if (s.cookie_common != p2p_latency_cookie_for(UBENCH_SECTION_COMMON)) {
            return false;
        }
        if (s.cookie_backend != p2p_latency_cookie_for(UBENCH_SECTION_BACKEND)) {
            return false;
        }
        /* The same image serves both participants. */
        return p2p_latency_common_valid(&s.common) &&
               (s.common.self == P2P_LATENCY_PARTICIPANT_A || s.common.self == P2P_LATENCY_PARTICIPANT_B);
    }

    static uint64_t result_addr(const Section& s) { return s.backend.own_result_addr; }

    static bool backend_init(
        Backend::state_t& state, const Section& s, uint64_t hartid, uint64_t run_value, p2p_latency_result_t& r) {
        return ::p2p_latency::backend_x280_noc::initialize(
            state, s.common, s.backend, hartid, run_value, r, P2P_LATENCY_BACKEND_ID_X280_X280_PEER_TILE);
    }

    static void backend_shutdown(Backend::state_t& state) { ::p2p_latency::backend_x280_noc::shutdown(state); }
};

}  // namespace

extern "C" int main(uint64_t hartid) {
    UBENCH_SECTION_STORAGE(p2p_latency_x280_noc_section_t, section) store;
    if (!x280_load_section<p2p_latency_x280_noc_section_t>(hartid, kParticipants, store.as_uint32)) {
        return 0;
    }
    const uint64_t run_value = x280_hart_control(hartid)->run_value;

    return ::p2p_latency::run_participant_from_section<X280PeerTile>(hartid, run_value, store.section);
}
