// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * p2p_latency_x280_same_tile.cpp -- X280 image for x280_same_tile member of the p2p_latency suite.
 *
 * ID 2: X280 hart <-> X280 hart, same tile, no NOC. Two harts of one L2CPU
 * participate. Hart 0 is participant A, hart 1 is participant B. Harts 2 and
 * 3 return 0 immediately -- their presence is participation under the
 * lifecycle harness (x280/README.md § 6, obligation 2) but they exchange
 * nothing.
 *
 * The args block is one p2p_latency_x280_same_tile_args_block_t. Every hart is told
 * the same base address (D-032), so the two participants pick their section
 * by hartid.
 *
 * Both participant sections have the same layout, so the template
 * instantiation of run_participant<> is the same for both harts. What
 * differs is what section each hart reads out of the block.
 *
 * All work is compile-time-specialised for:
 *   - Shape:        TOKEN_ONLY  (payload_size == 4)
 *   - PollMode:     TIGHT
 *   - TransmitMode: WAIT_FOR_COMPLETION
 *
 * Other combinations are separate template instantiations added by later
 * phases; they do not change any code in this file.
 *
 * The post-section-copy ceremony (validate, backend init, run_participant,
 * shutdown, release fence) lives in the shared entry-point template under
 * kernels/shared/p2p_latency_entry_cxx.hpp; this file supplies the section
 * copy and the ID 2 descriptor.
 */

extern "C" {
#include "x280_hart_protocol.h"
#include "rt/x280_idle.h"
}

#include "p2p_latency_entry_cxx.hpp"
#include "p2p_latency_frontend.h"
#include "p2p_latency_frontend_cxx.hpp"
#include "p2p_latency_layout.h"
#include "x280_same_tile_args.h"
#include "x280_section_load.hpp"
#include "backends/x280_x280_same_tile.hpp"

namespace {

/* ID 2 uses harts 0 and 1. Kept as constexpr so a change here is checked
 * against the assertion beside it rather than propagated as a magic number. */
constexpr uint64_t kParticipantAHart = 0u;
constexpr uint64_t kParticipantBHart = 1u;

static_assert(
    kParticipantAHart == 0u && kParticipantBHart == 1u,
    "hart 0 is A, hart 1 is B: this image's args block layout assumes that");

/* Member descriptor consumed by the shared entry-point template. Every field
 * here is either compile-time (Shape/Poll/Transmit) or a small forwarding
 * function; the state machine itself is in the frontend. */
struct X280SameTile {
    using Backend = ::p2p_latency::backend_x280_x280_same_tile::Backend;
    using Section = ::p2p_latency_x280_same_tile_section_t;

    static constexpr p2p_latency_transfer_shape_t SHAPE = P2P_LATENCY_TRANSFER_TOKEN_ONLY;
    static constexpr p2p_latency_poll_mode_t POLL_MODE = P2P_LATENCY_POLL_TIGHT;
    static constexpr p2p_latency_transmit_mode_t TX_MODE = P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION;

    static bool validate(const Section& s, uint64_t hartid) {
        if (s.cookie_common != p2p_latency_cookie_for(UBENCH_SECTION_COMMON)) {
            return false;
        }
        if (s.cookie_backend != p2p_latency_cookie_for(UBENCH_SECTION_BACKEND)) {
            return false;
        }
        if (!p2p_latency_common_valid(&s.common)) {
            return false;
        }
        /* The self field is host-owned. Cross-check it against hartid so a
         * misfiled section (host packing bug) fails here rather than as a
         * cross-writer collision on the mailboxes. */
        const uint32_t expected_self = (hartid == kParticipantAHart) ? static_cast<uint32_t>(P2P_LATENCY_PARTICIPANT_A)
                                                                     : static_cast<uint32_t>(P2P_LATENCY_PARTICIPANT_B);
        if (s.common.self != expected_self) {
            return false;
        }
        return true;
    }

    static uint64_t result_addr(const Section& s) { return s.backend.own_result_addr; }

    static bool backend_init(
        Backend::state_t& state, const Section& s, uint64_t hartid, uint64_t run_value, p2p_latency_result_t& r) {
        return ::p2p_latency::backend_x280_x280_same_tile::initialize(state, s.common, s.backend, hartid, run_value, r);
    }

    /* ID 2 has no TLB or transport to release; the backend owns no
     * resources beyond the state struct itself. */
    static void backend_shutdown(Backend::state_t&) {}
};

}  // namespace

extern "C" int main(uint64_t hartid) {
    /* Harts A and B load their own sections. The other harts load none:
     * under the lifecycle harness they still commit terminal success and
     * take the host's STOP; there is no participation beyond that. */
    UBENCH_SECTION_STORAGE(p2p_latency_x280_same_tile_section_t, section) store;
    if (!x280_load_section<p2p_latency_x280_same_tile_section_t>(hartid, kParticipantBHart + 1u, store.as_uint32)) {
        return 0;
    }
    const uint64_t run_value = x280_hart_control(hartid)->run_value;

    return ::p2p_latency::run_participant_from_section<X280SameTile>(hartid, run_value, store.section);
}
