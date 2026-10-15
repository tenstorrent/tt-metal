// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Shared paired-window backend for x280_peer_tile, tensix_peer_tile, erisc_peer_tile and drisc_peer_tile.
 * WAIT_FOR_DEPARTURE is unsupported; see docs/benchmarks/design.md, "Transmit-mode validity". */
#ifndef X280_P2P_LATENCY_BACKEND_X280_NOC_HPP
#define X280_P2P_LATENCY_BACKEND_X280_NOC_HPP

#include <cstdint>

#include "p2p_latency_frontend.h"
#include "p2p_latency_frontend_cxx.hpp"
#include "p2p_latency_layout.h"
#include "p2p_latency_x280_noc_args.h"

#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "rt/x280_measurement.h"
#include "rt/x280_noc_map.h"

namespace p2p_latency::backend_x280_noc {

struct timestamp_t {
    uint64_t cycles;
    uint64_t secondary;
};

using counter_delta_t = uint64_t;

struct state_t {
    /* LIM mailbox read after `fence ir, ir`. */
    volatile uint32_t* own_inbound_token;

    volatile uint32_t* peer_window_token;

    uint64_t hartid;
    uint64_t run_value;

    uint32_t tlb_index;

    uint32_t direction;
};

/* Configure and verify the paired-range TLB. Returns false on invalid
 * arguments, unsupported options, or TLB failure. */
inline bool initialize(
    state_t& state,
    const p2p_latency_common_args_t& common,
    const p2p_latency_x280_noc_backend_args_t& backend,
    uint64_t hartid,
    uint64_t run_value,
    p2p_latency_result_t& result,
    uint32_t backend_id) {
    if (common.payload_size != 4u) {
        return false;
    }
    if (backend.own_inbound_addr == 0u || backend.peer_inbound_addr == 0u || backend.own_result_addr == 0u) {
        return false;
    }
    if ((backend.own_inbound_addr % 4u) != 0u || (backend.peer_inbound_addr % 4u) != 0u) {
        return false;
    }
    if (!x280_tlb_paired_slot(backend.peer_tlb_index)) {
        return false;
    }

    volatile void* window = nullptr;
    const noc_window_status_t opened = noc_open_window_2m(
        backend.peer_tlb_index, backend.peer_noc, backend.peer_inbound_addr, P2P_LATENCY_X280_NOC_ORDERING, &window);
    if (opened != NOC_WINDOW_OPEN) {
        /* An unverified window is cleared, not retained: the host assigned this
         * paired slot to this run alone. A refused initialisation has no
         * shutdown and no result field for a failed clear, so the host's
         * readback of the slot (expect_window_cleared) is what checks it. */
        if (opened == NOC_WINDOW_UNVERIFIED) {
            (void)noc_clear_tlb_2m(backend.peer_tlb_index);
        }
        return false;
    }

    const uint32_t tok = p2p_latency_token_offset(common.payload_size);
    state.own_inbound_token = reinterpret_cast<volatile uint32_t*>(backend.own_inbound_addr + tok);
    state.peer_window_token =
        reinterpret_cast<volatile uint32_t*>(reinterpret_cast<uintptr_t>(window) + static_cast<uintptr_t>(tok));
    state.hartid = hartid;
    state.run_value = run_value;
    state.tlb_index = backend.peer_tlb_index;
    state.direction = 0u;

    p2p_latency::initialize_result(result, backend_id);
    return true;
}

inline void shutdown(state_t& state) { (void)noc_clear_tlb_2m(state.tlb_index); }

inline void bind_direction(state_t& state, p2p_latency_direction_t direction) {
    state.direction = static_cast<uint32_t>(direction);
}

inline timestamp_t measurement_now(const state_t&) {
    const x280_paired_t now = x280_paired_now();
    return timestamp_t{now.cpu, now.wall};
}

inline bool delta(const state_t&, timestamp_t t0, timestamp_t t1, counter_delta_t* primary) {
    return measurement_delta(t0.cycles, t1.cycles, primary) != 0;
}

template <p2p_latency_transfer_shape_t Shape, p2p_latency_transmit_mode_t TransmitMode>
inline void transmit(state_t& state, uint32_t token) {
    static_assert(Shape == P2P_LATENCY_TRANSFER_TOKEN_ONLY, "this backend supports only the token-only specialisation");
    static_assert(
        TransmitMode != P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE,
        "X280 has no established departure-only primitive; the design rejects this mode");

    *state.peer_window_token = token;
    if constexpr (TransmitMode == P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION) {
        x280_fence_iorw();
    }
}

/* LIM polling requires `fence ir, ir`
 * (rt/x280_idle.h). Returns true once expected arrives. The STOP check is
 * compiled in only under P2P_LATENCY_POLL_CHECKS_STOP, returning false on
 * STOP; without it a token that never arrives strands the hart until
 * `tt-smi -r`. */
template <p2p_latency_poll_mode_t PollMode>
inline bool poll(state_t& state, uint32_t expected) {
    /* Held in a register: the fence's "memory" clobber would otherwise reload
     * it from `state`, on the uncached LIM stack, every pass. */
    volatile uint32_t* const token = state.own_inbound_token;
    for (;;) {
        x280_fence_ir();
        if (*token == expected) {
            return true;
        }
#ifdef P2P_LATENCY_POLL_CHECKS_STOP
        if (x280_hart_stop_requested(state.hartid, state.run_value)) {
            return false;
        }
#endif
        if constexpr (PollMode == P2P_LATENCY_POLL_BACKOFF) {
            x280_pause();
        }
    }
}

template <p2p_latency_poll_mode_t PollMode>
inline bool poll_for(state_t& state, uint32_t expected, uint64_t max_ticks) {
    volatile uint32_t* const token = state.own_inbound_token;
    const uint64_t t0 = x280_rdcycle();
    for (;;) {
        x280_fence_ir();
        if (*token == expected) {
            return true;
        }
        if (x280_hart_stop_requested(state.hartid, state.run_value)) {
            return false;
        }
        if ((x280_rdcycle() - t0) > max_ticks) {
            return false;
        }
        if constexpr (PollMode == P2P_LATENCY_POLL_BACKOFF) {
            x280_pause();
        }
    }
}

inline uint32_t observed(state_t& state) {
    x280_fence_ir();
    return *state.own_inbound_token;
}

struct Backend {
    using state_t = ::p2p_latency::backend_x280_noc::state_t;
    using timestamp_t = ::p2p_latency::backend_x280_noc::timestamp_t;
    using counter_delta_t = ::p2p_latency::backend_x280_noc::counter_delta_t;

    static inline void bind_direction(state_t& s, p2p_latency_direction_t d) {
        ::p2p_latency::backend_x280_noc::bind_direction(s, d);
    }
    static inline timestamp_t measurement_now(const state_t& s) {
        return ::p2p_latency::backend_x280_noc::measurement_now(s);
    }
    static inline bool delta(const state_t& s, timestamp_t t0, timestamp_t t1, counter_delta_t* primary) {
        return ::p2p_latency::backend_x280_noc::delta(s, t0, t1, primary);
    }
    template <p2p_latency_transfer_shape_t Shape, p2p_latency_transmit_mode_t Tx>
    static inline void transmit(state_t& s, uint32_t token) {
        ::p2p_latency::backend_x280_noc::transmit<Shape, Tx>(s, token);
    }
    template <p2p_latency_poll_mode_t Pm>
    static inline bool poll(state_t& s, uint32_t expected) {
        return ::p2p_latency::backend_x280_noc::poll<Pm>(s, expected);
    }
    template <p2p_latency_poll_mode_t Pm>
    static inline bool poll_for(state_t& s, uint32_t expected, uint64_t max_ticks) {
        return ::p2p_latency::backend_x280_noc::poll_for<Pm>(s, expected, max_ticks);
    }
    static inline uint32_t observed(state_t& s) { return ::p2p_latency::backend_x280_noc::observed(s); }
};

}  // namespace p2p_latency::backend_x280_noc

#endif /* X280_P2P_LATENCY_BACKEND_X280_NOC_HPP */
