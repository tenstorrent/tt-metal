// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared RV32 Metal NoC backend for Tensix, idle ERISC and DRISC p2p peers.
#ifndef X280_P2P_LATENCY_BACKEND_METAL_NOC_PEER_TILE_HPP
#define X280_P2P_LATENCY_BACKEND_METAL_NOC_PEER_TILE_HPP

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "../../shared/p2p_latency_frontend.h"
#include "../../shared/p2p_latency_frontend_cxx.hpp"
#include "../../shared/p2p_latency_layout.h"
#include "../../../include/x280_noc_coord.h"
#include "../../../include/rt/measurement.h"

namespace p2p_latency::backend_metal_noc_peer_tile {

/*
 * TargetPolicy supplies deadline/performance reads and BACKEND_ID. Its counter remains local
 * to the participant; this backend performs no frequency conversion.
 */
template <typename TargetPolicy>
struct Backend {
    struct timestamp_t {
        uint64_t ticks;
        uint64_t secondary;
    };

    using counter_delta_t = uint64_t;

    struct state_t {
        // Peer NoC writes do not invalidate the local L1 cache; every poll
        // invalidates before reading this volatile local mailbox word.
        volatile tt_l1_ptr uint32_t* own_inbound_token;

        // Outbound writes source a word staged in this core's local L1.
        volatile tt_l1_ptr uint32_t* own_staging_token;

        // Address of the peer's inbound token in the peer's local address space.
        uint64_t peer_inbound_token;

        // Translated NoC coordinate consumed by NOC_XY_ADDR.
        x280_noc_coord_t peer_noc;
        uint32_t own_staging_addr32;

        // Generous emergency ceiling for the frontend's nominally unbounded
        // timed poll. Expiry makes poll return false, which ends the phase
        // TIMEOUT.
        uint64_t hang_deadline_ticks;
    };

    /*
     * Binds checked local and peer addresses and initializes the result record.
     * Addresses are byte addresses; the caller retains all pointed-to storage
     * until the Metal workload finishes. Returns false without transmitting on
     * an unsupported payload or malformed address.
     */
    static inline bool initialize(
        state_t& state,
        const p2p_latency_common_args_t& common,
        uint32_t own_inbound_addr,
        uint32_t own_staging_addr,
        uint32_t peer_inbound_addr_lo,
        uint32_t peer_inbound_addr_hi,
        x280_noc_coord_t peer_noc,
        uint64_t hang_deadline_ticks,
        p2p_latency_result_t& result) {
        if (common.payload_size != 4u) {
            return false;
        }
        if (own_inbound_addr == 0u || own_staging_addr == 0u) {
            return false;
        }
        if (peer_inbound_addr_lo == 0u && peer_inbound_addr_hi == 0u) {
            return false;
        }
        if ((own_inbound_addr % 4u) != 0u) {
            return false;
        }
        if ((own_staging_addr % 4u) != 0u) {
            return false;
        }
        if ((peer_inbound_addr_lo % 4u) != 0u) {
            return false;
        }
        if (!x280_noc_coord_valid(peer_noc)) {
            return false;
        }
        if (hang_deadline_ticks == 0u) {
            return false;
        }

        const uint32_t token_offset = p2p_latency_token_offset(common.payload_size);
        state.own_inbound_token = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own_inbound_addr + token_offset);
        state.own_staging_token = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own_staging_addr + token_offset);
        state.own_staging_addr32 = own_staging_addr + token_offset;
        state.peer_inbound_token =
            (static_cast<uint64_t>(peer_inbound_addr_hi) << 32) | static_cast<uint64_t>(peer_inbound_addr_lo);
        state.peer_inbound_token += token_offset;
        state.peer_noc = peer_noc;
        state.hang_deadline_ticks = hang_deadline_ticks;

        p2p_latency::initialize_result(result, TargetPolicy::BACKEND_ID);
        return true;
    }

    static inline void bind_direction(state_t&, p2p_latency_direction_t) {}

    static inline timestamp_t measurement_now(const state_t&) {
        timestamp_t stamp{TargetPolicy::read_measurement_clock(), UINT64_MAX};
        if (!measurement_cpu_cycles(&stamp.secondary)) {
            stamp.ticks = UINT64_MAX;
        }
        return stamp;
    }

    // UINT64_MAX ticks is a failed read (measurement_now), refused before the
    // subtraction, which would otherwise wrap it into a small interval.
    static inline bool delta(const state_t&, timestamp_t t0, timestamp_t t1, counter_delta_t* primary) {
        return t0.ticks != UINT64_MAX && t1.ticks != UINT64_MAX && measurement_delta(t0.ticks, t1.ticks, primary) != 0;
    }

    template <p2p_latency_transfer_shape_t Shape, p2p_latency_transmit_mode_t TransmitMode>
    static inline void transmit(state_t& state, uint32_t token) {
        static_assert(
            Shape == P2P_LATENCY_TRANSFER_TOKEN_ONLY, "Metal NoC peer supports only the token-only specialization");

        *state.own_staging_token = token;
        noc_async_write(
            state.own_staging_addr32,
            NOC_XY_ADDR(state.peer_noc.x, state.peer_noc.y, state.peer_inbound_token),
            sizeof(uint32_t));

        if constexpr (TransmitMode == P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION) {
            // Remote acknowledgement observed.
            noc_async_write_barrier();
        } else if constexpr (TransmitMode == P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE) {
            // Local NIU has emitted the packet.
            noc_async_writes_flushed();
        }
    }

    // A RISC's L0 data cache can serve a load from L1, and a peer NoC write does
    // not invalidate it (Tensix: BabyRISCV/MemoryOrdering.md, "L0 data cache";
    // no Blackhole document covers ERISC or DRISC: docs/measurement.md "Unconfirmed
    // hardware assumptions", item 13). The invalidate inside each
    // iteration is required even for a volatile load.
    // Returns true once expected arrives, false when hang_deadline_ticks
    // expires.
    template <p2p_latency_poll_mode_t PollMode>
    static inline bool poll(state_t& state, uint32_t expected) {
        const uint64_t t0 = TargetPolicy::read_clock();
        for (;;) {
            invalidate_l1_cache();
            if (*state.own_inbound_token == expected) {
                return true;
            }
            if (TargetPolicy::read_clock() - t0 > state.hang_deadline_ticks) {
                return false;
            }
        }
    }

    template <p2p_latency_poll_mode_t PollMode>
    static inline bool poll_for(state_t& state, uint32_t expected, uint64_t max_ticks) {
        const uint64_t t0 = TargetPolicy::read_clock();
        for (;;) {
            invalidate_l1_cache();
            if (*state.own_inbound_token == expected) {
                return true;
            }
            if (TargetPolicy::read_clock() - t0 > max_ticks) {
                return false;
            }
        }
    }

    static inline uint32_t observed(state_t& state) {
        invalidate_l1_cache();
        return *state.own_inbound_token;
    }
};

}  // namespace p2p_latency::backend_metal_noc_peer_tile

#endif  // X280_P2P_LATENCY_BACKEND_METAL_NOC_PEER_TILE_HPP
