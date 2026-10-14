// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared runtime-argument parser and frontend entry for Metal p2p peers.
#ifndef X280_P2P_LATENCY_METAL_PEER_KERNEL_HPP
#define X280_P2P_LATENCY_METAL_PEER_KERNEL_HPP

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "../shared/p2p_latency_frontend.h"
#include "../shared/p2p_latency_frontend_cxx.hpp"
#include "../shared/p2p_latency_layout.h"
#include "metal_peer_args.h"
#include "metal_peer_target.hpp"

namespace p2p_latency {

/*
 * MetalPeerTarget supplies Backend,
 * enter_transport_mode() and leave_transport_mode() under the exact-one target selection enforced by
 * metal_peer_target.hpp. The layout cookie is read and rejected before any
 * address or option slot is accessed. Local pointers are constructed only
 * after their non-zero and alignment checks pass.
 */
inline void run_metal_peer_kernel() {
    using Member = MetalPeerTarget;
    const uint32_t layout_cookie = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_LAYOUT_COOKIE);
    if (!ubench_layout_is_current(layout_cookie, P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE)) {
        return;
    }

    p2p_latency_common_args_t common{};
    common.self = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_SELF);
    common.first_direction = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_FIRST_DIRECTION);
    common.payload_size = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_PAYLOAD_SIZE);
    common.warmup_count = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_WARMUP_COUNT);
    common.sample_count = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_SAMPLE_COUNT);
    common.run_nonce = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_RUN_NONCE);
    const uint32_t poll_lo = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_LO);
    const uint32_t poll_hi = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_HI);
    common.warmup_max_poll_cycles = (static_cast<uint64_t>(poll_hi) << 32) | static_cast<uint64_t>(poll_lo);

    const uint32_t own_inbound = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_OWN_INBOUND_ADDR);
    const uint32_t own_staging = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_OWN_STAGING_ADDR);
    const uint32_t own_result = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_OWN_RESULT_ADDR);
    const uint32_t peer_inbound_lo = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_LO);
    const uint32_t peer_inbound_hi = get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_HI);
    const x280_noc_coord_t peer_noc{
        get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_X),
        get_arg_val<uint32_t>(P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_Y)};
    if (common.self != P2P_LATENCY_PARTICIPANT_B) {
        return;
    }
    if (!p2p_latency_common_valid(&common)) {
        return;
    }
    if (own_inbound == 0u || own_staging == 0u || own_result == 0u) {
        return;
    }
    if ((own_inbound % 4u) != 0u || (own_staging % 4u) != 0u || (own_result % 4u) != 0u) {
        return;
    }
    if (peer_inbound_lo == 0u && peer_inbound_hi == 0u) {
        return;
    }
    if ((peer_inbound_lo % 4u) != 0u) {
        return;
    }

    volatile tt_l1_ptr p2p_latency_result_t* result_v =
        reinterpret_cast<volatile tt_l1_ptr p2p_latency_result_t*>(own_result);
    p2p_latency_result_t& result = *const_cast<p2p_latency_result_t*>(result_v);

    using Backend = typename Member::Backend;
    typename Backend::state_t state{};
    if (!Backend::initialize(
            state,
            common,
            own_inbound,
            own_staging,
            peer_inbound_lo,
            peer_inbound_hi,
            peer_noc,
            Member::HANG_GUARD_TICKS,
            result)) {
        result.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_ARG_LAYOUT_MISMATCH);
        return;
    }

    if (!Member::enter_transport_mode()) {
        result.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_TRAP);
        return;
    }

    (void)p2p_latency::run_participant<
        Backend,
        P2P_LATENCY_TRANSFER_TOKEN_ONLY,
        P2P_LATENCY_POLL_TIGHT,
        P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION>(state, common, result);

    if (!Member::leave_transport_mode()) {
        result.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_TRAP);
    }
}

}  // namespace p2p_latency

#endif  // X280_P2P_LATENCY_METAL_PEER_KERNEL_HPP
