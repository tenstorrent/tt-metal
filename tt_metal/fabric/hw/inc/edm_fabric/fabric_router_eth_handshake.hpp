// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"

namespace erisc {
namespace datamover {
namespace handshake {

/*
 * Fabric-specific handshake functions with termination signal support.
 * These extend the base EDM handshake with the ability to exit early
 * when the host requests immediate termination, enabling graceful
 * recovery during fabric init/teardown.
 */

// FIX AD (#42429): LLDP-style symmetric handshake for fabric router.
// Both sides send MAGIC and poll for MAGIC — no sender/receiver distinction.
// IMPORTANT: prepare_handshake_state() MUST have been called during Object Setup
// (before edm_status = STARTED) so that local_value is zeroed and scratch contains MAGIC.
template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE void fabric_symmetric_handshake(
    uint32_t handshake_register_address,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    volatile tt_l1_ptr handshake_info_t* handshake_info =
        reinterpret_cast<volatile tt_l1_ptr handshake_info_t*>(handshake_register_address);

#if STRATEGY7_HANDSHAKE_BYPASS
    // FIX S7 (#42429): Host confirmed all ERISCs are alive via STARTED poll and wrote
    // handshake_bypass=1 to each ERISC's L1 handshake_info during configure_fabric_cores().
    // The ETH DMA handshake loop is redundant with the host barrier — skip it entirely.
    // This eliminates all handshake-related race conditions: TXQ races, STARTED deadlocks,
    // lost MAGIC overwrites, and the SENDER↔SENDER deadlock class.
    // Old handshake code preserved below (unreachable when bypass active) for rollback.
    if (handshake_info->handshake_bypass != 0) {
        return;
    }
#endif  // STRATEGY7_HANDSHAKE_BYPASS

    uint32_t local_val_addr = ((uint32_t)(&handshake_info->local_value)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t scratch_addr = ((uint32_t)(&handshake_info->scratch)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t count = 0;
    while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE &&
           !tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)) {
        if (count == HS_CONTEXT_SWITCH_TIMEOUT) {
            count = 0;

#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)
            run_routing();
#endif
        } else {
            count++;
            internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
            handshake_info->diag_send_count++;  // Strategy 11: track send attempts
        }
        invalidate_l1_cache();
    }
    // FIX HS1/HS2 (#42429): Post-loop final send (defense-in-depth).
    // With Fix A (early init), the erase race is eliminated. This final send remains
    // as belt-and-suspenders for the case where one side exits slightly before the
    // other has entered the loop.
    if (!tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)) {
        internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
    }
}

// Legacy fabric_sender_side_handshake — DEPRECATED, use prepare_handshake_state() + fabric_symmetric_handshake().
template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE void fabric_sender_side_handshake(
    uint32_t handshake_register_address,
    uint16_t my_mesh_id,
    uint8_t my_device_id,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    init_handshake_info(handshake_register_address, my_mesh_id, my_device_id);
    fabric_symmetric_handshake<RISC_CPU_DATA_CACHE_ENABLED>(
        handshake_register_address, termination_signal_ptr, HS_CONTEXT_SWITCH_TIMEOUT);
}

// Legacy fabric_receiver_side_handshake — DEPRECATED, use prepare_handshake_state() + fabric_symmetric_handshake().
template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE void fabric_receiver_side_handshake(
    uint32_t handshake_register_address,
    uint16_t my_mesh_id,
    uint8_t my_device_id,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    init_handshake_info(handshake_register_address, my_mesh_id, my_device_id);
    fabric_symmetric_handshake<RISC_CPU_DATA_CACHE_ENABLED>(
        handshake_register_address, termination_signal_ptr, HS_CONTEXT_SWITCH_TIMEOUT);
}

}  // namespace handshake
}  // namespace datamover
}  // namespace erisc
