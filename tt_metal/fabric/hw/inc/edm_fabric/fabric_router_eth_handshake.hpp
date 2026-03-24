// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE void fabric_sender_side_handshake(
    uint32_t handshake_register_address,
    uint16_t my_mesh_id,
    uint8_t my_device_id,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    volatile tt_l1_ptr handshake_info_t* handshake_info =
        init_handshake_info(handshake_register_address, my_mesh_id, my_device_id);
    uint32_t local_val_addr = ((uint32_t)(&handshake_info->local_value)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t scratch_addr = ((uint32_t)(&handshake_info->scratch)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t count = 0;
    while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE
#ifndef ARCH_WORMHOLE
           && !tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)
#endif
    ) {
        if (count == HS_CONTEXT_SWITCH_TIMEOUT) {
            count = 0;
            static_assert(PHYSICAL_AERISC_ID == 0, "run_routing() is only safe from ERISC0");
            run_routing();
        } else {
            count++;
            internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
        }
        invalidate_l1_cache();
    }
}

template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE void fabric_receiver_side_handshake(
    uint32_t handshake_register_address,
    uint16_t my_mesh_id,
    uint8_t my_device_id,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    volatile tt_l1_ptr handshake_info_t* handshake_info =
        init_handshake_info(handshake_register_address, my_mesh_id, my_device_id);
    uint32_t local_val_addr = ((uint32_t)(&handshake_info->local_value)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t scratch_addr = ((uint32_t)(&handshake_info->scratch)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t count = 0;
    while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE
#ifndef ARCH_WORMHOLE
           && !tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)
#endif
    ) {
        if (count == HS_CONTEXT_SWITCH_TIMEOUT) {
            count = 0;
            static_assert(PHYSICAL_AERISC_ID == 0, "run_routing() is only safe from ERISC0");
            run_routing();
        } else {
            count++;
        }
        invalidate_l1_cache();
    }
#ifndef ARCH_WORMHOLE
    if (!tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)) {
        internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
    }
#else
    internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
#endif
}

}  // namespace handshake
}  // namespace datamover
}  // namespace erisc
