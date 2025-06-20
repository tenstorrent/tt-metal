// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ethernet/dataflow_api.h"
#include <fabric_host_interface.h>

namespace erisc {
namespace datamover {

/*
 * Before any payload messages can be exchanged over the link, we must ensure that the other end
 * of the link is ready to start sending/receiving messages. We perform a handshake to ensure that's
 * case. Before handshaking, we make sure to clear any of the channel sync datastructures local
 * to our core.
 *
 * Important note about handshaking: the sender/master canNOT complete the handshake until all receiver
 * channels are initialized. Otherwise we have a race between channel initialization on the receiver side
 * and real payload data (and signals) using those channels.
 *
 * Note that the master and subordinate concepts here only apply in the context of handshaking and initialization
 * of the EDM. They do not apply during the main EDM execution loop.
 *
 * The basic protocol for the handshake is to use the reserved space at erisc_info[0] where the master writes
 * and sends payload available information to that channel. The receive must acknowledge that message and upon
 * doing so, considers the handshake complete.
 */

namespace handshake {

/* EDM Handshaking Mechanism:
 * 1. Both sides set their local_value register to 0.
 * 2. Both sides write a magic value to their scratch register.
 * 3. Handshake master repeatedly copies the magic value from the scratch register to the local_value of the remote
 * subordinate, until it sees the magic value in its local_value register.
 * 4. Handshake subordinate polls its local_value register until it sees the magic value written by the master. It
 * then copies the magic value from its scratch register to the master's local_value register, completing the handshake.
 */

static constexpr uint32_t A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH = 1000000000;
static constexpr uint32_t MAGIC_HANDSHAKE_VALUE = 0xAA;

// Data-Structure used for EDM to EDM Handshaking.
struct handshake_info_t {
    uint32_t local_value;  // Updated by remote
    uint32_t padding[3];   // Ensures 16B alignment for scratch register
    uint32_t scratch[4];   // TODO: This can be removed if we use a stream register for handshaking.
};

FORCE_INLINE volatile tt_l1_ptr handshake_info_t* init_handshake_info(uint32_t handshake_register_address) {
    volatile tt_l1_ptr handshake_info_t* handshake_info =
        reinterpret_cast<volatile tt_l1_ptr handshake_info_t*>(handshake_register_address);
    handshake_info->local_value = 0;
    handshake_info->scratch[0] = MAGIC_HANDSHAKE_VALUE;
    return handshake_info;
}

FORCE_INLINE void sender_side_handshake(
    uint32_t handshake_register_address, size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    volatile tt_l1_ptr handshake_info_t* handshake_info = init_handshake_info(handshake_register_address);
    uint32_t local_val_addr = ((uint32_t)(&handshake_info->local_value)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t scratch_addr = ((uint32_t)(&handshake_info->scratch)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t count = 0;
    while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE) {
        if (count == HS_CONTEXT_SWITCH_TIMEOUT) {
            count = 0;
            run_routing();
        } else {
            count++;
            internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
        }
        invalidate_l1_cache();
    }
}

FORCE_INLINE void receiver_side_handshake(
    uint32_t handshake_register_address, size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    volatile tt_l1_ptr handshake_info_t* handshake_info = init_handshake_info(handshake_register_address);
    uint32_t local_val_addr = ((uint32_t)(&handshake_info->local_value)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t scratch_addr = ((uint32_t)(&handshake_info->scratch)) / tt::tt_fabric::PACKET_WORD_SIZE_BYTES;
    uint32_t count = 0;
    while (handshake_info->local_value != MAGIC_HANDSHAKE_VALUE) {
        if (count == HS_CONTEXT_SWITCH_TIMEOUT) {
            count = 0;
            run_routing();
        } else {
            count++;
        }
        invalidate_l1_cache();
    }
    internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1);
}

namespace deprecated {

/* The split handshaking mechanism exposed through the following APIs is deprecated.
 * This was developed for non-persistent kernels targeting EDM cores, which are not
 * supported with TT-Fabric.
 * TODO: Remove these APIs once legacy CCL Ops are removed.
 *
 * The handshaking process is split into two parts for the sender/master and two parts for the
 * the subordinate. The handshake is broken into 2 parts so that the master can initiate the handshake
 * as early as possible so the message can be "in flight" over the ethernet link while other EDM
 * initialization is taking place.
 */

/*
 * Initialize base datastructures and values which are common to master and subordinate EDM cores.
 * The main memory region initialized here is the channel ack region offset 16B from the
 * base handshake address.
 *
 * This memory region serves a special purpose for flow control between EDM cores. This
 * 16B region is initialized to a fixed set of values. This region is used by receiver
 * EDM channels when sending first level acks to its corresponding sender EDM channel.
 *
 * See ChannelBuffer::eth_receiver_channel_ack for more information
 */
FORCE_INLINE void initialize_edm_common_datastructures(uint32_t handshake_register_address) {
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_register_address)[4] = 1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_register_address)[5] = 1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_register_address)[6] = 0x1c0ffee1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_register_address)[7] = 0x1c0ffee2;

    erisc_info->channels[0].receiver_ack = 0;
    for (uint32_t i = 1; i < eth_l1_mem::address_map::MAX_NUM_CONCURRENT_TRANSACTIONS; i++) {
        erisc_info->channels[i].bytes_sent = 0;
        erisc_info->channels[i].receiver_ack = 0;
    }
    *(volatile tt_l1_ptr uint32_t*)handshake_register_address = 0;
}

/*
 * As the designated master EDM core, initiate a handshake by sending a packet to reserved
 * memory region.
 */
FORCE_INLINE void sender_side_start(
    uint32_t handshake_register_address, size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    initialize_edm_common_datastructures(handshake_register_address);
    eth_wait_receiver_done(HS_CONTEXT_SWITCH_TIMEOUT);
    while (eth_txq_is_busy()) {
        asm volatile("nop");
    }
    eth_send_bytes(handshake_register_address, handshake_register_address, 16);
}

/*
 * As the designated master EDM core, wait for the acknowledgement from the subordinate EDM core
 */
FORCE_INLINE void sender_side_finish(
    uint32_t handshake_register_address, size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    eth_wait_for_receiver_done(HS_CONTEXT_SWITCH_TIMEOUT);
}

FORCE_INLINE void receiver_side_start(uint32_t handshake_register_address) {
    initialize_edm_common_datastructures(handshake_register_address);
}

/*
 * Return: true if subordinate EDM handshake core is able to complete the handshake with
 * an ack.
 */
FORCE_INLINE bool receiver_side_can_finish() { return eth_bytes_are_available_on_channel(0); }

/*
 * As the designated subordinate EDM core, send the acknowledgement to the master EDM core.
 * The subordinate EDM core shall only acknowledge after receiving the initial handshake packet
 * from the master EDM core.
 */
FORCE_INLINE void receiver_side_finish(
    uint32_t handshake_register_address, size_t HS_CONTEXT_SWITCH_TIMEOUT = A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH) {
    eth_wait_for_bytes(16, HS_CONTEXT_SWITCH_TIMEOUT);
    while (eth_txq_is_busy()) {
        asm volatile("nop");
    }
    eth_receiver_channel_done(0);
}

}  // namespace deprecated

}  // namespace handshake

}  // namespace datamover
}  // namespace erisc
