// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/hw/inc/ethernet/tunneling.h"
#include "tt_metal/hw/inc/ethernet/tt_eth_api.h"
#include <tuple>
#include <cstdint>
#include <cstddef>
#include "debug/dprint.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h"
constexpr uint32_t data_txq_id = get_compile_time_arg_val(0);
constexpr uint32_t ack_txq_id = get_compile_time_arg_val(1);
constexpr uint32_t PAYLOAD_SIZE = get_compile_time_arg_val(2);

static constexpr uint32_t CREDITS_STREAM_ID = 0;
static constexpr uint32_t ACK_STREAM_ID = 1;

void kernel_main() {
    size_t arg_idx = 0;
    uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    bool is_handshake_sender = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_credit_ack_src = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_credit_ack_dest = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(arg_idx++);
    int num_messages = get_arg_val<uint32_t>(arg_idx++);

    // Clear our counters for receiver credits src + dest
    *reinterpret_cast<volatile uint32_t*>(receiver_credit_ack_src) = 0;
    *reinterpret_cast<volatile uint32_t*>(receiver_credit_ack_dest) = 0;

    if constexpr (data_txq_id != ack_txq_id) {
        eth_enable_packet_mode(ack_txq_id);
    }

    init_ptr_val(CREDITS_STREAM_ID, 0);

    // Handshake to make sure it's safe to start sending
    if (is_handshake_sender) {
        erisc::datamover::handshake::sender_side_handshake(handshake_addr);
    } else {
        erisc::datamover::handshake::receiver_side_handshake(handshake_addr);
    }

    bool has_unsent_messages = true;
    bool has_unsent_acks = true;
    int num_messages_sent = 0;
    int num_acks_sent = 0;
    int last_printed_ack = 0;
    size_t idle_count = 0;
    while (has_unsent_messages || has_unsent_acks) {
        // Send Messages
        bool current_ack = *reinterpret_cast<volatile int32_t*>(receiver_credit_ack_dest);
        if (current_ack != last_printed_ack) {
            last_printed_ack = current_ack;
        }
        if (has_unsent_messages) {
            *reinterpret_cast<volatile uint32_t*>(local_eth_l1_src_addr) = num_messages_sent + 1;
            while (internal_::eth_txq_is_busy(data_txq_id)) {
            }
            internal_::eth_send_packet_bytes_unsafe(
                data_txq_id, local_eth_l1_src_addr, remote_eth_l1_dst_addr, PAYLOAD_SIZE);
            while (internal_::eth_txq_is_busy(data_txq_id)) {
            }
            remote_update_ptr_val<CREDITS_STREAM_ID, data_txq_id>(1);
            num_messages_sent++;
            has_unsent_messages = num_messages_sent < num_messages;
            idle_count = 0;
        }

        // Send Acks
        if (has_unsent_acks) {
            if (get_ptr_val<CREDITS_STREAM_ID>() > num_acks_sent) {
                *reinterpret_cast<volatile uint32_t*>(receiver_credit_ack_src) = num_acks_sent + 1;
                while (internal_::eth_txq_is_busy(ack_txq_id)) {
                }
                internal_::eth_send_packet_bytes_unsafe(
                    ack_txq_id, receiver_credit_ack_src, receiver_credit_ack_dest, 16);
                num_acks_sent++;
                has_unsent_acks = num_acks_sent < num_messages;
                idle_count = 0;
            }
        }
        idle_count++;
    }

    while (*reinterpret_cast<volatile int32_t*>(receiver_credit_ack_dest) < num_messages_sent) {
        invalidate_l1_cache();
    }

    // Validate that at the very least we got the correct last value in the payload buffer
    while (*reinterpret_cast<volatile int32_t*>(remote_eth_l1_dst_addr) != num_messages_sent) {
        invalidate_l1_cache();
    }
}
