// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstdint>
#include "lite_fabric.hpp"
#include "lite_fabric_constants.hpp"
#include "lite_fabric_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_transaction_id_tracker.hpp"
#include "lite_fabric_types.hpp"

namespace lite_fabric {

// Linked from main
extern bool on_mmio_chip;
extern volatile HostInterface* host_interface;
extern RemoteReceiverChannelsType remote_receiver_channels;
extern LocalSenderChannelsType local_sender_channels;
extern WriteTridTracker receiver_channel_0_trid_tracker;

extern OutboundReceiverChannelPointersTupleImpl outbound_to_receiver_channel_pointers_tuple;
extern ReceiverChannelPointersTupleImpl receiver_channel_pointers_tuple;

/////////////////////
// Sender Channel
/////////////////////
FORCE_INLINE void send_next_data(
    SenderEthChannelBuffer& sender_buffer_channel,
    lite_fabric::OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS_ARRAY[0]>& outbound_to_receiver_channel_pointers,
    ReceiverEthChannelBuffer& receiver_buffer_channel) {
    auto& remote_receiver_buffer_index = outbound_to_receiver_channel_pointers.remote_receiver_buffer_index;
    auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
    constexpr uint32_t sender_txq_id = 0;
    uint32_t src_addr = sender_buffer_channel.get_cached_next_buffer_slot_addr();

    volatile auto* pkt_header = reinterpret_cast<volatile lite_fabric::LiteFabricHeader*>(src_addr);
    size_t payload_size_bytes = pkt_header->get_payload_size_including_header();
    // Actual payload may be offset by an unaligned offset. Ensure we include this in the payload size
    // Buffer slots have 16B padding at the end which is unused.
    payload_size_bytes += pkt_header->unaligned_offset;
    payload_size_bytes = (payload_size_bytes + 15) & ~15;
    uint32_t dest_addr = receiver_buffer_channel.get_cached_next_buffer_slot_addr();
    pkt_header->src_ch_id = 0;

    while (internal_::eth_txq_is_busy(sender_txq_id));
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    host_interface->d2h.fabric_sender_channel_index =
        tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[0]>(host_interface->d2h.fabric_sender_channel_index);

    remote_receiver_buffer_index = tt::tt_fabric::BufferIndex{
        tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[0]>(remote_receiver_buffer_index.get())};
    receiver_buffer_channel.set_cached_next_buffer_slot_addr(
        receiver_buffer_channel.get_buffer_address(remote_receiver_buffer_index));
    sender_buffer_channel.set_cached_next_buffer_slot_addr(sender_buffer_channel.get_buffer_address(
        tt::tt_fabric::BufferIndex{(uint8_t)host_interface->d2h.fabric_sender_channel_index}));
    remote_receiver_num_free_slots--;
    // update the remote reg
    static constexpr uint32_t packets_to_forward = 1;
    while (internal_::eth_txq_is_busy(sender_txq_id));
    remote_update_ptr_val<to_receiver_0_pkts_sent_id, sender_txq_id>(packets_to_forward);
}

FORCE_INLINE void run_sender_channel_step() {
    auto& outbound_to_receiver_channel_pointers = outbound_to_receiver_channel_pointers_tuple.template get<0>();
    auto& local_sender_channel = lite_fabric::local_sender_channels.template get<0>();
    auto& remote_receiver_channel = remote_receiver_channels.template get<0>();
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    bool has_unsent_packet =
        host_interface->h2d.sender_host_write_index != host_interface->d2h.fabric_sender_channel_index;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    if (can_send) {
        send_next_data(local_sender_channel, outbound_to_receiver_channel_pointers, remote_receiver_channel);
    }

    // Process COMPLETIONs from receiver
    int32_t completions_since_last_check = get_ptr_val(to_sender_0_pkts_completed_id);
    if (completions_since_last_check) {
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        increment_local_update_ptr_val(to_sender_0_pkts_completed_id, -completions_since_last_check);
    }
}

/////////////////////
// Receiver Channel
/////////////////////
__attribute__((optimize("jump-tables"))) FORCE_INLINE void service_fabric_request(
    tt_l1_ptr lite_fabric::LiteFabricHeader* const packet_start,
    uint16_t payload_size_bytes,
    uint32_t transaction_id,
    tt::tt_fabric::EthChannelBuffer<lite_fabric::LiteFabricHeader, SENDER_NUM_BUFFERS_ARRAY[0]>&
        sender_buffer_channel) {
    invalidate_l1_cache();
    const auto& header = *packet_start;

    lite_fabric::NocSendTypeEnum noc_send_type = header.get_base_send_type();
    uint8_t noc_index = header.get_noc_index();
    if (static_cast<int>(noc_send_type) > static_cast<int>(lite_fabric::NocSendTypeEnum::NOC_SEND_TYPE_LAST)) {
        __builtin_unreachable();
    }
    switch (noc_send_type) {
        case lite_fabric::NocSendTypeEnum::NOC_UNICAST_WRITE: {
            const uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) +
                                                   sizeof(lite_fabric::LiteFabricHeader) + header.unaligned_offset;

            const auto dest_address = header.command_fields.noc_unicast.noc_address;

            noc_async_write_one_packet_with_trid<true, false>(
                payload_start_address,
                dest_address,
                payload_size_bytes,
                transaction_id,
                lite_fabric::local_chip_data_cmd_buf,
                noc_index,
                lite_fabric::forward_and_local_write_noc_vc);
        } break;

        case lite_fabric::NocSendTypeEnum::NOC_READ: {
            if (!on_mmio_chip) {
                const auto src_address = header.command_fields.noc_read.noc_address;
                // This assumes nobody else is using the sender channel on device 1 because
                // the tunnel depth is only 1 at the moment
                uint32_t dst_address = sender_buffer_channel.get_cached_next_buffer_slot_addr();
                uint32_t payload_dst_address =
                    dst_address + sizeof(lite_fabric::LiteFabricHeader) + header.unaligned_offset;
                // Create packet header for writing back
                tt_l1_ptr lite_fabric::LiteFabricHeader* packet_header_in_sender_ch =
                    reinterpret_cast<lite_fabric::LiteFabricHeader*>(dst_address);
                *packet_header_in_sender_ch = header;
                // Read the data into the buffer
                // This is safe only if the data at the sender buffer slot has been flushed out
                // We rely on the host to not do a read until the received data has been read out
                noc_async_read(src_address, payload_dst_address, payload_size_bytes, noc_index);
                noc_async_read_barrier(noc_index);

                // Tell ourselves there is data to send
                // NOTE: sender_buffer_channel index will be incremented in send_next_data
                host_interface->h2d.sender_host_write_index =
                    tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[0]>(
                        host_interface->h2d.sender_host_write_index);
            } else {
            }

        } break;

        default: {
            ASSERT(false);
        } break;
    };
}

// MUST CHECK !is_eth_txq_busy() before calling
FORCE_INLINE void receiver_send_completion_ack(uint8_t src_id) {
    while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ));
    remote_update_ptr_val<DEFAULT_ETH_TXQ>(to_sender_0_pkts_completed_id, 1);
}

FORCE_INLINE void run_receiver_channel_step() {
    auto& receiver_channel_pointers = receiver_channel_pointers_tuple.template get<0>();
    auto& local_sender_channel = lite_fabric::local_sender_channels.template get<0>();
    auto& remote_receiver_channel = remote_receiver_channels.template get<0>();
    auto pkts_received_since_last_check = get_ptr_val<to_receiver_0_pkts_sent_id>();
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    bool unwritten_packets = pkts_received_since_last_check != 0;

    if (unwritten_packets) {
        invalidate_l1_cache();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr lite_fabric::LiteFabricHeader* packet_header = const_cast<lite_fabric::LiteFabricHeader*>(
            remote_receiver_channel.template get_packet_header<lite_fabric::LiteFabricHeader>(receiver_buffer_index));

        receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);

        uint8_t trid = receiver_channel_0_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
            receiver_buffer_index);
        // lite fabric tunnel depth is 1 so any fabric cmds being sent here will be writes to/reads from this chip
        service_fabric_request(packet_header, packet_header->payload_size_bytes, trid, local_sender_channel);

        wr_sent_counter.increment();
        // decrement the to_receiver_0_pkts_sent_id stream register by 1 since current packet has been processed.
        increment_local_update_ptr_val<to_receiver_0_pkts_sent_id>(-1);
    }

    // flush and completion are fused, so we only need to update one of the counters
    // update completion since other parts of the code check against completion
    auto& completion_counter = receiver_channel_pointers.completion_counter;
    // Currently unclear if it's better to loop here or not...
    bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
    auto receiver_buffer_index = completion_counter.get_buffer_index();
    bool next_trid_flushed = receiver_channel_0_trid_tracker.transaction_flushed(receiver_buffer_index);
    bool can_send_completion = unflushed_writes && next_trid_flushed;
    if (on_mmio_chip) {
        can_send_completion =
            can_send_completion && (((host_interface->d2h.fabric_receiver_channel_index + 1) %
                                     RECEIVER_NUM_BUFFERS_ARRAY[0]) != host_interface->h2d.receiver_host_read_index);
    }

    if (can_send_completion) {
        receiver_send_completion_ack(receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
        receiver_channel_0_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
        completion_counter.increment();
        if (on_mmio_chip) {
            host_interface->d2h.fabric_receiver_channel_index =
                tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[0]>(
                    host_interface->d2h.fabric_receiver_channel_index);
        }
    }
}

}  // namespace lite_fabric
