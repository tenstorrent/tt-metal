// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstdint>
#include "tt_metal/lite_fabric/hw/inc/constants.hpp"
#include "tt_metal/lite_fabric/hw/inc/header.hpp"
#include "tt_metal/lite_fabric/hw/inc/host_interface.hpp"
#include "tt_metal/lite_fabric/hw/inc/types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

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
template <uint32_t CHANNEL_INDEX>
FORCE_INLINE void send_next_data(
    SenderEthChannelBuffer& sender_buffer_channel,
    lite_fabric::OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>&
        outbound_to_receiver_channel_pointers,
    ReceiverEthChannelBuffer& receiver_buffer_channel) {
    auto& remote_receiver_buffer_index = outbound_to_receiver_channel_pointers.remote_receiver_buffer_index;
    auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
    constexpr uint32_t sender_txq_id = 0;
    uint32_t src_addr = sender_buffer_channel.get_cached_next_buffer_slot_addr();

    volatile auto* pkt_header = reinterpret_cast<volatile lite_fabric::FabricLiteHeader*>(src_addr);

    if (pkt_header->get_base_send_type() == lite_fabric::NocSendTypeEnum::WRITE_REG) {
        const uint32_t reg_address = pkt_header->command_fields.write_reg.reg_address;
        const uint32_t reg_value = pkt_header->command_fields.write_reg.reg_value;
        while (internal_::eth_txq_is_busy(sender_txq_id));
        internal_::eth_write_remote_reg(sender_txq_id, reg_address, reg_value);
        // Continue to forward the packet to ensure pointers are synced
    }

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
        tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(
            host_interface->d2h.fabric_sender_channel_index);

    remote_receiver_buffer_index = tt::tt_fabric::BufferIndex{
        tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(remote_receiver_buffer_index.get())};
    receiver_buffer_channel.set_cached_next_buffer_slot_addr(
        receiver_buffer_channel.get_buffer_address(remote_receiver_buffer_index));
    sender_buffer_channel.set_cached_next_buffer_slot_addr(sender_buffer_channel.get_buffer_address(
        tt::tt_fabric::BufferIndex{(uint8_t)host_interface->d2h.fabric_sender_channel_index}));
    remote_receiver_num_free_slots--;
    // update the remote reg
    static constexpr uint32_t packets_to_forward = 1;
    while (internal_::eth_txq_is_busy(sender_txq_id));
    remote_update_ptr_val<to_receiver_pkts_sent_ids[CHANNEL_INDEX], sender_txq_id>(packets_to_forward);
}

template <uint32_t CHANNEL_INDEX>
FORCE_INLINE void run_sender_channel_step() {
    auto& outbound_to_receiver_channel_pointers =
        outbound_to_receiver_channel_pointers_tuple.template get<CHANNEL_INDEX>();
    auto& local_sender_channel = lite_fabric::local_sender_channels.template get<CHANNEL_INDEX>();
    auto& remote_receiver_channel = remote_receiver_channels.template get<CHANNEL_INDEX>();
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    bool has_unsent_packet =
        host_interface->h2d.sender_host_write_index != host_interface->d2h.fabric_sender_channel_index;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    if (can_send) {
        send_next_data<CHANNEL_INDEX>(
            local_sender_channel, outbound_to_receiver_channel_pointers, remote_receiver_channel);
    }

    // Process COMPLETIONs from receiver
    int32_t completions_since_last_check = get_ptr_val(to_sender_pkts_completed_ids[CHANNEL_INDEX]);
    if (completions_since_last_check) {
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        increment_local_update_ptr_val(to_sender_pkts_completed_ids[CHANNEL_INDEX], -completions_since_last_check);
    }
}

/////////////////////
// Receiver Channel
/////////////////////
template <uint32_t CHANNEL_INDEX>
__attribute__((optimize("jump-tables"))) FORCE_INLINE void service_fabric_request(
    tt_l1_ptr lite_fabric::FabricLiteHeader* const packet_start,
    uint16_t payload_size_bytes,
    uint32_t transaction_id,
    tt::tt_fabric::EthChannelBuffer<lite_fabric::FabricLiteHeader, SENDER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>&
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
                                                   sizeof(lite_fabric::FabricLiteHeader) + header.unaligned_offset;

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
                const uint64_t src_address = header.command_fields.noc_read.noc_address;
                // This assumes nobody else is using the sender channel on device 1 because
                // the tunnel depth is only 1 at the moment
                uint32_t dst_header_address = sender_buffer_channel.get_cached_next_buffer_slot_addr();
                // Create packet header for writing back
                tt_l1_ptr lite_fabric::FabricLiteHeader* packet_header_in_sender_ch =
                    reinterpret_cast<lite_fabric::FabricLiteHeader*>(dst_header_address);
                *packet_header_in_sender_ch = header;
                // Read the data into the buffer
                // This is safe only if the data at the sender buffer slot has been flushed out
                // We rely on the host to not do a read until the received data has been read out
                // When doing reads, ensure that the lower bits of the src_address and payload_dst_address are the same
                // we will let the host know of the data offset by setting header.unaligned_offset

                // Calculate natural payload location (immediately after header)
                uint32_t natural_payload_address = dst_header_address + sizeof(lite_fabric::FabricLiteHeader);

                // Get the lower 6 bits that we need to match from source address
                uint32_t src_alignment = src_address & (GLOBAL_ALIGNMENT - 1);  // Lower 6 bits of source
                uint32_t natural_alignment =
                    natural_payload_address & (GLOBAL_ALIGNMENT - 1);  // Lower 6 bits of natural location

                // Calculate offset needed to align to source's lower 6 bits
                uint32_t alignment_offset;
                if (src_alignment >= natural_alignment) {
                    alignment_offset = src_alignment - natural_alignment;
                } else {
                    alignment_offset = GLOBAL_ALIGNMENT + src_alignment - natural_alignment;
                }

                // Final aligned payload address
                uint32_t payload_dst_address = natural_payload_address + alignment_offset;

                // Store the offset for the host to know where the actual data starts
                packet_header_in_sender_ch->unaligned_offset = alignment_offset;

                noc_async_read(src_address, payload_dst_address, payload_size_bytes, noc_index);
                noc_async_read_barrier(noc_index);

                // Tell ourselves there is data to send
                // NOTE: sender_buffer_channel index will be incremented in send_next_data
                host_interface->h2d.sender_host_write_index =
                    tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(
                        host_interface->h2d.sender_host_write_index);
            }
        } break;

        case lite_fabric::NocSendTypeEnum::WRITE_REG: {
            // Do nothing. Sender directly wrote to us with eth_write_remote_reg
        } break;

        default: {
            ASSERT(false);
        } break;
    };
}

template <uint32_t CHANNEL_INDEX>
FORCE_INLINE void run_receiver_channel_step() {
    auto& receiver_channel_pointers = receiver_channel_pointers_tuple.template get<CHANNEL_INDEX>();
    auto& local_sender_channel = lite_fabric::local_sender_channels.template get<CHANNEL_INDEX>();
    auto& remote_receiver_channel = remote_receiver_channels.template get<CHANNEL_INDEX>();
    auto pkts_received_since_last_check = get_ptr_val<to_receiver_pkts_sent_ids[CHANNEL_INDEX]>();
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    bool unwritten_packets = pkts_received_since_last_check != 0;

    if (unwritten_packets) {
        invalidate_l1_cache();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr lite_fabric::FabricLiteHeader* packet_header = const_cast<lite_fabric::FabricLiteHeader*>(
            remote_receiver_channel.template get_packet_header<lite_fabric::FabricLiteHeader>(receiver_buffer_index));

        receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);

        uint8_t trid = receiver_channel_0_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
            receiver_buffer_index);
        // lite fabric tunnel depth is 1 so any fabric cmds being sent here will be writes to/reads from this chip
        service_fabric_request<CHANNEL_INDEX>(
            packet_header, packet_header->payload_size_bytes, trid, local_sender_channel);

        wr_sent_counter.increment();
        // decrement the to_receiver_0_pkts_sent_id stream register by 1 since current packet has been processed.
        increment_local_update_ptr_val<to_receiver_pkts_sent_ids[CHANNEL_INDEX]>(-1);
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
            can_send_completion &&
            (((host_interface->d2h.fabric_receiver_channel_index + 1) % RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]) !=
             host_interface->h2d.receiver_host_read_index);
    }

    if (can_send_completion) {
        // Completion pointer is to the host
        while (internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ));
        remote_update_ptr_val<DEFAULT_ETH_TXQ>(to_sender_pkts_completed_ids[CHANNEL_INDEX], 1);

        receiver_channel_0_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
        completion_counter.increment();
        if (on_mmio_chip) {
            host_interface->d2h.fabric_receiver_channel_index =
                tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[CHANNEL_INDEX]>(
                    host_interface->d2h.fabric_receiver_channel_index);
        }
    }
}

}  // namespace lite_fabric
