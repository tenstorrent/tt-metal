// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ttnn/deprecated/tt_dnn/op_library/ccl/edm/erisc_async_datamover.hpp"

#define DONT_STRIDE_IN_ETH_BUFFER 0

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */


// Initiate DRAM write -> advances  write pointer
template <bool dest_is_dram>
void write_chunk(
    const uint32_t eth_l1_buffer_address_base,
    const uint32_t num_pages,
    const uint32_t num_pages_per_l1_buffer,
    const uint32_t page_size,
    uint32_t &page_index,
    const InterleavedAddrGen<dest_is_dram> &dest_address_generator) {
    uint32_t local_eth_l1_curr_src_addr = eth_l1_buffer_address_base;
    uint32_t end_page_index = std::min(page_index + num_pages_per_l1_buffer, num_pages);
    for (; page_index < end_page_index; ++page_index) {
        // read source address
        uint64_t dest_noc_addr = get_noc_addr(page_index, dest_address_generator);
        noc_async_write(local_eth_l1_curr_src_addr, dest_noc_addr, page_size);
        // read dest addr
        #if DONT_STRIDE_IN_ETH_BUFFER == 0
        local_eth_l1_curr_src_addr += page_size;
        #endif
    }
}


template <uint32_t MAX_NUM_CHANNELS, bool dest_is_dram>
bool eth_initiate_noc_write_sequence(
    std::array<uint32_t, MAX_NUM_CHANNELS> &transaction_channel_receiver_buffer_addresses,
    erisc::datamover::QueueIndexPointer<uint8_t> &noc_writer_buffer_wrptr,
    erisc::datamover::QueueIndexPointer<uint8_t> &noc_writer_buffer_ackptr,
    const erisc::datamover::QueueIndexPointer<uint8_t> eth_receiver_wrptr,
    const erisc::datamover::QueueIndexPointer<uint8_t> eth_receiver_ackptr,

    const uint32_t num_pages,
    const uint32_t num_pages_per_l1_buffer,
    const uint32_t page_size,
    uint32_t &page_index,
    const InterleavedAddrGen<dest_is_dram> &dest_address_generator) {
    bool did_something = false;
    bool noc_write_is_in_progress = erisc::datamover::deprecated::receiver_is_noc_write_in_progress(noc_writer_buffer_wrptr, noc_writer_buffer_ackptr);

    if (!noc_write_is_in_progress) {
        bool next_payload_received = noc_writer_buffer_wrptr != eth_receiver_wrptr;
        if (next_payload_received) {
            // Can initialize a new write if data is at this buffer location (eth num_bytes != 0)
            // and the receiver ackptr != next write pointer
            // // DPRINT << "rx: accepting payload, sending receive ack on channel " << (uint32_t)noc_writer_buffer_wrptr << "\n";
            write_chunk<dest_is_dram>(
                transaction_channel_receiver_buffer_addresses[noc_writer_buffer_wrptr.index()],
                num_pages,
                num_pages_per_l1_buffer,
                page_size,
                page_index,
                dest_address_generator);
            noc_writer_buffer_wrptr.increment();
            did_something = true;
        }
    }

    return did_something;
}

void kernel_main() {
    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);
    constexpr std::uint32_t total_num_message_sends = get_compile_time_arg_val(2);
    constexpr std::uint32_t NUM_TRANSACTION_BUFFERS = get_compile_time_arg_val(3);
    constexpr bool dest_is_dram = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t MAX_NUM_CHANNELS = NUM_TRANSACTION_BUFFERS;
    // Handshake first before timestamping to make sure we aren't measuring any
    // dispatch/setup times for the kernels on both sides of the link.

    const std::uint32_t eth_channel_sync_ack_addr = get_arg_val<uint32_t>(0);
    const std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(1);
    const std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(2);
    const std::uint32_t dest_addr = get_arg_val<uint32_t>(3);
    const std::uint32_t page_size = get_arg_val<uint32_t>(4);
    const std::uint32_t num_pages = get_arg_val<uint32_t>(5);
    erisc::datamover::eth_setup_handshake(remote_eth_l1_dst_addr, false);

    const InterleavedAddrGen<dest_is_dram> dest_address_generator = {
        .bank_base_address = dest_addr, .page_size = page_size};

    erisc::datamover::QueueIndexPointer<uint8_t> noc_writer_buffer_ackptr(MAX_NUM_CHANNELS);
    erisc::datamover::QueueIndexPointer<uint8_t> noc_writer_buffer_wrptr(MAX_NUM_CHANNELS);
    erisc::datamover::QueueIndexPointer<uint8_t> eth_receiver_rdptr(MAX_NUM_CHANNELS);
    erisc::datamover::QueueIndexPointer<uint8_t> eth_receiver_ackptr(MAX_NUM_CHANNELS);

    {
        DeviceZoneScopedN("eth_latency");
        std::array<uint32_t, NUM_TRANSACTION_BUFFERS> transaction_channel_remote_buffer_addresses;
        erisc::datamover::initialize_transaction_buffer_addresses<NUM_TRANSACTION_BUFFERS>(
            MAX_NUM_CHANNELS,
            remote_eth_l1_dst_addr,
            num_bytes_per_send,
            transaction_channel_remote_buffer_addresses);
        std::array<uint32_t, NUM_TRANSACTION_BUFFERS> transaction_channel_local_buffer_addresses;
        erisc::datamover::initialize_transaction_buffer_addresses<NUM_TRANSACTION_BUFFERS>(
            MAX_NUM_CHANNELS,
            local_eth_l1_src_addr,
            num_bytes_per_send,
            transaction_channel_local_buffer_addresses);

        constexpr uint32_t SWITCH_INTERVAL = 100000;
        uint32_t page_index = 0;
        const uint32_t num_pages_per_l1_buffer = num_bytes_per_send / page_size;

        bool write_in_flight = false;
        uint32_t num_eth_sends_acked = 0;
        uint32_t count = 0;
        uint32_t num_context_switches = 0;
        uint32_t max_num_context_switches = 1000;
        bool printed_hang = false;
        uint32_t num_receives_acked = 0;

        while (num_eth_sends_acked < total_num_message_sends && page_index < num_pages) {
            bool did_something = false;

            bool received = erisc::datamover::deprecated::receiver_eth_accept_payload_sequence(
                                noc_writer_buffer_wrptr, noc_writer_buffer_ackptr, eth_receiver_rdptr, eth_receiver_ackptr, eth_channel_sync_ack_addr);
            num_receives_acked = received ? num_receives_acked + 1 : num_receives_acked;
            did_something = received || did_something;

            did_something = eth_initiate_noc_write_sequence<MAX_NUM_CHANNELS, dest_is_dram>(
                                transaction_channel_local_buffer_addresses,
                                noc_writer_buffer_wrptr,
                                noc_writer_buffer_ackptr,
                                eth_receiver_rdptr,
                                eth_receiver_ackptr,

                                num_pages,
                                num_pages_per_l1_buffer,
                                page_size,
                                page_index,
                                dest_address_generator) ||
                            did_something;

            did_something = erisc::datamover::deprecated::receiver_noc_read_worker_completion_check_sequence(
                                noc_writer_buffer_wrptr, noc_writer_buffer_ackptr, noc_index) ||
                            did_something;

            did_something =
                erisc::datamover::deprecated::receiver_eth_send_ack_to_sender_sequence(
                    noc_writer_buffer_wrptr, noc_writer_buffer_ackptr, eth_receiver_rdptr, eth_receiver_ackptr, num_eth_sends_acked) ||
                did_something;

            if (!did_something) {

                if (count++ > SWITCH_INTERVAL) {
                    count = 0;
                    run_routing();
                    num_context_switches++;
                    if (num_context_switches > max_num_context_switches) {
                        if (!printed_hang) {
                            DPRINT << "rx: HANG\n";
                            DPRINT << "rx: HANG num_eth_sends_acked " << (uint32_t)num_eth_sends_acked << "\n";
                            DPRINT << "rx: HANG total_num_message_sends " << (uint32_t)total_num_message_sends << "\n";
                            for (uint32_t i = 0; i < MAX_NUM_CHANNELS; i++) {

                                DPRINT << "rx: HANG channel [" << i << "] bytes_sent " << erisc_info->channels[0].bytes_sent << "\n";
                                DPRINT << "rx: HANG channel [" << i << "] bytes_receiver_ack " << erisc_info->channels[0].receiver_ack << "\n";
                                DPRINT << "rx: HANG eth_is_receiver_channel_send_acked (" << i << ") " << (eth_is_receiver_channel_send_acked(i) ? "true" : "false") << "\n";
                                DPRINT << "rx: HANG eth_is_receiver_channel_send_done(" << i << ") " << (eth_is_receiver_channel_send_done(i) ? "true" : "false") << "\n";
                            }
                            DPRINT << "rx: HANG noc_writer_buffer_ackptr " << (uint32_t)noc_writer_buffer_ackptr.index() << "\n";
                            DPRINT << "rx: HANG (raw) noc_writer_buffer_ackptr " << (uint32_t)noc_writer_buffer_ackptr.raw_index() << "\n";
                            DPRINT << "rx: HANG noc_writer_buffer_wrptr " << (uint32_t)noc_writer_buffer_wrptr.index() << "\n";
                            DPRINT << "rx: HANG (raw) noc_writer_buffer_wrptr " << (uint32_t)noc_writer_buffer_wrptr.raw_index() << "\n";
                            DPRINT << "rx: HANG eth_receiver_rdptr " << (uint32_t)eth_receiver_rdptr.index() << "\n";
                            DPRINT << "rx: HANG (raw) eth_receiver_rdptr " << (uint32_t)eth_receiver_rdptr.raw_index() << "\n";
                            DPRINT << "rx: HANG eth_receiver_ackptr " << (uint32_t)eth_receiver_ackptr.index() << "\n";
                            DPRINT << "rx: HANG (raw) eth_receiver_ackptr " << (uint32_t)eth_receiver_ackptr.raw_index() << "\n";
                            DPRINT << "rx: HANG num_receives_acked " << (uint32_t)num_receives_acked << "\n";
                            printed_hang = true;
                            num_context_switches = 0;
                        }
                    }
                } else {
                    count++;
                }
            } else {
                num_context_switches = 0;
            }
        }

        while (!ncrisc_noc_nonposted_writes_sent(noc_index));
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index));

        DPRINT << "rx: DONE\n";
        DPRINT << "rx: DONE eth_sends_completed " << (uint32_t)num_eth_sends_acked << "\n";
        DPRINT << "rx: DONE total_num_message_sends " << (uint32_t)total_num_message_sends << "\n";

        DPRINT << "rx: DONE num_eth_sends_acked " << (uint32_t)num_eth_sends_acked << "\n";
        DPRINT << "rx: DONE total_num_message_sends " << (uint32_t)total_num_message_sends << "\n";
        for (uint32_t i = 0; i < MAX_NUM_CHANNELS; i++) {

            DPRINT << "rx: DONE channel [" << i << "] bytes_sent " << erisc_info->channels[0].bytes_sent << "\n";
            DPRINT << "rx: DONE channel [" << i << "] bytes_receiver_ack " << erisc_info->channels[0].receiver_ack << "\n";
            DPRINT << "rx: DONE eth_is_receiver_channel_send_acked (" << i << ") " << (eth_is_receiver_channel_send_acked(i) ? "true" : "false") << "\n";
            DPRINT << "rx: DONE eth_is_receiver_channel_send_done(" << i << ") " << (eth_is_receiver_channel_send_done(i) ? "true" : "false") << "\n";
        }
        DPRINT << "rx: DONE noc_writer_buffer_ackptr " << (uint32_t)noc_writer_buffer_ackptr.index() << "\n";
        DPRINT << "rx: DONE (raw) noc_writer_buffer_ackptr " << (uint32_t)noc_writer_buffer_ackptr.raw_index() << "\n";
        DPRINT << "rx: DONE noc_writer_buffer_wrptr " << (uint32_t)noc_writer_buffer_wrptr.index() << "\n";
        DPRINT << "rx: DONE (raw) noc_writer_buffer_wrptr " << (uint32_t)noc_writer_buffer_wrptr.raw_index() << "\n";
        DPRINT << "rx: DONE eth_receiver_rdptr " << (uint32_t)eth_receiver_rdptr.index() << "\n";
        DPRINT << "rx: DONE (raw) eth_receiver_rdptr " << (uint32_t)eth_receiver_rdptr.raw_index() << "\n";
        DPRINT << "rx: DONE eth_receiver_ackptr " << (uint32_t)eth_receiver_ackptr.index() << "\n";
        DPRINT << "rx: DONE (raw) eth_receiver_ackptr " << (uint32_t)eth_receiver_ackptr.raw_index() << "\n";
        DPRINT << "rx: DONE num_receives_acked " << (uint32_t)num_receives_acked << "\n";
    }
}
