// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ttnn/deprecated/tt_dnn/op_library/ccl/edm/erisc_async_datamover.hpp"

#define ENABLE_L1_BUFFER_OVERLAP 0
// #define ENABLE_L1_BUFFER_OVERLAP 1
// #define EMULATE_DRAM_READ_CYCLES 1
#define EMULATE_DRAM_READ_CYCLES 0
// #define DONT_STRIDE_IN_ETH_BUFFER 1
#define DONT_STRIDE_IN_ETH_BUFFER 0

template <bool src_is_dram>
void read_chunk(
    uint32_t eth_l1_buffer_address_base,
    uint32_t num_pages,
    uint32_t num_pages_per_l1_buffer,
    uint32_t page_size,
    uint32_t &page_index,
    const InterleavedAddrGen<src_is_dram> &source_address_generator
)
{
    uint32_t local_eth_l1_curr_src_addr = eth_l1_buffer_address_base;
    uint32_t end_page_index = std::min(page_index + num_pages_per_l1_buffer, num_pages);
    for (; page_index < end_page_index; ++page_index) {
        // read source address
        uint64_t src_noc_addr = get_noc_addr(page_index, source_address_generator);
        noc_async_read(src_noc_addr, local_eth_l1_curr_src_addr, page_size);
        // read dest addr
        #if DONT_STRIDE_IN_ETH_BUFFER == 0
        local_eth_l1_curr_src_addr += page_size;
        #endif
    }
}


template <uint8_t MAX_CONCURRENT_TRANSACTIONS, bool src_is_dram>
FORCE_INLINE  bool noc_read_data_sequence(
    std::array<uint32_t, MAX_CONCURRENT_TRANSACTIONS> &transaction_channel_sender_buffer_addresses,
    uint32_t num_bytes_per_send,
    erisc::datamover::QueueIndexPointer<uint8_t> &noc_reader_buffer_wrptr,
    erisc::datamover::QueueIndexPointer<uint8_t> &noc_reader_buffer_ackptr,
    const erisc::datamover::QueueIndexPointer<uint8_t> eth_sender_rdptr,
    const erisc::datamover::QueueIndexPointer<uint8_t> eth_sender_ackptr,
    const uint8_t noc_index,
    const InterleavedAddrGen<src_is_dram> &source_address_generator,
    const uint32_t page_size,
    const uint32_t num_pages_per_l1_buffer,
    const uint32_t num_pages,
    uint32_t &page_index
    ) {
    bool did_something = false;

    bool noc_read_is_in_progress =
        erisc::datamover::deprecated::sender_is_noc_read_in_progress(noc_reader_buffer_wrptr, noc_reader_buffer_ackptr);
    bool more_data_to_read = page_index < num_pages;
    if (!noc_read_is_in_progress && more_data_to_read) {
        // We can only If a noc read is in progress, we can't issue another noc read
        bool next_buffer_available = !erisc::datamover::deprecated::sender_buffer_pool_full(
            noc_reader_buffer_wrptr, noc_reader_buffer_ackptr, eth_sender_rdptr, eth_sender_ackptr);
        if (next_buffer_available) {

            // Queue up another read
            // non blocking - issues noc_async_read
            // issue_read_chunk(noc_reader_buffer_wrptr, ...);
            #if EMULATE_DRAM_READ_CYCLES == 1
            issue_read_chunk();
            #else

            // DPRINT << "tx: reading data into L1 buffer on channel " << (uint32_t)noc_reader_buffer_wrptr << "\n";
            read_chunk<src_is_dram>(
                transaction_channel_sender_buffer_addresses[noc_reader_buffer_wrptr.index()], // eth_l1_buffer_address_base
                num_pages,
                num_pages_per_l1_buffer,
                page_size,
                page_index,
                source_address_generator
            );
            #endif
            noc_reader_buffer_wrptr.increment();

            did_something = true;
        }
    }

    return did_something;
}


void kernel_main() {
    // COMPILE TIME ARGS
    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);
    constexpr std::uint32_t total_num_message_sends = get_compile_time_arg_val(2);
    constexpr std::uint32_t NUM_TRANSACTION_BUFFERS = get_compile_time_arg_val(3);
    constexpr bool src_is_dram = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t MAX_NUM_CHANNELS = NUM_TRANSACTION_BUFFERS;

    // COMPILE TIME ARG VALIDATION
    static_assert(MAX_NUM_CHANNELS > 1, "Implementation currently doesn't support single buffering");

    // RUNTIME ARGS
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);

    std::uint32_t src_addr = get_arg_val<uint32_t>(2);
    std::uint32_t page_size = get_arg_val<uint32_t>(3);
    std::uint32_t num_pages = get_arg_val<uint32_t>(4);

    erisc::datamover::QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr(MAX_NUM_CHANNELS);
    erisc::datamover::QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr(MAX_NUM_CHANNELS);
    erisc::datamover::QueueIndexPointer<uint8_t> eth_sender_rdptr(MAX_NUM_CHANNELS);
    erisc::datamover::QueueIndexPointer<uint8_t> eth_sender_ackptr(MAX_NUM_CHANNELS);

    // Handshake with the other erisc first so we don't include dispatch time
    // in our measurements
    erisc::datamover::eth_setup_handshake(local_eth_l1_src_addr, true);

    // const InterleavedAddrGenFast<src_is_dram> s = {
    //     .bank_base_address = src_addr, .page_size = page_size, .data_format = df};

    const InterleavedAddrGen<src_is_dram> source_address_generator = {
        .bank_base_address = src_addr, .page_size = page_size};

    // SETUP DATASTRUCTURES
    std::array<uint32_t, MAX_NUM_CHANNELS> transaction_channel_sender_buffer_addresses;
    std::array<uint32_t, MAX_NUM_CHANNELS> transaction_channel_receiver_buffer_addresses;\
    erisc::datamover::initialize_transaction_buffer_addresses<MAX_NUM_CHANNELS>(
        MAX_NUM_CHANNELS,
        local_eth_l1_src_addr,
        num_bytes_per_send,
        transaction_channel_sender_buffer_addresses);
    erisc::datamover::initialize_transaction_buffer_addresses<MAX_NUM_CHANNELS>(
        MAX_NUM_CHANNELS,
        remote_eth_l1_dst_addr,
        num_bytes_per_send,
        transaction_channel_receiver_buffer_addresses);

    uint32_t eth_sends_completed = 0;

    constexpr uint32_t SWITCH_INTERVAL = 100000;
    uint32_t count = 0;
    uint32_t page_index = 0;
    uint32_t num_pages_per_l1_buffer = num_bytes_per_send / page_size;
    uint32_t num_context_switches = 0;
    uint32_t max_num_context_switches = 10000;
    bool printed_hang = false;
    uint32_t total_eth_sends = 0;
    while (eth_sends_completed < total_num_message_sends) {
        bool did_something = false;

        did_something = erisc::datamover::deprecated::sender_noc_receive_payload_ack_check_sequence(
                            noc_reader_buffer_wrptr,
                            noc_reader_buffer_ackptr,
                            noc_index) || did_something;

        did_something = noc_read_data_sequence<MAX_NUM_CHANNELS, src_is_dram>(
                            transaction_channel_sender_buffer_addresses,
                            num_bytes_per_send,
                            noc_reader_buffer_wrptr,
                            noc_reader_buffer_ackptr,
                            eth_sender_rdptr,
                            eth_sender_ackptr,
                            noc_index,
                            source_address_generator,
                            page_size,
                            num_pages_per_l1_buffer,
                            num_pages,
                            page_index) || did_something;

        bool sent_eth_data = erisc::datamover::deprecated::sender_eth_send_data_sequence<MAX_NUM_CHANNELS>(
                            transaction_channel_sender_buffer_addresses,
                            transaction_channel_receiver_buffer_addresses,
                            local_eth_l1_src_addr,
                            remote_eth_l1_dst_addr,
                            num_bytes_per_send,  // bytes to send from this buffer over eth link
                            num_bytes_per_send,  // break the end-to-end send into messages of this size
                            num_bytes_per_send_word_size,
                            noc_reader_buffer_wrptr,
                            noc_reader_buffer_ackptr,
                            eth_sender_rdptr,
                            eth_sender_ackptr);
        total_eth_sends = sent_eth_data ? total_eth_sends + 1 : total_eth_sends;
        did_something = sent_eth_data || did_something;


        did_something = erisc::datamover::deprecated::sender_eth_check_receiver_ack_sequence(
                            noc_reader_buffer_wrptr,
                            noc_reader_buffer_ackptr,
                            eth_sender_rdptr,
                            eth_sender_ackptr,
                            eth_sends_completed) ||
                        did_something;

        if (!did_something) {
            if (count++ > SWITCH_INTERVAL) {
                count = 0;
                run_routing();
                num_context_switches++;
                if (num_context_switches > max_num_context_switches) {
                    if (!printed_hang) {
                        DPRINT << "tx: HANG\n";
                        DPRINT << "tx: HANG eth_sends_completed " << eth_sends_completed << "\n";
                        DPRINT << "tx: HANG noc_reader_buffer_wrptr " << (uint32_t)noc_reader_buffer_ackptr.index() << "\n";
                        DPRINT << "tx: HANG (raw) noc_reader_buffer_wrptr " << (uint32_t)noc_reader_buffer_ackptr.raw_index() << "\n";
                        DPRINT << "tx: HANG noc_reader_buffer_ackptr " << (uint32_t)noc_reader_buffer_wrptr.index() << "\n";
                        DPRINT << "tx: HANG (raw) noc_reader_buffer_ackptr " << (uint32_t)noc_reader_buffer_wrptr.raw_index() << "\n";
                        DPRINT << "tx: HANG eth_sender_rdptr " << (uint32_t)eth_sender_rdptr.index() << "\n";
                        DPRINT << "tx: HANG (raw) eth_sender_rdptr " << (uint32_t)eth_sender_rdptr.raw_index() << "\n";
                        DPRINT << "tx: HANG eth_sender_ackptr " << (uint32_t)eth_sender_ackptr.index() << "\n";
                        DPRINT << "tx: HANG (raw) eth_sender_ackptr " << (uint32_t)eth_sender_ackptr.raw_index() << "\n";
                        DPRINT << "tx: HANG total_eth_sends " << (uint32_t)total_eth_sends << "\n";
                        for (uint32_t i = 0; i < MAX_NUM_CHANNELS; i++) {
                            DPRINT << "tx: HANG channel [" << i << "] bytes_sent " << erisc_info->channels[0].bytes_sent << "\n";
                            DPRINT << "tx: HANG channel [" << i << "] bytes_receiver_ack " << erisc_info->channels[0].receiver_ack << "\n";
                            DPRINT << "tx: HANG eth_is_receiver_channel_send_acked (" << i << ") " << (eth_is_receiver_channel_send_acked(i) ? "true" : "false") << "\n";
                            DPRINT << "tx: HANG eth_is_receiver_channel_send_done(" << i << ") " << (eth_is_receiver_channel_send_done(i) ? "true" : "false") << "\n";
                        }
                        // bool noc_read_is_in_progress =
                        //     is_noc_read_in_progress(noc_reader_buffer_wrptr, noc_reader_buffer_ackptr);
                        // bool more_data_to_read = page_index < num_pages;
                        // bool next_buffer_available = !buffer_pool_full<MAX_NUM_CHANNELS>(
                        //     noc_reader_buffer_wrptr, noc_reader_buffer_ackptr, eth_sender_rdptr, eth_sender_ackptr);
                        // DPRINT << "tx: HANG noc_read_is_in_progress " << (noc_read_is_in_progress ? "true" : "false") << "\n";
                        // DPRINT << "tx: HANG more_data_to_read " << (more_data_to_read ? "true" : "false") << "\n";
                        // DPRINT << "tx: HANG next_buffer_available " << (next_buffer_available ? "true" : "false") << "\n";
                        num_context_switches = 0;
                        printed_hang = true;
                    }
                }
            } else {
                count++;
            }
        } else {
            num_context_switches = 0;
        }
    }


    DPRINT << "tx: DONE\n";
    DPRINT << "tx: DONE eth_sends_completed " << (uint32_t)eth_sends_completed << "\n";
    DPRINT << "tx: DONE total_num_message_sends " << (uint32_t)total_num_message_sends << "\n";
}
