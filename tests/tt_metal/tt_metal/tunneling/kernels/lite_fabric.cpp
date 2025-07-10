// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/pause.h"
#include "eth_chan_noc_mapping.h"
#include "lite_fabric.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

// taken from fabric_erisc_datamover.cpp ... commonize!
// Forward‐declare the Impl primary template:
template <template <uint8_t> class ChannelType, auto& BufferSizes, typename Seq>
struct ChannelPointersTupleImpl;

// Provide the specialization that actually holds the tuple and `get<>`:
template <template <uint8_t> class ChannelType, auto& BufferSizes, size_t... Is>
struct ChannelPointersTupleImpl<ChannelType, BufferSizes, std::index_sequence<Is...>> {
    std::tuple<ChannelType<BufferSizes[Is]>...> channel_ptrs;

    template <size_t I>
    constexpr auto& get() {
        return std::get<I>(channel_ptrs);
    }
};

// Simplify the “builder” so that make() returns the Impl<…> directly:
template <template <uint8_t> class ChannelType, auto& BufferSizes>
struct ChannelPointersTuple {
    static constexpr size_t N = std::size(BufferSizes);

    static constexpr auto make() {
        return ChannelPointersTupleImpl<ChannelType, BufferSizes, std::make_index_sequence<N>>{};
    }
};

static constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_free_slots_stream_ids = {
    tunneling::sender_channel_0_free_slots_stream_id};

/*
 * Tracks receiver channel pointers (from sender side)
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct OutboundReceiverChannelPointers {
    uint32_t num_free_slots = RECEIVER_NUM_BUFFERS;
    tt::tt_fabric::BufferIndex remote_receiver_buffer_index{0};
    size_t cached_next_buffer_slot_addr = 0;

    FORCE_INLINE bool has_space_for_packet() const { return num_free_slots; }
};

/*
 * Tracks receiver channel pointers (from receiver side). Must call reset() before using.
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct ReceiverChannelPointers {
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> wr_sent_counter;
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> wr_flush_counter;
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> ack_counter;
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> completion_counter;
    std::array<uint8_t, RECEIVER_NUM_BUFFERS> src_chan_ids;

    FORCE_INLINE void set_src_chan_id(tt::tt_fabric::BufferIndex buffer_index, uint8_t src_chan_id) {
        src_chan_ids[buffer_index.get()] = src_chan_id;
    }

    FORCE_INLINE uint8_t get_src_chan_id(tt::tt_fabric::BufferIndex buffer_index) const {
        return src_chan_ids[buffer_index.get()];
    }

    FORCE_INLINE void reset() {
        wr_sent_counter.reset();
        wr_flush_counter.reset();
        ack_counter.reset();
        completion_counter.reset();
    }
};

FORCE_INLINE void send_next_data(
    tt::tt_fabric::EthChannelBuffer<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>& sender_buffer_channel,
    tt::tt_fabric::EdmChannelWorkerInterface<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>& sender_worker_interface,
    OutboundReceiverChannelPointers<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& outbound_to_receiver_channel_pointers,
    tt::tt_fabric::EthChannelBuffer<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& receiver_buffer_channel) {
    auto& remote_receiver_buffer_index = outbound_to_receiver_channel_pointers.remote_receiver_buffer_index;
    auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
    auto& local_sender_write_counter = sender_worker_interface.local_write_counter;
    constexpr uint32_t sender_txq_id = 0;
    uint32_t src_addr = sender_buffer_channel.get_cached_next_buffer_slot_addr();

    volatile auto* pkt_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(src_addr);
    size_t payload_size_bytes = pkt_header->get_payload_size_including_header();
    auto dest_addr = receiver_buffer_channel.get_cached_next_buffer_slot_addr();
    pkt_header->src_ch_id = 0;

    while (internal_::eth_txq_is_busy(sender_txq_id));
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    // local_sender_write_counter.index =
    //     tt::tt_fabric::BufferIndex{tt::tt_fabric::wrap_increment<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>(local_sender_write_counter.index.get())};

    remote_receiver_buffer_index = tt::tt_fabric::BufferIndex{
        tt::tt_fabric::wrap_increment<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>(remote_receiver_buffer_index.get())};
    receiver_buffer_channel.set_cached_next_buffer_slot_addr(
        receiver_buffer_channel.get_buffer_address(remote_receiver_buffer_index));
    // sender_buffer_channel.set_cached_next_buffer_slot_addr(
    //     sender_buffer_channel.get_buffer_address(local_sender_write_counter.get_buffer_index()));
    remote_receiver_num_free_slots--;
    // update the remote reg
    static constexpr uint32_t packets_to_forward = 1;
    while (internal_::eth_txq_is_busy(sender_txq_id));
    remote_update_ptr_val<tunneling::to_receiver_0_pkts_sent_id, sender_txq_id>(packets_to_forward);
}

FORCE_INLINE void run_sender_channel_step(
    tt::tt_fabric::EthChannelBuffer<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>& local_sender_channel,
    tt::tt_fabric::EdmChannelWorkerInterface<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>&
        local_sender_channel_worker_interface,
    OutboundReceiverChannelPointers<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& outbound_to_receiver_channel_pointers,
    tt::tt_fabric::EthChannelBuffer<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& remote_receiver_channel,
    uint32_t sender_channel_free_slots_stream_id) {
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    uint32_t free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    bool has_unsent_packet = free_slots != tunneling::SENDER_NUM_BUFFERS_ARRAY[0];
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    if (can_send) {
        send_next_data(
            local_sender_channel,
            local_sender_channel_worker_interface,
            outbound_to_receiver_channel_pointers,
            remote_receiver_channel);
        increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
    }

    // Process COMPLETIONs from receiver
    int32_t completions_since_last_check = get_ptr_val(to_sender_0_pkts_completed_id);
    if (completions_since_last_check) {
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        increment_local_update_ptr_val(to_sender_0_pkts_completed_id, -completions_since_last_check);
        // local_sender_channel_worker_interface
        //     .template update_persistent_connection_copy_of_free_slots<0>(
        //         completions_since_last_check);
    }
}

// MUST CHECK !is_eth_txq_busy() before calling
FORCE_INLINE void receiver_send_completion_ack(uint8_t src_id) {
    while (internal_::eth_txq_is_busy(receiver_txq_id));
    remote_update_ptr_val<receiver_txq_id>(to_sender_0_pkts_completed_id, 1);
}

FORCE_INLINE void run_receiver_channel_step(
    tt::tt_fabric::EthChannelBuffer<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& local_receiver_channel,
    ReceiverChannelPointers<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& receiver_channel_pointers,
    WriteTransactionIdTracker<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0], tunneling::NUM_TRANSACTION_IDS, 0>&
        receiver_channel_trid_tracker) {
    auto pkts_received_since_last_check = get_ptr_val<to_receiver_0_pkts_sent_id>();
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    bool unwritten_packets = pkts_received_since_last_check != 0;

    if (unwritten_packets) {
        invalidate_l1_cache();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        // ROUTING_FIELDS_TYPE cached_routing_fields = packet_header->routing_fields;

        receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);

        uint8_t trid = receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
            receiver_buffer_index);
        // lite fabric tunnel depth is 1 so any fabric cmds being sent here will be writes to/reads from this chip
        execute_chip_unicast_to_local_chip(packet_header, packet_header->payload_size_bytes, trid, 0);

        wr_sent_counter.increment();
        // decrement the to_receiver_0_pkts_sent_id stream register by 1 since current packet has been processed.
        increment_local_update_ptr_val<tunneling::to_receiver_0_pkts_sent_id>(-1);
    }

    // flush and completion are fused, so we only need to update one of the counters
    // update completion since other parts of the code check against completion
    auto& completion_counter = receiver_channel_pointers.completion_counter;
    // Currently unclear if it's better to loop here or not...
    bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
    auto receiver_buffer_index = completion_counter.get_buffer_index();
    bool next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
    bool can_send_completion = unflushed_writes && next_trid_flushed;
    // if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
    //     can_send_completion = can_send_completion && !internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ);
    // }
    if (can_send_completion) {
        receiver_send_completion_ack(receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
        receiver_channel_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
        completion_counter.increment();
    }
}

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t lite_fabric_config_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t lf_local_sender_0_channel_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t lf_local_sender_channel_0_connection_info_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t lf_local_receiver_0_channel_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t lf_remote_receiver_0_channel_buffer_address = get_arg_val<uint32_t>(arg_idx++);

    const size_t lf_local_sender_channel_0_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    auto lf_sender0_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));

    constexpr uint32_t channel_buffer_size = 4096 + sizeof(PACKET_HEADER_TYPE);
    static_assert(channel_buffer_size == 4128, "Expected channel buffer size to be 4128B");

    volatile tunneling::lite_fabric_config_t* lite_fabric_config =
        reinterpret_cast<volatile tunneling::lite_fabric_config_t*>(lite_fabric_config_addr);

    // One send buffer and one receiver buffer
    init_ptr_val<tunneling::to_receiver_0_pkts_sent_id>(0);
    init_ptr_val<tunneling::to_sender_0_pkts_acked_id>(0);
    init_ptr_val<tunneling::to_sender_0_pkts_completed_id>(0);

    // host can read stream register because it needs to check this...?
    init_ptr_val<tunneling::sender_channel_0_free_slots_stream_id>(tunneling::SENDER_NUM_BUFFERS_ARRAY[0]);

    //  initialize the statically allocated "semaphores"
    // *reinterpret_cast<volatile uint32_t*>(lf_local_sender_channel_0_connection_semaphore_addr) = 0;
    // *lf_sender0_worker_semaphore_ptr = 0;

    std::array<uint32_t, tunneling::NUM_SENDER_CHANNELS> local_sender_channel_free_slots_stream_ids_ordered;

    auto remote_receiver_channels = tt::tt_fabric::EthChannelBuffers<tunneling::RECEIVER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<tunneling::NUM_RECEIVER_CHANNELS>{});

    auto local_receiver_channels = tt::tt_fabric::EthChannelBuffers<tunneling::RECEIVER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<tunneling::NUM_RECEIVER_CHANNELS>{});

    auto local_sender_channels = tt::tt_fabric::EthChannelBuffers<tunneling::SENDER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<tunneling::NUM_SENDER_CHANNELS>{});

    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& local_sender_buffer_addresses = {
        lf_local_sender_0_channel_address};
    const std::array<size_t, tunneling::NUM_RECEIVER_CHANNELS>& local_receiver_buffer_addresses = {
        lf_local_receiver_0_channel_buffer_address};
    const std::array<size_t, tunneling::NUM_RECEIVER_CHANNELS>& remote_receiver_buffer_addresses = {
        lf_remote_receiver_0_channel_buffer_address};

    std::array<size_t, tunneling::NUM_SENDER_CHANNELS> local_sender_flow_control_semaphores = {
        reinterpret_cast<size_t>(lf_sender0_worker_semaphore_ptr)};
    std::array<size_t, tunneling::NUM_SENDER_CHANNELS> local_sender_connection_live_semaphore_addresses = {
        lf_local_sender_channel_0_connection_semaphore_addr};

    // use same addr space for host to lite fabric edm connection
    std::array<size_t, tunneling::NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        lf_local_sender_channel_0_connection_info_addr};

    for (size_t i = 0; i < tunneling::NUM_SENDER_CHANNELS; i++) {
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
            local_sender_connection_info_addresses[i]);
        connection_worker_info_ptr->edm_read_counter = 0;
    }

    // create the sender channnel worker interfaces with input array of number of buffers
    auto local_sender_channel_worker_interfaces =
        tt::tt_fabric::EdmChannelWorkerInterfaces<tunneling::SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<tunneling::NUM_SENDER_CHANNELS>{});

    for (size_t i = 0; i < tunneling::NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_free_slots_stream_ids_ordered[i] = sender_channel_free_slots_stream_ids[i];
    }

    // initialize the remote receiver channel buffers
    remote_receiver_channels.init(
        remote_receiver_buffer_addresses.data(),
        channel_buffer_size,
        sizeof(PACKET_HEADER_TYPE),
        receiver_channel_base_id);

    // initialize the local receiver channel buffers
    local_receiver_channels.init(
        local_receiver_buffer_addresses.data(),
        channel_buffer_size,
        sizeof(PACKET_HEADER_TYPE),
        tunneling::receiver_channel_base_id);

    // initialize the local sender channel worker interfaces
    local_sender_channels.init(
        local_sender_buffer_addresses.data(),
        channel_buffer_size,
        sizeof(PACKET_HEADER_TYPE),
        tunneling::sender_channel_base_id);

    // initialize the local sender channel worker interfaces
    // init_local_sender_channel_worker_interfaces(
    //     local_sender_connection_live_semaphore_addresses,
    //     local_sender_connection_info_addresses,
    //     local_sender_channel_worker_interfaces,
    //     local_sender_flow_control_semaphores);

    WriteTransactionIdTracker<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0], tunneling::NUM_TRANSACTION_IDS, 0>
        receiver_channel_0_trid_tracker;

    auto outbound_to_receiver_channel_pointers =
        ChannelPointersTuple<OutboundReceiverChannelPointers, tunneling::RECEIVER_NUM_BUFFERS_ARRAY>::make();
    auto outbound_to_receiver_channel_pointer_ch0 = outbound_to_receiver_channel_pointers.template get<0>();

    auto receiver_channel_pointers =
        ChannelPointersTuple<ReceiverChannelPointers, tunneling::RECEIVER_NUM_BUFFERS_ARRAY>::make();
    auto receiver_channel_pointers_ch0 = receiver_channel_pointers.template get<0>();
    receiver_channel_pointers_ch0.reset();

    // ------------------------ Done all the initializations ------------------------

    // ------------------------ Do local and neighbour handshake ------------------------
    tunneling::do_init_and_handshake_sequence(lite_fabric_config_addr);
    // ------------------------ Done local and neighbour handshake ------------------------

    // ------------------------ Main loop ------------------------
    while (lite_fabric_config->termination_signal == 0) {
        invalidate_l1_cache();

        run_sender_channel_step(
            local_sender_channels.template get<0>(),
            local_sender_channel_worker_interfaces.template get<0>(),
            outbound_to_receiver_channel_pointer_ch0,
            remote_receiver_channels.template get<0>(),
            local_sender_channel_free_slots_stream_ids_ordered[0]);

        run_receiver_channel_step(
            local_receiver_channels.template get<0>(), receiver_channel_pointers_ch0, receiver_channel_0_trid_tracker);
    }

    DPRINT << "Got the termination signal "
           << get_stream_reg_read_addr<tunneling::sender_channel_0_free_slots_stream_id>() << ENDL();

    // each mmio eth will get termination signal from host and then sends it to neighbour. neighbour will also send it
    // back... send termination signal to the neighbour eth core ... just send the entire config and have the remote eth
    // core check termination signal
    internal_::eth_send_packet<false>(
        0, lite_fabric_config_addr >> 4, lite_fabric_config_addr >> 4, sizeof(tunneling::lite_fabric_config_t) >> 4);

    // *reinterpret_cast<volatile uint32_t*>(lf_local_sender_channel_0_connection_semaphore_addr) = 99;
    // *lf_sender0_worker_semaphore_ptr = 99;

    // update noc apis to not use trids
    receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();

    // re-init the noc counters as the noc api used is not incrementing them
    ncrisc_noc_counters_init();

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
