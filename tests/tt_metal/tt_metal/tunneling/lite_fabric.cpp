// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "tt_metal/api/tt-metalium/hal_types.hpp"
#include "blackhole/dev_mem_map.h"
#include "blackhole/noc_nonblocking_api.h"
#include "dataflow_api.h"
#include "eth_chan_noc_mapping.h"
#include "lite_fabric.hpp"
#include "firmware_common.h"
#include "init-fsm-basic.h"
#include "lite_fabric_constants.hpp"
#include "lite_fabric_channels.hpp"
#include "lite_fabric_channel_util.hpp"
#include "lite_fabric_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

#define BEGIN_MAIN_FUNCTION()                                                           \
    IF_NOT_METAL_LAUNCH(int main())                                                     \
    IF_METAL_LAUNCH(void kernel_main()) {                                               \
        IF_NOT_METAL_LAUNCH(configure_csr();)                                           \
        IF_NOT_METAL_LAUNCH(noc_index = NOC_INDEX;)                                     \
        IF_NOT_METAL_LAUNCH(do_crt1((uint32_t*)MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH);) \
        IF_NOT_METAL_LAUNCH(noc_bank_table_init(MEM_AERISC_BANK_TO_NOC_SCRATCH);)       \
        IF_NOT_METAL_LAUNCH(risc_init();)                                               \
        IF_NOT_METAL_LAUNCH(noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);)                     \
        IF_NOT_METAL_LAUNCH(for (uint32_t n = 0; n < NUM_NOCS; n++) { noc_local_state_init(n); })

// End the main function
#define END_MAIN_FUNCTION()        \
    IF_NOT_METAL_LAUNCH(return 0;) \
    IF_METAL_LAUNCH(return;)       \
    }

#if !defined(tt_l1_ptr)
#define tt_l1_ptr __attribute__((rvtt_l1_ptr))
#endif

// Define METAL_LAUNCH if lite_fabric is being launched by Metal
#if !defined(METAL_LAUNCH)
#define IF_NOT_METAL_LAUNCH(x) x
#define IF_METAL_LAUNCH(x)

uint8_t noc_index __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[tt::tt_metal::NumHalProgrammableCoreTypes] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

// Not initialized anywhere yet
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

#else

#include "debug/dprint.h"

#define IF_NOT_METAL_LAUNCH(x)
#define IF_METAL_LAUNCH(x) x
#endif

FORCE_INLINE void send_next_data(
    tt::tt_fabric::EthChannelBuffer<lite_fabric::LiteFabricHeader, SENDER_NUM_BUFFERS_ARRAY[0]>& sender_buffer_channel,
    volatile lite_fabric::HostToLiteFabricInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE>& host_interface,
    OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS_ARRAY[0]>& outbound_to_receiver_channel_pointers,
    tt::tt_fabric::EthChannelBuffer<lite_fabric::LiteFabricHeader, RECEIVER_NUM_BUFFERS_ARRAY[0]>&
        receiver_buffer_channel,
    bool on_mmio_chip) {
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
    DPRINT << "S: Forward Buffer 0x" << HEX() << src_addr << " " << DEC() << (uint32_t)payload_size_bytes << "B to 0x"
           << HEX() << dest_addr << DEC() << ENDL();
    pkt_header->src_ch_id = 0;

    while (internal_::eth_txq_is_busy(sender_txq_id));
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    host_interface.d2h.fabric_sender_channel_index =
        tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[0]>(host_interface.d2h.fabric_sender_channel_index);

    remote_receiver_buffer_index = tt::tt_fabric::BufferIndex{
        tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[0]>(remote_receiver_buffer_index.get())};
    receiver_buffer_channel.set_cached_next_buffer_slot_addr(
        receiver_buffer_channel.get_buffer_address(remote_receiver_buffer_index));
    sender_buffer_channel.set_cached_next_buffer_slot_addr(sender_buffer_channel.get_buffer_address(
        tt::tt_fabric::BufferIndex{(uint8_t)host_interface.d2h.fabric_sender_channel_index}));
    remote_receiver_num_free_slots--;
    // update the remote reg
    static constexpr uint32_t packets_to_forward = 1;
    while (internal_::eth_txq_is_busy(sender_txq_id));
    remote_update_ptr_val<to_receiver_0_pkts_sent_id, sender_txq_id>(packets_to_forward);
}

FORCE_INLINE void run_sender_channel_step(
    tt::tt_fabric::EthChannelBuffer<lite_fabric::LiteFabricHeader, SENDER_NUM_BUFFERS_ARRAY[0]>& local_sender_channel,
    volatile lite_fabric::HostToLiteFabricInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE>& host_interface,
    OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS_ARRAY[0]>& outbound_to_receiver_channel_pointers,
    tt::tt_fabric::EthChannelBuffer<lite_fabric::LiteFabricHeader, RECEIVER_NUM_BUFFERS_ARRAY[0]>&
        remote_receiver_channel,
    bool on_mmio_chip) {
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    bool has_unsent_packet =
        host_interface.h2d.sender_host_write_index != host_interface.d2h.fabric_sender_channel_index;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    if (can_send) {
        DPRINT << "S: host write index: " << (uint32_t)host_interface.h2d.sender_host_write_index
               << ", fabric read index: " << (uint32_t)host_interface.d2h.fabric_sender_channel_index << ENDL();
        DPRINT << "S: Receiver has space for packet: " << (uint32_t)receiver_has_space_for_packet
               << ", has unsent packet: " << (uint32_t)has_unsent_packet << ", can send: " << (uint32_t)can_send
               << ENDL();
        send_next_data(
            local_sender_channel,
            host_interface,
            outbound_to_receiver_channel_pointers,
            remote_receiver_channel,
            on_mmio_chip);
    }

    // Process COMPLETIONs from receiver
    int32_t completions_since_last_check = get_ptr_val(to_sender_0_pkts_completed_id);
    if (completions_since_last_check) {
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        increment_local_update_ptr_val(to_sender_0_pkts_completed_id, -completions_since_last_check);
    }
}

__attribute__((optimize("jump-tables"))) FORCE_INLINE void service_fabric_request(
    tt_l1_ptr lite_fabric::LiteFabricHeader* const packet_start,
    uint16_t payload_size_bytes,
    uint32_t transaction_id,
    volatile lite_fabric::LiteFabricMemoryMap* lite_fabric,
    tt::tt_fabric::EthChannelBuffer<lite_fabric::LiteFabricHeader, SENDER_NUM_BUFFERS_ARRAY[0]>& sender_buffer_channel,
    bool on_mmio_chip) {
    invalidate_l1_cache();
    const auto& header = *packet_start;

    lite_fabric::NocSendType noc_send_type = header.noc_send_type;
    if (noc_send_type > lite_fabric::NocSendType::NOC_SEND_TYPE_LAST) {
        __builtin_unreachable();
    }
    switch (noc_send_type) {
        case lite_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) +
                                                   sizeof(lite_fabric::LiteFabricHeader) + header.unaligned_offset;

            const auto dest_address = header.command_fields.noc_unicast.noc_address;
            DPRINT << "R: NOC_UNICAST_WRITE Source Address: " << HEX() << payload_start_address
                   << " Destination Address: " << HEX() << dest_address << " Size: " << DEC()
                   << (uint32_t)payload_size_bytes << ENDL();

            noc_async_write_one_packet_with_trid<true, false>(
                payload_start_address,
                dest_address,
                payload_size_bytes,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
        } break;

        case lite_fabric::NocSendType::NOC_READ: {
            if (!on_mmio_chip) {
                auto& host_interface = lite_fabric->host_interface;
                const auto src_address = header.command_fields.noc_read.noc_address;
                // This assumes nobody else is using the sender channel on device 1 because
                // the tunnel depth is only 1 at the moment
                uint32_t dst_address = sender_buffer_channel.get_cached_next_buffer_slot_addr();
                uint32_t payload_dst_address =
                    dst_address + sizeof(lite_fabric::LiteFabricHeader) + header.unaligned_offset;

                DPRINT << "R: NOC_READ src_address: " << HEX() << src_address << " dst_address: " << dst_address
                       << DEC() << " event: " << header.command_fields.noc_read.event << ENDL();

                // Create packet header for writing back
                tt_l1_ptr lite_fabric::LiteFabricHeader* packet_header_in_sender_ch =
                    reinterpret_cast<lite_fabric::LiteFabricHeader*>(dst_address);
                *packet_header_in_sender_ch = header;
                // Read the data into the buffer
                // This is safe only if the data at the sender buffer slot has been flushed out
                // We rely on the host to not do a read until the received data has been read out
                DPRINT << "Read into buffer" << ENDL();
                noc_async_read(src_address, payload_dst_address, payload_size_bytes);
                noc_async_read_barrier();
                DPRINT << "Read into buffer done" << ENDL();

                // Tell ourselves there is data to send
                // NOTE: sender_buffer_channel index will be incremented in send_next_data
                host_interface.h2d.sender_host_write_index = tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS_ARRAY[0]>(
                    host_interface.h2d.sender_host_write_index);
            } else {
                DPRINT << "NOC_READ Event " << DEC() << header.command_fields.noc_read.event << " Address " << HEX()
                       << (uint32_t)packet_start << ENDL();
            }

        } break;

        default: {
            ASSERT(false);
        } break;
    };
}

// MUST CHECK !is_eth_txq_busy() before calling
FORCE_INLINE void receiver_send_completion_ack(uint8_t src_id) {
    while (internal_::eth_txq_is_busy(receiver_txq_id));
    remote_update_ptr_val<receiver_txq_id>(to_sender_0_pkts_completed_id, 1);
}

FORCE_INLINE void run_receiver_channel_step(
    tt::tt_fabric::EthChannelBuffer<lite_fabric::LiteFabricHeader, RECEIVER_NUM_BUFFERS_ARRAY[0]>&
        remote_receiver_channel,
    ReceiverChannelPointers<RECEIVER_NUM_BUFFERS_ARRAY[0]>& receiver_channel_pointers,
    WriteTransactionIdTracker<RECEIVER_NUM_BUFFERS_ARRAY[0], NUM_TRANSACTION_IDS, 0>& receiver_channel_trid_tracker,
    volatile lite_fabric::LiteFabricMemoryMap* lite_fabric,
    tt::tt_fabric::EthChannelBuffer<lite_fabric::LiteFabricHeader, SENDER_NUM_BUFFERS_ARRAY[0]>& local_sender_channel,
    bool on_mmio_chip) {
    auto pkts_received_since_last_check = get_ptr_val<to_receiver_0_pkts_sent_id>();
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    bool unwritten_packets = pkts_received_since_last_check != 0;
    volatile lite_fabric::HostToLiteFabricInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE>& host_interface =
        lite_fabric->host_interface;

    if (unwritten_packets) {
        invalidate_l1_cache();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr lite_fabric::LiteFabricHeader* packet_header = const_cast<lite_fabric::LiteFabricHeader*>(
            remote_receiver_channel.template get_packet_header<lite_fabric::LiteFabricHeader>(receiver_buffer_index));

        DPRINT << "R: rcvr buffer index " << (uint32_t)receiver_buffer_index << " from addr " << HEX()
               << (uint32_t)(remote_receiver_channel.get_buffer_address(receiver_buffer_index)) << DEC() << ENDL();

        receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);

        uint8_t trid = receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
            receiver_buffer_index);
        // lite fabric tunnel depth is 1 so any fabric cmds being sent here will be writes to/reads from this chip
        service_fabric_request(
            packet_header, packet_header->payload_size_bytes, trid, lite_fabric, local_sender_channel, on_mmio_chip);

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
    bool next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
    bool can_send_completion = unflushed_writes && next_trid_flushed;
    if (on_mmio_chip) {
        can_send_completion =
            can_send_completion && (((host_interface.d2h.fabric_receiver_channel_index + 1) %
                                     RECEIVER_NUM_BUFFERS_ARRAY[0]) != host_interface.h2d.receiver_host_read_index);
    }

    if (can_send_completion) {
        receiver_send_completion_ack(receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
        receiver_channel_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
        completion_counter.increment();
        if (on_mmio_chip) {
            host_interface.d2h.fabric_receiver_channel_index =
                tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[0]>(
                    host_interface.d2h.fabric_receiver_channel_index);
        }
    }
}

BEGIN_MAIN_FUNCTION() {
    invalidate_l1_cache();

    auto structs = reinterpret_cast<volatile lite_fabric::LiteFabricMemoryMap*>(MEM_AERISC_FABRIC_LITE_CONFIG);

    volatile lite_fabric::HostToLiteFabricInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE>& host_interface =
        structs->host_interface;

    const uint32_t lf_local_sender_0_channel_address = (uintptr_t)&structs->sender_channel_buffer;
    const uint32_t lf_local_sender_channel_0_connection_info_addr = (uintptr_t)&structs->sender_location_info;
    const uint32_t lf_remote_receiver_0_channel_buffer_address = (uintptr_t)&structs->receiver_channel_buffer;
    const uint32_t lf_local_sender_channel_0_connection_semaphore_addr =
        (uintptr_t)&structs->sender_connection_live_semaphore;
    auto lf_sender0_worker_semaphore_ptr =
        reinterpret_cast<volatile uint32_t*>((uintptr_t)&structs->sender_flow_control_semaphore);

    // One send buffer and one receiver buffer
    init_ptr_val<to_receiver_0_pkts_sent_id>(0);
    init_ptr_val<to_sender_0_pkts_acked_id>(0);
    init_ptr_val<to_sender_0_pkts_completed_id>(0);

    auto remote_receiver_channels =
        tt::tt_fabric::EthChannelBuffers<lite_fabric::LiteFabricHeader, RECEIVER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_RECEIVER_CHANNELS>{});

    auto local_sender_channels =
        tt::tt_fabric::EthChannelBuffers<lite_fabric::LiteFabricHeader, SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});

    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& local_sender_buffer_addresses = {
        lf_local_sender_0_channel_address};
    const std::array<size_t, NUM_RECEIVER_CHANNELS>& remote_receiver_buffer_addresses = {
        lf_remote_receiver_0_channel_buffer_address};

    // use same addr space for host to lite fabric edm connection
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        lf_local_sender_channel_0_connection_info_addr};

    // initialize the remote receiver channel buffers
    remote_receiver_channels.init(
        remote_receiver_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(lite_fabric::LiteFabricHeader),
        RECEIVER_CHANNEL_BASE_ID);
    lite_fabric::init_receiver_headers(remote_receiver_channels);

    // initialize the local sender channel worker interfaces
    local_sender_channels.init(
        local_sender_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(lite_fabric::LiteFabricHeader),
        SENDER_CHANNEL_BASE_ID);

    WriteTransactionIdTracker<RECEIVER_NUM_BUFFERS_ARRAY[0], NUM_TRANSACTION_IDS, 0> receiver_channel_0_trid_tracker;

    auto outbound_to_receiver_channel_pointers =
        ChannelPointersTuple<OutboundReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make();
    auto outbound_to_receiver_channel_pointer_ch0 = outbound_to_receiver_channel_pointers.template get<0>();

    auto receiver_channel_pointers = ChannelPointersTuple<ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make();
    auto receiver_channel_pointers_ch0 = receiver_channel_pointers.template get<0>();
    receiver_channel_pointers_ch0.reset();

    // Must initialize to match what's on the host
    structs->host_interface.init();

    bool on_mmio_chip = structs->config.is_mmio;
    DPRINT << "Routing Enabled " << structs->config.routing_enabled
           << " Init on MMIO = " << (uint32_t)structs->config.is_mmio << " Host IF at 0x" << HEX()
           << (uint32_t)&host_interface << DEC() << ENDL();

    lite_fabric::routing_init(&structs->config);

    while (structs->config.routing_enabled) {
        invalidate_l1_cache();

        run_sender_channel_step(
            local_sender_channels.template get<0>(),
            host_interface,
            outbound_to_receiver_channel_pointer_ch0,
            remote_receiver_channels.template get<0>(),
            on_mmio_chip);

        invalidate_l1_cache();
        run_receiver_channel_step(
            remote_receiver_channels.template get<0>(),
            receiver_channel_pointers_ch0,
            receiver_channel_0_trid_tracker,
            structs,
            local_sender_channels.template get<0>(),
            on_mmio_chip);
    }

    receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();

    ncrisc_noc_counters_init();

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    DPRINT << "lite_fabric: out " << (uint32_t)structs->config.routing_enabled << ENDL();

#if !defined(METAL_LAUNCH)
    lite_fabric::ConnectedRisc1Interface::assert_connected_dm1_reset();
#endif

    structs->config.current_state = lite_fabric::InitState::TERMINATED;
}
END_MAIN_FUNCTION()
