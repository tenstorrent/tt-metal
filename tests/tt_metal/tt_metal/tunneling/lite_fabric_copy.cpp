// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "risc_common.h"
#include "noc.h"
#include "noc_overlay_parameters.h"
#include "ckernel_structs.h"
#include "stream_io_map.h"
#include "c_tensix_core.h"
#include "tdma_xmov.h"
#include "firmware_common.h"
#include "dev_msgs.h"
#include "risc_attribs.h"
#include "dataflow_api.h"
#include "ethernet/dataflow_api.h"
#include "ethernet/tunneling.h"
#include "blackhole/dev_mem_map.h"
#include "blackhole/noc_nonblocking_api.h"
#include "tt_metal/hw/inc/blackhole/core_config.h"
#include "lite_fabric_constants.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "tests/tt_metal/tt_metal/tunneling/fabric_lite.hpp"
#include "debug/dprint.h"
#include "init-fsm-basic.h"

#if !defined(tt_l1_ptr)
#define tt_l1_ptr __attribute__((rvtt_l1_ptr))
#endif

#if !defined(NUM_NOCS)
#define NUM_NOCS 2
#endif

#if !defined(NOC_INDEX)
#define NOC_INDEX 0
#endif

// Define METAL_LAUNCH if lite_fabric is being launched by Metal
#if !defined(METAL_LAUNCH)
#define IF_NOT_METAL_LAUNCH(x) x
#define IF_METAL_LAUNCH(x)

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* crta_l1_base __attribute__((used));
uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

// Not initialized anywhere yet
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

#else
#define IF_NOT_METAL_LAUNCH(x)
#define IF_METAL_LAUNCH(x) x
#endif

bool did_something = false;

// Define a main function for lite_fabric being compiled as firmware or kernel
#define BEGIN_MAIN_FUNCTION()                                                           \
    IF_NOT_METAL_LAUNCH(int main())                                                     \
    IF_METAL_LAUNCH(void kernel_main()) {                                               \
        IF_NOT_METAL_LAUNCH(configure_csr();)                                           \
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

constexpr uint32_t SOFT_RESET_ADDR = 0xFFB121B0;

// Interface to the connected RISC1 processor via ethernet
struct ConnectedRisc1Interface {
    static constexpr uint32_t k_SoftResetAddr = 0xFFB121B0;
    static constexpr uint32_t k_ResetPcAddr = SUBORDINATE_AERISC_RESET_PC;

    template <uint32_t TXQ = 0>
    inline void send_packet(uint32_t src_addr, uint32_t dst_addr, uint32_t n) const {
        do {
            invalidate_l1_cache();
        } while (internal_::eth_txq_is_busy(TXQ));
        internal_::eth_send_packet(TXQ, src_addr >> 4, dst_addr >> 4, n);
    }

    // Put the connected RISC1 into reset
    inline void assert_connected_dm1_reset() const {
        constexpr uint32_t k_ResetValue = 0x47000;
        internal_::eth_write_remote_reg(0, k_SoftResetAddr, k_ResetValue);
    }

    // Take the connected RISC1 out of reset
    inline void deassert_connected_dm1_reset() const {
        constexpr uint32_t k_ResetValue = 0;
        internal_::eth_write_remote_reg(0, k_SoftResetAddr, k_ResetValue);
    }

    inline void set_pc(uint32_t pc) const { internal_::eth_write_remote_reg(0, k_ResetPcAddr, pc); }
};

constexpr uint32_t local_sender_0_channel_address =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, sender_channel_buffer);
static_assert(local_sender_0_channel_address % 16 == 0);

constexpr uint32_t remote_receiver_0_channel_address =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, receiver_channel_buffer);
static_assert(remote_receiver_0_channel_address % 16 == 0);

constexpr uint32_t local_receiver_0_channel_address = remote_receiver_0_channel_address;

constexpr uint32_t local_sender_channel_0_connection_info_addr =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, sender_location_info);
static_assert(local_sender_channel_0_connection_info_addr % 16 == 0);

constexpr uint32_t sender_flow_control_semaphore_address =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, sender_flow_control_semaphore);
static_assert(sender_flow_control_semaphore_address % 16 == 0);

constexpr uint32_t sender_connection_live_semaphore_address =
    MEM_AERISC_FABRIC_LITE_CONFIG + offsetof(lite_fabric::LiteFabricMemoryMap, sender_connection_live_semaphore);
static_assert(sender_connection_live_semaphore_address % 16 == 0);

template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool SKIP_CONNECTION_LIVENESS_CHECK,
    uint8_t SENDER_NUM_BUFFERS,
    uint8_t RECEIVER_NUM_BUFFERS>
inline void run_sender_channel_step(
    tt::tt_fabric::EthChannelBuffer<SENDER_NUM_BUFFERS>& local_sender_channel,
    tt::tt_fabric::EdmChannelWorkerInterface<SENDER_NUM_BUFFERS>& local_sender_channel_worker_interface,
    tt::tt_fabric::OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS>& outbound_to_receiver_channel_pointers,
    tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>& remote_receiver_channel,
    bool& channel_connection_established,
    uint32_t sender_channel_free_slots_stream_id,
    SenderChannelFromReceiverCredits& sender_channel_from_receiver_credits) {
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    uint32_t free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    bool has_unsent_packet = free_slots != SENDER_NUM_BUFFERS;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;
    if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        can_send = can_send && !internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ);
    }
    if constexpr (enable_first_level_ack) {
        bool sender_backpressured_from_sender_side = free_slots == 0;
        can_send = can_send && !sender_backpressured_from_sender_side;
    }

    if (can_send) {
        // send_next_data
        auto& remote_receiver_buffer_index = outbound_to_receiver_channel_pointers.remote_receiver_buffer_index;
        auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
        auto& local_sender_write_counter = local_sender_channel_worker_interface.local_write_counter;

        uint32_t src_addr = local_sender_channel.get_cached_next_buffer_slot_addr();

        volatile auto* pkt_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(src_addr);
        size_t payload_size_bytes = pkt_header->get_payload_size_including_header();
        auto dest_addr = remote_receiver_channel.get_cached_next_buffer_slot_addr();
        pkt_header->src_ch_id = 0;  // Sender channel index

        if constexpr (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
            while (internal_::eth_txq_is_busy(sender_txq_id)) {
            };
        }

        DPRINT << "Sending packet " << HEX() << src_addr << " -> " << (uint64_t)dest_addr << ENDL();
        internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

        if constexpr (SKIP_CONNECTION_LIVENESS_CHECK) {
            // For persistent connections, we don't need to increment the counter, we only care about the
            // buffer index, so we only increment it directly
            local_sender_write_counter.index = tt::tt_fabric::BufferIndex{
                tt::tt_fabric::wrap_increment<SENDER_NUM_BUFFERS>(local_sender_write_counter.index.get())};
        } else {
            local_sender_write_counter.increment();
        }

        remote_receiver_buffer_index = tt::tt_fabric::BufferIndex{
            tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS>(remote_receiver_buffer_index.get())};
        remote_receiver_channel.set_cached_next_buffer_slot_addr(
            remote_receiver_channel.get_buffer_address(remote_receiver_buffer_index));
        local_sender_channel.set_cached_next_buffer_slot_addr(
            local_sender_channel.get_buffer_address(local_sender_write_counter.get_buffer_index()));
        remote_receiver_num_free_slots--;
        // update the remote reg
        static constexpr uint32_t packets_to_forward = 1;
        while (internal_::eth_txq_is_busy(sender_txq_id)) {
        };
        remote_update_ptr_val<to_receiver_packets_sent_streams[0], sender_txq_id>(packets_to_forward);
    }

    // Process completions
    int32_t completions_since_last_check =
        sender_channel_from_receiver_credits.get_num_unprocessed_completions_from_receiver();
    if (completions_since_last_check) {
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        sender_channel_from_receiver_credits.increment_num_processed_completions(completions_since_last_check);
        if constexpr (!enable_first_level_ack) {
            if constexpr (SKIP_CONNECTION_LIVENESS_CHECK) {
                local_sender_channel_worker_interface
                    .template update_persistent_connection_copy_of_free_slots<enable_ring_support>(
                        completions_since_last_check);
            } else {
                // Connection liveness checks are only done for connections that are not persistent
                // For those connections, it's unsafe to use free-slots counters held in stream registers
                // due to the lack of race avoidant connection protocol. Therefore, we update our read counter
                // instead because these connections will be read/write counter based instead
                local_sender_channel_worker_interface.increment_local_read_counter(completions_since_last_check);
                if (channel_connection_established) {
                    local_sender_channel_worker_interface.notify_worker_of_read_counter_update();
                } else {
                    local_sender_channel_worker_interface.copy_read_counter_to_worker_location_info();
                    // If not connected, we update the read counter in L1 as well so the next connecting worker
                    // is more likely to see space available as soon as it tries connecting
                }
            }
        }
    }
}

// !!!WARNING!!! - MAKE SURE CONSUMER HAS SPACE BEFORE CALLING
template <uint8_t rx_channel_id>
FORCE_INLINE void receiver_forward_packet(
    // TODO: have a separate cached copy of the packet header to save some additional L1 loads
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    uint8_t transaction_id) {
    constexpr bool ENABLE_STATEFUL_NOC_APIS = false;
    invalidate_l1_cache();  // Make sure we have the latest packet header in L1
    uint32_t routing = cached_routing_fields.value & tt::tt_fabric::LowLatencyRoutingFields::FIELD_MASK;
    uint16_t payload_size_bytes = packet_start->payload_size_bytes;
    switch (routing) {
        case tt::tt_fabric::LowLatencyRoutingFields::WRITE_ONLY:
            execute_chip_unicast_to_local_chip(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
            break;
        default: {
            ASSERT(false);
        }
    }
}

template <uint8_t SENDER_NUM_BUFFERS>
FORCE_INLINE bool can_forward_packet_completely(
    ROUTING_FIELDS_TYPE cached_routing_fields,
    tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>& downstream_edm_interface) {
    // We always check if it is the terminal mcast packet value. We can do this because all unicast packets have the
    // mcast terminal value masked in to the routing field. This simplifies the check here to a single compare.
    bool deliver_locally_only = (cached_routing_fields.value & tt::tt_fabric::LowLatencyRoutingFields::FIELD_MASK) ==
                                tt::tt_fabric::LowLatencyRoutingFields::WRITE_ONLY;
    return deliver_locally_only || downstream_edm_interface.edm_has_space_for_packet();
}

template <
    uint8_t receiver_channel,
    uint8_t to_receiver_pkts_sent_id,
    typename WriteTridTracker,
    uint8_t RECEIVER_NUM_BUFFERS,
    uint8_t DOWNSTREAM_SENDER_NUM_BUFFERS>
void run_receiver_channel_step(
    tt::tt_fabric::EthChannelBuffer<RECEIVER_NUM_BUFFERS>& local_receiver_channel,
    std::array<tt::tt_fabric::EdmToEdmSender<DOWNSTREAM_SENDER_NUM_BUFFERS>, NUM_USED_RECEIVER_CHANNELS>&
        downstream_edm_interface,
    tt::tt_fabric::ReceiverChannelPointers<RECEIVER_NUM_BUFFERS>& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    ReceiverChannelResponseCreditSender& receiver_channel_response_credit_sender) {
    auto pkts_received_since_last_check = get_ptr_val<to_receiver_pkts_sent_id>();
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    bool unwritten_packets;
    if constexpr (enable_first_level_ack) {
        auto& ack_counter = receiver_channel_pointers.ack_counter;
        bool pkts_received = pkts_received_since_last_check > 0;
        ASSERT(receiver_channel_pointers.completion_counter - ack_counter < RECEIVER_NUM_BUFFERS);
        if (pkts_received) {
            // currently only support processing one packet at a time, so we only decrement by 1
            increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
            receiver_send_received_ack(
                receiver_channel_response_credit_sender, ack_counter.get_buffer_index(), local_receiver_channel);
            ack_counter.increment();
        }
        unwritten_packets = !wr_sent_counter.is_caught_up_to(ack_counter);
    } else {
        unwritten_packets = pkts_received_since_last_check != 0;
    }

    if (unwritten_packets) {
        invalidate_l1_cache();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        ROUTING_FIELDS_TYPE cached_routing_fields;
#if !defined(FABRIC_2D) || !defined(DYNAMIC_ROUTING_ENABLED)
        cached_routing_fields = packet_header->routing_fields;
#endif

        receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
        uint32_t hop_cmd;
        bool can_send_to_all_local_chip_receivers =
            can_forward_packet_completely(cached_routing_fields, downstream_edm_interface[receiver_channel]);

        if (can_send_to_all_local_chip_receivers) {
            did_something = true;
            uint8_t trid = receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                receiver_buffer_index);
            receiver_forward_packet<receiver_channel>(packet_header, cached_routing_fields, trid);
            wr_sent_counter.increment();
            // decrement the to_receiver_pkts_sent_id stream register by 1 since current packet has been processed.
            increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
        }
    }

    // flush and completion are fused, so we only need to update one of the counters
    // update completion since other parts of the code check against completion
    auto& completion_counter = receiver_channel_pointers.completion_counter;
    // Currently unclear if it's better to loop here or not...
    bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
    auto receiver_buffer_index = completion_counter.get_buffer_index();
    bool next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
    bool can_send_completion = unflushed_writes && next_trid_flushed;
    if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
        can_send_completion = can_send_completion && !internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ);
    }
    if (can_send_completion) {
        receiver_send_completion_ack(
            receiver_channel_response_credit_sender, receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
        receiver_channel_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
        completion_counter.increment();
    }
};

BEGIN_MAIN_FUNCTION() {
    invalidate_l1_cache();
    // Test values
    auto dptr = reinterpret_cast<volatile uint32_t*>(0x20000);
    dptr[0] = 0xdeadbeef;
    dptr[1] = 0xcafecafe;

    auto structs = reinterpret_cast<volatile lite_fabric::LiteFabricMemoryMap*>(MEM_AERISC_FABRIC_LITE_CONFIG);

    // Initialize stream registers
    init_ptr_val<to_receiver_packets_sent_streams[0]>(0);
    init_ptr_val<to_sender_packets_acked_streams[0]>(0);
    init_ptr_val<to_sender_packets_completed_streams[0]>(0);
    init_ptr_val<sender_channel_free_slots_stream_ids[0]>(CHANNEL_BUFFER_SLOTS);  // LOCAL
    init_ptr_val<receiver_channel_0_free_slots_from_east_stream_id>(CHANNEL_BUFFER_SLOTS);
    init_ptr_val<receiver_channel_0_free_slots_from_west_stream_id>(CHANNEL_BUFFER_SLOTS);
    init_ptr_val<receiver_channel_0_free_slots_from_north_stream_id>(CHANNEL_BUFFER_SLOTS);
    init_ptr_val<receiver_channel_0_free_slots_from_south_stream_id>(CHANNEL_BUFFER_SLOTS);

    WriteTransactionIdTracker<RECEIVER_NUM_BUFFERS_ARRAY[0], NUM_TRANSACTION_IDS, 0> receiver_channel_0_trid_tracker;

    auto local_receiver_channels = tt::tt_fabric::EthChannelBuffers<RECEIVER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_RECEIVER_CHANNELS>{});

    auto remote_receiver_channels = tt::tt_fabric::EthChannelBuffers<RECEIVER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_RECEIVER_CHANNELS>{});

    auto local_sender_channels = tt::tt_fabric::EthChannelBuffers<SENDER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_SENDER_CHANNELS>{});

    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& local_sender_buffer_addresses = {local_sender_0_channel_address};
    const std::array<size_t, NUM_RECEIVER_CHANNELS>& local_receiver_buffer_addresses = {
        local_receiver_0_channel_address};
    const std::array<size_t, NUM_RECEIVER_CHANNELS>& remote_receiver_buffer_addresses = {
        remote_receiver_0_channel_address};

    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_flow_control_semaphore_addresses = {
        sender_flow_control_semaphore_address};
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_live_semaphore_addresses = {
        sender_connection_live_semaphore_address};
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        local_sender_channel_0_connection_info_addr};

    std::array<uint32_t, NUM_SENDER_CHANNELS> local_sender_channel_free_slots_stream_ids_ordered;

    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
            local_sender_connection_info_addresses[i]);
        connection_worker_info_ptr->edm_read_counter = 0;
    }

    // populate_local_sender_channel_free_slots_stream_id_ordered_map
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_free_slots_stream_ids_ordered[i] = sender_channel_free_slots_stream_ids[i];
    }

    local_receiver_channels.init(
        local_receiver_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(PACKET_HEADER_TYPE),
        RECEIVER_CHANNEL_BASE_ID);

    remote_receiver_channels.init(
        remote_receiver_buffer_addresses.data(),
        CHANNEL_BUFFER_SIZE,
        sizeof(PACKET_HEADER_TYPE),
        RECEIVER_CHANNEL_BASE_ID);

    local_sender_channels.init(
        local_sender_buffer_addresses.data(), CHANNEL_BUFFER_SIZE, sizeof(PACKET_HEADER_TYPE), SENDER_CHANNEL_BASE_ID);

    auto local_sender_channel_worker_interfaces =
        tt::tt_fabric::EdmChannelWorkerInterfaces<SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});

    // Unused. NO downstream EdmToEdm
    std::array<tt::tt_fabric::EdmToEdmSender<DOWNSTREAM_SENDER_NUM_BUFFERS>, NUM_USED_RECEIVER_CHANNELS>
        downstream_edm_noc_interfaces;

    tt::tt_fabric::init_local_sender_channel_worker_interfaces<NUM_SENDER_CHANNELS>(
        local_sender_connection_live_semaphore_addresses,
        local_sender_connection_info_addresses,
        local_sender_channel_worker_interfaces,
        local_sender_flow_control_semaphore_addresses);

    auto outbound_to_receiver_channel_pointers = tt::tt_fabric::
        ChannelPointersTuple<tt::tt_fabric::OutboundReceiverChannelPointers, REMOTE_RECEIVER_NUM_BUFFERS_ARRAY>::make();
    auto receiver_channel_pointers =
        tt::tt_fabric::ChannelPointersTuple<tt::tt_fabric::ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make();

    // Select sender and receiver 0
    auto outbound_to_receiver_channel_pointer_ch0 = outbound_to_receiver_channel_pointers.template get<0>();
    auto remote_receiver_channel = remote_receiver_channels.template get<0>();
    auto local_receiver_channel = local_receiver_channels.template get<0>();
    auto local_sender_channel = local_sender_channels.template get<0>();
    auto receiver_channel_pointers_ch0 = receiver_channel_pointers.template get<0>();
    receiver_channel_pointers_ch0.reset();
    auto local_sender_channel_worker_interface = local_sender_channel_worker_interfaces.template get<0>();

    auto receiver_channel_response_credit_senders =
        init_receiver_channel_response_credit_senders<NUM_RECEIVER_CHANNELS>();
    auto sender_channel_from_receiver_credits =
        init_sender_channel_from_receiver_credits_flow_controllers<NUM_SENDER_CHANNELS>();

    std::array<bool, NUM_SENDER_CHANNELS> channel_connection_established =
        initialize_array<NUM_SENDER_CHANNELS, bool, false>();
    std::array<uint8_t, num_eth_ports> port_direction_table;  // Not used for 1D fabric

    DPRINT << "Lite Fabric Init MMIO = " << (uint32_t)structs->config.is_mmio << ENDL();

    structs->host_interface.init();

    lite_fabric::routing_init(&structs->config);

    while (structs->config.routing_enabled) {
        invalidate_l1_cache();

        run_sender_channel_step<
            0,
            to_receiver_packets_sent_streams[0],
            SKIP_CONNECTION_LIVENESS_CHECK,
            SENDER_NUM_BUFFERS_ARRAY[0],
            RECEIVER_NUM_BUFFERS_ARRAY[0]>(
            local_sender_channel,
            local_sender_channel_worker_interface,
            outbound_to_receiver_channel_pointer_ch0,
            remote_receiver_channel,
            channel_connection_established[0],
            local_sender_channel_free_slots_stream_ids_ordered[0],
            sender_channel_from_receiver_credits[0]);

        run_receiver_channel_step<0, to_receiver_packets_sent_streams[0]>(
            local_receiver_channel,
            downstream_edm_noc_interfaces,
            receiver_channel_pointers_ch0,
            receiver_channel_0_trid_tracker,
            port_direction_table,  // Unused for 1D fabric
            receiver_channel_response_credit_senders[0]);
    }

    receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();

    ncrisc_noc_counters_init();

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
END_MAIN_FUNCTION()
