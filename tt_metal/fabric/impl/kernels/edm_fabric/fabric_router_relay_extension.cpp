// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

#include <cstddef>
#include <array>
// clang-format on

namespace tt::tt_fabric {
template <uint8_t FABRIC_RELAY_CHANNEL_NUM_BUFFERS>
using FabricRelayChannelBuffer = EthChannelBuffer<PACKET_HEADER_TYPE, FABRIC_RELAY_CHANNEL_NUM_BUFFERS>;

template <uint8_t FABRIC_RELAY_CHANNEL_NUM_BUFFERS>
using FabricRelayStaticSizedChannelWorkerInterface =
    StaticSizedSenderChannelWorkerInterface<tt::tt_fabric::worker_handshake_noc, FABRIC_RELAY_CHANNEL_NUM_BUFFERS>;

using FabricRelayChannelClientLocationInfo = EDMChannelWorkerLocationInfo;

template <uint8_t FABRIC_RELAY_CHANNEL_NUM_BUFFERS>
using WorkerToFabricRelaySender = WorkerToFabricEdmSenderImpl<false, FABRIC_RELAY_CHANNEL_NUM_BUFFERS>;

using FabricRelayStatus = EDMStatus;

template <uint8_t NUM_EDM_BUFFERS>
using FabricRelayToMuxSender = WorkerToFabricEdmSenderImpl<false, NUM_EDM_BUFFERS>;
}  // namespace tt::tt_fabric

constexpr uint8_t NUM_BUFFERS = get_compile_time_arg_val(0);
constexpr size_t BUFFER_SIZE_BYTES = get_compile_time_arg_val(1);
constexpr size_t status_address = get_compile_time_arg_val(2);
constexpr size_t termination_signal_address = get_compile_time_arg_val(3);
constexpr size_t connection_info_base_address = get_compile_time_arg_val(4);
constexpr size_t connection_handshake_base_address = get_compile_time_arg_val(5);
constexpr size_t sender_flow_control_base_address = get_compile_time_arg_val(6);
constexpr size_t channels_base_l1_address = get_compile_time_arg_val(7);
constexpr size_t channel_stream_id = get_compile_time_arg_val(8);
constexpr size_t NUM_ITERS_BETWEEN_TEARDOWN_CHECKS = get_compile_time_arg_val(9);
// Mux connection info compile args (note: mux_noc_x/y use my_x[0]/my_y[0] instead)
constexpr size_t mux_buffer_base_addr = get_compile_time_arg_val(10);
constexpr size_t mux_num_buffers = get_compile_time_arg_val(11);
constexpr size_t mux_connection_handshake_addr = get_compile_time_arg_val(12);
constexpr size_t mux_worker_location_info_addr = get_compile_time_arg_val(13);
constexpr size_t mux_buffer_size_bytes = get_compile_time_arg_val(14);
constexpr size_t mux_buffer_index_addr = get_compile_time_arg_val(15);
constexpr size_t mux_relay_flow_control_semaphore_addr = get_compile_time_arg_val(16);
constexpr size_t mux_relay_teardown_semaphore_addr = get_compile_time_arg_val(17);
constexpr size_t mux_relay_buffer_index_semaphore_addr = get_compile_time_arg_val(18);
constexpr size_t mux_free_slots_stream_id = get_compile_time_arg_val(19);
constexpr size_t local_mux_status_address = get_compile_time_arg_val(20);
constexpr size_t router_noc_x = get_compile_time_arg_val(21);
constexpr size_t router_noc_y = get_compile_time_arg_val(22);
constexpr size_t fabric_router_sync_address = get_compile_time_arg_val(23);

template <uint8_t NUM_BUFFERS>
void wait_for_static_connection_to_ready(
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface) {
    while (!connect_is_requested(*worker_interface.connection_live_semaphore)) {
        invalidate_l1_cache();
    }

    worker_interface.cache_producer_noc_addr();
}

template <uint8_t NUM_BUFFERS>
void setup_channel(
    tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>* channel_ptr,
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>* worker_interface_ptr,
    bool& channel_connection_established,
    size_t buffer_size_bytes,
    size_t channel_base_address,
    size_t connection_info_address,
    size_t connection_handshake_address,
    size_t sender_flow_control_address,
    StreamId my_channel_free_slots_stream_id) {
    new (channel_ptr) tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>(
        channel_base_address, buffer_size_bytes, sizeof(PACKET_HEADER_TYPE));
    init_ptr_val(my_channel_free_slots_stream_id, NUM_BUFFERS);

    auto connection_worker_info_ptr =
        reinterpret_cast<volatile tt::tt_fabric::FabricRelayChannelClientLocationInfo*>(connection_info_address);

    new (worker_interface_ptr) tt::tt_fabric::FabricRelayStaticSizedChannelWorkerInterface<NUM_BUFFERS>(
        connection_worker_info_ptr,
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(sender_flow_control_address),
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_handshake_address),
        0 /* unused, sender_sync_noc_cmd_buf */,
        NUM_BUFFERS);  //

    channel_connection_established = false;
}

__attribute__((optimize("jump-tables"))) FORCE_INLINE void execute_noc_txn_to_local_chip(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* const packet_header) {
    const auto& header = *packet_header;
    uint16_t payload_size_bytes = header.payload_size_bytes;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_header) + sizeof(PACKET_HEADER_TYPE);

    tt::tt_fabric::NocSendType noc_send_type = header.noc_send_type;
    if (noc_send_type > tt::tt_fabric::NocSendType::NOC_SEND_TYPE_LAST) {
        __builtin_unreachable();
    }
    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const auto noc_addr = header.command_fields.unicast_write.noc_address;
            auto noc_addr_components = get_noc_address_components(noc_addr);
            const auto dest_noc_x = noc_addr_components.first.x;
            const auto dest_noc_y = noc_addr_components.first.y;
            const auto dest_address = noc_addr_components.second;
            DPRINT << "Relay Send NOC_UNICAST_WRITE to noc_x:" << (uint)dest_noc_x << " noc_y:" << (uint)dest_noc_y
                   << " addr:" << dest_address << " payload_size_bytes:" << payload_size_bytes << ENDL();
            noc_async_write_one_packet(payload_start_address, noc_addr, payload_size_bytes);
            // temporily place here until we have txn id support
            noc_async_writes_flushed();
            DPRINT << "Relay Send NOC_UNICAST_WRITE Done" << ENDL();
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            const auto noc_addr = header.command_fields.unicast_seminc.noc_address;
            auto noc_addr_components = get_noc_address_components(noc_addr);
            const auto dest_noc_x = noc_addr_components.first.x;
            const auto dest_noc_y = noc_addr_components.first.y;
            const auto dest_address = noc_addr_components.second;
            const auto increment = header.command_fields.unicast_seminc.val;
            DPRINT << "Relay Send NOC_UNICAST_ATOMIC_INC to noc_x:" << (uint)dest_noc_x << " noc_y:" << (uint)dest_noc_y
                   << " addr:" << dest_address << " increment:" << increment << ENDL();
            if (header.command_fields.unicast_seminc.flush) {
                noc_async_write_barrier();
            }
            noc_semaphore_inc<true>(noc_addr, increment);
            // temporily place here until we have txn id support
            noc_async_atomic_barrier();

        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {
            const auto noc_addr = header.command_fields.unicast_inline_write.noc_address;
            auto noc_addr_components = get_noc_address_components(noc_addr);
            const auto dest_noc_x = noc_addr_components.first.x;
            const auto dest_noc_y = noc_addr_components.first.y;
            const auto dest_address = noc_addr_components.second;
            const auto value = header.command_fields.unicast_inline_write.value;
            DPRINT << "Relay Send NOC_UNICAST_INLINE_WRITE to noc_x:" << (uint)dest_noc_x
                   << " noc_y:" << (uint)dest_noc_y << " addr:" << dest_address << " value:" << value << ENDL();
            noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(noc_addr, value);
            // temporily place here until we have txn id support
            noc_async_writes_flushed();
        } break;
        default: {
            ASSERT(false);
        } break;
    };
}

template <uint8_t NUM_BUFFERS, uint8_t NUM_MUX_BUFFERS>
void forward_data(
    tt::tt_fabric::FabricRelayChannelBuffer<NUM_BUFFERS>& channel,
    tt::tt_fabric::FabricRelayStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface,
    tt::tt_fabric::FabricRelayToMuxSender<NUM_MUX_BUFFERS>& mux_connection,
    bool& channel_connection_established,
    StreamId my_channel_free_slots_stream_id) {
    bool has_unsent_payload = get_ptr_val(my_channel_free_slots_stream_id.get()) != NUM_BUFFERS;
    if (has_unsent_payload) {
        DPRINT << "Relay has_unsent_payload " << (uint)has_unsent_payload << ENDL();
        size_t buffer_address = channel.get_buffer_address(worker_interface.local_write_counter.get_buffer_index());
        invalidate_l1_cache();
        auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(buffer_address);

        execute_noc_txn_to_local_chip(packet_header);

        worker_interface.local_write_counter.increment();
        worker_interface.local_read_counter.increment();

        increment_local_update_ptr_val(my_channel_free_slots_stream_id.get(), 1);

        constexpr bool enable_deadlock_avoidance = true;  // not used
        worker_interface.template notify_persistent_connection_of_free_space<enable_deadlock_avoidance>(1);
    }
}

void kernel_main() {
    size_t rt_args_idx = 0;

    auto status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_address);
    status_ptr[0] = tt::tt_fabric::FabricRelayStatus::STARTED;

    // clear out memory regions
    auto num_regions_to_clear = get_arg_val<uint32_t>(rt_args_idx++);
    for (uint32_t i = 0; i < num_regions_to_clear; i++) {
        auto address = get_arg_val<uint32_t>(rt_args_idx++);
        auto size = get_arg_val<uint32_t>(rt_args_idx++);
        zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(address), size);
    }

    // Construct the mux connection from compile time args directly
    // Note: Mux is on the same core, so use my_x[0] and my_y[0] for NOC coordinates
    constexpr bool is_persistent_fabric = true;
    auto mux_connection = tt::tt_fabric::FabricRelayToMuxSender<mux_num_buffers>(
        is_persistent_fabric,
        my_x[0],
        my_y[0],
        mux_buffer_base_addr,
        static_cast<uint8_t>(mux_num_buffers),
        mux_connection_handshake_addr,
        mux_worker_location_info_addr,
        static_cast<uint16_t>(mux_buffer_size_bytes),
        mux_buffer_index_addr,
        reinterpret_cast<volatile uint32_t*>(mux_relay_flow_control_semaphore_addr),
        reinterpret_cast<volatile uint32_t*>(mux_relay_teardown_semaphore_addr),
        mux_relay_buffer_index_semaphore_addr,
        static_cast<uint32_t>(mux_free_slots_stream_id),
        StreamId{static_cast<uint32_t>(mux_free_slots_stream_id)});

    tt::tt_fabric::FabricRelayChannelBuffer<NUM_BUFFERS> channel;
    tt::tt_fabric::FabricRelayStaticSizedChannelWorkerInterface<NUM_BUFFERS> worker_interface;
    bool channel_connection_established;

    size_t channel_base_address = channels_base_l1_address;
    size_t connection_info_address = connection_info_base_address;
    size_t connection_handshake_address = connection_handshake_base_address;
    size_t sender_flow_control_address = sender_flow_control_base_address;

    setup_channel<NUM_BUFFERS>(
        &channel,
        &worker_interface,
        channel_connection_established,
        BUFFER_SIZE_BYTES,
        channel_base_address,
        connection_info_address,
        connection_handshake_address,
        sender_flow_control_address,
        StreamId{channel_stream_id});

    // this termination signal will be set by the mux kernel during teardown, so first mux receives host teardown
    // signal, then it tells relay to teardown.
    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_address);

    // before connecting to mux, wait for mux status to turn into READY_FOR_TRAFFIC
    DPRINT << "Relay waiting for MUX READY_FOR_TRAFFIC" << ENDL();
    volatile auto mux_status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_mux_status_address);
    while (*mux_status_ptr != tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC) {
        invalidate_l1_cache();
    }
    DPRINT << "Relay waiting for MUX READY_FOR_TRAFFIC Done" << ENDL();

    // this is connecting to the local mux - (always non idle eth)
    mux_connection.open<false>();

    // signal the fabric router (this is the router that is connecting to the relay) the relay is ready
    DPRINT << "Relay signal Router " << (uint)router_noc_x << " " << (uint)router_noc_y << ENDL();
    auto noc_addr = get_noc_addr(router_noc_x, router_noc_y, fabric_router_sync_address);
    noc_semaphore_inc(noc_addr, 1);

    DPRINT << "Relay waiting for Router connection" << ENDL();
    wait_for_static_connection_to_ready<NUM_BUFFERS>(worker_interface);
    DPRINT << "Relay waiting for Router connection Done" << ENDL();

    status_ptr[0] = tt::tt_fabric::FabricRelayStatus::READY_FOR_TRAFFIC;

    DPRINT << "Relay starting Loop" << ENDL();
    while (!got_immediate_termination_signal(termination_signal_ptr)) {
        for (size_t i = 0; i < NUM_ITERS_BETWEEN_TEARDOWN_CHECKS; i++) {
            forward_data<NUM_BUFFERS, mux_num_buffers>(
                channel, worker_interface, mux_connection, channel_connection_established, StreamId{channel_stream_id});
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    status_ptr[0] = tt::tt_fabric::FabricRelayStatus::TERMINATED;
}
