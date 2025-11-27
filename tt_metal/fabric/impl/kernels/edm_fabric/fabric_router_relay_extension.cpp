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
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"
#include "tt_metal/fabric/hw/inc/udm/udm_memory_pool.hpp"

#include <cstddef>
#include <array>
// clang-format on

namespace tt::tt_fabric {
template <uint8_t FABRIC_RELAY_CHANNEL_NUM_BUFFERS>
using FabricRelayChannelBuffer = EthChannelBuffer<PACKET_HEADER_TYPE, FABRIC_RELAY_CHANNEL_NUM_BUFFERS>;

template <uint8_t FABRIC_RELAY_CHANNEL_NUM_BUFFERS>
using FabricRelayStaticSizedChannelWorkerInterface =
    StaticSizedSenderChannelWorkerInterface<noc_index, FABRIC_RELAY_CHANNEL_NUM_BUFFERS>;

using FabricRelayChannelClientLocationInfo = EDMChannelWorkerLocationInfo;

template <uint8_t FABRIC_RELAY_CHANNEL_NUM_BUFFERS>
using WorkerToFabricRelaySender = WorkerToFabricEdmSenderImpl<false, FABRIC_RELAY_CHANNEL_NUM_BUFFERS>;

using FabricRelayStatus = EDMStatus;

template <uint8_t NUM_EDM_BUFFERS>
using FabricRelayToMuxSender = WorkerToFabricEdmSenderImpl<false, NUM_EDM_BUFFERS>;
}  // namespace tt::tt_fabric

static_assert(noc_index == 1, "Relay kernel requires noc_index to be 1 for correct noc address calculation");

// Scalar configuration values
constexpr size_t status_address = get_compile_time_arg_val(0);
constexpr size_t termination_signal_address = get_compile_time_arg_val(1);
constexpr size_t channel_stream_id = get_compile_time_arg_val(2);
constexpr size_t NUM_ITERS_BETWEEN_TEARDOWN_CHECKS = get_compile_time_arg_val(3);
constexpr size_t mux_num_buffers = get_compile_time_arg_val(4);
constexpr size_t mux_buffer_size_bytes = get_compile_time_arg_val(5);
constexpr size_t downstream_mux_status_readback_address = get_compile_time_arg_val(6);
constexpr uint32_t NUM_MUX_CONNECTIONS = get_compile_time_arg_val(7);
constexpr uint32_t NUM_CHANNEL_TYPES = get_compile_time_arg_val(8);

// Per-channel-type arrays start at index 9
// Relay only has one channel type (ROUTER_CHANNEL), so arrays have size 1
constexpr uint32_t NUM_CHANNELS_ARRAY_START_IDX = 9;
constexpr uint32_t NUM_BUFFERS_ARRAY_START_IDX = NUM_CHANNELS_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t BUFFER_SIZE_ARRAY_START_IDX = NUM_BUFFERS_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t CONNECTION_INFO_BASE_ADDR_ARRAY_START_IDX = BUFFER_SIZE_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t CONNECTION_HANDSHAKE_BASE_ADDR_ARRAY_START_IDX =
    CONNECTION_INFO_BASE_ADDR_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t FLOW_CONTROL_BASE_ADDR_ARRAY_START_IDX =
    CONNECTION_HANDSHAKE_BASE_ADDR_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t CHANNEL_BUFFER_BASE_ADDR_ARRAY_START_IDX =
    FLOW_CONTROL_BASE_ADDR_ARRAY_START_IDX + NUM_CHANNEL_TYPES;

// Extract relay channel configuration (relay has only 1 channel type with 1 channel)
constexpr uint32_t NUM_CHANNELS = get_compile_time_arg_val(NUM_CHANNELS_ARRAY_START_IDX);
constexpr uint8_t NUM_BUFFERS = get_compile_time_arg_val(NUM_BUFFERS_ARRAY_START_IDX);
constexpr size_t BUFFER_SIZE_BYTES = get_compile_time_arg_val(BUFFER_SIZE_ARRAY_START_IDX);
constexpr size_t connection_info_base_address = get_compile_time_arg_val(CONNECTION_INFO_BASE_ADDR_ARRAY_START_IDX);
constexpr size_t connection_handshake_base_address =
    get_compile_time_arg_val(CONNECTION_HANDSHAKE_BASE_ADDR_ARRAY_START_IDX);
constexpr size_t sender_flow_control_base_address = get_compile_time_arg_val(FLOW_CONTROL_BASE_ADDR_ARRAY_START_IDX);
constexpr size_t channels_base_l1_address = get_compile_time_arg_val(CHANNEL_BUFFER_BASE_ADDR_ARRAY_START_IDX);

// Mux connection arrays start immediately after channel type arrays
constexpr uint32_t MUX_ACTIVE_START_IDX = CHANNEL_BUFFER_BASE_ADDR_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t MUX_NOC_X_START_IDX = MUX_ACTIVE_START_IDX + NUM_MUX_CONNECTIONS;
constexpr uint32_t MUX_NOC_Y_START_IDX = MUX_NOC_X_START_IDX + NUM_MUX_CONNECTIONS;
constexpr uint32_t MUX_BUFFER_BASE_ADDR_START_IDX = MUX_NOC_Y_START_IDX + NUM_MUX_CONNECTIONS;
constexpr uint32_t MUX_CONNECTION_HANDSHAKE_ADDR_START_IDX = MUX_BUFFER_BASE_ADDR_START_IDX + NUM_MUX_CONNECTIONS;
constexpr uint32_t MUX_WORKER_LOCATION_INFO_ADDR_START_IDX =
    MUX_CONNECTION_HANDSHAKE_ADDR_START_IDX + NUM_MUX_CONNECTIONS;
constexpr uint32_t MUX_BUFFER_INDEX_ADDR_START_IDX = MUX_WORKER_LOCATION_INFO_ADDR_START_IDX + NUM_MUX_CONNECTIONS;
constexpr uint32_t MUX_RELAY_FLOW_CONTROL_SEMAPHORE_ADDR_START_IDX =
    MUX_BUFFER_INDEX_ADDR_START_IDX + NUM_MUX_CONNECTIONS;
constexpr uint32_t MUX_RELAY_TEARDOWN_SEMAPHORE_ADDR_START_IDX =
    MUX_RELAY_FLOW_CONTROL_SEMAPHORE_ADDR_START_IDX + NUM_MUX_CONNECTIONS;
constexpr uint32_t MUX_RELAY_BUFFER_INDEX_SEMAPHORE_ADDR_START_IDX =
    MUX_RELAY_TEARDOWN_SEMAPHORE_ADDR_START_IDX + NUM_MUX_CONNECTIONS;
constexpr uint32_t MUX_FREE_SLOTS_STREAM_ID_START_IDX =
    MUX_RELAY_BUFFER_INDEX_SEMAPHORE_ADDR_START_IDX + NUM_MUX_CONNECTIONS;

// Initialize mux connection arrays using compile-time argument helper
constexpr std::array<uint32_t, NUM_MUX_CONNECTIONS> mux_active =
    fill_array_with_next_n_args<uint32_t, MUX_ACTIVE_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<uint32_t, NUM_MUX_CONNECTIONS> mux_noc_x =
    fill_array_with_next_n_args<uint32_t, MUX_NOC_X_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<uint32_t, NUM_MUX_CONNECTIONS> mux_noc_y =
    fill_array_with_next_n_args<uint32_t, MUX_NOC_Y_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_MUX_CONNECTIONS> mux_buffer_base_addr =
    fill_array_with_next_n_args<size_t, MUX_BUFFER_BASE_ADDR_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_MUX_CONNECTIONS> mux_connection_handshake_addr =
    fill_array_with_next_n_args<size_t, MUX_CONNECTION_HANDSHAKE_ADDR_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_MUX_CONNECTIONS> mux_worker_location_info_addr =
    fill_array_with_next_n_args<size_t, MUX_WORKER_LOCATION_INFO_ADDR_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_MUX_CONNECTIONS> mux_buffer_index_addr =
    fill_array_with_next_n_args<size_t, MUX_BUFFER_INDEX_ADDR_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_MUX_CONNECTIONS> mux_relay_flow_control_semaphore_addr =
    fill_array_with_next_n_args<size_t, MUX_RELAY_FLOW_CONTROL_SEMAPHORE_ADDR_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_MUX_CONNECTIONS> mux_relay_teardown_semaphore_addr =
    fill_array_with_next_n_args<size_t, MUX_RELAY_TEARDOWN_SEMAPHORE_ADDR_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_MUX_CONNECTIONS> mux_relay_buffer_index_semaphore_addr =
    fill_array_with_next_n_args<size_t, MUX_RELAY_BUFFER_INDEX_SEMAPHORE_ADDR_START_IDX, NUM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_MUX_CONNECTIONS> mux_free_slots_stream_id =
    fill_array_with_next_n_args<size_t, MUX_FREE_SLOTS_STREAM_ID_START_IDX, NUM_MUX_CONNECTIONS>();

// Final compile-time arguments (computed incrementally from last array)
constexpr size_t LOCAL_MUX_STATUS_ADDRESS_IDX = MUX_FREE_SLOTS_STREAM_ID_START_IDX + NUM_MUX_CONNECTIONS;
constexpr size_t mux_status_address = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX);
constexpr size_t udm_memory_pool_base_address = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 1);
constexpr size_t udm_memory_pool_slot_size = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 2);
constexpr size_t udm_memory_pool_num_slots = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 3);
constexpr size_t direction = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 4);
constexpr size_t router_noc_x = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 5);
constexpr size_t router_noc_y = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 6);
constexpr size_t fabric_router_sync_address = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 7);

// Mux connection array indices: [0]=local, [1]=downstream_en, [2]=downstream_ws
constexpr uint32_t LOCAL_MUX_IDX = 0;
constexpr uint32_t DOWNSTREAM_EN_MUX_IDX = 1;
constexpr uint32_t DOWNSTREAM_WS_MUX_IDX = 2;

template <uint8_t NUM_BUFFERS>
void wait_for_static_connection_to_ready(
    tt::tt_fabric::FabricRelayStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface) {
    while (!connect_is_requested(*worker_interface.connection_live_semaphore)) {
        invalidate_l1_cache();
    }

    worker_interface.cache_producer_noc_addr();
}

FORCE_INLINE void wait_for_mux_endpoint_ready(
    uint8_t mux_noc_x, uint8_t mux_noc_y, size_t mux_status_address, uint32_t mux_status_readback_address) {
    uint64_t noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, mux_status_address);
    auto ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mux_status_readback_address);

    ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
    do {
        noc_async_read_one_packet(noc_addr, mux_status_readback_address, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
    } while (ptr[0] != tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC);
}

template <uint8_t NUM_BUFFERS>
void setup_channel(
    tt::tt_fabric::FabricRelayChannelBuffer<NUM_BUFFERS>* channel_ptr,
    tt::tt_fabric::FabricRelayStaticSizedChannelWorkerInterface<NUM_BUFFERS>* worker_interface_ptr,
    bool& channel_connection_established,
    size_t buffer_size_bytes,
    size_t channel_base_address,
    size_t connection_info_address,
    size_t connection_handshake_address,
    size_t sender_flow_control_address,
    StreamId my_channel_free_slots_stream_id) {
    new (channel_ptr) tt::tt_fabric::FabricRelayChannelBuffer<NUM_BUFFERS>(
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

template <uint32_t Direction, typename RelayToMuxSenderType, size_t NumMuxConnections, typename UDMMemoryPoolType>
__attribute__((optimize("jump-tables"))) FORCE_INLINE void execute_noc_txn_to_local_chip(
    std::array<RelayToMuxSenderType, NumMuxConnections>& mux_connections,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* const packet_header,
    UDMMemoryPoolType& memory_pool) {
    const auto& header = *packet_header;
    uint16_t payload_size_bytes = header.payload_size_bytes;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_header) + sizeof(PACKET_HEADER_TYPE);

    tt::tt_fabric::NocSendType noc_send_type = header.noc_send_type;
    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const auto noc_addr = header.command_fields.unicast_write.noc_address;
            noc_async_write_one_packet(payload_start_address, noc_addr, payload_size_bytes);
            // temporarily place here until we have txn id support
            noc_async_writes_flushed();
            // writes done, send ack back
            tt::tt_fabric::udm::fabric_fast_write_ack<Direction>(mux_connections, packet_header);
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            const auto noc_addr = header.command_fields.unicast_seminc.noc_address;
            const auto increment = header.command_fields.unicast_seminc.val;
            if (header.command_fields.unicast_seminc.flush) {
                noc_async_write_barrier();
            }
            noc_semaphore_inc(noc_addr, increment);
            // writes done, send ack back
            tt::tt_fabric::udm::fabric_fast_atomic_ack<Direction>(mux_connections, packet_header);

        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {
            const auto noc_addr = header.command_fields.unicast_inline_write.noc_address;
            const auto value = header.command_fields.unicast_inline_write.value;
            noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(noc_addr, value);
            // temporarily place here until we have txn id support
            noc_async_writes_flushed();
            // writes done, send ack back
            tt::tt_fabric::udm::fabric_fast_write_ack<Direction>(mux_connections, packet_header);
        } break;

        case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const auto dest_address = header.command_fields.unicast_seminc_fused.noc_address;
            noc_async_write_one_packet(payload_start_address, dest_address, payload_size_bytes);

            const uint64_t semaphore_dest_address = header.command_fields.unicast_seminc_fused.semaphore_noc_address;
            const auto increment = header.command_fields.unicast_seminc_fused.val;
            if (header.command_fields.unicast_seminc_fused.flush) {
                noc_async_write_barrier();
            }
            noc_semaphore_inc(semaphore_dest_address, increment);
            // writes done, send ack back - Do we also need to send atomic inc response back?
            tt::tt_fabric::udm::fabric_fast_write_ack<Direction>(mux_connections, packet_header);
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE: {
            size_t offset = 0;
            size_t chunk_size;
            for (size_t i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; ++i) {
                if (i == NOC_SCATTER_WRITE_MAX_CHUNKS - 1) {
                    chunk_size = payload_size_bytes - offset;
                } else {
                    chunk_size = header.command_fields.unicast_scatter_write.chunk_size[i];
                }
                const auto dest_address = header.command_fields.unicast_scatter_write.noc_address[i];
                noc_async_write_one_packet(payload_start_address + offset, dest_address, chunk_size);
                offset += chunk_size;
            }
            // temporarily place here until we have txn id support
            noc_async_writes_flushed();
            // writes done, send ack back
            tt::tt_fabric::udm::fabric_fast_write_ack<Direction>(mux_connections, packet_header);
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_READ: {
            const auto noc_addr = header.command_fields.unicast_read.noc_address;
            const auto size_bytes = header.udm_control.read.size_bytes;
            const auto num_slots = memory_pool.slots_needed(size_bytes);
            // wait for space, allocate slots, and fill with noc_async_read (handles wrap)
            while (!memory_pool.cb_has_enough_slots(num_slots)) {
            }
            memory_pool.cb_allocate_and_fill_slots(noc_addr, size_bytes);
            // temporarily place here until we have txn id support
            noc_async_read_barrier();
            // reads done, send ack back (reads slot by slot from memory pool)
            tt::tt_fabric::udm::fabric_fast_read_any_len_ack<Direction>(mux_connections, packet_header, memory_pool);
            // free slots (FIFO)
            memory_pool.cb_deallocate_slots(num_slots);
        } break;
        default: {
            ASSERT(false);
        } break;
    };
}

template <uint32_t Direction, uint8_t NUM_BUFFERS, uint8_t NUM_MUX_BUFFERS, typename UDMMemoryPoolType>
void forward_data(
    tt::tt_fabric::FabricRelayChannelBuffer<NUM_BUFFERS>& channel,
    tt::tt_fabric::FabricRelayStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface,
    std::array<tt::tt_fabric::FabricRelayToMuxSender<NUM_MUX_BUFFERS>, NUM_MUX_CONNECTIONS>& mux_connections,
    bool& channel_connection_established,
    StreamId my_channel_free_slots_stream_id,
    UDMMemoryPoolType& memory_pool) {
    bool has_unsent_payload = get_ptr_val(my_channel_free_slots_stream_id.get()) != NUM_BUFFERS;
    if (has_unsent_payload) {
        size_t buffer_address = channel.get_buffer_address(worker_interface.local_write_counter.get_buffer_index());
        invalidate_l1_cache();
        auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(buffer_address);

        execute_noc_txn_to_local_chip<Direction>(mux_connections, packet_header, memory_pool);

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

    // Initialize UDM memory pool for read response data
    tt::tt_fabric::udm::
        UDMMemoryPool<udm_memory_pool_base_address, udm_memory_pool_slot_size, udm_memory_pool_num_slots>
            memory_pool;

    // clear out memory regions
    auto num_regions_to_clear = get_arg_val<uint32_t>(rt_args_idx++);
    for (uint32_t i = 0; i < num_regions_to_clear; i++) {
        auto address = get_arg_val<uint32_t>(rt_args_idx++);
        auto size = get_arg_val<uint32_t>(rt_args_idx++);
        zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(address), size);
    }

    // Construct mux connections using arrays: [0]=local, [1]=downstream_en, [2]=downstream_ws
    // Use placement new to construct only active connections (like downstream_edm_noc_interfaces in router)
    std::array<tt::tt_fabric::FabricRelayToMuxSender<mux_num_buffers>, NUM_MUX_CONNECTIONS> mux_connections;

    // Create each mux connection if active
    constexpr bool is_persistent = true;  // All active connections are persistent

    for (uint32_t i = 0; i < NUM_MUX_CONNECTIONS; i++) {
        if (mux_active[i]) {
            new (&mux_connections[i]) tt::tt_fabric::FabricRelayToMuxSender<mux_num_buffers>(
                is_persistent,
                mux_noc_x[i],
                mux_noc_y[i],
                mux_buffer_base_addr[i],
                static_cast<uint8_t>(mux_num_buffers),
                mux_connection_handshake_addr[i],
                mux_worker_location_info_addr[i],
                static_cast<uint16_t>(mux_buffer_size_bytes),
                mux_buffer_index_addr[i],
                reinterpret_cast<volatile uint32_t*>(mux_relay_flow_control_semaphore_addr[i]),
                reinterpret_cast<volatile uint32_t*>(mux_relay_teardown_semaphore_addr[i]),
                mux_relay_buffer_index_semaphore_addr[i],
                static_cast<uint32_t>(mux_free_slots_stream_id[i]),
                StreamId{static_cast<uint32_t>(mux_free_slots_stream_id[i]) /* unused */});
        }
    }

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
    volatile auto mux_status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mux_status_address);
    while (*mux_status_ptr != tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC) {
        invalidate_l1_cache();
    }

    // before connecting to downstream mux, wait for mux status to turn into READY_FOR_TRAFFIC
    for (uint32_t i = 1; i < NUM_MUX_CONNECTIONS; i++) {
        if (mux_active[i]) {
            wait_for_mux_endpoint_ready(
                mux_connections[i].edm_noc_x,
                mux_connections[i].edm_noc_y,
                mux_status_address,
                downstream_mux_status_readback_address);
        }
    }

    // Open all active mux connections (local: same core, downstream: remote cores)
    for (uint32_t i = 0; i < NUM_MUX_CONNECTIONS; i++) {
        if (mux_active[i]) {
            mux_connections[i].open();
        }
    }

    // signal the fabric router (this is the router that is connecting to the relay) the relay is ready
    auto noc_addr = get_noc_addr(router_noc_x, router_noc_y, fabric_router_sync_address);
    noc_semaphore_inc(noc_addr, 1);

    wait_for_static_connection_to_ready<NUM_BUFFERS>(worker_interface);

    status_ptr[0] = tt::tt_fabric::FabricRelayStatus::READY_FOR_TRAFFIC;

    while (!got_immediate_termination_signal(termination_signal_ptr)) {
        for (size_t i = 0; i < NUM_ITERS_BETWEEN_TEARDOWN_CHECKS; i++) {
            forward_data<direction, NUM_BUFFERS, mux_num_buffers>(
                channel,
                worker_interface,
                mux_connections,
                channel_connection_established,
                StreamId{channel_stream_id},
                memory_pool);
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    status_ptr[0] = tt::tt_fabric::FabricRelayStatus::TERMINATED;
}
