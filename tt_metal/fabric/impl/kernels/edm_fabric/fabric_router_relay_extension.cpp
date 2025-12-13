// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "api/dataflow/dataflow_api.h"
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
#include "tt_metal/fabric/hw/inc/udm/udm_registered_response_pool.hpp"

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
constexpr size_t udm_registered_response_pool_base_address = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 5);
constexpr size_t udm_registered_response_pool_num_slots = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 6);
constexpr size_t router_noc_x = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 7);
constexpr size_t router_noc_y = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 8);
constexpr size_t fabric_router_sync_address = get_compile_time_arg_val(LOCAL_MUX_STATUS_ADDRESS_IDX + 9);

// Mux connection array indices: [0]=local, [1]=downstream_en, [2]=downstream_ws
constexpr uint32_t LOCAL_MUX_IDX = 0;
constexpr uint32_t DOWNSTREAM_EN_MUX_IDX = 1;
constexpr uint32_t DOWNSTREAM_WS_MUX_IDX = 2;

constexpr bool ENABLE_RISC_CPU_DATA_CACHE = true;

template <uint8_t NUM_BUFFERS>
void wait_for_static_connection_to_ready(
    tt::tt_fabric::FabricRelayStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface) {
    while (!connect_is_requested(*worker_interface.connection_live_semaphore)) {
        invalidate_l1_cache();
    }

    worker_interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE>();
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

// Helper function to register a response in the response pool
// Precondition: response_pool.has_space() must be true before calling.
// This invariant is critical for the deadlock avoidance mechanism.
template <uint32_t Direction, tt::tt_fabric::NocSendType noc_send_type, typename RegisteredResponsePoolType>
FORCE_INLINE void register_write_or_atomic_response(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* const packet_header, RegisteredResponsePoolType& response_pool) {
    ASSERT(response_pool.has_space());
    // Only register non-posted operations
    if (!packet_header->udm_control.write.posted) {
        volatile tt::tt_fabric::udm::RegisteredResponse* response = response_pool.get_unregistered_slot();
        uint8_t mux_idx =
            tt::tt_fabric::udm::select_relay_to_mux_connection<Direction>(packet_header->udm_control.write.src_chip_id);
        response->template populate_from_header<noc_send_type>(packet_header->udm_control.write, mux_idx);
        response_pool.register_response();
    }
}

// Helper function to register a read response in the response pool
// Precondition: response_pool.has_space() must be true before calling.
// This invariant is critical for the deadlock avoidance mechanism.
template <uint32_t Direction, typename RegisteredResponsePoolType, typename UDMMemoryPoolType>
FORCE_INLINE void register_read_response(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* const packet_header,
    RegisteredResponsePoolType& response_pool,
    UDMMemoryPoolType& memory_pool) {
    ASSERT(response_pool.has_space());
    ASSERT(packet_header->udm_control.read.size_bytes > 0);

    const uint64_t read_noc_addr = packet_header->command_fields.unicast_read.noc_address;
    uint8_t mux_idx =
        tt::tt_fabric::udm::select_relay_to_mux_connection<Direction>(packet_header->udm_control.read.src_chip_id);

    volatile tt::tt_fabric::udm::RegisteredResponse* response = response_pool.get_unregistered_slot();
    response->template populate_from_header<tt::tt_fabric::NocSendType::NOC_UNICAST_READ>(
        packet_header->udm_control.read, mux_idx, read_noc_addr);

    response_pool.register_response();
}

template <
    uint32_t Direction,
    typename RelayToMuxSenderType,
    size_t NumMuxConnections,
    typename UDMMemoryPoolType,
    typename RegisteredResponsePoolType>
__attribute__((optimize("jump-tables"))) FORCE_INLINE void execute_noc_txn_or_register_response(
    std::array<RelayToMuxSenderType, NumMuxConnections>& mux_connections,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* const packet_header,
    UDMMemoryPoolType& memory_pool,
    RegisteredResponsePoolType& response_pool) {
    const auto& header = *packet_header;
    uint16_t payload_size_bytes = header.payload_size_bytes;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_header) + sizeof(PACKET_HEADER_TYPE);

    switch (header.noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const auto noc_addr = header.command_fields.unicast_write.noc_address;
            noc_async_write_one_packet(payload_start_address, noc_addr, payload_size_bytes);
            // temporarily place here until we have txn id support
            noc_async_writes_flushed();
            // Register response for non-posted writes
            register_write_or_atomic_response<Direction, tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE>(
                packet_header, response_pool);
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            const auto noc_addr = header.command_fields.unicast_seminc.noc_address;
            const auto increment = header.command_fields.unicast_seminc.val;
            if (header.command_fields.unicast_seminc.flush) {
                noc_async_write_barrier();
            }
            noc_semaphore_inc(noc_addr, increment);
            // Register response for non-posted atomics
            register_write_or_atomic_response<Direction, tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC>(
                packet_header, response_pool);
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {
            const auto noc_addr = header.command_fields.unicast_inline_write.noc_address;
            const auto value = header.command_fields.unicast_inline_write.value;
            noc_inline_dw_write(noc_addr, value);
            // temporarily place here until we have txn id support
            noc_async_writes_flushed();
            // Register response for non-posted writes
            register_write_or_atomic_response<Direction, tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE>(
                packet_header, response_pool);
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
            // Register response for non-posted fused atomics
            register_write_or_atomic_response<Direction, tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC>(
                packet_header, response_pool);
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
            // Register response for non-posted scatter writes
            register_write_or_atomic_response<Direction, tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE>(
                packet_header, response_pool);
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_READ: {
            // Register the read response
            register_read_response<Direction>(packet_header, response_pool, memory_pool);
        } break;
        default: {
            ASSERT(false);
        } break;
    };
}

// Main function to process inbound packets from the local router
template <
    uint32_t Direction,
    uint8_t NUM_BUFFERS,
    uint8_t NUM_MUX_BUFFERS,
    typename UDMMemoryPoolType,
    typename RegisteredResponsePoolType>
void process_inbound_packet(
    tt::tt_fabric::FabricRelayChannelBuffer<NUM_BUFFERS>& channel,
    tt::tt_fabric::FabricRelayStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface,
    std::array<tt::tt_fabric::FabricRelayToMuxSender<NUM_MUX_BUFFERS>, NUM_MUX_CONNECTIONS>& mux_connections,
    bool& channel_connection_established,
    StreamId my_channel_free_slots_stream_id,
    UDMMemoryPoolType& memory_pool,
    RegisteredResponsePoolType& response_pool) {
    bool has_unsent_payload = get_ptr_val(my_channel_free_slots_stream_id.get()) != NUM_BUFFERS;
    if (has_unsent_payload) {
        size_t buffer_address = channel.get_buffer_address(worker_interface.local_write_counter.get_buffer_index());
        invalidate_l1_cache();
        auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(buffer_address);

        execute_noc_txn_or_register_response<Direction>(mux_connections, packet_header, memory_pool, response_pool);

        worker_interface.local_write_counter.increment();
        worker_interface.local_read_counter.increment();

        increment_local_update_ptr_val(my_channel_free_slots_stream_id.get(), 1);

        constexpr bool enable_deadlock_avoidance = true;  // not used
        worker_interface.template notify_persistent_connection_of_free_space<enable_deadlock_avoidance>(1);
    }
}

// Helper: Send write/atomic response using existing fabric_fast_ack functions
template <uint32_t Direction, typename RelayToMuxSenderType, size_t NumMuxConnections>
FORCE_INLINE bool send_write_or_atomic_response(
    std::array<RelayToMuxSenderType, NumMuxConnections>& mux_connections,
    volatile tt::tt_fabric::udm::RegisteredResponse* response) {
    uint32_t mux_idx = response->mux_index;
    bool has_space_for_packet = mux_connections[mux_idx].edm_has_space_for_packet();
    if (has_space_for_packet) {
        switch (response->noc_send_type) {
            case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC:
            case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC:
                tt::tt_fabric::udm::fabric_fast_atomic_ack(mux_connections[mux_idx], response);
                break;
            case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE:
            case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE:
            case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE:
                tt::tt_fabric::udm::fabric_fast_write_ack(mux_connections[mux_idx], response);
                break;
            default: ASSERT(false); break;
        }
    }
    return has_space_for_packet;
}

// Helper: Try to allocate memory pool slots for current read response chunk
// Returns true if we allocated any slots (caller should then send them)
template <typename UDMMemoryPoolType>
FORCE_INLINE void allocate_read_memory(
    UDMMemoryPoolType& memory_pool, volatile tt::tt_fabric::udm::RegisteredResponse* response) {
    memory_pool.cb_allocate_and_fill_slots(response);
}

// Helper: Send ONE read response slot using fabric_fast_read_ack
// Returns true if ALL slots have been sent (response complete)
template <uint32_t Direction, typename RelayToMuxSenderType, size_t NumMuxConnections, typename UDMMemoryPoolType>
FORCE_INLINE bool send_read_response_data(
    std::array<RelayToMuxSenderType, NumMuxConnections>& mux_connections,
    UDMMemoryPoolType& memory_pool,
    volatile tt::tt_fabric::udm::RegisteredResponse* response) {
    uint32_t mux_idx = response->mux_index;
    bool response_completed = false;
    bool has_space_for_packet = mux_connections[mux_idx].edm_has_space_for_packet();
    if (has_space_for_packet && response->has_data_to_send()) {
        constexpr uint32_t num_slots_dealloc = 1;
        // Send one slot (helper handles fused atomic on last slot)
        tt::tt_fabric::udm::fabric_fast_read_ack(mux_connections[mux_idx], response, memory_pool);

        // Deallocate ONE slot from memory pool
        memory_pool.cb_deallocate_slots(num_slots_dealloc);

        // Update response: advance dest address, decrement counters
        response_completed = response->complete_send(memory_pool.get_slot_size());
    }
    return response_completed;
}

// Helper: Process read response - allocate memory and send data back in chunks
// Returns true when ALL chunks have been sent (response complete)
template <uint32_t Direction, typename RelayToMuxSenderType, size_t NumMuxConnections, typename UDMMemoryPoolType>
FORCE_INLINE bool process_read_response(
    std::array<RelayToMuxSenderType, NumMuxConnections>& mux_connections,
    UDMMemoryPoolType& memory_pool,
    volatile tt::tt_fabric::udm::RegisteredResponse* response) {
    // Allocate more slots if needed and space available
    allocate_read_memory(memory_pool, response);

    // Send if we have bytes in memory pool
    bool response_completed = send_read_response_data<Direction>(mux_connections, memory_pool, response);
    return response_completed;
}

// Main function to process responses from the registered response pool
template <
    uint32_t Direction,
    typename RelayToMuxSenderType,
    size_t NumMuxConnections,
    typename UDMMemoryPoolType,
    typename RegisteredResponsePoolType>
FORCE_INLINE void process_response(
    std::array<RelayToMuxSenderType, NumMuxConnections>& mux_connections,
    UDMMemoryPoolType& memory_pool,
    RegisteredResponsePoolType& response_pool) {
    if (!response_pool.is_empty()) {
        // Get the current response to process
        volatile tt::tt_fabric::udm::RegisteredResponse* response = response_pool.get_registered_slot();

        bool response_complete = false;
        if (response->noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_READ) {
            // Handle read response - allocate memory, read data, send back
            response_complete = process_read_response<Direction>(mux_connections, memory_pool, response);
        } else {
            // Handle write/atomic response
            response_complete = send_write_or_atomic_response<Direction>(mux_connections, response);
        }
        // Unregister response if complete
        if (response_complete) {
            response_pool.unregister_response();
        }
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

    // Initialize the registered response pool for tracking pending responses
    tt::tt_fabric::udm::
        RegisteredResponsePool<udm_registered_response_pool_base_address, udm_registered_response_pool_num_slots>
            response_pool;

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

    while (!got_immediate_termination_signal<true>(termination_signal_ptr)) {
        for (size_t i = 0; i < NUM_ITERS_BETWEEN_TEARDOWN_CHECKS; i++) {
            process_inbound_packet<direction, NUM_BUFFERS, mux_num_buffers>(
                channel,
                worker_interface,
                mux_connections,
                channel_connection_established,
                StreamId{channel_stream_id},
                memory_pool,
                response_pool);

            // Process registered responses (send acks for completed operations)
            process_response<direction, tt::tt_fabric::FabricRelayToMuxSender<mux_num_buffers>, NUM_MUX_CONNECTIONS>(
                mux_connections, memory_pool, response_pool);
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    status_ptr[0] = tt::tt_fabric::FabricRelayStatus::TERMINATED;
}
