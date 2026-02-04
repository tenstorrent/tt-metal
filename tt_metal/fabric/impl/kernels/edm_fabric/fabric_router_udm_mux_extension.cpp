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
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"

#include <cstddef>
#include <array>
// clang-format on

// ========== Scalar configuration values from MuxConfig::get_compile_time_args ==========
constexpr size_t NUM_TOTAL_CHANNELS = get_compile_time_arg_val(0);
constexpr size_t status_address = get_compile_time_arg_val(1);
constexpr size_t termination_signal_address = get_compile_time_arg_val(2);
constexpr size_t local_fabric_router_status_address = get_compile_time_arg_val(3);
constexpr size_t fabric_router_status_address = get_compile_time_arg_val(4);
constexpr uint8_t NUM_EDM_BUFFERS = get_compile_time_arg_val(5);
constexpr size_t NUM_FULL_SIZE_CHANNELS_ITERS = get_compile_time_arg_val(6);
constexpr size_t NUM_ITERS_BETWEEN_TEARDOWN_CHECKS = get_compile_time_arg_val(7);
constexpr ProgrammableCoreType CORE_TYPE = static_cast<ProgrammableCoreType>(get_compile_time_arg_val(8));
constexpr bool wait_for_fabric_endpoint = get_compile_time_arg_val(9) == 1;
constexpr uint32_t NUM_DOWNSTREAM_MUX_CONNECTIONS = get_compile_time_arg_val(10);
constexpr uint32_t NUM_CHANNEL_TYPES = get_compile_time_arg_val(11);

// ========== Per-channel-type arrays (7 arrays × NUM_CHANNEL_TYPES entries) ==========
constexpr uint32_t NUM_CHANNELS_ARRAY_START_IDX = 12;
constexpr uint32_t NUM_BUFFERS_ARRAY_START_IDX = NUM_CHANNELS_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t BUFFER_SIZE_ARRAY_START_IDX = NUM_BUFFERS_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t CONNECTION_INFO_BASE_ADDR_ARRAY_START_IDX = BUFFER_SIZE_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t CONNECTION_HANDSHAKE_BASE_ADDR_ARRAY_START_IDX =
    CONNECTION_INFO_BASE_ADDR_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t FLOW_CONTROL_BASE_ADDR_ARRAY_START_IDX =
    CONNECTION_HANDSHAKE_BASE_ADDR_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t CHANNEL_BUFFER_BASE_ADDR_ARRAY_START_IDX =
    FLOW_CONTROL_BASE_ADDR_ARRAY_START_IDX + NUM_CHANNEL_TYPES;

// ========== Extract per-channel-type configuration ==========
// In UDM mux mode, we have 3 channel types:
// [0] = WORKER_CHANNEL
// [1] = RELAY_TO_MUX_CHANNEL
// [2] = MUX_TO_MUX_CHANNEL
constexpr std::array<uint32_t, NUM_CHANNEL_TYPES> num_channels_per_type =
    fill_array_with_next_n_args<uint32_t, NUM_CHANNELS_ARRAY_START_IDX, NUM_CHANNEL_TYPES>();
constexpr std::array<uint32_t, NUM_CHANNEL_TYPES> num_buffers_per_type =
    fill_array_with_next_n_args<uint32_t, NUM_BUFFERS_ARRAY_START_IDX, NUM_CHANNEL_TYPES>();
constexpr std::array<size_t, NUM_CHANNEL_TYPES> buffer_size_per_type =
    fill_array_with_next_n_args<size_t, BUFFER_SIZE_ARRAY_START_IDX, NUM_CHANNEL_TYPES>();
constexpr std::array<size_t, NUM_CHANNEL_TYPES> connection_info_base_addrs =
    fill_array_with_next_n_args<size_t, CONNECTION_INFO_BASE_ADDR_ARRAY_START_IDX, NUM_CHANNEL_TYPES>();
constexpr std::array<size_t, NUM_CHANNEL_TYPES> connection_handshake_base_addrs =
    fill_array_with_next_n_args<size_t, CONNECTION_HANDSHAKE_BASE_ADDR_ARRAY_START_IDX, NUM_CHANNEL_TYPES>();
constexpr std::array<size_t, NUM_CHANNEL_TYPES> flow_control_base_addrs =
    fill_array_with_next_n_args<size_t, FLOW_CONTROL_BASE_ADDR_ARRAY_START_IDX, NUM_CHANNEL_TYPES>();
constexpr std::array<size_t, NUM_CHANNEL_TYPES> channel_buffer_base_addrs =
    fill_array_with_next_n_args<size_t, CHANNEL_BUFFER_BASE_ADDR_ARRAY_START_IDX, NUM_CHANNEL_TYPES>();

// ========== Downstream mux connection arrays (11 arrays from MuxConfig) ==========
// These come right after the per-channel-type arrays in MuxConfig::get_compile_time_args
constexpr uint32_t DOWNSTREAM_MUX_ARRAYS_START_IDX = CHANNEL_BUFFER_BASE_ADDR_ARRAY_START_IDX + NUM_CHANNEL_TYPES;
constexpr uint32_t DOWNSTREAM_MUX_ACTIVE_START_IDX = DOWNSTREAM_MUX_ARRAYS_START_IDX;
constexpr uint32_t DOWNSTREAM_MUX_NOC_X_START_IDX = DOWNSTREAM_MUX_ACTIVE_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr uint32_t DOWNSTREAM_MUX_NOC_Y_START_IDX = DOWNSTREAM_MUX_NOC_X_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr uint32_t DOWNSTREAM_MUX_BUFFER_BASE_ADDR_START_IDX =
    DOWNSTREAM_MUX_NOC_Y_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr uint32_t DOWNSTREAM_MUX_CONNECTION_HANDSHAKE_ADDR_START_IDX =
    DOWNSTREAM_MUX_BUFFER_BASE_ADDR_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr uint32_t DOWNSTREAM_MUX_WORKER_LOCATION_INFO_ADDR_START_IDX =
    DOWNSTREAM_MUX_CONNECTION_HANDSHAKE_ADDR_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr uint32_t DOWNSTREAM_MUX_BUFFER_INDEX_ADDR_START_IDX =
    DOWNSTREAM_MUX_WORKER_LOCATION_INFO_ADDR_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr uint32_t DOWNSTREAM_MUX_FLOW_CONTROL_SEMAPHORE_ADDR_START_IDX =
    DOWNSTREAM_MUX_BUFFER_INDEX_ADDR_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr uint32_t DOWNSTREAM_MUX_TEARDOWN_SEMAPHORE_ADDR_START_IDX =
    DOWNSTREAM_MUX_FLOW_CONTROL_SEMAPHORE_ADDR_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr uint32_t DOWNSTREAM_MUX_BUFFER_INDEX_SEMAPHORE_ADDR_START_IDX =
    DOWNSTREAM_MUX_TEARDOWN_SEMAPHORE_ADDR_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr uint32_t DOWNSTREAM_MUX_FREE_SLOTS_STREAM_ID_START_IDX =
    DOWNSTREAM_MUX_BUFFER_INDEX_SEMAPHORE_ADDR_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;

// Initialize downstream mux connection arrays using compile-time argument helper
constexpr std::array<uint32_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_active =
    fill_array_with_next_n_args<uint32_t, DOWNSTREAM_MUX_ACTIVE_START_IDX, NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<uint32_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_noc_x =
    fill_array_with_next_n_args<uint32_t, DOWNSTREAM_MUX_NOC_X_START_IDX, NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<uint32_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_noc_y =
    fill_array_with_next_n_args<uint32_t, DOWNSTREAM_MUX_NOC_Y_START_IDX, NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_buffer_base_addr =
    fill_array_with_next_n_args<size_t, DOWNSTREAM_MUX_BUFFER_BASE_ADDR_START_IDX, NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_connection_handshake_addr =
    fill_array_with_next_n_args<
        size_t,
        DOWNSTREAM_MUX_CONNECTION_HANDSHAKE_ADDR_START_IDX,
        NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_worker_location_info_addr =
    fill_array_with_next_n_args<
        size_t,
        DOWNSTREAM_MUX_WORKER_LOCATION_INFO_ADDR_START_IDX,
        NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_buffer_index_addr =
    fill_array_with_next_n_args<size_t, DOWNSTREAM_MUX_BUFFER_INDEX_ADDR_START_IDX, NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_flow_control_semaphore_addr =
    fill_array_with_next_n_args<
        size_t,
        DOWNSTREAM_MUX_FLOW_CONTROL_SEMAPHORE_ADDR_START_IDX,
        NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_teardown_semaphore_addr =
    fill_array_with_next_n_args<
        size_t,
        DOWNSTREAM_MUX_TEARDOWN_SEMAPHORE_ADDR_START_IDX,
        NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_buffer_index_semaphore_addr =
    fill_array_with_next_n_args<
        size_t,
        DOWNSTREAM_MUX_BUFFER_INDEX_SEMAPHORE_ADDR_START_IDX,
        NUM_DOWNSTREAM_MUX_CONNECTIONS>();
constexpr std::array<size_t, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_free_slots_stream_id =
    fill_array_with_next_n_args<
        size_t,
        DOWNSTREAM_MUX_FREE_SLOTS_STREAM_ID_START_IDX,
        NUM_DOWNSTREAM_MUX_CONNECTIONS>();

// A channel is a "traffic injection channel" if it is a sender channel that is adding *new*
// traffic to this dimension/ring. Examples include channels service worker traffic and
// sender channels that receive traffic from a "turn" (e.g. an EAST channel receiving traffic from NORTH)
// This attribute is necessary to support bubble flow control.
// Bubble flow control flag comes after downstream mux arrays, before upstream routers
constexpr uint32_t BUBBLE_FLOW_CONTROL_START_IDX =
    DOWNSTREAM_MUX_FREE_SLOTS_STREAM_ID_START_IDX + NUM_DOWNSTREAM_MUX_CONNECTIONS;
constexpr bool enable_bubble_flow_control = get_compile_time_arg_val(BUBBLE_FLOW_CONTROL_START_IDX) != 0;
constexpr size_t BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS = 2;

// Upstream routers and sync address (computed incrementally from bubble flow control)
constexpr size_t NUM_UPSTREAM_ROUTERS_IDX = BUBBLE_FLOW_CONTROL_START_IDX + 1;
constexpr size_t num_upstream_routers = get_compile_time_arg_val(NUM_UPSTREAM_ROUTERS_IDX);
constexpr size_t fabric_router_sync_address = get_compile_time_arg_val(NUM_UPSTREAM_ROUTERS_IDX + 1);

constexpr size_t CHANNEL_STREAM_IDS_START_IDX = NUM_UPSTREAM_ROUTERS_IDX + 2;

constexpr size_t NOC_ALIGN_PADDING_BYTES = 12;

// ========== Channel type indices (must match ChannelTypes enum order in host) ==========
// In UDM mux mode:
constexpr uint32_t WORKER_CHANNEL_TYPE_IDX = 0;        // WORKER_CHANNEL
constexpr uint32_t RELAY_TO_MUX_CHANNEL_TYPE_IDX = 1;  // RELAY_TO_MUX_CHANNEL
constexpr uint32_t MUX_TO_MUX_CHANNEL_TYPE_IDX = 2;    // MUX_TO_MUX_CHANNEL

// Extract per-type configuration
constexpr uint32_t NUM_WORKER_CHANNELS = num_channels_per_type[WORKER_CHANNEL_TYPE_IDX];
constexpr uint32_t NUM_RELAY_TO_MUX_CHANNELS = num_channels_per_type[RELAY_TO_MUX_CHANNEL_TYPE_IDX];
constexpr uint32_t NUM_MUX_TO_MUX_CHANNELS = num_channels_per_type[MUX_TO_MUX_CHANNEL_TYPE_IDX];

constexpr uint32_t NUM_BUFFERS_WORKER = num_buffers_per_type[WORKER_CHANNEL_TYPE_IDX];
constexpr uint32_t NUM_BUFFERS_RELAY_TO_MUX = num_buffers_per_type[RELAY_TO_MUX_CHANNEL_TYPE_IDX];
constexpr uint32_t NUM_BUFFERS_MUX_TO_MUX = num_buffers_per_type[MUX_TO_MUX_CHANNEL_TYPE_IDX];

constexpr size_t BUFFER_SIZE_WORKER = buffer_size_per_type[WORKER_CHANNEL_TYPE_IDX];
constexpr size_t BUFFER_SIZE_RELAY_TO_MUX = buffer_size_per_type[RELAY_TO_MUX_CHANNEL_TYPE_IDX];
constexpr size_t BUFFER_SIZE_MUX_TO_MUX = buffer_size_per_type[MUX_TO_MUX_CHANNEL_TYPE_IDX];

namespace tt::tt_fabric {
using FabricMuxToEdmSender = WorkerToFabricEdmSenderImpl<false, NUM_EDM_BUFFERS>;
using FabricMuxToMuxSender = WorkerToFabricEdmSenderImpl<false, NUM_BUFFERS_MUX_TO_MUX>;
}  // namespace tt::tt_fabric

static_assert(noc_index == 0, "Mux kernel requires noc_index to be 0 so relay kernel can use 1");

template <uint8_t NUM_BUFFERS>
void wait_for_static_connection_to_ready(
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface) {
    while (!connect_is_requested(*worker_interface.connection_live_semaphore)) {
        invalidate_l1_cache();
    }

    worker_interface.template cache_producer_noc_addr<true>();
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
    tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>* channel_ptr,
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>* worker_interface_ptr,
    bool& channel_connection_established,
    uint8_t channel_id,
    size_t buffer_size_bytes,
    size_t& channel_base_address,
    size_t& connection_info_address,
    size_t& connection_handshake_address,
    size_t& sender_flow_control_address,
    StreamId my_channel_free_slots_stream_id,
    bool is_persistent_channel) {
    new (channel_ptr) tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>(
        channel_base_address, buffer_size_bytes, sizeof(PACKET_HEADER_TYPE));
    channel_base_address += NUM_BUFFERS * buffer_size_bytes;
    init_ptr_val(my_channel_free_slots_stream_id, NUM_BUFFERS);

    auto connection_worker_info_ptr =
        reinterpret_cast<volatile tt::tt_fabric::FabricMuxChannelClientLocationInfo*>(connection_info_address);
    connection_info_address += sizeof(tt::tt_fabric::FabricMuxChannelClientLocationInfo);

    new (worker_interface_ptr) tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>(
        connection_worker_info_ptr,
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(sender_flow_control_address),
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_handshake_address),
        0 /* unused, sender_sync_noc_cmd_buf */,
        tt::tt_fabric::MUX_TO_WORKER_INTERFACE_STARTING_READ_COUNTER_VALUE);  // for udm mux, the initial read counter
                                                                              // is always 0
    sender_flow_control_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
    connection_handshake_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;

    channel_connection_established = false;
}

template <uint8_t NUM_BUFFERS, uint32_t Direction>
void forward_data(
    tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>& channel,
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface,
    tt::tt_fabric::FabricMuxToEdmSender& fabric_connection,
    std::array<tt::tt_fabric::FabricMuxToMuxSender, NUM_DOWNSTREAM_MUX_CONNECTIONS>& downstream_mux_connections,
    bool& channel_connection_established,
    StreamId my_channel_free_slots_stream_id,
    bool is_persistent_channel,
    bool is_injection_channel) {
    bool has_unsent_payload = get_ptr_val(my_channel_free_slots_stream_id.get()) != NUM_BUFFERS;
    bool send_packet = has_unsent_payload;
    if constexpr (enable_bubble_flow_control) {
        if (send_packet && is_injection_channel) {
            send_packet = fabric_connection
                              .edm_has_space_for_packet<BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS>();
        }
    }
    if (send_packet) {
        size_t buffer_address = channel.get_buffer_address(worker_interface.local_write_counter.get_buffer_index());
        invalidate_l1_cache();
        auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(buffer_address);

        bool can_forward = tt::tt_fabric::udm::forward_to_downstream_mux_or_local_router<Direction>(
            packet_header, fabric_connection, downstream_mux_connections);
        if (can_forward) {
            worker_interface.local_write_counter.increment();
            worker_interface.local_read_counter.increment();

            // not handling/processing acks for now, re-evaluate if needed
            increment_local_update_ptr_val(my_channel_free_slots_stream_id.get(), 1);

            noc_async_writes_flushed();

            if (is_persistent_channel) {
                worker_interface.notify_worker_of_read_counter_update();
            } else if (channel_connection_established) {
                worker_interface.notify_worker_of_read_counter_update();
            }
        }
    }

    if (!is_persistent_channel) {
        tt::tt_fabric::check_worker_connections<tt::tt_fabric::USE_DYNAMIC_CREDIT_ADDR, true>(
            worker_interface, channel_connection_established, my_channel_free_slots_stream_id.get());
    }
}

void kernel_main() {
    size_t rt_args_idx = 0;

    // In UDM mode, we don't need upstream router coordinates since the relay handles signaling
    // The mux only signals to the local fabric router via the fabric_connection

    auto status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_address);
    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::STARTED;

    // clear out memory regions
    auto num_regions_to_clear = get_arg_val<uint32_t>(rt_args_idx++);
    for (uint32_t i = 0; i < num_regions_to_clear; i++) {
        auto address = get_arg_val<uint32_t>(rt_args_idx++);
        auto size = get_arg_val<uint32_t>(rt_args_idx++);
        zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(address), size);
    }

    auto fabric_connection = tt::tt_fabric::FabricMuxToEdmSender::build_from_args<CORE_TYPE>(rt_args_idx);

    // ========== Create channel arrays grouped by type ==========
    // Relay-to-mux channels (RELAY_TO_MUX_CHANNEL)
    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_RELAY_TO_MUX>, NUM_RELAY_TO_MUX_CHANNELS>
        relay_to_mux_channels;
    std::array<
        tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS_RELAY_TO_MUX>,
        NUM_RELAY_TO_MUX_CHANNELS>
        relay_to_mux_channel_interfaces;
    std::array<bool, NUM_RELAY_TO_MUX_CHANNELS> relay_to_mux_channel_connection_established;

    // Mux-to-mux channels (MUX_TO_MUX_CHANNEL)
    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_MUX_TO_MUX>, NUM_MUX_TO_MUX_CHANNELS>
        mux_to_mux_channels;
    std::array<
        tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS_MUX_TO_MUX>,
        NUM_MUX_TO_MUX_CHANNELS>
        mux_to_mux_channel_interfaces;
    std::array<bool, NUM_MUX_TO_MUX_CHANNELS> mux_to_mux_channel_connection_established;

    // ========== Parse stream IDs and persistent flags (grouped by type) ==========
    constexpr std::array<uint32_t, NUM_WORKER_CHANNELS> worker_stream_ids =
        fill_array_with_next_n_args<uint32_t, CHANNEL_STREAM_IDS_START_IDX, NUM_WORKER_CHANNELS>();
    constexpr std::array<uint32_t, NUM_RELAY_TO_MUX_CHANNELS> relay_to_mux_stream_ids = fill_array_with_next_n_args<
        uint32_t,
        CHANNEL_STREAM_IDS_START_IDX + NUM_WORKER_CHANNELS,
        NUM_RELAY_TO_MUX_CHANNELS>();
    constexpr std::array<uint32_t, NUM_MUX_TO_MUX_CHANNELS> mux_to_mux_stream_ids = fill_array_with_next_n_args<
        uint32_t,
        CHANNEL_STREAM_IDS_START_IDX + NUM_WORKER_CHANNELS + NUM_RELAY_TO_MUX_CHANNELS,
        NUM_MUX_TO_MUX_CHANNELS>();

    constexpr size_t IS_PERSISTENT_CHANNELS_START_IDX = CHANNEL_STREAM_IDS_START_IDX + NUM_TOTAL_CHANNELS;
    constexpr std::array<uint32_t, NUM_WORKER_CHANNELS> worker_is_persistent =
        fill_array_with_next_n_args<uint32_t, IS_PERSISTENT_CHANNELS_START_IDX, NUM_WORKER_CHANNELS>();
    constexpr std::array<uint32_t, NUM_RELAY_TO_MUX_CHANNELS> relay_to_mux_is_persistent = fill_array_with_next_n_args<
        uint32_t,
        IS_PERSISTENT_CHANNELS_START_IDX + NUM_WORKER_CHANNELS,
        NUM_RELAY_TO_MUX_CHANNELS>();
    constexpr std::array<uint32_t, NUM_MUX_TO_MUX_CHANNELS> mux_to_mux_is_persistent = fill_array_with_next_n_args<
        uint32_t,
        IS_PERSISTENT_CHANNELS_START_IDX + NUM_WORKER_CHANNELS + NUM_RELAY_TO_MUX_CHANNELS,
        NUM_MUX_TO_MUX_CHANNELS>();

    // Relay termination signal address (for mux to signal relay during teardown)
    constexpr size_t RELAY_TERMINATION_SIGNAL_IDX = IS_PERSISTENT_CHANNELS_START_IDX + NUM_TOTAL_CHANNELS;
    constexpr size_t relay_termination_signal_address = get_compile_time_arg_val(RELAY_TERMINATION_SIGNAL_IDX);

    // Direction
    constexpr size_t direction = get_compile_time_arg_val(RELAY_TERMINATION_SIGNAL_IDX + 1);

    // Whether this mux has a fabric router to connect to
    // False for missing directions (inter-mux forwarding only, no actual router)
    constexpr bool has_fabric_router = get_compile_time_arg_val(RELAY_TERMINATION_SIGNAL_IDX + 2) == 1;

    // Channel storage address (L1 address for storing worker channel arrays only)
    constexpr size_t channel_storage_base_address = get_compile_time_arg_val(RELAY_TERMINATION_SIGNAL_IDX + 3);

    // ========== Set up L1-based storage pointers for worker channels only ==========
    constexpr size_t worker_channels_storage_size =
        NUM_WORKER_CHANNELS * sizeof(tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_WORKER>);
    constexpr size_t worker_interfaces_storage_size =
        NUM_WORKER_CHANNELS * sizeof(tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS_WORKER>);
    constexpr size_t total_worker_storage_size = worker_channels_storage_size + worker_interfaces_storage_size;

    // Verify 4KB is enough for worker channel storage
    static_assert(total_worker_storage_size <= 4096, "Worker channel storage exceeds 4KB L1 allocation");

    size_t storage_offset = channel_storage_base_address;
    auto worker_channels = reinterpret_cast<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_WORKER>*>(storage_offset);
    storage_offset += worker_channels_storage_size;
    auto worker_channel_interfaces =
        reinterpret_cast<tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS_WORKER>*>(
            storage_offset);

    // Worker channel connection status (stack-allocated)
    bool worker_channel_connection_established[NUM_WORKER_CHANNELS];

    // ========== Setup worker channels (WORKER_CHANNEL) ==========
    size_t worker_channel_base_address = channel_buffer_base_addrs[WORKER_CHANNEL_TYPE_IDX];
    size_t worker_connection_info_address = connection_info_base_addrs[WORKER_CHANNEL_TYPE_IDX];
    size_t worker_connection_handshake_address = connection_handshake_base_addrs[WORKER_CHANNEL_TYPE_IDX];
    size_t worker_flow_control_address = flow_control_base_addrs[WORKER_CHANNEL_TYPE_IDX];

    for (uint32_t i = 0; i < NUM_WORKER_CHANNELS; i++) {
        setup_channel<NUM_BUFFERS_WORKER>(
            &worker_channels[i],
            &worker_channel_interfaces[i],
            worker_channel_connection_established[i],
            i,
            BUFFER_SIZE_WORKER,
            worker_channel_base_address,
            worker_connection_info_address,
            worker_connection_handshake_address,
            worker_flow_control_address,
            StreamId{worker_stream_ids[i]},
            worker_is_persistent[i] == 1);
    }

    // ========== Setup relay-to-mux channels (RELAY_TO_MUX_CHANNEL) ==========
    size_t relay_to_mux_channel_base_address = channel_buffer_base_addrs[RELAY_TO_MUX_CHANNEL_TYPE_IDX];
    size_t relay_to_mux_connection_info_address = connection_info_base_addrs[RELAY_TO_MUX_CHANNEL_TYPE_IDX];
    size_t relay_to_mux_connection_handshake_address = connection_handshake_base_addrs[RELAY_TO_MUX_CHANNEL_TYPE_IDX];
    size_t relay_to_mux_flow_control_address = flow_control_base_addrs[RELAY_TO_MUX_CHANNEL_TYPE_IDX];

    for (uint32_t i = 0; i < NUM_RELAY_TO_MUX_CHANNELS; i++) {
        setup_channel<NUM_BUFFERS_RELAY_TO_MUX>(
            &relay_to_mux_channels[i],
            &relay_to_mux_channel_interfaces[i],
            relay_to_mux_channel_connection_established[i],
            i,
            BUFFER_SIZE_RELAY_TO_MUX,
            relay_to_mux_channel_base_address,
            relay_to_mux_connection_info_address,
            relay_to_mux_connection_handshake_address,
            relay_to_mux_flow_control_address,
            StreamId{relay_to_mux_stream_ids[i]},
            relay_to_mux_is_persistent[i] == 1);
    }

    // ========== Setup mux-to-mux channels (MUX_TO_MUX_CHANNEL) ==========
    size_t mux_to_mux_channel_base_address = channel_buffer_base_addrs[MUX_TO_MUX_CHANNEL_TYPE_IDX];
    size_t mux_to_mux_connection_info_address = connection_info_base_addrs[MUX_TO_MUX_CHANNEL_TYPE_IDX];
    size_t mux_to_mux_connection_handshake_address = connection_handshake_base_addrs[MUX_TO_MUX_CHANNEL_TYPE_IDX];
    size_t mux_to_mux_flow_control_address = flow_control_base_addrs[MUX_TO_MUX_CHANNEL_TYPE_IDX];

    for (uint32_t i = 0; i < NUM_MUX_TO_MUX_CHANNELS; i++) {
        setup_channel<NUM_BUFFERS_MUX_TO_MUX>(
            &mux_to_mux_channels[i],
            &mux_to_mux_channel_interfaces[i],
            mux_to_mux_channel_connection_established[i],
            i,
            BUFFER_SIZE_MUX_TO_MUX,
            mux_to_mux_channel_base_address,
            mux_to_mux_connection_info_address,
            mux_to_mux_connection_handshake_address,
            mux_to_mux_flow_control_address,
            StreamId{mux_to_mux_stream_ids[i]},
            mux_to_mux_is_persistent[i] == 1);
    }

    // Construct downstream mux connections similar to how relay kernel does it for mux_connections
    // Use placement new to construct only active connections
    std::array<tt::tt_fabric::FabricMuxToMuxSender, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_connections;

    // Create each downstream mux connection if active
    constexpr bool is_persistent = true;  // All active downstream mux connections are persistent

    for (uint32_t i = 0; i < NUM_DOWNSTREAM_MUX_CONNECTIONS; i++) {
        if (downstream_mux_active[i]) {
            new (&downstream_mux_connections[i]) tt::tt_fabric::FabricMuxToMuxSender(
                is_persistent,
                downstream_mux_noc_x[i],
                downstream_mux_noc_y[i],
                downstream_mux_buffer_base_addr[i],
                static_cast<uint8_t>(NUM_BUFFERS_MUX_TO_MUX),
                downstream_mux_connection_handshake_addr[i],
                downstream_mux_worker_location_info_addr[i],
                static_cast<uint16_t>(BUFFER_SIZE_MUX_TO_MUX),
                downstream_mux_buffer_index_addr[i],
                reinterpret_cast<volatile uint32_t*>(downstream_mux_flow_control_semaphore_addr[i]),
                reinterpret_cast<volatile uint32_t*>(downstream_mux_teardown_semaphore_addr[i]),
                downstream_mux_buffer_index_semaphore_addr[i],
                static_cast<uint32_t>(downstream_mux_free_slots_stream_id[i]),
                StreamId{static_cast<uint32_t>(downstream_mux_free_slots_stream_id[i]) /* unused */});
        }
    }

    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_address);

    // In UDM mode, mux does NOT signal upstream routers - the relay will do that
    // (upstream routers connect to relay, not mux)

    // Only wait for and open fabric router connection if we have a router
    if constexpr (has_fabric_router) {
        // wait for fabric router to be ready before setting up the connection
        if constexpr (wait_for_fabric_endpoint) {
            tt::tt_fabric::wait_for_fabric_endpoint_ready(
                fabric_connection.edm_noc_x,
                fabric_connection.edm_noc_y,
                fabric_router_status_address,
                local_fabric_router_status_address);
        }

        fabric_connection.open<false>();
    }

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC;

    // Before connecting to downstream muxes, wait for their status to turn into READY_FOR_TRAFFIC
    // Use status_address (our own status address) as the readback location
    for (uint32_t i = 0; i < NUM_DOWNSTREAM_MUX_CONNECTIONS; i++) {
        if (downstream_mux_active[i]) {
            wait_for_mux_endpoint_ready(
                downstream_mux_noc_x[i],
                downstream_mux_noc_y[i],
                status_address,                       // Read downstream mux's status (same structure as ours)
                local_fabric_router_status_address);  // Use as temporary readback buffer
        }
    }

    // Open all active downstream mux connections
    for (uint32_t i = 0; i < NUM_DOWNSTREAM_MUX_CONNECTIONS; i++) {
        if (downstream_mux_active[i]) {
            downstream_mux_connections[i].open();
        }
    }

    // Wait for persistent channels to be ready
    for (uint32_t i = 0; i < NUM_WORKER_CHANNELS; i++) {
        if (worker_is_persistent[i] == 1) {
            wait_for_static_connection_to_ready<NUM_BUFFERS_WORKER>(worker_channel_interfaces[i]);
        }
    }

    for (uint32_t i = 0; i < NUM_RELAY_TO_MUX_CHANNELS; i++) {
        if (relay_to_mux_is_persistent[i] == 1) {
            wait_for_static_connection_to_ready<NUM_BUFFERS_RELAY_TO_MUX>(relay_to_mux_channel_interfaces[i]);
        }
    }

    for (uint32_t i = 0; i < NUM_MUX_TO_MUX_CHANNELS; i++) {
        if (mux_to_mux_is_persistent[i] == 1) {
            wait_for_static_connection_to_ready<NUM_BUFFERS_MUX_TO_MUX>(mux_to_mux_channel_interfaces[i]);
        }
    }

    while (!got_immediate_termination_signal<true>(termination_signal_ptr)) {
        for (size_t i = 0; i < NUM_ITERS_BETWEEN_TEARDOWN_CHECKS; i++) {
            // Process worker channels (WORKER_CHANNEL)
            for (uint32_t channel_id = 0; channel_id < NUM_WORKER_CHANNELS; channel_id++) {
                constexpr bool is_injection_channel = true;
                forward_data<NUM_BUFFERS_WORKER, direction>(
                    worker_channels[channel_id],
                    worker_channel_interfaces[channel_id],
                    fabric_connection,
                    downstream_mux_connections,
                    worker_channel_connection_established[channel_id],
                    StreamId{worker_stream_ids[channel_id]},
                    worker_is_persistent[channel_id] == 1,
                    is_injection_channel);
            }

            // Process relay-to-mux channels (RELAY_TO_MUX_CHANNEL)
            for (uint32_t channel_id = 0; channel_id < NUM_RELAY_TO_MUX_CHANNELS; channel_id++) {
                constexpr bool is_injection_channel = true;
                forward_data<NUM_BUFFERS_RELAY_TO_MUX, direction>(
                    relay_to_mux_channels[channel_id],
                    relay_to_mux_channel_interfaces[channel_id],
                    fabric_connection,
                    downstream_mux_connections,
                    relay_to_mux_channel_connection_established[channel_id],
                    StreamId{relay_to_mux_stream_ids[channel_id]},
                    relay_to_mux_is_persistent[channel_id] == 1,
                    is_injection_channel);
            }

            // Process mux-to-mux channels (MUX_TO_MUX_CHANNEL)
            for (uint32_t channel_id = 0; channel_id < NUM_MUX_TO_MUX_CHANNELS; channel_id++) {
                constexpr bool is_injection_channel = true;
                forward_data<NUM_BUFFERS_MUX_TO_MUX, direction>(
                    mux_to_mux_channels[channel_id],
                    mux_to_mux_channel_interfaces[channel_id],
                    fabric_connection,
                    downstream_mux_connections,
                    mux_to_mux_channel_connection_established[channel_id],
                    StreamId{mux_to_mux_stream_ids[channel_id]},
                    mux_to_mux_is_persistent[channel_id] == 1,
                    is_injection_channel);
            }
        }
    }

    // Signal relay to terminate (mux and relay are on the same core, so just write to L1 directly)
    volatile auto relay_termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(relay_termination_signal_address);
    *relay_termination_signal_ptr = tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE;

    if constexpr (has_fabric_router) {
        fabric_connection.close();
    }
    noc_async_write_barrier();
    noc_async_posted_writes_flushed();
    noc_async_atomic_barrier();

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
}
