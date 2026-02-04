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

static_assert(NUM_EDM_BUFFERS < 256, "NUM_EDM_BUFFERS too big, maybe CT args are misconfigured");
static_assert(
    NUM_FULL_SIZE_CHANNELS_ITERS < 256, "NUM_FULL_SIZE_CHANNELS_ITERS too big, maybe CT args are misconfigured");
static_assert(
    NUM_ITERS_BETWEEN_TEARDOWN_CHECKS < 256,
    "NUM_ITERS_BETWEEN_TEARDOWN_CHECKS too big, maybe CT args are misconfigured");

// ========== Per-channel-type arrays (7 arrays × NUM_CHANNEL_TYPES entries) ==========
// Legacy MUX mode has 2 channel types: WORKER_CHANNEL and ROUTER_CHANNEL
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
// In Legacy MUX mode, we have 2 channel types:
// [0] = WORKER_CHANNEL
// [1] = ROUTER_CHANNEL
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
// Note: Legacy MUX mode has NUM_DOWNSTREAM_MUX_CONNECTIONS = 0, so these arrays are empty
constexpr uint32_t DOWNSTREAM_MUX_ARRAYS_START_IDX = CHANNEL_BUFFER_BASE_ADDR_ARRAY_START_IDX + NUM_CHANNEL_TYPES;

// A channel is a "traffic injection channel" if it is a sender channel that is adding *new*
// traffic to this dimension/ring. Examples include channels service worker traffic and
// sender channels that receive traffic from a "turn" (e.g. an EAST channel receiving traffic from NORTH)
// This attribute is necessary to support bubble flow control.
constexpr uint32_t BUBBLE_FLOW_CONTROL_START_IDX = DOWNSTREAM_MUX_ARRAYS_START_IDX + (NUM_DOWNSTREAM_MUX_CONNECTIONS * 11);
constexpr bool enable_bubble_flow_control = get_compile_time_arg_val(BUBBLE_FLOW_CONTROL_START_IDX) != 0;
constexpr size_t BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS = 2;

// ========== Builder-added args (from MuxBuilder::get_compile_time_args) ==========
// Upstream routers and sync address follow after downstream mux arrays
constexpr size_t NUM_UPSTREAM_ROUTERS_IDX = BUBBLE_FLOW_CONTROL_START_IDX + 1;
constexpr size_t num_upstream_routers = get_compile_time_arg_val(NUM_UPSTREAM_ROUTERS_IDX);
constexpr size_t fabric_router_sync_address = get_compile_time_arg_val(NUM_UPSTREAM_ROUTERS_IDX + 1);

constexpr size_t CHANNEL_STREAM_IDS_START_IDX = NUM_UPSTREAM_ROUTERS_IDX + 2;

constexpr size_t NOC_ALIGN_PADDING_BYTES = 12;

// ========== Channel type indices (must match ChannelTypes enum order in host) ==========
// In Legacy MUX mode:
constexpr uint32_t WORKER_CHANNEL_TYPE_IDX = 0;  // WORKER_CHANNEL
constexpr uint32_t ROUTER_CHANNEL_TYPE_IDX = 1;  // ROUTER_CHANNEL

// Extract per-type configuration
constexpr uint32_t NUM_WORKER_CHANNELS = num_channels_per_type[WORKER_CHANNEL_TYPE_IDX];
constexpr uint32_t NUM_ROUTER_CHANNELS = num_channels_per_type[ROUTER_CHANNEL_TYPE_IDX];

constexpr uint32_t NUM_BUFFERS_WORKER = num_buffers_per_type[WORKER_CHANNEL_TYPE_IDX];
constexpr uint32_t NUM_BUFFERS_ROUTER = num_buffers_per_type[ROUTER_CHANNEL_TYPE_IDX];

constexpr size_t BUFFER_SIZE_WORKER = buffer_size_per_type[WORKER_CHANNEL_TYPE_IDX];
constexpr size_t BUFFER_SIZE_ROUTER = buffer_size_per_type[ROUTER_CHANNEL_TYPE_IDX];

constexpr bool ENABLE_RISC_CPU_DATA_CACHE = true;
namespace tt::tt_fabric {
using FabricMuxToEdmSender = WorkerToFabricEdmSenderImpl<false, NUM_EDM_BUFFERS>;
}  // namespace tt::tt_fabric

template <uint8_t NUM_BUFFERS>
void wait_for_static_connection_to_ready(
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface) {
    while (!connect_is_requested(*worker_interface.connection_live_semaphore)) {
        invalidate_l1_cache();
    }

    worker_interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE>();
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
        is_persistent_channel ? NUM_BUFFERS : tt::tt_fabric::MUX_TO_WORKER_INTERFACE_STARTING_READ_COUNTER_VALUE);  //
    sender_flow_control_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
    connection_handshake_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;

    channel_connection_established = false;
}

template <uint8_t NUM_BUFFERS>
void forward_data(
    tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>& channel,
    tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS>& worker_interface,
    tt::tt_fabric::FabricMuxToEdmSender& fabric_connection,
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

        fabric_connection.wait_for_empty_write_slot();

        fabric_connection.send_payload_flush_non_blocking_from_address(
            (uint32_t)packet_header, packet_header->get_payload_size_including_header());

        worker_interface.local_write_counter.increment();
        worker_interface.local_read_counter.increment();

        // not handling/processing acks for now, re-evaluate if needed
        increment_local_update_ptr_val(my_channel_free_slots_stream_id.get(), 1);

        noc_async_writes_flushed();
        if (is_persistent_channel) {
            constexpr bool enable_deadlock_avoidance = true;  // not used
            worker_interface.template notify_persistent_connection_of_free_space<enable_deadlock_avoidance>(1);
        } else if (channel_connection_established) {
            worker_interface.notify_worker_of_read_counter_update();
        }
    }

    if (!is_persistent_channel) {
        tt::tt_fabric::check_worker_connections<tt::tt_fabric::USE_DYNAMIC_CREDIT_ADDR, true>(
            worker_interface, channel_connection_established, my_channel_free_slots_stream_id.get());
    }
}

void kernel_main() {
    size_t rt_args_idx = 0;

    std::array<uint8_t, num_upstream_routers> upstream_noc_x;
    std::array<uint8_t, num_upstream_routers> upstream_noc_y;
    if constexpr (num_upstream_routers > 0) {
        for (uint32_t i = 0; i < num_upstream_routers; i++) {
            upstream_noc_x[i] = (uint8_t)get_arg_val<uint32_t>(rt_args_idx++);
        }
        for (uint32_t i = 0; i < num_upstream_routers; i++) {
            upstream_noc_y[i] = (uint8_t)get_arg_val<uint32_t>(rt_args_idx++);
        }
    }

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
    // Worker channels (WORKER_CHANNEL)
    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_WORKER>, NUM_WORKER_CHANNELS> worker_channels;
    std::array<tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS_WORKER>, NUM_WORKER_CHANNELS>
        worker_channel_interfaces;
    std::array<bool, NUM_WORKER_CHANNELS> worker_channel_connection_established;

    // Router channels (ROUTER_CHANNEL)
    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_ROUTER>, NUM_ROUTER_CHANNELS> router_channels;
    std::array<tt::tt_fabric::FabricMuxStaticSizedChannelWorkerInterface<NUM_BUFFERS_ROUTER>, NUM_ROUTER_CHANNELS>
        router_channel_interfaces;
    std::array<bool, NUM_ROUTER_CHANNELS> router_channel_connection_established;

    // ========== Parse stream IDs and persistent flags (grouped by type) ==========
    constexpr std::array<uint32_t, NUM_WORKER_CHANNELS> worker_stream_ids =
        fill_array_with_next_n_args<uint32_t, CHANNEL_STREAM_IDS_START_IDX, NUM_WORKER_CHANNELS>();
    constexpr std::array<uint32_t, NUM_ROUTER_CHANNELS> router_stream_ids = fill_array_with_next_n_args<
        uint32_t,
        CHANNEL_STREAM_IDS_START_IDX + NUM_WORKER_CHANNELS,
        NUM_ROUTER_CHANNELS>();

    constexpr size_t IS_PERSISTENT_CHANNELS_START_IDX = CHANNEL_STREAM_IDS_START_IDX + NUM_TOTAL_CHANNELS;
    constexpr std::array<uint32_t, NUM_WORKER_CHANNELS> worker_is_persistent =
        fill_array_with_next_n_args<uint32_t, IS_PERSISTENT_CHANNELS_START_IDX, NUM_WORKER_CHANNELS>();
    constexpr std::array<uint32_t, NUM_ROUTER_CHANNELS> router_is_persistent = fill_array_with_next_n_args<
        uint32_t,
        IS_PERSISTENT_CHANNELS_START_IDX + NUM_WORKER_CHANNELS,
        NUM_ROUTER_CHANNELS>();

    // ========== Injection status arrays (for bubble flow control in MUX mode) ==========
    constexpr size_t INJECTION_STATUS_START_IDX = IS_PERSISTENT_CHANNELS_START_IDX + NUM_TOTAL_CHANNELS;
    constexpr std::array<uint32_t, NUM_WORKER_CHANNELS> worker_channel_injection_status =
        fill_array_with_next_n_args<uint32_t, INJECTION_STATUS_START_IDX, NUM_WORKER_CHANNELS>();
    constexpr std::array<uint32_t, NUM_ROUTER_CHANNELS> router_channel_injection_status =
        fill_array_with_next_n_args<uint32_t, INJECTION_STATUS_START_IDX + NUM_WORKER_CHANNELS, NUM_ROUTER_CHANNELS>();

    // Direction (last compile-time argument, after injection status arrays)
    constexpr size_t direction = get_compile_time_arg_val(INJECTION_STATUS_START_IDX + NUM_TOTAL_CHANNELS);

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

    // ========== Setup router channels (ROUTER_CHANNEL) ==========
    size_t router_channel_base_address = channel_buffer_base_addrs[ROUTER_CHANNEL_TYPE_IDX];
    size_t router_connection_info_address = connection_info_base_addrs[ROUTER_CHANNEL_TYPE_IDX];
    size_t router_connection_handshake_address = connection_handshake_base_addrs[ROUTER_CHANNEL_TYPE_IDX];
    size_t router_flow_control_address = flow_control_base_addrs[ROUTER_CHANNEL_TYPE_IDX];

    for (uint32_t i = 0; i < NUM_ROUTER_CHANNELS; i++) {
        setup_channel<NUM_BUFFERS_ROUTER>(
            &router_channels[i],
            &router_channel_interfaces[i],
            router_channel_connection_established[i],
            i,
            BUFFER_SIZE_ROUTER,
            router_channel_base_address,
            router_connection_info_address,
            router_connection_handshake_address,
            router_flow_control_address,
            StreamId{router_stream_ids[i]},
            router_is_persistent[i] == 1);
    }

    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_address);

    // signal the upstream routers the mux is ready
    if constexpr (num_upstream_routers > 0) {
        for (uint32_t i = 0; i < num_upstream_routers; i++) {
            auto noc_addr = get_noc_addr(upstream_noc_x[i], upstream_noc_y[i], fabric_router_sync_address);
            noc_semaphore_inc(noc_addr, 1);
        }
    }

    // wait for fabric router to be ready before setting up the connection
    if constexpr (wait_for_fabric_endpoint) {
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_connection.edm_noc_x,
            fabric_connection.edm_noc_y,
            fabric_router_status_address,
            local_fabric_router_status_address);
    }

    constexpr bool use_worker_allocated_credit_address = CORE_TYPE == ProgrammableCoreType::IDLE_ETH;
    fabric_connection.open<use_worker_allocated_credit_address>();

    // Wait for persistent channels to be ready
    for (uint32_t i = 0; i < NUM_WORKER_CHANNELS; i++) {
        if (worker_is_persistent[i] == 1) {
            wait_for_static_connection_to_ready<NUM_BUFFERS_WORKER>(worker_channel_interfaces[i]);
        }
    }

    for (uint32_t i = 0; i < NUM_ROUTER_CHANNELS; i++) {
        if (router_is_persistent[i] == 1) {
            wait_for_static_connection_to_ready<NUM_BUFFERS_ROUTER>(router_channel_interfaces[i]);
        }
    }

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC;

#if defined(COMPILE_FOR_IDLE_ERISC)
    uint32_t heartbeat = 0;
#endif
    while (!got_immediate_termination_signal<true>(termination_signal_ptr)) {
        for (size_t i = 0; i < NUM_ITERS_BETWEEN_TEARDOWN_CHECKS; i++) {
            // Process worker channels (WORKER_CHANNEL)
            for (uint32_t channel_id = 0; channel_id < NUM_WORKER_CHANNELS; channel_id++) {
                forward_data<NUM_BUFFERS_WORKER>(
                    worker_channels[channel_id],
                    worker_channel_interfaces[channel_id],
                    fabric_connection,
                    worker_channel_connection_established[channel_id],
                    StreamId{worker_stream_ids[channel_id]},
                    worker_is_persistent[channel_id] == 1,
                    worker_channel_injection_status[channel_id]);
            }

            // Process router channels (ROUTER_CHANNEL)
            for (uint32_t channel_id = 0; channel_id < NUM_ROUTER_CHANNELS; channel_id++) {
                forward_data<NUM_BUFFERS_ROUTER>(
                    router_channels[channel_id],
                    router_channel_interfaces[channel_id],
                    fabric_connection,
                    router_channel_connection_established[channel_id],
                    StreamId{router_stream_ids[channel_id]},
                    router_is_persistent[channel_id] == 1,
                    router_channel_injection_status[channel_id]);
            }
        }
#if defined(COMPILE_FOR_IDLE_ERISC)
        RISC_POST_HEARTBEAT(heartbeat);
#endif
    }

    fabric_connection.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
}
