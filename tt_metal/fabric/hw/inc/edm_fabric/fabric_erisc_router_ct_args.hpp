// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_trimming.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_code_profiling.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_static_channels_ct_args.hpp"
#include "hostdev/fabric_telemetry_msgs.h"
#include "api/alignment.h"

#include <array>
#include <utility>

// Convenience macro for named compile-time arg lookup
#define NAMED_CT_ARG(name) get_named_compile_time_arg_val(name)

// CHANNEL CONSTANTS
// ETH TXQ SELECTION

constexpr size_t NUM_ROUTER_CARDINAL_DIRECTIONS = 4;

// ============================================================================
// Stream IDs (from named compile-time arguments)
// ============================================================================
constexpr uint32_t to_receiver_0_pkts_sent_id = NAMED_CT_ARG("TO_RECEIVER_0_PKTS_SENT_ID");
constexpr uint32_t to_receiver_1_pkts_sent_id = NAMED_CT_ARG("TO_RECEIVER_1_PKTS_SENT_ID");
constexpr uint32_t to_sender_0_pkts_acked_id = NAMED_CT_ARG("TO_SENDER_0_PKTS_ACKED_ID");
constexpr uint32_t to_sender_1_pkts_acked_id = NAMED_CT_ARG("TO_SENDER_1_PKTS_ACKED_ID");
constexpr uint32_t to_sender_2_pkts_acked_id = NAMED_CT_ARG("TO_SENDER_2_PKTS_ACKED_ID");
constexpr uint32_t to_sender_3_pkts_acked_id = NAMED_CT_ARG("TO_SENDER_3_PKTS_ACKED_ID");
constexpr uint32_t to_sender_0_pkts_completed_id = NAMED_CT_ARG("TO_SENDER_0_PKTS_COMPLETED_ID");
constexpr uint32_t to_sender_1_pkts_completed_id = NAMED_CT_ARG("TO_SENDER_1_PKTS_COMPLETED_ID");
constexpr uint32_t to_sender_2_pkts_completed_id = NAMED_CT_ARG("TO_SENDER_2_PKTS_COMPLETED_ID");
constexpr uint32_t to_sender_3_pkts_completed_id = NAMED_CT_ARG("TO_SENDER_3_PKTS_COMPLETED_ID");
constexpr uint32_t to_sender_4_pkts_completed_id = NAMED_CT_ARG("TO_SENDER_4_PKTS_COMPLETED_ID");
constexpr uint32_t to_sender_5_pkts_completed_id = NAMED_CT_ARG("TO_SENDER_5_PKTS_COMPLETED_ID");
constexpr uint32_t to_sender_6_pkts_completed_id = NAMED_CT_ARG("TO_SENDER_6_PKTS_COMPLETED_ID");
constexpr uint32_t to_sender_7_pkts_completed_id = NAMED_CT_ARG("TO_SENDER_7_PKTS_COMPLETED_ID");
constexpr uint32_t vc_0_free_slots_from_downstream_edge_1_stream_id =
    NAMED_CT_ARG("VC0_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_1_STREAM_ID");
constexpr uint32_t vc_0_free_slots_from_downstream_edge_2_stream_id =
    NAMED_CT_ARG("VC0_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_2_STREAM_ID");
constexpr uint32_t vc_0_free_slots_from_downstream_edge_3_stream_id =
    NAMED_CT_ARG("VC0_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_3_STREAM_ID");
constexpr uint32_t vc_0_free_slots_from_downstream_edge_4_stream_id =
    NAMED_CT_ARG("VC0_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_4_STREAM_ID");
constexpr uint32_t vc_1_free_slots_from_downstream_edge_1_stream_id =
    NAMED_CT_ARG("VC1_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_1_STREAM_ID");
constexpr uint32_t vc_1_free_slots_from_downstream_edge_2_stream_id =
    NAMED_CT_ARG("VC1_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_2_STREAM_ID");
constexpr uint32_t vc_1_free_slots_from_downstream_edge_3_stream_id =
    NAMED_CT_ARG("VC1_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_3_STREAM_ID");
constexpr uint32_t vc_1_free_slots_from_downstream_edge_4_stream_id =
    NAMED_CT_ARG("VC1_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_4_STREAM_ID");
constexpr uint32_t sender_channel_0_free_slots_stream_id = NAMED_CT_ARG("SENDER_CHANNEL_0_FREE_SLOTS_STREAM_ID");
constexpr uint32_t sender_channel_1_free_slots_stream_id = NAMED_CT_ARG("SENDER_CHANNEL_1_FREE_SLOTS_STREAM_ID");
constexpr uint32_t sender_channel_2_free_slots_stream_id = NAMED_CT_ARG("SENDER_CHANNEL_2_FREE_SLOTS_STREAM_ID");
constexpr uint32_t sender_channel_3_free_slots_stream_id = NAMED_CT_ARG("SENDER_CHANNEL_3_FREE_SLOTS_STREAM_ID");
constexpr uint32_t sender_channel_4_free_slots_stream_id = NAMED_CT_ARG("SENDER_CHANNEL_4_FREE_SLOTS_STREAM_ID");
constexpr uint32_t sender_channel_5_free_slots_stream_id = NAMED_CT_ARG("SENDER_CHANNEL_5_FREE_SLOTS_STREAM_ID");
constexpr uint32_t sender_channel_6_free_slots_stream_id = NAMED_CT_ARG("SENDER_CHANNEL_6_FREE_SLOTS_STREAM_ID");
constexpr uint32_t sender_channel_7_free_slots_stream_id = NAMED_CT_ARG("SENDER_CHANNEL_7_FREE_SLOTS_STREAM_ID");
constexpr uint32_t tensix_relay_local_free_slots_stream_id = NAMED_CT_ARG("TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID");
constexpr uint32_t MULTI_RISC_TEARDOWN_SYNC_STREAM_ID = NAMED_CT_ARG("MULTI_RISC_TEARDOWN_SYNC_STREAM_ID");
constexpr uint32_t ETH_RETRAIN_LINK_SYNC_STREAM_ID = NAMED_CT_ARG("ETH_RETRAIN_LINK_SYNC_STREAM_ID");

// ============================================================================
// Maximum channel counts
// ============================================================================
constexpr size_t MAX_NUM_SENDER_CHANNELS = NAMED_CT_ARG("MAX_NUM_SENDER_CHANNELS");
constexpr size_t MAX_NUM_RECEIVER_CHANNELS = NAMED_CT_ARG("MAX_NUM_RECEIVER_CHANNELS");
// VC0 and VC1 channel counts depend on router type:
// Z_ROUTER: 5 VC0 + 4 VC1 = 9 total
// MESH: 4 VC0 + 4 VC1 = 8 total (with some unused)
constexpr size_t MAX_NUM_SENDER_CHANNELS_VC0 = (MAX_NUM_SENDER_CHANNELS >= 9) ? 5 : 4;
constexpr size_t MAX_NUM_SENDER_CHANNELS_VC1 = MAX_NUM_SENDER_CHANNELS - MAX_NUM_SENDER_CHANNELS_VC0;
constexpr size_t VC1_SENDER_CHANNEL_START = MAX_NUM_SENDER_CHANNELS_VC0;

// ============================================================================
// Downstream tensix connections
// ============================================================================
constexpr uint32_t num_ds_or_local_tensix_connections = NAMED_CT_ARG("NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS");

// ============================================================================
// Main configuration
// ============================================================================
constexpr size_t NUM_SENDER_CHANNELS = NAMED_CT_ARG("NUM_SENDER_CHANNELS");
constexpr size_t NUM_RECEIVER_CHANNELS = NAMED_CT_ARG("NUM_RECEIVER_CHANNELS");
constexpr size_t NUM_DOWNSTREAM_CHANNELS = NAMED_CT_ARG("NUM_DOWNSTREAM_CHANNELS");
constexpr size_t NUM_DOWNSTREAM_SENDERS_VC0 = NAMED_CT_ARG("NUM_DOWNSTREAM_SENDERS_VC0");
constexpr size_t NUM_DOWNSTREAM_SENDERS_VC1 = NAMED_CT_ARG("NUM_DOWNSTREAM_SENDERS_VC1");
constexpr bool wait_for_host_signal = NAMED_CT_ARG("WAIT_FOR_HOST_SIGNAL");

static_assert(
    NUM_RECEIVER_CHANNELS <= NUM_SENDER_CHANNELS,
    "NUM_RECEIVER_CHANNELS must be less than or equal to NUM_SENDER_CHANNELS");
static_assert(
    NUM_RECEIVER_CHANNELS <= MAX_NUM_RECEIVER_CHANNELS,
    "NUM_RECEIVER_CHANNELS must be less than or equal to MAX_NUM_RECEIVER_CHANNELS");
static_assert(
    NUM_SENDER_CHANNELS <= MAX_NUM_SENDER_CHANNELS,
    "NUM_SENDER_CHANNELS must be less than or equal to MAX_NUM_SENDER_CHANNELS");
static_assert(wait_for_host_signal == 0 || wait_for_host_signal == 1, "wait_for_host_signal must be 0 or 1");

constexpr uint32_t SWITCH_INTERVAL =
#ifndef DEBUG_PRINT_ENABLED
    NAMED_CT_ARG("SWITCH_INTERVAL");
#else
    0;
#endif
constexpr bool fuse_receiver_flush_and_completion_ptr = NAMED_CT_ARG("FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR");
constexpr bool enable_deadlock_avoidance = NAMED_CT_ARG("ENABLE_DEADLOCK_AVOIDANCE");
constexpr bool is_intermesh_router = NAMED_CT_ARG("IS_INTERMESH_ROUTER");
constexpr bool is_handshake_sender = NAMED_CT_ARG("IS_HANDSHAKE_SENDER") != 0;
constexpr size_t handshake_addr = NAMED_CT_ARG("HANDSHAKE_ADDR");

static_assert(fuse_receiver_flush_and_completion_ptr == 1, "fuse_receiver_flush_and_completion_ptr must be 0");

constexpr size_t VC0_RECEIVER_CHANNEL = 0;
constexpr size_t VC1_RECEIVER_CHANNEL = 1;

// Doesn't REALLY matter but for consistency I picked the next available ID
constexpr size_t worker_info_offset_past_connection_semaphore = 32;

// the size of one of the buffers within a sender channel
// For example if `channel_buffer_size` = 4k, with `SENDER_NUM_BUFFERS` = 2
// then the total amount of buffering for that channel is 8k.
// This `channel_buffer_size` includes packet headers in the size.
// This should be renamed to channel_slot_size_bytes to avoid confusion/ambiguity:
//   "is it the payload size or does it include the packet header size?"
constexpr size_t channel_buffer_size = NAMED_CT_ARG("CHANNEL_BUFFER_SIZE");
static_assert(channel_buffer_size <= UINT16_MAX, "channel_buffer_size exceeds uint16_t telemetry field");
constexpr bool fabric_tensix_extension_mux_mode = NAMED_CT_ARG("FABRIC_TENSIX_EXTENSION_MUX_MODE");
constexpr bool skip_src_ch_id_update = fabric_tensix_extension_mux_mode;

constexpr bool ENABLE_FIRST_LEVEL_ACK_VC0 = NAMED_CT_ARG("ENABLE_FIRST_LEVEL_ACK_VC0");
constexpr bool ENABLE_FIRST_LEVEL_ACK_VC1 = NAMED_CT_ARG("ENABLE_FIRST_LEVEL_ACK_VC1");
constexpr bool ENABLE_RISC_CPU_DATA_CACHE = NAMED_CT_ARG("ENABLE_RISC_CPU_DATA_CACHE");
constexpr bool z_router_enabled = NAMED_CT_ARG("Z_ROUTER_ENABLED");
constexpr size_t VC0_DOWNSTREAM_EDM_SIZE = NAMED_CT_ARG("VC0_DOWNSTREAM_EDM_SIZE");
constexpr size_t VC1_DOWNSTREAM_EDM_SIZE = NAMED_CT_ARG("VC1_DOWNSTREAM_EDM_SIZE");
constexpr size_t ACTUAL_VC0_SENDER_CHANNELS = NAMED_CT_ARG("ACTUAL_VC0_SENDER_CHANNELS");
constexpr size_t ACTUAL_VC1_SENDER_CHANNELS = NAMED_CT_ARG("ACTUAL_VC1_SENDER_CHANNELS");

// Remote channel info (always available; 0 when inactive)
constexpr size_t remote_worker_sender_channel = NAMED_CT_ARG("REMOTE_WORKER_SENDER_CHANNEL");

// UDM mode (always available; 0 when inactive)
constexpr bool udm_mode = NAMED_CT_ARG("UDM_MODE") != 0;
constexpr uint32_t LOCAL_RELAY_NUM_BUFFERS = NAMED_CT_ARG("LOCAL_RELAY_NUM_BUFFERS");

// ============================================================================
// Channel allocations (positional args, starting at index 0)
// ============================================================================
constexpr size_t CHANNEL_ALLOCATIONS_IDX = 0;

using channel_allocs = ChannelAllocations<CHANNEL_ALLOCATIONS_IDX, NUM_SENDER_CHANNELS, NUM_RECEIVER_CHANNELS>;

constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_TO_ENTRY_IDX = channel_allocs::sender_channel_to_entry_index;
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_TO_ENTRY_IDX =
    channel_allocs::receiver_channel_to_entry_index;

// Remote channel allocations (after local channels + marker)
constexpr size_t REMOTE_CHANNEL_START_MARKER_IDX = CHANNEL_ALLOCATIONS_IDX + channel_allocs::GET_NUM_ARGS_CONSUMED();
static_assert(
    get_compile_time_arg_val(REMOTE_CHANNEL_START_MARKER_IDX) == 0xabaddad6,
    "Remote channel start marker not found. This implies some arguments were misaligned between host and device. "
    "Double check the CT args.");

constexpr size_t REMOTE_CHANNEL_ALLOCATIONS_IDX = REMOTE_CHANNEL_START_MARKER_IDX + 1;
using eth_remote_channel_allocs = ChannelAllocations<REMOTE_CHANNEL_ALLOCATIONS_IDX, 0, NUM_RECEIVER_CHANNELS>;

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> REMOTE_RECEIVER_TO_ENTRY_IDX =
    eth_remote_channel_allocs::receiver_channel_to_entry_index;

// Downstream sender num buffers (after remote channel allocations + marker)
constexpr size_t DOWNSTREAM_SENDER_NUM_BUFFERS_SPECIAL_TAG_IDX =
    REMOTE_CHANNEL_ALLOCATIONS_IDX + eth_remote_channel_allocs::GET_NUM_ARGS_CONSUMED();
static_assert(
    get_compile_time_arg_val(DOWNSTREAM_SENDER_NUM_BUFFERS_SPECIAL_TAG_IDX) == 0xabaddad7,
    "DOWNSTREAM_SENDER_NUM_BUFFERS_SPECIAL_TAG_IDX not found. This implies some arguments were misaligned between host "
    "and device. Double check the CT args.");

// Downstream sender num buffers comes after channel mappings
// Sized to MAX_NUM_RECEIVER_CHANNELS (one per VC) to match host side array
constexpr size_t DOWNSTREAM_SENDER_NUM_BUFFERS_IDX = DOWNSTREAM_SENDER_NUM_BUFFERS_SPECIAL_TAG_IDX + 1;
constexpr std::array<size_t, MAX_NUM_RECEIVER_CHANNELS> DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY =
    fill_array_with_next_n_args<size_t, DOWNSTREAM_SENDER_NUM_BUFFERS_IDX, MAX_NUM_RECEIVER_CHANNELS>();
// TODO: remove DOWNSTREAM_SENDER_NUM_BUFFERS and use TMP on downstream sender channels.
constexpr size_t DOWNSTREAM_SENDER_NUM_BUFFERS_VC0 = DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY[0];
constexpr size_t DOWNSTREAM_SENDER_NUM_BUFFERS_VC1 = DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY[1];

constexpr size_t ANOTHER_SPECIAL_TAG_2 = 0xabaddad9;
constexpr size_t ANOTHER_SPECIAL_TAG_2_IDX = DOWNSTREAM_SENDER_NUM_BUFFERS_IDX + MAX_NUM_RECEIVER_CHANNELS;
static_assert(
    get_compile_time_arg_val(ANOTHER_SPECIAL_TAG_2_IDX) == ANOTHER_SPECIAL_TAG_2,
    "ANOTHER_SPECIAL_TAG_2 not found. This implies some arguments were misaligned between host and device. Double "
    "check the CT args.");

// ============================================================================
// Sender channel connection info addresses (named args)
// ============================================================================
constexpr size_t local_sender_channel_0_connection_info_addr = NAMED_CT_ARG("LOCAL_SENDER_CH_0_CONN_INFO_ADDR");
constexpr size_t local_sender_channel_1_connection_info_addr = NAMED_CT_ARG("LOCAL_SENDER_CH_1_CONN_INFO_ADDR");
constexpr size_t local_sender_channel_2_connection_info_addr = NAMED_CT_ARG("LOCAL_SENDER_CH_2_CONN_INFO_ADDR");
constexpr size_t local_sender_channel_3_connection_info_addr = NAMED_CT_ARG("LOCAL_SENDER_CH_3_CONN_INFO_ADDR");
constexpr size_t local_sender_channel_4_connection_info_addr = NAMED_CT_ARG("LOCAL_SENDER_CH_4_CONN_INFO_ADDR");
constexpr size_t local_sender_channel_5_connection_info_addr = NAMED_CT_ARG("LOCAL_SENDER_CH_5_CONN_INFO_ADDR");
constexpr size_t local_sender_channel_6_connection_info_addr = NAMED_CT_ARG("LOCAL_SENDER_CH_6_CONN_INFO_ADDR");
constexpr size_t local_sender_channel_7_connection_info_addr = NAMED_CT_ARG("LOCAL_SENDER_CH_7_CONN_INFO_ADDR");
constexpr size_t local_sender_channel_8_connection_info_addr = NAMED_CT_ARG("LOCAL_SENDER_CH_8_CONN_INFO_ADDR");

// ============================================================================
// Status pointers
// ============================================================================
// TODO: CONVERT TO SEMAPHORE
constexpr uint32_t termination_signal_addr = NAMED_CT_ARG("TERMINATION_SIGNAL_ADDR");
constexpr uint32_t edm_local_sync_ptr_addr = wait_for_host_signal ? NAMED_CT_ARG("EDM_LOCAL_SYNC_PTR_ADDR") : 0;
constexpr uint32_t edm_local_tensix_sync_ptr_addr = NAMED_CT_ARG("EDM_LOCAL_TENSIX_SYNC_PTR_ADDR");
constexpr uint32_t edm_status_ptr_addr = NAMED_CT_ARG("EDM_STATUS_PTR_ADDR");

// for blackhole we need to disable the noc flush in inline writes to L1 for better perf. For wormhole this flag is not
// used.
constexpr bool enable_read_counter_update_noc_flush = false;
constexpr uint32_t notify_worker_of_read_counter_update_src_address =
    NAMED_CT_ARG("NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR");

// ============================================================================
// Channel servicing flags
// ============================================================================
constexpr std::array<bool, MAX_NUM_SENDER_CHANNELS> is_sender_channel_serviced = {
    static_cast<bool>(NAMED_CT_ARG("IS_SENDER_CHANNEL_0_SERVICED")),
    static_cast<bool>(NAMED_CT_ARG("IS_SENDER_CHANNEL_1_SERVICED")),
    static_cast<bool>(NAMED_CT_ARG("IS_SENDER_CHANNEL_2_SERVICED")),
    static_cast<bool>(NAMED_CT_ARG("IS_SENDER_CHANNEL_3_SERVICED")),
    static_cast<bool>(NAMED_CT_ARG("IS_SENDER_CHANNEL_4_SERVICED")),
    static_cast<bool>(NAMED_CT_ARG("IS_SENDER_CHANNEL_5_SERVICED")),
    static_cast<bool>(NAMED_CT_ARG("IS_SENDER_CHANNEL_6_SERVICED")),
    static_cast<bool>(NAMED_CT_ARG("IS_SENDER_CHANNEL_7_SERVICED")),
    static_cast<bool>(NAMED_CT_ARG("IS_SENDER_CHANNEL_8_SERVICED")),
};
constexpr std::array<bool, MAX_NUM_RECEIVER_CHANNELS> is_receiver_channel_serviced = {
    static_cast<bool>(NAMED_CT_ARG("IS_RECEIVER_CHANNEL_0_SERVICED")),
    static_cast<bool>(NAMED_CT_ARG("IS_RECEIVER_CHANNEL_1_SERVICED")),
};

// ============================================================================
// RISC configuration
// ============================================================================
constexpr bool enable_ethernet_handshake = NAMED_CT_ARG("ENABLE_ETHERNET_HANDSHAKE") != 0;
constexpr bool enable_context_switch = NAMED_CT_ARG("ENABLE_CONTEXT_SWITCH") != 0;
constexpr bool enable_interrupts = NAMED_CT_ARG("ENABLE_INTERRUPTS") != 0;
constexpr size_t sender_txq_id = NAMED_CT_ARG("SENDER_TXQ_ID");
constexpr size_t receiver_txq_id = NAMED_CT_ARG("RECEIVER_TXQ_ID");
constexpr bool multi_txq_enabled = sender_txq_id != receiver_txq_id;

constexpr size_t iterations_between_ctx_switch_and_teardown_checks =
    NAMED_CT_ARG("ITERATIONS_BETWEEN_CTX_SWITCH_AND_TEARDOWN_CHECKS");
constexpr size_t is_2d_fabric = NAMED_CT_ARG("IS_2D_FABRIC");
constexpr size_t my_direction = NAMED_CT_ARG("MY_DIRECTION");
constexpr size_t num_eth_ports = NAMED_CT_ARG("NUM_ETH_PORTS");

// If true, the sender channel will spin inside send_next_data until the eth_txq is not busy, rather than checking
// eth_txq_is_busy() being false as a prerequisite for sending the next packet
constexpr bool ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA = NAMED_CT_ARG("ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA") != 0;
constexpr bool ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK =
    NAMED_CT_ARG("ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK") != 0;

constexpr size_t DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD =
    NAMED_CT_ARG("DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD");

// Context switch timeouts
constexpr size_t DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT =
#ifndef DEBUG_PRINT_ENABLED
    NAMED_CT_ARG("DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT");
#else
    128;
#endif
constexpr bool IDLE_CONTEXT_SWITCHING = NAMED_CT_ARG("IDLE_CONTEXT_SWITCHING") != 0;

constexpr size_t MY_ETH_CHANNEL = NAMED_CT_ARG("MY_ETH_CHANNEL");

constexpr size_t MY_ERISC_ID = NAMED_CT_ARG("MY_ERISC_ID");
constexpr size_t NUM_ACTIVE_ERISCS = NAMED_CT_ARG("NUM_ACTIVE_ERISCS");
static_assert(MY_ERISC_ID < NUM_ACTIVE_ERISCS, "MY_ERISC_ID must be less than NUM_ACTIVE_ERISCS");

// Defines if packet header updates (as the packet header traverses its route) are done on the receiver side or the
// sender side. If true, then the receiver channel updates the packet header before forwarding it. If false, the sender
// channel updates the packet header before sending it over Ethernet.
constexpr bool UPDATE_PKT_HDR_ON_RX_CH = NAMED_CT_ARG("UPDATE_PKT_HDR_ON_RX_CH") != 0;

constexpr bool FORCE_ALL_PATHS_TO_USE_SAME_NOC = NAMED_CT_ARG("FORCE_ALL_PATHS_TO_USE_SAME_NOC") != 0;

constexpr bool is_intermesh_router_on_edge = NAMED_CT_ARG("IS_INTERMESH_ROUTER_ON_EDGE") != 0;
constexpr bool is_intramesh_router_on_edge = NAMED_CT_ARG("IS_INTRAMESH_ROUTER_ON_EDGE") != 0;

// ============================================================================
// Sender channel per-channel arrays
// Arrays are sized to NUM_SENDER_CHANNELS, built from MAX-sized intermediaries.
// ============================================================================
static constexpr std::array<bool, MAX_NUM_SENDER_CHANNELS> sender_ch_live_check_skip_all_ = {
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_0_LIVE_CHECK_SKIP")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_1_LIVE_CHECK_SKIP")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_2_LIVE_CHECK_SKIP")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_3_LIVE_CHECK_SKIP")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_4_LIVE_CHECK_SKIP")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_5_LIVE_CHECK_SKIP")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_6_LIVE_CHECK_SKIP")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_7_LIVE_CHECK_SKIP")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_8_LIVE_CHECK_SKIP")),
};
constexpr std::array<bool, NUM_SENDER_CHANNELS> sender_ch_live_check_skip =
    take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, bool>(sender_ch_live_check_skip_all_);

// A channel is a "traffic injection channel" if it is a sender channel that is adding *new*
// traffic to this dimension/ring. Examples include channels service worker traffic and
// sender channels that receive traffic from a "turn" (e.g. an EAST channel receiving traffic from NORTH)
// This attribute is necessary to support bubble flow control.
static constexpr std::array<bool, MAX_NUM_SENDER_CHANNELS> sender_channel_is_traffic_injection_channel_all_ = {
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_0_IS_INJECTION")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_1_IS_INJECTION")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_2_IS_INJECTION")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_3_IS_INJECTION")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_4_IS_INJECTION")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_5_IS_INJECTION")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_6_IS_INJECTION")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_7_IS_INJECTION")),
    static_cast<bool>(NAMED_CT_ARG("SENDER_CH_8_IS_INJECTION")),
};
constexpr std::array<bool, NUM_SENDER_CHANNELS> sender_channel_is_traffic_injection_channel =
    take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, bool>(
        sender_channel_is_traffic_injection_channel_all_);
constexpr size_t BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS = 2;

static constexpr std::array<size_t, MAX_NUM_SENDER_CHANNELS> sender_channel_ack_noc_ids_all_ = {
    static_cast<size_t>(NAMED_CT_ARG("SENDER_CH_0_ACK_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("SENDER_CH_1_ACK_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("SENDER_CH_2_ACK_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("SENDER_CH_3_ACK_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("SENDER_CH_4_ACK_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("SENDER_CH_5_ACK_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("SENDER_CH_6_ACK_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("SENDER_CH_7_ACK_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("SENDER_CH_8_ACK_NOC_ID")),
};
constexpr std::array<size_t, NUM_SENDER_CHANNELS> sender_channel_ack_noc_ids =
    take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(sender_channel_ack_noc_ids_all_);

static constexpr std::array<uint8_t, MAX_NUM_SENDER_CHANNELS> sender_channel_ack_cmd_buf_ids_all_ = {
    static_cast<uint8_t>(NAMED_CT_ARG("SENDER_CH_0_ACK_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("SENDER_CH_1_ACK_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("SENDER_CH_2_ACK_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("SENDER_CH_3_ACK_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("SENDER_CH_4_ACK_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("SENDER_CH_5_ACK_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("SENDER_CH_6_ACK_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("SENDER_CH_7_ACK_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("SENDER_CH_8_ACK_CMD_BUF_ID")),
};
constexpr std::array<uint8_t, NUM_SENDER_CHANNELS> sender_channel_ack_cmd_buf_ids =
    take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, uint8_t>(sender_channel_ack_cmd_buf_ids_all_);

// ============================================================================
// Receiver channel per-channel arrays
// ============================================================================
static constexpr std::array<size_t, MAX_NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_noc_ids_all_ = {
    static_cast<size_t>(NAMED_CT_ARG("RX_CH_0_FWD_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("RX_CH_1_FWD_NOC_ID")),
};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_noc_ids =
    take_first_n_elements<NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, size_t>(
        receiver_channel_forwarding_noc_ids_all_);

static constexpr std::array<uint8_t, MAX_NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_data_cmd_buf_ids_all_ = {
    static_cast<uint8_t>(NAMED_CT_ARG("RX_CH_0_FWD_DATA_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("RX_CH_1_FWD_DATA_CMD_BUF_ID")),
};
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_data_cmd_buf_ids =
    take_first_n_elements<NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, uint8_t>(
        receiver_channel_forwarding_data_cmd_buf_ids_all_);

static constexpr std::array<uint8_t, MAX_NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_sync_cmd_buf_ids_all_ = {
    static_cast<uint8_t>(NAMED_CT_ARG("RX_CH_0_FWD_SYNC_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("RX_CH_1_FWD_SYNC_CMD_BUF_ID")),
};
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_sync_cmd_buf_ids =
    take_first_n_elements<NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, uint8_t>(
        receiver_channel_forwarding_sync_cmd_buf_ids_all_);

static constexpr std::array<size_t, MAX_NUM_RECEIVER_CHANNELS> receiver_channel_local_write_noc_ids_all_ = {
    static_cast<size_t>(NAMED_CT_ARG("RX_CH_0_LOCAL_WRITE_NOC_ID")),
    static_cast<size_t>(NAMED_CT_ARG("RX_CH_1_LOCAL_WRITE_NOC_ID")),
};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> receiver_channel_local_write_noc_ids =
    take_first_n_elements<NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, size_t>(
        receiver_channel_local_write_noc_ids_all_);

static constexpr std::array<uint8_t, MAX_NUM_RECEIVER_CHANNELS> receiver_channel_local_write_cmd_buf_ids_all_ = {
    static_cast<uint8_t>(NAMED_CT_ARG("RX_CH_0_LOCAL_WRITE_CMD_BUF_ID")),
    static_cast<uint8_t>(NAMED_CT_ARG("RX_CH_1_LOCAL_WRITE_CMD_BUF_ID")),
};
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> receiver_channel_local_write_cmd_buf_ids =
    take_first_n_elements<NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, uint8_t>(
        receiver_channel_local_write_cmd_buf_ids_all_);

// ============================================================================
// Telemetry
// ============================================================================
constexpr bool ENABLE_FABRIC_TELEMETRY = static_cast<bool>(NAMED_CT_ARG("ENABLE_FABRIC_TELEMETRY"));

constexpr uint8_t FABRIC_TELEMETRY_STATS_MASK = static_cast<uint8_t>(NAMED_CT_ARG("FABRIC_TELEMETRY_STATS_MASK"));
constexpr bool FABRIC_TELEMETRY_ROUTER_STATE =
    ENABLE_FABRIC_TELEMETRY &&
    ((FABRIC_TELEMETRY_STATS_MASK & static_cast<uint8_t>(DynamicStatistics::ROUTER_STATE)) != 0);
constexpr bool FABRIC_TELEMETRY_BANDWIDTH =
    ENABLE_FABRIC_TELEMETRY &&
    ((FABRIC_TELEMETRY_STATS_MASK & static_cast<uint8_t>(DynamicStatistics::BANDWIDTH)) != 0);
constexpr bool FABRIC_TELEMETRY_HEARTBEAT_TX =
    ENABLE_FABRIC_TELEMETRY &&
    ((FABRIC_TELEMETRY_STATS_MASK & static_cast<uint8_t>(DynamicStatistics::HEARTBEAT_TX)) != 0);
constexpr bool FABRIC_TELEMETRY_HEARTBEAT_RX =
    ENABLE_FABRIC_TELEMETRY &&
    ((FABRIC_TELEMETRY_STATS_MASK & static_cast<uint8_t>(DynamicStatistics::HEARTBEAT_RX)) != 0);
constexpr bool FABRIC_TELEMETRY_ANY_DYNAMIC_STAT = FABRIC_TELEMETRY_ROUTER_STATE || FABRIC_TELEMETRY_BANDWIDTH ||
                                                   FABRIC_TELEMETRY_HEARTBEAT_TX || FABRIC_TELEMETRY_HEARTBEAT_RX;

constexpr PerfTelemetryRecorderType perf_telemetry_mode =
    static_cast<PerfTelemetryRecorderType>(NAMED_CT_ARG("PERF_TELEMETRY_MODE"));

constexpr size_t perf_telemetry_buffer_addr = NAMED_CT_ARG("PERF_TELEMETRY_BUFFER_ADDR");

// ============================================================================
// Code Profiling
// ============================================================================
constexpr uint32_t code_profiling_enabled_timers_bitfield = NAMED_CT_ARG("CODE_PROFILING_ENABLED_TIMERS");
constexpr size_t code_profiling_buffer_base_addr = NAMED_CT_ARG("CODE_PROFILING_BUFFER_ADDR");

// ============================================================================
// Multi-TXQ credit counters (always available; 0 when inactive)
// ============================================================================
constexpr size_t to_sender_remote_ack_counters_base_address = NAMED_CT_ARG("TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR");
constexpr size_t to_sender_remote_completion_counters_base_address =
    NAMED_CT_ARG("TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR");

// To optimize for CPU bottleneck instructions, instead of sending acks individually, based on the specific credit
// addresses, the router instead will send all credits at once. This eliminates a handful of instructions per ack. This
// behaviour is completely safe when using these unbounded counter credits because the credits are unbounded unsigned
// counters. Any overflow materializes as a roll back to zero, and subtractions are safe with unsigned.
constexpr size_t to_senders_credits_base_address =
    std::min(to_sender_remote_ack_counters_base_address, to_sender_remote_completion_counters_base_address);

constexpr size_t local_receiver_ack_counters_base_address = NAMED_CT_ARG("LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR");
constexpr size_t local_receiver_completion_counters_base_address =
    NAMED_CT_ARG("LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR");

constexpr size_t local_receiver_credits_base_address =
    std::min(local_receiver_ack_counters_base_address, local_receiver_completion_counters_base_address);
// the two arrays are contiguous in memory. so we take the size of the first and then double it
constexpr size_t total_number_of_receiver_to_sender_credit_num_bytes =
    (std::max(local_receiver_ack_counters_base_address, local_receiver_completion_counters_base_address) -
     local_receiver_credits_base_address) *
    2;
static_assert(
    align_power_of_2(total_number_of_receiver_to_sender_credit_num_bytes, ETH_WORD_SIZE_BYTES) ==
        total_number_of_receiver_to_sender_credit_num_bytes,
    "total_number_of_receiver_to_sender_credit_num_bytes must be aligned to ETH_WORD_SIZE_BYTES");

static_assert(
    !multi_txq_enabled || to_sender_remote_ack_counters_base_address != 0,
    "to_sender_remote_ack_counters_base_address must be valid");
static_assert(
    !multi_txq_enabled || to_sender_remote_completion_counters_base_address != 0,
    "to_sender_remote_completion_counters_base_address must be valid");
static_assert(
    !multi_txq_enabled || local_receiver_ack_counters_base_address != 0,
    "local_receiver_ack_counters_base_address must be valid");
static_assert(
    !multi_txq_enabled || local_receiver_completion_counters_base_address != 0,
    "local_receiver_completion_counters_base_address must be valid");

// ============================================================================
// Host signal args (named; always available, 0 when wait_for_host_signal is false)
// ============================================================================
// TODO: Add type safe getter
constexpr bool is_local_handshake_master =
    wait_for_host_signal ? (NAMED_CT_ARG("IS_LOCAL_HANDSHAKE_MASTER") != 0) : false;
constexpr uint32_t local_handshake_master_eth_chan =
    wait_for_host_signal ? NAMED_CT_ARG("LOCAL_HANDSHAKE_MASTER_ETH_CHAN") : 0;
constexpr uint32_t num_local_edms = wait_for_host_signal ? NAMED_CT_ARG("NUM_LOCAL_EDMS") : 0;
constexpr uint32_t edm_channels_mask = wait_for_host_signal ? NAMED_CT_ARG("EDM_CHANNELS_MASK") : 0;

template <size_t SLOT_SIZE_BYTES, size_t PACKET_HEADER_SIZE_BYTES>
struct BufferSlot {
    static constexpr size_t size_bytes = SLOT_SIZE_BYTES;
    static constexpr size_t header_size_bytes = PACKET_HEADER_SIZE_BYTES;
    static constexpr size_t max_payload_size_bytes = size_bytes - header_size_bytes;
};

using buffer_slot = BufferSlot<channel_buffer_size, sizeof(PACKET_HEADER_TYPE)>;

constexpr size_t NUM_FORWARDED_SENDER_CHANNELS = NUM_SENDER_CHANNELS - 1;

//////////////////////////////////////////////////////////////////////////////////////////
////                CT ARGS FETCHING DONE
//////////////////////////////////////////////////////////////////////////////////////////

constexpr size_t sender_channel_base_id = 0;
constexpr size_t receiver_channel_base_id = NUM_SENDER_CHANNELS;

// TRANSACTION IDS
// TODO: Pass this value from host
constexpr uint8_t NUM_TRANSACTION_IDS = enable_deadlock_avoidance ? 8 : 4;

constexpr std::array<uint8_t, MAX_NUM_RECEIVER_CHANNELS> RX_CH_TRID_STARTS =
    initialize_receiver_channel_trid_starts<MAX_NUM_RECEIVER_CHANNELS, NUM_TRANSACTION_IDS>();

constexpr std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> to_receiver_packets_sent_streams =
    take_first_n_elements<MAX_NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS>{to_receiver_0_pkts_sent_id, to_receiver_1_pkts_sent_id});

// not in symbol table - because not used
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_packets_acked_streams =
    take_first_n_elements<MAX_NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>{
            // VC0
            to_sender_0_pkts_acked_id,
            to_sender_1_pkts_acked_id,
            to_sender_2_pkts_acked_id,
            to_sender_3_pkts_acked_id,
            // VC1
            0,  // Padding upto MAX_NUM_SENDER_CHANNELS. VC1 does not use first level acks.
            0,
            0,
            0});

// data section
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_packets_completed_streams =
    take_first_n_elements<MAX_NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>{
            to_sender_0_pkts_completed_id,
            to_sender_1_pkts_completed_id,
            to_sender_2_pkts_completed_id,
            to_sender_3_pkts_completed_id,
            to_sender_4_pkts_completed_id,
            to_sender_5_pkts_completed_id,
            to_sender_6_pkts_completed_id,
            to_sender_7_pkts_completed_id});

// Miscellaneous configuration

// TODO: move this to compile time args if we need to enable it
constexpr bool enable_trid_flush_check_on_noc_txn = false;

constexpr bool is_persistent_fabric = true;

namespace tt::tt_fabric {
static_assert(
    receiver_channel_local_write_noc_ids[0] == edm_to_local_chip_noc,
    "edm_to_local_chip_noc must equal to receiver_channel_local_write_noc_ids");
static constexpr uint8_t edm_to_downstream_noc = receiver_channel_forwarding_noc_ids[0];
#ifdef ARCH_BLACKHOLE
static constexpr uint8_t worker_handshake_noc = noc_index;
#else
static constexpr uint8_t worker_handshake_noc = sender_channel_ack_noc_ids[0];
#endif
constexpr bool local_chip_noc_equals_downstream_noc =
    receiver_channel_forwarding_noc_ids[0] == receiver_channel_local_write_noc_ids[0];
static constexpr uint8_t local_chip_data_cmd_buf = receiver_channel_local_write_cmd_buf_ids[0];
static constexpr uint8_t forward_and_local_write_noc_vc = NAMED_CT_ARG("EDM_NOC_VC");

// Helpers to extract num_slots from a channel's allocation entry
template <typename Allocs, auto& ChannelToEntryIndex, size_t ChannelIdx>
constexpr size_t get_channel_num_slots() {
    constexpr size_t entry_idx = ChannelToEntryIndex[ChannelIdx];
    return Allocs::template Entry<entry_idx>::num_slots;
}

template <typename Allocs, auto& ChannelToEntryIndex, size_t ChannelIdx>
constexpr size_t get_channel_remote_num_slots() {
    constexpr size_t entry_idx = ChannelToEntryIndex[ChannelIdx];
    return Allocs::template Entry<entry_idx>::remote_num_slots;
}

// Build arrays by inspecting each channel's allocation entry
template <typename Allocs, auto& ChannelToEntryIndex, size_t NumChannels, size_t... Indices>
constexpr std::array<size_t, NumChannels> build_num_slots_array_impl(std::index_sequence<Indices...>) {
    return {get_channel_num_slots<Allocs, ChannelToEntryIndex, Indices>()...};
}

template <typename Allocs, auto& ChannelToEntryIndex, size_t NumChannels>
constexpr std::array<size_t, NumChannels> build_num_slots_array() {
    return build_num_slots_array_impl<Allocs, ChannelToEntryIndex, NumChannels>(
        std::make_index_sequence<NumChannels>{});
}

template <typename Allocs, auto& ChannelToEntryIndex, size_t NumChannels, size_t... Indices>
constexpr std::array<size_t, NumChannels> build_remote_num_slots_array_impl(std::index_sequence<Indices...>) {
    return {get_channel_remote_num_slots<Allocs, ChannelToEntryIndex, Indices>()...};
}

template <typename Allocs, auto& ChannelToEntryIndex, size_t NumChannels>
constexpr std::array<size_t, NumChannels> build_remote_num_slots_array() {
    return build_remote_num_slots_array_impl<Allocs, ChannelToEntryIndex, NumChannels>(
        std::make_index_sequence<NumChannels>{});
}

constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_NUM_BUFFERS_ARRAY =
    build_num_slots_array<channel_allocs, SENDER_TO_ENTRY_IDX, NUM_SENDER_CHANNELS>();

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_NUM_BUFFERS_ARRAY =
    build_num_slots_array<channel_allocs, RECEIVER_TO_ENTRY_IDX, NUM_RECEIVER_CHANNELS>();

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> REMOTE_RECEIVER_NUM_BUFFERS_ARRAY = build_num_slots_array<
    eth_remote_channel_allocs,
    eth_remote_channel_allocs::receiver_channel_to_entry_index,
    NUM_RECEIVER_CHANNELS>();

}  // namespace tt::tt_fabric

//-------------------------------- Channel Trimming --------------------------------//
// channel trimming is a feature that allows the router to trim channels that are not used.
// this is useful for reducing the amount of compute needed by the router.

// RX channel forwarding disable flags (from imported trimming profile)
constexpr bool disable_rx_ch0_forwarding = get_named_compile_time_arg_val("DISABLE_RX_CH0_FORWARDING") != 0;
constexpr bool disable_rx_ch1_forwarding = get_named_compile_time_arg_val("DISABLE_RX_CH1_FORWARDING") != 0;
constexpr std::array<bool, MAX_NUM_RECEIVER_CHANNELS> is_receiver_channel_forwarding_disabled = {
    disable_rx_ch0_forwarding, disable_rx_ch1_forwarding};

constexpr bool ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE =
    get_named_compile_time_arg_val("ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE");
constexpr size_t RESOURCE_USAGE_CAPTURE_OUTPUT_L1_ADDRESS =
    ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE
        ? get_named_compile_time_arg_val("RESOURCE_USAGE_CAPTURE_OUTPUT_L1_ADDRESS")
        : 0;

using ChannelTrimmingUsagePtr = tt::tt_fabric::FabricDatapathUsageL1Ptr<
    ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE,
    RESOURCE_USAGE_CAPTURE_OUTPUT_L1_ADDRESS,
    MAX_NUM_RECEIVER_CHANNELS,
    MAX_NUM_SENDER_CHANNELS>;
constexpr ChannelTrimmingUsagePtr channel_trimming_usage_recorder{};

//-------------------------------- Credit Amortization --------------------------------//
constexpr uint32_t SENDER_CREDIT_AMORTIZATION_FREQUENCY =
    get_named_compile_time_arg_val("SENDER_CREDIT_AMORTIZATION_FREQUENCY");
constexpr uint32_t RECEIVER_CREDIT_AMORTIZATION_FREQUENCY =
    get_named_compile_time_arg_val("RECEIVER_CREDIT_AMORTIZATION_FREQUENCY");
constexpr bool super_speedy_mode =
    SENDER_CREDIT_AMORTIZATION_FREQUENCY > 0 && RECEIVER_CREDIT_AMORTIZATION_FREQUENCY > 0;
