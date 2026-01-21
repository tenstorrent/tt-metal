// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_elastic_channels_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_code_profiling.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_static_channels_ct_args.hpp"
#include "hostdev/fabric_telemetry_msgs.h"
#include "api/alignment.h"

#include <array>
#include <utility>

// CHANNEL CONSTANTS
// ETH TXQ SELECTION

constexpr size_t NUM_ROUTER_CARDINAL_DIRECTIONS = 4;

constexpr bool SPECIAL_MARKER_CHECK_ENABLED = true;

// Stream IDs from compile time arguments (first 28 arguments)
constexpr size_t STREAM_ID_ARGS_START_IDX = 0;
constexpr uint32_t to_receiver_0_pkts_sent_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 0);
constexpr uint32_t to_receiver_1_pkts_sent_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 1);
constexpr uint32_t to_sender_0_pkts_acked_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 2);
constexpr uint32_t to_sender_1_pkts_acked_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 3);
constexpr uint32_t to_sender_2_pkts_acked_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 4);
constexpr uint32_t to_sender_3_pkts_acked_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 5);
constexpr uint32_t to_sender_0_pkts_completed_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 6);
constexpr uint32_t to_sender_1_pkts_completed_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 7);
constexpr uint32_t to_sender_2_pkts_completed_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 8);
constexpr uint32_t to_sender_3_pkts_completed_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 9);
constexpr uint32_t to_sender_4_pkts_completed_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 10);
constexpr uint32_t to_sender_5_pkts_completed_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 11);
constexpr uint32_t to_sender_6_pkts_completed_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 12);
constexpr uint32_t to_sender_7_pkts_completed_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 13);
constexpr uint32_t vc_0_free_slots_from_downstream_edge_1_stream_id =
    get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 14);
constexpr uint32_t vc_0_free_slots_from_downstream_edge_2_stream_id =
    get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 15);
constexpr uint32_t vc_0_free_slots_from_downstream_edge_3_stream_id =
    get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 16);
constexpr uint32_t vc_0_free_slots_from_downstream_edge_4_stream_id =
    get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 17);
constexpr uint32_t vc_1_free_slots_from_downstream_edge_1_stream_id =
    get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 18);
constexpr uint32_t vc_1_free_slots_from_downstream_edge_2_stream_id =
    get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 19);
constexpr uint32_t vc_1_free_slots_from_downstream_edge_3_stream_id =
    get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 20);
constexpr uint32_t vc_1_free_slots_from_downstream_edge_4_stream_id =
    get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 21);
constexpr uint32_t sender_channel_0_free_slots_stream_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 22);
constexpr uint32_t sender_channel_1_free_slots_stream_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 23);
constexpr uint32_t sender_channel_2_free_slots_stream_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 24);
constexpr uint32_t sender_channel_3_free_slots_stream_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 25);
constexpr uint32_t sender_channel_4_free_slots_stream_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 26);
constexpr uint32_t sender_channel_5_free_slots_stream_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 27);
constexpr uint32_t sender_channel_6_free_slots_stream_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 28);
constexpr uint32_t sender_channel_7_free_slots_stream_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 29);
constexpr uint32_t tensix_relay_local_free_slots_stream_id = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 30);
constexpr uint32_t MULTI_RISC_TEARDOWN_SYNC_STREAM_ID = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 31);
constexpr uint32_t ETH_RETRAIN_LINK_SYNC_STREAM_ID = get_compile_time_arg_val(STREAM_ID_ARGS_START_IDX + 32);

// Special marker after stream IDs
constexpr size_t STREAM_IDS_END_MARKER_IDX = STREAM_ID_ARGS_START_IDX + 33;
constexpr size_t STREAM_IDS_END_MARKER = 0xFFEE0001;
static_assert(
    !SPECIAL_MARKER_CHECK_ENABLED || get_compile_time_arg_val(STREAM_IDS_END_MARKER_IDX) == STREAM_IDS_END_MARKER,
    "Stream IDs end marker not found. This implies some arguments were misaligned between host and device. Double "
    "check the CT args.");

// Maximum channel counts (from builder_config::num_max_sender_channels and builder_config::num_max_receiver_channels)
constexpr size_t MAX_NUM_SENDER_CHANNELS_IDX = STREAM_IDS_END_MARKER_IDX + 1;
constexpr size_t MAX_NUM_SENDER_CHANNELS = get_compile_time_arg_val(MAX_NUM_SENDER_CHANNELS_IDX);
constexpr size_t MAX_NUM_RECEIVER_CHANNELS_IDX = MAX_NUM_SENDER_CHANNELS_IDX + 1;
constexpr size_t MAX_NUM_RECEIVER_CHANNELS = get_compile_time_arg_val(MAX_NUM_RECEIVER_CHANNELS_IDX);
// VC0 and VC1 channel counts depend on router type:
// Z_ROUTER: 5 VC0 + 4 VC1 = 9 total
// MESH: 4 VC0 + 4 VC1 = 8 total (with some unused)
// These are computed from MAX_NUM_SENDER_CHANNELS
constexpr size_t MAX_NUM_SENDER_CHANNELS_VC0 = (MAX_NUM_SENDER_CHANNELS >= 9) ? 5 : 4;  // 5 if Z router, else 4
constexpr size_t MAX_NUM_SENDER_CHANNELS_VC1 = MAX_NUM_SENDER_CHANNELS - MAX_NUM_SENDER_CHANNELS_VC0;  // Remainder
constexpr size_t VC1_SENDER_CHANNEL_START = MAX_NUM_SENDER_CHANNELS_VC0;

// Downstream tensix connections argument (after stream IDs, marker, and max channel counts)
constexpr size_t NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS_IDX = MAX_NUM_RECEIVER_CHANNELS_IDX + 1;
constexpr uint32_t num_ds_or_local_tensix_connections =
    get_compile_time_arg_val(NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS_IDX);

// Main configuration arguments (after stream IDs, marker, max channel counts, and downstream tensix connections)
constexpr size_t SENDER_CHANNEL_NOC_CONFIG_START_IDX = NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS_IDX + 1;
constexpr size_t NUM_SENDER_CHANNELS = get_compile_time_arg_val(SENDER_CHANNEL_NOC_CONFIG_START_IDX);
constexpr size_t NUM_RECEIVER_CHANNELS_CT_ARG_IDX = SENDER_CHANNEL_NOC_CONFIG_START_IDX + 1;
constexpr size_t NUM_RECEIVER_CHANNELS = get_compile_time_arg_val(NUM_RECEIVER_CHANNELS_CT_ARG_IDX);
constexpr size_t NUM_FORWARDING_PATHS_CT_ARG_IDX = NUM_RECEIVER_CHANNELS_CT_ARG_IDX + 1;
constexpr size_t NUM_DOWNSTREAM_CHANNELS = get_compile_time_arg_val(NUM_FORWARDING_PATHS_CT_ARG_IDX);
constexpr size_t NUM_DOWNSTREAM_SENDERS_VC0_CT_ARG_IDX = NUM_FORWARDING_PATHS_CT_ARG_IDX + 1;
constexpr size_t NUM_DOWNSTREAM_SENDERS_VC0 = get_compile_time_arg_val(NUM_DOWNSTREAM_SENDERS_VC0_CT_ARG_IDX);
constexpr size_t NUM_DOWNSTREAM_SENDERS_VC1_CT_ARG_IDX = NUM_DOWNSTREAM_SENDERS_VC0_CT_ARG_IDX + 1;
constexpr size_t NUM_DOWNSTREAM_SENDERS_VC1 = get_compile_time_arg_val(NUM_DOWNSTREAM_SENDERS_VC1_CT_ARG_IDX);
constexpr size_t wait_for_host_signal_IDX = NUM_DOWNSTREAM_SENDERS_VC1_CT_ARG_IDX + 1;
constexpr bool wait_for_host_signal = get_compile_time_arg_val(wait_for_host_signal_IDX);
constexpr size_t MAIN_CT_ARGS_START_IDX = wait_for_host_signal_IDX + 1;

static_assert(
    NUM_RECEIVER_CHANNELS <= NUM_SENDER_CHANNELS,
    "NUM_RECEIVER_CHANNELS must be less than or equal to NUM_SENDER_CHANNELS");
static_assert(
    NUM_RECEIVER_CHANNELS <= MAX_NUM_RECEIVER_CHANNELS,
    "NUM_RECEIVER_CHANNELS must be less than or equal to MAX_NUM_RECEIVER_CHANNELS");
static_assert(
    NUM_SENDER_CHANNELS <= MAX_NUM_SENDER_CHANNELS,
    "NUM_SENDER_CHANNELS must be less than or equal to MAX_NUM_SENDER_CHANNELS");
static_assert(
    wait_for_host_signal_IDX == 42,
    "wait_for_host_signal_IDX must be 41 (32 stream IDs + 1 marker + 2 max channel counts + 1 tensix connections + 6 "
    "config args: num_sender_channels, num_receiver_channels, num_fwd_paths, num_downstream_senders_vc0, "
    "num_downstream_senders_vc1, wait_for_host_signal)");
static_assert(
    get_compile_time_arg_val(wait_for_host_signal_IDX) == 0 || get_compile_time_arg_val(wait_for_host_signal_IDX) == 1,
    "wait_for_host_signal must be 0 or 1");
static_assert(
    MAIN_CT_ARGS_START_IDX == 43,
    "MAIN_CT_ARGS_START_IDX must be 42 (32 stream IDs + 1 marker + 2 max channel counts + 1 tensix connections + 6 "
    "config args: num_sender_channels, num_receiver_channels, num_fwd_paths, num_downstream_senders_vc0, "
    "num_downstream_senders_vc1, wait_for_host_signal)");

constexpr uint32_t SWITCH_INTERVAL =
#ifndef DEBUG_PRINT_ENABLED
    get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 0);
#else
    0;
#endif
constexpr bool fuse_receiver_flush_and_completion_ptr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 1);
constexpr bool enable_deadlock_avoidance = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 2);
constexpr bool is_intermesh_router = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 3);
constexpr bool is_handshake_sender = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 4) != 0;
constexpr size_t handshake_addr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 5);

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
constexpr size_t channel_buffer_size = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 6);
constexpr bool fabric_tensix_extension_mux_mode = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 7);
constexpr bool skip_src_ch_id_update = fabric_tensix_extension_mux_mode;

constexpr bool ENABLE_FIRST_LEVEL_ACK_VC0 = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 8);
constexpr bool ENABLE_FIRST_LEVEL_ACK_VC1 = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 9);
constexpr bool ENABLE_RISC_CPU_DATA_CACHE = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 10);
constexpr bool z_router_enabled = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 11);
constexpr size_t VC0_DOWNSTREAM_EDM_SIZE = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 12);
constexpr size_t VC1_DOWNSTREAM_EDM_SIZE = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 13);
constexpr size_t ACTUAL_VC0_SENDER_CHANNELS = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 14);
constexpr size_t ACTUAL_VC1_SENDER_CHANNELS = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 15);

constexpr size_t REMOTE_CHANNEL_INFO_START_IDX = MAIN_CT_ARGS_START_IDX + 16;
constexpr size_t remote_worker_sender_channel =
    conditional_get_compile_time_arg<skip_src_ch_id_update, REMOTE_CHANNEL_INFO_START_IDX>();

constexpr size_t UDM_MODE_IDX = REMOTE_CHANNEL_INFO_START_IDX + (skip_src_ch_id_update ? 1 : 0);
constexpr bool udm_mode = get_compile_time_arg_val(UDM_MODE_IDX) != 0;

constexpr size_t LOCAL_TENSIX_RELAY_INFO_START_IDX = UDM_MODE_IDX + 1;
constexpr uint32_t LOCAL_RELAY_NUM_BUFFERS =
    conditional_get_compile_time_arg<udm_mode, LOCAL_TENSIX_RELAY_INFO_START_IDX>();

constexpr size_t ANOTHER_SPECIAL_TAG = 0xabcd9876;
constexpr size_t ANOTHER_SPECIAL_TAG_IDX = LOCAL_TENSIX_RELAY_INFO_START_IDX + (udm_mode ? 1 : 0);
static_assert(
    get_compile_time_arg_val(ANOTHER_SPECIAL_TAG_IDX) == ANOTHER_SPECIAL_TAG,
    "ANOTHER_SPECIAL_TAG not found. This implies some arguments were misaligned between host and device. Double check the CT args.");

// ========== NEW MULTI-POOL CHANNEL PARSING ==========
// Parse channel pool collection (replaces old static-only parsing)
constexpr size_t CHANNEL_POOL_COLLECTION_IDX = ANOTHER_SPECIAL_TAG_IDX + 1;

// For now, we only support single static pool configuration for backward compatibility
// Pool CT args structure:
//   - num_pools (1 arg)
//   - pool_types[] (1 arg for single pool)
//   - Static pool data (23 args): 5 sender buffers + 2 receiver buffers + 2 remote receiver buffers +
//                                  5 sender addrs + 4 receiver addrs + 5 remote sender addrs
// Total: 1 + 1 + 23 = 25 args
using channel_pools_args =
    ChannelPoolCollection<CHANNEL_POOL_COLLECTION_IDX, NUM_SENDER_CHANNELS, NUM_RECEIVER_CHANNELS>;
constexpr size_t NUM_POOLS = channel_pools_args::num_channel_pools;
// Parse channel-to-pool mappings (after all pool data)
constexpr size_t CHANNEL_MAPPINGS_START_SPECIAL_TAG_IDX  = CHANNEL_POOL_COLLECTION_IDX + channel_pools_args::GET_NUM_ARGS_CONSUMED();
static_assert(
    get_compile_time_arg_val(CHANNEL_MAPPINGS_START_SPECIAL_TAG_IDX) == 0xabaddad8,
    "CHANNEL_MAPPINGS_START_SPECIAL_TAG_IDX not found. This implies some arguments were misaligned between host and device. Double check the CT args.");

constexpr size_t CHANNEL_MAPPINGS_START_IDX = CHANNEL_MAPPINGS_START_SPECIAL_TAG_IDX + 1;
constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_TO_POOL_IDX = channel_pools_args::sender_channel_to_pool_index;
constexpr std::array<FabricChannelPoolType, NUM_SENDER_CHANNELS> SENDER_TO_POOL_TYPE = fill_array_with_next_n_args<
    FabricChannelPoolType,
    CHANNEL_MAPPINGS_START_IDX + NUM_SENDER_CHANNELS,
    NUM_SENDER_CHANNELS>();
static_assert(all_elements_satisfy(SENDER_TO_POOL_TYPE, [](FabricChannelPoolType pool_type) { return pool_type <= FabricChannelPoolType::ELASTIC; }), "SENDER_TO_POOL_TYPE must be less than or equal to FabricChannelPoolType::ELASTIC");
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_TO_POOL_IDX = channel_pools_args::receiver_channel_to_pool_index;
static_assert(all_elements_satisfy(RECEIVER_TO_POOL_IDX, [](size_t pool_idx) { return pool_idx < NUM_POOLS; }), "RECEIVER_TO_POOL_IDX must be less than NUM_POOLS");
constexpr std::array<FabricChannelPoolType, NUM_RECEIVER_CHANNELS> RECEIVER_TO_POOL_TYPE = fill_array_with_next_n_args<
    FabricChannelPoolType,

    // We accidentally double emit the *_TO_POOL_TYPE arrays so we skip past some unused args
    CHANNEL_MAPPINGS_START_IDX + (2 * NUM_SENDER_CHANNELS) + NUM_RECEIVER_CHANNELS,
    NUM_RECEIVER_CHANNELS>();
static_assert(all_elements_satisfy(RECEIVER_TO_POOL_TYPE, [](FabricChannelPoolType pool_type) { return pool_type <= FabricChannelPoolType::ELASTIC; }));

// Parse remote channel pool data (after channel-to-pool mappings)
constexpr size_t REMOTE_CHANNEL_POOL_START_MARKER_IDX = CHANNEL_MAPPINGS_START_IDX + 2 * (NUM_SENDER_CHANNELS + NUM_RECEIVER_CHANNELS);
static_assert(
    get_compile_time_arg_val(REMOTE_CHANNEL_POOL_START_MARKER_IDX) == 0xabaddad6,
    "Remote channel pool start marker not found. This implies some arguments were misaligned between host and device. Double check the CT args.");

// Parse remote channel pool collection (follows same structure as local channels)
// The remote multi-pool allocator emits pool data in the same format as local channels
constexpr size_t REMOTE_CHANNEL_POOL_IDX = REMOTE_CHANNEL_POOL_START_MARKER_IDX + 1;
using eth_remote_channel_pools_args = ChannelPoolCollection<REMOTE_CHANNEL_POOL_IDX, 0, NUM_RECEIVER_CHANNELS>;

static constexpr size_t REMOTE_CHANNEL_MAPPINGS_START_IDX =
    REMOTE_CHANNEL_POOL_IDX + eth_remote_channel_pools_args::GET_NUM_ARGS_CONSUMED();
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> REMOTE_RECEIVER_TO_POOL_IDX =
    eth_remote_channel_pools_args::receiver_channel_to_pool_index;
constexpr size_t NUM_REMOTE_POOLS = eth_remote_channel_pools_args::num_channel_pools;
constexpr std::array<FabricChannelPoolType, NUM_RECEIVER_CHANNELS> REMOTE_RECEIVER_TO_POOL_TYPE =
    fill_array_with_next_n_args<
        FabricChannelPoolType,
        // We accidentally double emit the *_TO_POOL_TYPE arrays so we skip past some unused args
        REMOTE_CHANNEL_MAPPINGS_START_IDX + NUM_RECEIVER_CHANNELS,
        NUM_RECEIVER_CHANNELS>();
static_assert(all_elements_satisfy(REMOTE_RECEIVER_TO_POOL_TYPE, [](FabricChannelPoolType pool_type) {
    return pool_type <= FabricChannelPoolType::ELASTIC;
}));

// Calculate how many args the remote channel pool consumes
constexpr size_t DOWNSTREAM_SENDER_NUM_BUFFERS_SPECIAL_TAG_IDX =
    REMOTE_CHANNEL_MAPPINGS_START_IDX + 2 * NUM_RECEIVER_CHANNELS;
static_assert(
    get_compile_time_arg_val(DOWNSTREAM_SENDER_NUM_BUFFERS_SPECIAL_TAG_IDX) == 0xabaddad7,
    "DOWNSTREAM_SENDER_NUM_BUFFERS_SPECIAL_TAG_IDX not found. This implies some arguments were misaligned between host and device. Double check the CT args.");

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
    "ANOTHER_SPECIAL_TAG_2 not found. This implies some arguments were misaligned between host and device. Double check the CT args.");

constexpr size_t MAIN_CT_ARGS_IDX_1 = ANOTHER_SPECIAL_TAG_2_IDX + 1;
constexpr size_t local_sender_channel_0_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 0);
constexpr size_t local_sender_channel_1_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 1);
constexpr size_t local_sender_channel_2_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 2);
constexpr size_t local_sender_channel_3_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 3);
constexpr size_t local_sender_channel_4_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 4);
constexpr size_t local_sender_channel_5_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 5);
constexpr size_t local_sender_channel_6_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 6);
constexpr size_t local_sender_channel_7_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 7);
constexpr size_t local_sender_channel_8_connection_info_addr =
    get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 8);  // 9th channel for Z routers

// TODO: CONVERT TO SEMAPHORE
constexpr size_t MAIN_CT_ARGS_IDX_2 = MAIN_CT_ARGS_IDX_1 + MAX_NUM_SENDER_CHANNELS;
constexpr uint32_t termination_signal_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_2);
constexpr uint32_t edm_local_sync_ptr_addr =
    wait_for_host_signal ? get_compile_time_arg_val(MAIN_CT_ARGS_IDX_2 + 1) : 0;
constexpr uint32_t edm_local_tensix_sync_ptr_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_2 + 2);
constexpr uint32_t edm_status_ptr_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_2 + 3);

// for blackhole we need to disable the noc flush in inline writes to L1 for better perf. For wormhole this flag is not
// used.
constexpr bool enable_read_counter_update_noc_flush = false;
constexpr uint32_t notify_worker_of_read_counter_update_src_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_2 + 4);
constexpr size_t NOTIFY_WORKER_SRC_ADDR_MARKER_IDX = MAIN_CT_ARGS_IDX_2 + 5;
constexpr size_t NOTIFY_WORKER_SRC_ADDR_MARKER = 0x7A9B3C4D;
static_assert(
    !SPECIAL_MARKER_CHECK_ENABLED ||
        get_compile_time_arg_val(NOTIFY_WORKER_SRC_ADDR_MARKER_IDX) == NOTIFY_WORKER_SRC_ADDR_MARKER,
    "Notify worker marker not found. This implies some arguments were misaligned between host and device. Double "
    "check the CT args.");

constexpr size_t sender_channel_serviced_args_idx = MAIN_CT_ARGS_IDX_2 + 6;
constexpr std::array<bool, MAX_NUM_SENDER_CHANNELS> is_sender_channel_serviced =
    fill_array_with_next_n_args<bool, sender_channel_serviced_args_idx, MAX_NUM_SENDER_CHANNELS>();
constexpr size_t receiver_channel_serviced_args_idx =
    sender_channel_serviced_args_idx + is_sender_channel_serviced.size();
constexpr std::array<bool, MAX_NUM_RECEIVER_CHANNELS> is_receiver_channel_serviced =
    fill_array_with_next_n_args<bool, receiver_channel_serviced_args_idx, MAX_NUM_RECEIVER_CHANNELS>();
constexpr size_t MAIN_CT_ARGS_IDX_5 = receiver_channel_serviced_args_idx + is_receiver_channel_serviced.size();

constexpr bool enable_ethernet_handshake = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5) != 0;
constexpr bool enable_context_switch = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 1) != 0;
constexpr bool enable_interrupts = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 2) != 0;
constexpr size_t sender_txq_id = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 3);
constexpr size_t receiver_txq_id = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 4);
constexpr bool multi_txq_enabled = sender_txq_id != receiver_txq_id;

constexpr size_t iterations_between_ctx_switch_and_teardown_checks = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 5);
constexpr size_t is_2d_fabric = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 6);
constexpr size_t my_direction = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 7);
constexpr size_t num_eth_ports = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 8);

// If true, the sender channel will spin inside send_next_data until the eth_txq is not busy, rather than checking
// eth_txq_is_busy() being false as a prerequisite for sending the next packet
constexpr bool ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 9) != 0;
constexpr bool ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 10) != 0;

constexpr size_t DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 11);

// Context switch timeouts
constexpr size_t DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT =
#ifndef DEBUG_PRINT_ENABLED
    get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 12);
#else
    128;
#endif
constexpr bool IDLE_CONTEXT_SWITCHING = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 13) != 0;

constexpr size_t MY_ETH_CHANNEL = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 14);

constexpr size_t MY_ERISC_ID = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 15);
constexpr size_t NUM_ACTIVE_ERISCS = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 16);
static_assert(MY_ERISC_ID < NUM_ACTIVE_ERISCS, "MY_ERISC_ID must be less than NUM_ACTIVE_ERISCS");

// Defines if packet header updates (as the packet header traverses its route) are done on the receiver side or the
// sender side. If true, then the receiver channel updates the packet header before forwarding it. If false, the sender
// channel updates the packet header before sending it over Ethernet.
constexpr bool UPDATE_PKT_HDR_ON_RX_CH = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 17) != 0;

constexpr bool FORCE_ALL_PATHS_TO_USE_SAME_NOC = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 18) != 0;

constexpr bool is_intermesh_router_on_edge = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 19) != 0;
constexpr bool is_intramesh_router_on_edge = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 20) != 0;

constexpr size_t SPECIAL_MARKER_0_IDX = MAIN_CT_ARGS_IDX_5 + 21;
constexpr size_t SPECIAL_MARKER_0 = 0x00c0ffee;
static_assert(
    !SPECIAL_MARKER_CHECK_ENABLED || get_compile_time_arg_val(SPECIAL_MARKER_0_IDX) == SPECIAL_MARKER_0,
    "Special marker 0 not found. This implies some arguments were misaligned between host and device. Double check the "
    "CT args.");

constexpr size_t SKIP_LIVENESS_CHECK_ARG_IDX = SPECIAL_MARKER_0_IDX + SPECIAL_MARKER_CHECK_ENABLED;
constexpr std::array<bool, NUM_SENDER_CHANNELS> sender_ch_live_check_skip =
    fill_array_with_next_n_args<bool, SKIP_LIVENESS_CHECK_ARG_IDX, NUM_SENDER_CHANNELS>();

// A channel is a "traffic injection channel" if it is a sender channel that is adding *new*
// traffic to this dimension/ring. Examples include channels service worker traffic and
// sender channels that receive traffic from a "turn" (e.g. an EAST channel receiving traffic from NORTH)
// This attribute is necessary to support bubble flow control.
constexpr size_t SENDER_CHANNEL_IS_INJECTION_CHANNEL_START_IDX = SKIP_LIVENESS_CHECK_ARG_IDX + NUM_SENDER_CHANNELS;
constexpr std::array<bool, NUM_SENDER_CHANNELS> sender_channel_is_traffic_injection_channel =
    fill_array_with_next_n_args<bool, SENDER_CHANNEL_IS_INJECTION_CHANNEL_START_IDX, NUM_SENDER_CHANNELS>();
constexpr size_t BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS = 2;

constexpr size_t SENDER_CHANNEL_ACK_NOC_IDS_START_IDX =
    SENDER_CHANNEL_IS_INJECTION_CHANNEL_START_IDX + NUM_SENDER_CHANNELS;
constexpr size_t SENDER_CHANNEL_ACK_CMD_BUF_IDS_START_IDX = SENDER_CHANNEL_ACK_NOC_IDS_START_IDX + NUM_SENDER_CHANNELS;
constexpr std::array<size_t, NUM_SENDER_CHANNELS> sender_channel_ack_noc_ids =
    fill_array_with_next_n_args<size_t, SENDER_CHANNEL_ACK_NOC_IDS_START_IDX, NUM_SENDER_CHANNELS>();
constexpr std::array<uint8_t, NUM_SENDER_CHANNELS> sender_channel_ack_cmd_buf_ids =
    fill_array_with_next_n_args<uint8_t, SENDER_CHANNEL_ACK_CMD_BUF_IDS_START_IDX, NUM_SENDER_CHANNELS>();

constexpr size_t RX_CH_FWD_NOC_IDS_START_IDX = SENDER_CHANNEL_ACK_CMD_BUF_IDS_START_IDX + NUM_SENDER_CHANNELS;
constexpr size_t RX_CH_FWD_DATA_CMD_BUF_IDS_START_IDX = RX_CH_FWD_NOC_IDS_START_IDX + NUM_RECEIVER_CHANNELS;
constexpr size_t RX_CH_FWD_SYNC_CMD_BUF_IDS_START_IDX = RX_CH_FWD_DATA_CMD_BUF_IDS_START_IDX + NUM_RECEIVER_CHANNELS;
constexpr size_t RX_CH_LOCAL_WRITE_NOC_ID_IDX = RX_CH_FWD_SYNC_CMD_BUF_IDS_START_IDX + NUM_RECEIVER_CHANNELS;
constexpr size_t RX_CH_LOCAL_WRITE_CMD_BUF_ID_IDX = RX_CH_LOCAL_WRITE_NOC_ID_IDX + NUM_RECEIVER_CHANNELS;

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_noc_ids =
    fill_array_with_next_n_args<size_t, RX_CH_FWD_NOC_IDS_START_IDX, NUM_RECEIVER_CHANNELS>();
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_data_cmd_buf_ids =
    fill_array_with_next_n_args<uint8_t, RX_CH_FWD_DATA_CMD_BUF_IDS_START_IDX, NUM_RECEIVER_CHANNELS>();
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_sync_cmd_buf_ids =
    fill_array_with_next_n_args<uint8_t, RX_CH_FWD_SYNC_CMD_BUF_IDS_START_IDX, NUM_RECEIVER_CHANNELS>();
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> receiver_channel_local_write_noc_ids =
    fill_array_with_next_n_args<size_t, RX_CH_LOCAL_WRITE_NOC_ID_IDX, NUM_RECEIVER_CHANNELS>();
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> receiver_channel_local_write_cmd_buf_ids =
    fill_array_with_next_n_args<uint8_t, RX_CH_LOCAL_WRITE_CMD_BUF_ID_IDX, NUM_RECEIVER_CHANNELS>();

// TODO: Add a special marker in CT args so we don't misalign unintentionally
constexpr size_t EDM_NOC_VC_IDX = RX_CH_LOCAL_WRITE_CMD_BUF_ID_IDX + NUM_RECEIVER_CHANNELS;
constexpr size_t SPECIAL_MARKER_1_IDX = EDM_NOC_VC_IDX + 1;
constexpr size_t SPECIAL_MARKER_1 = 0x10c0ffee;
static_assert(
    !SPECIAL_MARKER_CHECK_ENABLED || get_compile_time_arg_val(SPECIAL_MARKER_1_IDX) == SPECIAL_MARKER_1,
    "Special marker 1 not found. This implies some arguments were misaligned between host and device. Double check the "
    "CT args.");

///////////////////////////////////////////////
// Telemetry
constexpr size_t PERF_TELEMETRY_MODE_IDX = SPECIAL_MARKER_1_IDX + SPECIAL_MARKER_CHECK_ENABLED;

constexpr bool ENABLE_FABRIC_TELEMETRY = static_cast<bool>(get_compile_time_arg_val(PERF_TELEMETRY_MODE_IDX));

constexpr uint8_t FABRIC_TELEMETRY_STATS_MASK =
    static_cast<uint8_t>(get_compile_time_arg_val(PERF_TELEMETRY_MODE_IDX + 1));
constexpr bool FABRIC_TELEMETRY_BANDWIDTH =
    ENABLE_FABRIC_TELEMETRY &&
    ((FABRIC_TELEMETRY_STATS_MASK & static_cast<uint8_t>(DynamicStatistics::BANDWIDTH)) != 0);
constexpr bool FABRIC_TELEMETRY_HEARTBEAT_TX =
    ENABLE_FABRIC_TELEMETRY &&
    ((FABRIC_TELEMETRY_STATS_MASK & static_cast<uint8_t>(DynamicStatistics::HEARTBEAT_TX)) != 0);
constexpr bool FABRIC_TELEMETRY_HEARTBEAT_RX =
    ENABLE_FABRIC_TELEMETRY &&
    ((FABRIC_TELEMETRY_STATS_MASK & static_cast<uint8_t>(DynamicStatistics::HEARTBEAT_RX)) != 0);
constexpr bool FABRIC_TELEMETRY_ANY_DYNAMIC_STAT =
    FABRIC_TELEMETRY_BANDWIDTH || FABRIC_TELEMETRY_HEARTBEAT_TX || FABRIC_TELEMETRY_HEARTBEAT_RX;

constexpr PerfTelemetryRecorderType perf_telemetry_mode =
    static_cast<PerfTelemetryRecorderType>(get_compile_time_arg_val(PERF_TELEMETRY_MODE_IDX + 2));

constexpr size_t PERF_TELEMETRY_BUFFER_ADDR_IDX = PERF_TELEMETRY_MODE_IDX + 3;
constexpr size_t perf_telemetry_buffer_addr = get_compile_time_arg_val(PERF_TELEMETRY_BUFFER_ADDR_IDX);


///////////////////////////////////////////////
// Code Profiling
constexpr size_t CODE_PROFILING_ENABLED_TIMERS_IDX = PERF_TELEMETRY_BUFFER_ADDR_IDX + 1;
constexpr uint32_t code_profiling_enabled_timers_bitfield = get_compile_time_arg_val(CODE_PROFILING_ENABLED_TIMERS_IDX);

constexpr size_t CODE_PROFILING_BUFFER_ADDR_IDX = CODE_PROFILING_ENABLED_TIMERS_IDX + 1;
constexpr size_t code_profiling_buffer_base_addr = get_compile_time_arg_val(CODE_PROFILING_BUFFER_ADDR_IDX);

constexpr size_t SPECIAL_MARKER_2A_IDX = CODE_PROFILING_BUFFER_ADDR_IDX + 1;
constexpr size_t SPECIAL_MARKER_2A = 0x20c0ffee;
static_assert(
    !SPECIAL_MARKER_CHECK_ENABLED || get_compile_time_arg_val(SPECIAL_MARKER_2A_IDX) == SPECIAL_MARKER_2A,
    "Special marker 2A not found. This implies some arguments were misaligned between host and device. Double check the "
    "CT args.");

constexpr size_t TO_SENDER_CREDIT_COUNTERS_START_IDX = SPECIAL_MARKER_2A_IDX + SPECIAL_MARKER_CHECK_ENABLED;

constexpr size_t to_sender_remote_ack_counters_base_address =
    conditional_get_compile_time_arg<multi_txq_enabled, TO_SENDER_CREDIT_COUNTERS_START_IDX>();

constexpr size_t to_sender_remote_completion_counters_base_address =
    conditional_get_compile_time_arg<multi_txq_enabled, TO_SENDER_CREDIT_COUNTERS_START_IDX + 1>();

// To optimize for CPU bottleneck instructions, instead of sending acks individually, based on the specific credit
// addresses, the router instead will send all credits at once. This eliminates a handful of instructions per ack. This
// behaviour is completely safe when using these unbounded counter credits because the credits are unbounded unsigned
// counters. Any overflow materializes as a roll back to zero, and subtractions are safe with unsigned.
constexpr size_t to_senders_credits_base_address =
    std::min(to_sender_remote_ack_counters_base_address, to_sender_remote_completion_counters_base_address);

constexpr size_t local_receiver_ack_counters_base_address =
    conditional_get_compile_time_arg<multi_txq_enabled, TO_SENDER_CREDIT_COUNTERS_START_IDX + 2>();

constexpr size_t local_receiver_completion_counters_base_address =
    conditional_get_compile_time_arg<multi_txq_enabled, TO_SENDER_CREDIT_COUNTERS_START_IDX + 3>();

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

constexpr size_t SPECIAL_MARKER_3_IDX = TO_SENDER_CREDIT_COUNTERS_START_IDX + (multi_txq_enabled ? 4 : 0);
constexpr size_t SPECIAL_MARKER_3 = 0x30c0ffee;
static_assert(
    !SPECIAL_MARKER_CHECK_ENABLED || get_compile_time_arg_val(SPECIAL_MARKER_3_IDX) == SPECIAL_MARKER_3,
    "Special marker 2 not found. This implies some arguments were misaligned between host and device. Double check the "
    "CT args.");

constexpr size_t HOST_SIGNAL_ARGS_START_IDX = SPECIAL_MARKER_3_IDX + SPECIAL_MARKER_CHECK_ENABLED;
// static_assert(HOST_SIGNAL_ARGS_START_IDX == 56, "HOST_SIGNAL_ARGS_START_IDX must be 56");
// TODO: Add type safe getter
constexpr bool is_local_handshake_master =
    bool(conditional_get_compile_time_arg<wait_for_host_signal, HOST_SIGNAL_ARGS_START_IDX + 0>());
constexpr uint32_t local_handshake_master_eth_chan =
    conditional_get_compile_time_arg<wait_for_host_signal, HOST_SIGNAL_ARGS_START_IDX + 1>();
constexpr uint32_t num_local_edms =
    conditional_get_compile_time_arg<wait_for_host_signal, HOST_SIGNAL_ARGS_START_IDX + 2>();
constexpr uint32_t edm_channels_mask =
    conditional_get_compile_time_arg<wait_for_host_signal, HOST_SIGNAL_ARGS_START_IDX + 3>();

template <size_t SLOT_SIZE_BYTES, size_t PACKET_HEADER_SIZE_BYTES>
struct BufferSlot {
    static constexpr size_t size_bytes = SLOT_SIZE_BYTES;
    static constexpr size_t header_size_bytes = PACKET_HEADER_SIZE_BYTES;
    static constexpr size_t max_payload_size_bytes = size_bytes - header_size_bytes;
};

using buffer_slot = BufferSlot<channel_buffer_size, sizeof(PACKET_HEADER_TYPE)>;

constexpr uint32_t ELASTIC_CHANNELS_CT_ARG_START_IDX = HOST_SIGNAL_ARGS_START_IDX + 4;
using FWDED_SENDER_ELASTIC_CHANNELS_INFO =
    tt::tt_fabric::elastic_channels::RouterElasticChannelsCtArgs<ELASTIC_CHANNELS_CT_ARG_START_IDX, buffer_slot::size_bytes>;


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
static constexpr uint8_t forward_and_local_write_noc_vc = get_compile_time_arg_val(EDM_NOC_VC_IDX);

// ----------------------------------------------------------------------------- //
// --------------------------------- PLACEHOLDER ------------------------------- //
// ---------------------- UNTIL ELASTIC CHANNELS IMPLEMENTED ------------------- //
// --------------------------------- ISSUE #26311 ------------------------------ //
constexpr size_t CHUNK_N_PKTS = 0;
constexpr std::array<bool, NUM_SENDER_CHANNELS> IS_ELASTIC_SENDER_CHANNEL =
    initialize_array<NUM_SENDER_CHANNELS, bool, false>();

// Helper to extract num_slots from a channel's pool (returns 0 for non-static pools)
template <typename ChannelPoolCollection, auto& ChannelToPoolIndex, size_t ChannelIdx>
constexpr size_t get_channel_num_slots() {
    constexpr size_t pool_idx = ChannelToPoolIndex[ChannelIdx];
    constexpr auto pool_type = static_cast<FabricChannelPoolType>(
        ChannelPoolCollection::channel_pool_types[pool_idx]);

    // If static pool, extract num_slots; otherwise default to 0
    if constexpr (pool_type == FabricChannelPoolType::STATIC) {
        using PoolType = std::tuple_element_t<pool_idx, typename ChannelPoolCollection::PoolsTuple>;
        return PoolType::num_slots;
    } else {
        return 0;
    }
}

// Helper to extract remote_num_slots from a channel's pool (returns 0 for non-static pools)
template <typename ChannelPoolCollection, auto& ChannelToPoolIndex, size_t ChannelIdx>
constexpr size_t get_channel_remote_num_slots() {
    constexpr size_t pool_idx = ChannelToPoolIndex[ChannelIdx];
    constexpr auto pool_type = static_cast<FabricChannelPoolType>(
        ChannelPoolCollection::channel_pool_types[pool_idx]);

    // If static pool, extract remote_num_slots; otherwise default to 0
    if constexpr (pool_type == FabricChannelPoolType::STATIC) {
        using PoolType = std::tuple_element_t<pool_idx, typename ChannelPoolCollection::PoolsTuple>;
        return PoolType::remote_num_slots;
    } else {
        return 0;
    }
}

// Build array by inspecting each channel's pool
template <typename ChannelPoolCollection, auto& ChannelToPoolIndex, size_t NumChannels, size_t... Indices>
constexpr std::array<size_t, NumChannels> build_num_slots_array_impl(std::index_sequence<Indices...>) {
    return {get_channel_num_slots<ChannelPoolCollection, ChannelToPoolIndex, Indices>()...};
}

template <typename ChannelPoolCollection, auto& ChannelToPoolIndex, size_t NumChannels>
constexpr std::array<size_t, NumChannels> build_num_slots_array() {
    return build_num_slots_array_impl<ChannelPoolCollection, ChannelToPoolIndex, NumChannels>(
        std::make_index_sequence<NumChannels>{});
}

// Build remote num slots array by inspecting each channel's pool
template <typename ChannelPoolCollection, auto& ChannelToPoolIndex, size_t NumChannels, size_t... Indices>
constexpr std::array<size_t, NumChannels> build_remote_num_slots_array_impl(std::index_sequence<Indices...>) {
    return {get_channel_remote_num_slots<ChannelPoolCollection, ChannelToPoolIndex, Indices>()...};
}

template <typename ChannelPoolCollection, auto& ChannelToPoolIndex, size_t NumChannels>
constexpr std::array<size_t, NumChannels> build_remote_num_slots_array() {
    return build_remote_num_slots_array_impl<ChannelPoolCollection, ChannelToPoolIndex, NumChannels>(
        std::make_index_sequence<NumChannels>{});
}

// Backward compatibility arrays - no longer used by multi-pool implementation
// These are kept for backward compatibility with code that hasn't migrated yet
// The actual buffer counts are now extracted directly from pool data
constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_NUM_BUFFERS_ARRAY = build_num_slots_array<
    channel_pools_args,
    SENDER_TO_POOL_IDX /*channel_pools_args::sender_channel_to_pool_index*/,
    NUM_SENDER_CHANNELS>();

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_NUM_BUFFERS_ARRAY = build_num_slots_array<
    channel_pools_args,
    channel_pools_args::receiver_channel_to_pool_index,
    NUM_RECEIVER_CHANNELS>();

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> REMOTE_RECEIVER_NUM_BUFFERS_ARRAY = build_num_slots_array<
    eth_remote_channel_pools_args,
    eth_remote_channel_pools_args::receiver_channel_to_pool_index,
    NUM_RECEIVER_CHANNELS>();

}  // namespace tt::tt_fabric
