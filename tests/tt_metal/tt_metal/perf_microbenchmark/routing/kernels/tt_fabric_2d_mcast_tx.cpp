// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

#ifdef TEST_ENABLE_FABRIC_TRACING
#include "tt_metal/tools/profiler/experimental/fabric_event_profiler.hpp"
#endif

// clang-format on

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t target_address = get_compile_time_arg_val(2);
constexpr uint32_t mcast_mode = get_compile_time_arg_val(4);
constexpr bool is_2d_fabric = get_compile_time_arg_val(5);
constexpr bool use_dynamic_routing = get_compile_time_arg_val(6);

inline void setup_connection_and_headers(
    tt::tt_fabric::WorkerToFabricEdmSender& connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint64_t noc_dest_addr,
    uint32_t packet_payload_size_bytes) {
    // connect to edm
    connection.open();
    packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_addr}, packet_payload_size_bytes);
}

inline void send_packet(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint64_t noc_dest_addr,
    uint32_t source_l1_buffer_address,
    uint32_t packet_payload_size_bytes,
    uint32_t seed,
    tt::tt_fabric::WorkerToFabricEdmSender& connection) {
#ifndef BENCHMARK_MODE
    packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_addr}, packet_payload_size_bytes);
    // fill packet data for sanity testing
    tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
    fill_packet_data(start_addr, packet_payload_size_bytes / 16, seed);
    tt_l1_ptr uint32_t* last_word_addr =
        reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address + packet_payload_size_bytes - 4);
#endif
    connection.wait_for_empty_write_slot();
#ifdef TEST_ENABLE_FABRIC_TRACING
    RECORD_FABRIC_HEADER(packet_header);
#endif
    connection.send_payload_without_header_non_blocking_from_address(
        source_l1_buffer_address, packet_payload_size_bytes);
    connection.send_payload_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

void set_mcast_header(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    eth_chan_directions trunk_direction,
    uint16_t trunk_hops,
    uint16_t e_hops,
    uint16_t w_hops) {
    uint16_t n_hops = 0;
    uint16_t s_hops = 0;

    if (trunk_direction == eth_chan_directions::NORTH) {
        n_hops = trunk_hops;
    } else if (trunk_direction == eth_chan_directions::SOUTH) {
        s_hops = trunk_hops;
    }

    fabric_set_mcast_route((LowLatencyMeshPacketHeader*)packet_header, 0, 0, e_hops, w_hops, n_hops, s_hops);
}

inline void teardown_connection(tt::tt_fabric::WorkerToFabricEdmSender& connection) { connection.close(); }

void kernel_main() {
    using namespace tt::tt_fabric;

    size_t rt_args_idx = 0;
    uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t rx_noc_encoding = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t ew_dim = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t my_dev_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t fwd_dev_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t fwd_mesh_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t north_trunk_hops = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t north_trunk_branch_hops = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender north_trunk_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    uint32_t bwd_dev_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t south_trunk_hops = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t south_trunk_branch_hops = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender south_trunk_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* north_packet_header =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* south_packet_header =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    uint64_t noc_dest_addr = get_noc_addr_helper(rx_noc_encoding, target_address);
    zero_l1_buf((uint32_t*)packet_header_buffer_address, sizeof(PACKET_HEADER_TYPE) * 2);

    if constexpr (mcast_mode & 0x1) {
        // North trunk present
        set_mcast_header(
            north_packet_header,
            eth_chan_directions::NORTH,
            north_trunk_hops,
            north_trunk_branch_hops & 0xFFFF,
            north_trunk_branch_hops >> 16);
        setup_connection_and_headers(
            north_trunk_connection, north_packet_header, noc_dest_addr, packet_payload_size_bytes);
    }
    if constexpr (mcast_mode & 0x2) {
        // South trunk present
        set_mcast_header(
            south_packet_header,
            eth_chan_directions::SOUTH,
            south_trunk_hops,
            south_trunk_branch_hops & 0xFFFF,
            south_trunk_branch_hops >> 16);
        setup_connection_and_headers(
            south_trunk_connection, south_packet_header, noc_dest_addr, packet_payload_size_bytes);
    }

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    // loop over for num packets
    for (uint32_t i = 0; i < num_packets; i++) {
#ifndef BENCHMARK_MODE
        time_seed = prng_next(time_seed);
#endif
        if constexpr (mcast_mode & 0x1) {
            // North Trunk Mcast
            send_packet(
                north_packet_header,
                noc_dest_addr,
                source_l1_buffer_address,
                packet_payload_size_bytes,
                time_seed,
                north_trunk_connection);
        }
        if constexpr (mcast_mode & 0x2) {
            // South Trunk Mcast
            send_packet(
                south_packet_header,
                noc_dest_addr,
                source_l1_buffer_address,
                packet_payload_size_bytes,
                time_seed,
                south_trunk_connection);
        }

#ifndef BENCHMARK_MODE
        noc_dest_addr += packet_payload_size_bytes;
#endif
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    if constexpr (mcast_mode & 0x1) {
        teardown_connection(north_trunk_connection);
    }
    if constexpr (mcast_mode & 0x2) {
        teardown_connection(south_trunk_connection);
    }

    noc_async_write_barrier();

    uint64_t bytes_sent = packet_payload_size_bytes * num_packets;

    // write out results
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_sent;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_sent >> 32;
}
