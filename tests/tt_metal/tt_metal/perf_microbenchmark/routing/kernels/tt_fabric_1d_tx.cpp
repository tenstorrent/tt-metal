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
constexpr bool mcast_mode = get_compile_time_arg_val(3);
constexpr bool is_2d_fabric = get_compile_time_arg_val(4);
constexpr bool use_dynamic_routing = get_compile_time_arg_val(5);

inline void setup_connection_and_headers(
    tt::tt_fabric::WorkerToFabricEdmSender& connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t hops,
    uint64_t noc_dest_addr,
    uint32_t packet_payload_size_bytes) {
    // connect to edm
    connection.open();
    if constexpr (!is_2d_fabric) {
        if constexpr (mcast_mode) {
            packet_header->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(hops)});
        } else {
            packet_header->to_chip_unicast(static_cast<uint8_t>(hops));
        }
    }

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
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, const eth_chan_directions& direction, uint32_t num_hops) {
    uint16_t e_num_hops = 0;
    uint16_t w_num_hops = 0;
    uint16_t n_num_hops = 0;
    uint16_t s_num_hops = 0;

    if (direction == eth_chan_directions::EAST) {
        e_num_hops = num_hops;
    } else if (direction == eth_chan_directions::WEST) {
        w_num_hops = num_hops;
    } else if (direction == eth_chan_directions::NORTH) {
        n_num_hops = num_hops;
    } else if (direction == eth_chan_directions::SOUTH) {
        s_num_hops = num_hops;
    }

    fabric_set_mcast_route(
        (LowLatencyMeshPacketHeader*)packet_header, 0, 0, e_num_hops, w_num_hops, n_num_hops, s_num_hops);
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

    uint64_t noc_dest_addr = get_noc_addr_helper(rx_noc_encoding, target_address);

    tt::tt_fabric::WorkerToFabricEdmSender fwd_fabric_connection;
    tt::tt_fabric::WorkerToFabricEdmSender bwd_fabric_connection;

    volatile tt_l1_ptr PACKET_HEADER_TYPE* fwd_packet_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* bwd_packet_header;

    if constexpr (mcast_mode) {
        uint32_t mcast_fwd_hops = get_arg_val<uint32_t>(rt_args_idx++);

        fwd_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

        uint32_t bwd_dev_id = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t mcast_bwd_hops = get_arg_val<uint32_t>(rt_args_idx++);
        bwd_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

        fwd_packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
        bwd_packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
        zero_l1_buf((uint32_t*)packet_header_buffer_address, sizeof(PACKET_HEADER_TYPE) * 2);

        if constexpr (is_2d_fabric) {
            set_mcast_header(fwd_packet_header, (eth_chan_directions)fwd_fabric_connection.direction, mcast_fwd_hops);
            set_mcast_header(bwd_packet_header, (eth_chan_directions)bwd_fabric_connection.direction, mcast_bwd_hops);
        }

        setup_connection_and_headers(
            fwd_fabric_connection, fwd_packet_header, mcast_fwd_hops, noc_dest_addr, packet_payload_size_bytes);

        setup_connection_and_headers(
            bwd_fabric_connection, bwd_packet_header, mcast_bwd_hops, noc_dest_addr, packet_payload_size_bytes);

    } else {
        uint32_t unicast_hops = get_arg_val<uint32_t>(rt_args_idx++);

        fwd_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

        fwd_packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
        zero_l1_buf((uint32_t*)packet_header_buffer_address, sizeof(PACKET_HEADER_TYPE));

        if constexpr (is_2d_fabric) {
            if constexpr (use_dynamic_routing) {
                fabric_set_unicast_route(
                    (MeshPacketHeader*)packet_header_buffer_address,
                    (eth_chan_directions)fwd_fabric_connection.direction,
                    my_dev_id,
                    fwd_dev_id,
                    fwd_mesh_id,
                    ew_dim);
            } else {
                fabric_set_unicast_route(
                    (LowLatencyMeshPacketHeader*)packet_header_buffer_address,
                    (eth_chan_directions)fwd_fabric_connection.direction,
                    my_dev_id,
                    fwd_dev_id,
                    fwd_mesh_id,
                    ew_dim);
            }
        }

        setup_connection_and_headers(
            fwd_fabric_connection, fwd_packet_header, unicast_hops, noc_dest_addr, packet_payload_size_bytes);
    }

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    // loop over for num packets
    for (uint32_t i = 0; i < num_packets; i++) {
#ifndef BENCHMARK_MODE
        time_seed = prng_next(time_seed);
#endif
        if constexpr (mcast_mode) {
            // fwd packet
            send_packet(
                fwd_packet_header,
                noc_dest_addr,
                source_l1_buffer_address,
                packet_payload_size_bytes,
                time_seed,
                fwd_fabric_connection);

            // bwd packet
            send_packet(
                bwd_packet_header,
                noc_dest_addr,
                source_l1_buffer_address,
                packet_payload_size_bytes,
                time_seed,
                bwd_fabric_connection);
        } else {
            send_packet(
                fwd_packet_header,
                noc_dest_addr,
                source_l1_buffer_address,
                packet_payload_size_bytes,
                time_seed,
                fwd_fabric_connection);
        }
#ifndef BENCHMARK_MODE
        noc_dest_addr += packet_payload_size_bytes;
#endif
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    if constexpr (mcast_mode) {
        teardown_connection(fwd_fabric_connection);
        teardown_connection(bwd_fabric_connection);
    } else {
        teardown_connection(fwd_fabric_connection);
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
