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

uint32_t target_address = get_compile_time_arg_val(2);
constexpr bool use_dram_dst = get_compile_time_arg_val(3);
constexpr bool is_2d_fabric = get_compile_time_arg_val(4);
constexpr bool use_dynamic_routing = get_compile_time_arg_val(5);
constexpr bool is_chip_multicast = get_compile_time_arg_val(6);
constexpr bool additional_dir = get_compile_time_arg_val(7);

inline void setup_header_routing_1d(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, uint32_t start_distance, uint32_t range) {
    if constexpr (is_chip_multicast) {
        packet_header->to_chip_multicast(
            MulticastRoutingCommandHeader{static_cast<uint8_t>(start_distance), static_cast<uint8_t>(range)});
    } else {
        packet_header->to_chip_unicast(static_cast<uint8_t>(start_distance));
    }
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

    // dst_dev_id is ignored since Low Latency Mesh Fabric does not support arbitrary 2D Mcasts yet
    // dst_mesh_id is ignored since Low Latency Mesh Fabric is not used for Inter-Mesh Routing
    fabric_set_mcast_route(
        (LowLatencyMeshPacketHeader*)packet_header, 0, 0, e_num_hops, w_num_hops, n_num_hops, s_num_hops);
}

inline void setup_header_routing_2d(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    eth_chan_directions direction,
    uint32_t range,
    uint32_t my_dev_id,
    uint32_t dst_dev_id,
    uint32_t dst_mesh_id,
    uint32_t ew_dim) {
    if constexpr (is_chip_multicast) {
        static_assert(
            !(is_2d_fabric && is_chip_multicast &&
              use_dynamic_routing));  // dynamic routing not supported for 2D multicast in this test
        set_mcast_header(packet_header, direction, range);
    } else {
        if constexpr (use_dynamic_routing) {
            fabric_set_unicast_route(
                (MeshPacketHeader*)packet_header,
                direction,  // Ignored: Dynamic Routing does not need outgoing_direction specified
                my_dev_id,  // Ignored: Dynamic Routing does not need src chip ID
                dst_dev_id,
                dst_mesh_id,
                ew_dim);  // Ignored: Dynamic Routing does not need mesh dimensions
        } else {
            fabric_set_unicast_route(
                (LowLatencyMeshPacketHeader*)packet_header,
                direction,
                my_dev_id,
                dst_dev_id,
                dst_mesh_id,  // Ignored since Low Latency Mesh Fabric is not used for Inter-Mesh Routing
                ew_dim);
        }
    }
}

inline void setup_header_noc_unicast_write(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t packet_payload_size_bytes,
    uint32_t dest_addr,
    uint8_t noc_x_start,
    uint8_t noc_y_start) {
    packet_header->to_noc_unicast_write(
        NocUnicastCommandHeader{get_noc_addr(noc_x_start, noc_y_start, dest_addr)}, packet_payload_size_bytes);
}

inline void setup_header_noc_unicast_write_dram(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t packet_payload_size_bytes,
    uint32_t dest_addr,
    uint32_t dest_bank_id) {
    packet_header->to_noc_unicast_write(
        NocUnicastCommandHeader{get_noc_addr_from_bank_id<true>(dest_bank_id, dest_addr)}, packet_payload_size_bytes);
}

inline void setup_header_noc_unicast_atomic_inc(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t notification_mailbox_address,
    uint8_t noc_x_start,
    uint8_t noc_y_start) {
    packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
        get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
        1 /* increment value */,
        std::numeric_limits<uint16_t>::max(),
        true /*flush*/});
}

inline void send_notification(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, tt::tt_fabric::WorkerToFabricEdmSender& connection) {
    // Notify mailbox that the packets have been sent
    connection.wait_for_empty_write_slot();
#ifdef TEST_ENABLE_FABRIC_TRACING
    RECORD_FABRIC_HEADER(packet_header);
#endif
    connection.send_payload_flush_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

inline void send_packet(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t source_l1_buffer_address,
    uint32_t packet_payload_size_bytes,
    uint32_t seed,
    tt::tt_fabric::WorkerToFabricEdmSender& connection) {
#ifndef BENCHMARK_MODE
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

// connect to edm
inline void setup_connection(tt::tt_fabric::WorkerToFabricEdmSender& connection) { connection.open(); }

inline void teardown_connection(tt::tt_fabric::WorkerToFabricEdmSender& connection) { connection.close(); }

void kernel_main() {
    using namespace tt::tt_fabric;

    size_t rt_args_idx = 0;
    uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_args_idx++);

    // noc transfer info
    uint32_t noc_x_start = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t noc_y_start = get_arg_val<uint32_t>(rt_args_idx++);

    // routing info
    uint32_t ew_dim = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t my_dev_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t fwd_start_distance = get_arg_val<uint32_t>(rt_args_idx++);  // for 1d only
    uint32_t fwd_range = get_arg_val<uint32_t>(rt_args_idx++);           // for multicast only
    uint32_t fwd_dev_id = get_arg_val<uint32_t>(rt_args_idx++);          // for 2d unicast only
    uint32_t fwd_mesh_id = get_arg_val<uint32_t>(rt_args_idx++);         // for 2d unicast only

    // DRAM destination args
    uint32_t dest_bank_id;
    uint32_t dest_dram_addr;
    uint32_t notification_mailbox_address;

    if constexpr (use_dram_dst) {
        dest_bank_id = get_arg_val<uint32_t>(rt_args_idx++);
        dest_dram_addr = get_arg_val<uint32_t>(rt_args_idx++);
        notification_mailbox_address = get_arg_val<uint32_t>(rt_args_idx++);
    }

    uint32_t bwd_start_distance;
    uint32_t bwd_range;
    uint32_t bwd_dev_id;
    uint32_t bwd_mesh_id;

    tt::tt_fabric::WorkerToFabricEdmSender fwd_fabric_connection;
    tt::tt_fabric::WorkerToFabricEdmSender bwd_fabric_connection;

    volatile tt_l1_ptr PACKET_HEADER_TYPE* fwd_packet_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* bwd_packet_header;

    /***************** setup forward dir *****************/
    fwd_fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    fwd_packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    zero_l1_buf((uint32_t*)packet_header_buffer_address, sizeof(PACKET_HEADER_TYPE));

    setup_connection(fwd_fabric_connection);

    if constexpr (!is_2d_fabric) {  // 1D
        setup_header_routing_1d(fwd_packet_header, fwd_start_distance, fwd_range);
        DPRINT << "fwd_start_distance" << fwd_start_distance << ", fwd_range" << fwd_range << ENDL();
    } else {  // 2D
        setup_header_routing_2d(
            fwd_packet_header,
            (eth_chan_directions)fwd_fabric_connection.direction,
            fwd_range,
            my_dev_id,
            fwd_dev_id,
            fwd_mesh_id,
            ew_dim);
    }

    /***************** setup bawkward dir *****************/
    if constexpr (additional_dir) {
        // routing info for additional direction
        bwd_start_distance = get_arg_val<uint32_t>(rt_args_idx++);  // for 1d only
        bwd_range = get_arg_val<uint32_t>(rt_args_idx++);           // for multicast only
        bwd_dev_id = get_arg_val<uint32_t>(rt_args_idx++);          // for 2d unicast only
        bwd_mesh_id = get_arg_val<uint32_t>(rt_args_idx++);         // for 2d unicast only

        bwd_fabric_connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

        bwd_packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

        zero_l1_buf((uint32_t*)packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE), sizeof(PACKET_HEADER_TYPE));

        setup_connection(bwd_fabric_connection);

        if constexpr (!is_2d_fabric) {  // 1D
            setup_header_routing_1d(bwd_packet_header, bwd_start_distance, bwd_range);
        } else {  // 2D
            setup_header_routing_2d(
                bwd_packet_header,
                (eth_chan_directions)bwd_fabric_connection.direction,
                bwd_range,
                my_dev_id,
                bwd_dev_id,
                bwd_mesh_id,
                ew_dim);
        }
    }

    /***************** send packets *****************/
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    // loop over for num packets
    for (uint32_t i = 0; i < num_packets; i++) {
#ifndef BENCHMARK_MODE
        time_seed = prng_next(time_seed);

        if constexpr (use_dram_dst) {
            // Calculate current DRAM destination address for this packet
            uint32_t current_dram_addr = dest_dram_addr + (i * packet_payload_size_bytes);

            setup_header_noc_unicast_write_dram(
                fwd_packet_header, packet_payload_size_bytes, current_dram_addr, dest_bank_id);

            if constexpr (additional_dir) {
                setup_header_noc_unicast_write_dram(
                    bwd_packet_header, packet_payload_size_bytes, current_dram_addr, dest_bank_id);
            }
        } else {
            setup_header_noc_unicast_write(
                fwd_packet_header, packet_payload_size_bytes, target_address, noc_x_start, noc_y_start);

            if constexpr (additional_dir) {
                setup_header_noc_unicast_write(
                    bwd_packet_header, packet_payload_size_bytes, target_address, noc_x_start, noc_y_start);
            }
            target_address += packet_payload_size_bytes;
        }
#endif

        // fwd packet
        send_packet(
            fwd_packet_header, source_l1_buffer_address, packet_payload_size_bytes, time_seed, fwd_fabric_connection);

        if constexpr (additional_dir) {
            // bwd packet
            send_packet(
                bwd_packet_header,
                source_l1_buffer_address,
                packet_payload_size_bytes,
                time_seed,
                bwd_fabric_connection);
        }
    }

    /* Use atomic increment to flush writes for simplicity*/
    if constexpr (use_dram_dst) {
        // Ensure all DRAM packets are written before sending notification
        noc_async_write_barrier();

        // Send notification that packets have been sent
        setup_header_noc_unicast_atomic_inc(fwd_packet_header, notification_mailbox_address, noc_x_start, noc_y_start);
        if constexpr (additional_dir) {
            setup_header_noc_unicast_atomic_inc(
                bwd_packet_header, notification_mailbox_address, noc_x_start, noc_y_start);
        }
        send_notification(fwd_packet_header, fwd_fabric_connection);
        if constexpr (additional_dir) {
            send_notification(bwd_packet_header, bwd_fabric_connection);
        }
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    teardown_connection(fwd_fabric_connection);
    if constexpr (additional_dir) {
        teardown_connection(bwd_fabric_connection);
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
