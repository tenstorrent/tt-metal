// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/tools/profiler/fabric_event_profiler.hpp"
#include "tt_metal/fabric/hw/inc/mesh/api.h"

// clang-format on

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t target_address = get_compile_time_arg_val(2);

using namespace tt::tt_fabric;
namespace mesh_exp = tt::tt_fabric::mesh::experimental;
using mesh_exp::MeshMcastRange;

bool make_range(
    MeshMcastRange& range, eth_chan_directions dir, uint16_t trunk_hops, uint16_t branch_east, uint16_t branch_west) {
    if (trunk_hops == 0) {
        return false;
    }
    range = MeshMcastRange{0, 0, 0, 0};
    switch (dir) {
        case eth_chan_directions::NORTH:
            range.n = static_cast<uint8_t>(trunk_hops);
            range.e = static_cast<uint8_t>(branch_east);
            range.w = static_cast<uint8_t>(branch_west);
            break;
        case eth_chan_directions::SOUTH:
            range.s = static_cast<uint8_t>(trunk_hops);
            range.e = static_cast<uint8_t>(branch_east);
            range.w = static_cast<uint8_t>(branch_west);
            break;
        case eth_chan_directions::EAST: range.e = static_cast<uint8_t>(trunk_hops); break;
        case eth_chan_directions::WEST: range.w = static_cast<uint8_t>(trunk_hops); break;
        default: break;
    }
    return true;
}

void kernel_main() {
    size_t rt_args_idx = 0;
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t rx_noc_encoding = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t north_trunk_hops = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t north_trunk_branch_hops = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t south_trunk_hops = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t south_trunk_branch_hops = get_arg_val<uint32_t>(rt_args_idx++);
    uint16_t east_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(rt_args_idx++));
    uint16_t west_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(rt_args_idx++));

    MeshMcastRange ranges[RoutingPlaneConnectionManager::MaxConnections];
    uint32_t num_routes = 0;

    num_routes += make_range(
        ranges[num_routes],
        eth_chan_directions::NORTH,
        north_trunk_hops,
        static_cast<uint16_t>(north_trunk_branch_hops & 0xFFFF),
        static_cast<uint16_t>(north_trunk_branch_hops >> 16));
    num_routes += make_range(
        ranges[num_routes],
        eth_chan_directions::SOUTH,
        south_trunk_hops,
        static_cast<uint16_t>(south_trunk_branch_hops & 0xFFFF),
        static_cast<uint16_t>(south_trunk_branch_hops >> 16));
    num_routes += make_range(ranges[num_routes], eth_chan_directions::WEST, west_hops, 0, 0);
    num_routes += make_range(ranges[num_routes], eth_chan_directions::EAST, east_hops, 0, 0);

    ASSERT(num_routes > 0);

    tt::tt_fabric::RoutingPlaneConnectionManager connection_manager;
    mesh_exp::open_connections(connection_manager, num_routes, rt_args_idx);

    uint8_t route_id = PacketHeaderPool::allocate_header_n(num_routes);
    uint64_t noc_dest_addr = get_noc_addr_helper(rx_noc_encoding, target_address);

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    for (uint32_t i = 0; i < num_packets; ++i) {
#ifndef BENCHMARK_MODE
        time_seed = prng_next(time_seed);
        tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
        fill_packet_data(start_addr, packet_payload_size_bytes / 16, time_seed);
#endif
        mesh_exp::fabric_multicast_noc_unicast_write(
            connection_manager,
            route_id,
            ranges,
            source_l1_buffer_address,
            packet_payload_size_bytes,
            tt::tt_fabric::NocUnicastCommandHeader{noc_dest_addr});

#ifndef BENCHMARK_MODE
        noc_dest_addr += packet_payload_size_bytes;
#endif
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    mesh_exp::close_connections(connection_manager);
    noc_async_write_barrier();

    uint64_t bytes_sent = packet_payload_size_bytes * num_packets;

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = static_cast<uint32_t>(cycles_elapsed);
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = static_cast<uint32_t>(cycles_elapsed >> 32);
    test_results[TT_FABRIC_WORD_CNT_INDEX] = static_cast<uint32_t>(bytes_sent);
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = static_cast<uint32_t>(bytes_sent >> 32);
}
