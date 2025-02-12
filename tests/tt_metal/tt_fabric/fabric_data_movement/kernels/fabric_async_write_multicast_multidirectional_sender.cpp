// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_fabric/hw/inc/tt_fabric.h"
#include "tt_fabric/hw/inc/tt_fabric_interface.h"
#include "tt_fabric/hw/inc/tt_fabric_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

// clang-format on

using namespace tt::tt_fabric;

volatile fabric_client_interface_t* client_interface;

uint64_t xy_local_addr;

void kernel_main() {
    uint32_t rt_args_idx = 0;
    // Fabric configuration specific arguments
    uint32_t client_interface_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t gk_interface_addr_l = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t gk_interface_addr_h = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    uint32_t src_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_noc_offset = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t num_bytes = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t e_dst_mesh_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t e_dst_device_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t e_depth = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t e_router_noc_xy = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t w_dst_mesh_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t w_dst_device_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t w_depth = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t w_router_noc_xy = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    // uint32_t n_dst_mesh_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    // uint32_t n_dst_device_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    // uint32_t n_depth = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    // uint32_t n_router_noc_xy = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    // uint32_t s_dst_mesh_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    // uint32_t s_dst_device_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    // uint32_t s_depth = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    // uint32_t s_router_noc_xy = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    constexpr uint32_t num_dirs = 2;  // 4

    uint64_t dst_noc_addr = get_noc_addr_helper(dst_noc_offset, dst_addr);
    uint32_t packet_size_bytes = num_bytes + PACKET_HEADER_SIZE_BYTES;
    fabric_async_write_multicast_add_header(
        src_addr,  // source address in sender’s memory
        e_dst_mesh_id,
        e_dst_device_id,
        dst_noc_addr,       // destination write address
        packet_size_bytes,  // number of bytes to write to remote destination
        e_depth,
        0,
        0,
        0);

    // make sure fabric node gatekeeper is available.
    fabric_endpoint_init<false>(client_interface_addr, gk_interface_addr_l, gk_interface_addr_h);
    for (uint32_t i = 1; i < num_dirs; i++) {
        copy_l1_buf((uint32_t*)client_interface, (uint32_t*)(client_interface + i), sizeof(fabric_client_interface_t));
    }

    fabric_setup_pull_request(
        src_addr,          // source address in sender’s memory
        packet_size_bytes  // number of bytes to write to remote destination
    );

    fabric_send_pull_request<false>(e_router_noc_xy, e_dst_mesh_id, e_dst_device_id);
    fabric_wait_for_pull_request_bytes_flushed(PACKET_HEADER_SIZE_BYTES);
    packet_header_t* packet_header = (packet_header_t*)(src_addr);

    // West Mcast
    client_interface++;

    packet_header->routing.dst_mesh_id = w_dst_mesh_id;
    packet_header->routing.dst_dev_id = w_dst_device_id;
    packet_header->packet_parameters.mcast_parameters.east = 0;
    packet_header->packet_parameters.mcast_parameters.west = w_depth;
    // make sure fabric node gatekeeper is available.
    fabric_endpoint_init<false>(client_interface_addr, gk_interface_addr_l, gk_interface_addr_h);

    fabric_setup_pull_request(
        src_addr,          // source address in sender’s memory
        packet_size_bytes  // number of bytes to write to remote destination
    );

    fabric_send_pull_request<false>(w_router_noc_xy, w_dst_mesh_id, w_dst_device_id);
    // fabric_wait_for_pull_request_bytes_flushed(PACKET_HEADER_SIZE_BYTES);

    // // North Mcast
    // client_interface++;

    // packet_header->routing.dst_mesh_id = n_dst_mesh_id;
    // packet_header->routing.dst_dev_id = n_dst_device_id;
    // packet_header->packet_parameters.mcast_parameters.west = 0;
    // packet_header->packet_parameters.mcast_parameters.north = n_depth;
    // // make sure fabric node gatekeeper is available.
    // fabric_endpoint_init<false>(client_interface_addr, gk_interface_addr_l, gk_interface_addr_h);

    // fabric_setup_pull_request(
    //     src_addr,     // source address in sender’s memory
    //     packet_size_bytes  // number of bytes to write to remote destination
    // );

    // fabric_send_pull_request<false>(n_router_noc_xy, n_dst_mesh_id, n_dst_device_id);
    // fabric_wait_for_pull_request_bytes_flushed(PACKET_HEADER_SIZE_BYTES);

    // // South Mcast
    // client_interface++;

    // packet_header->routing.dst_mesh_id = s_dst_mesh_id;
    // packet_header->routing.dst_dev_id = s_dst_device_id;
    // packet_header->packet_parameters.mcast_parameters.north = 0;
    // packet_header->packet_parameters.mcast_parameters.south = s_depth;
    // // make sure fabric node gatekeeper is available.
    // fabric_endpoint_init<false>(client_interface_addr, gk_interface_addr_l, gk_interface_addr_h);

    // fabric_setup_pull_request(
    //     src_addr,     // source address in sender’s memory
    //     packet_size_bytes  // number of bytes to write to remote destination
    // );

    // fabric_send_pull_request<false>(s_router_noc_xy, s_dst_mesh_id, s_dst_device_id);

    // Flush all pull requests
    client_interface = (volatile fabric_client_interface_t*)client_interface_addr;
    for (uint32_t i = 0; i < num_dirs; i++) {
        fabric_wait_for_pull_request_flushed();
        client_interface++;
    }
}
