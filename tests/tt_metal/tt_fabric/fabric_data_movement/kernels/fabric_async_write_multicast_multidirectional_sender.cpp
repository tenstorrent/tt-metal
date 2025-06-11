// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/test_common.hpp"

using namespace tt::tt_fabric;

void kernel_main() {
    constexpr uint32_t client_interface_cb = get_compile_time_arg_val(0);
    constexpr tt::tt_fabric::ClientDataMode data_mode =
        static_cast<tt::tt_fabric::ClientDataMode>(get_compile_time_arg_val(1));
    constexpr uint32_t test_mode = get_compile_time_arg_val(2);

    uint32_t rt_args_idx = 0;
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
    uint32_t outbound_eth_chan = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    constexpr uint32_t num_dirs = 2;

    uint64_t dst_noc_addr = get_noc_addr_helper(dst_noc_offset, dst_addr);
    uint32_t packet_size_bytes = num_bytes + PACKET_HEADER_SIZE_BYTES;

    uint32_t client_interface_addr = get_write_ptr(client_interface_cb);
    volatile fabric_pull_client_interface_t* client_interface =
        (volatile fabric_pull_client_interface_t*)client_interface_addr;
    for (uint32_t i = 0; i < num_dirs; i++) {
        fabric_endpoint_init(client_interface + i, 0 /* unused */);
    }

    fabric_async_write_multicast<data_mode, AsyncWriteMode::ALL, RoutingType::ROUTER_XY>(
        client_interface,
        e_router_noc_xy,
        src_addr,  // source address in sender’s memory
        e_dst_mesh_id,
        e_dst_device_id,
        dst_noc_addr,       // destination write address
        packet_size_bytes,  // number of bytes to write to remote destination
        e_depth,
        0,
        0,
        0);

    // West Mcast
    if constexpr (data_mode == ClientDataMode::PACKETIZED_DATA) {
        // Wait for packet header to be flushed since we will reuse it for the next mcast direction
        fabric_wait_for_pull_request_bytes_flushed(client_interface, PACKET_HEADER_SIZE_BYTES);
        client_interface++;
        packet_header_t* packet_header = (packet_header_t*)(src_addr);
        packet_header->routing.dst_mesh_id = w_dst_mesh_id;
        packet_header->routing.dst_dev_id = w_dst_device_id;
        packet_header->packet_parameters.mcast_parameters.east = 0;
        packet_header->packet_parameters.mcast_parameters.west = w_depth;

        fabric_async_write_multicast<data_mode, AsyncWriteMode::ADD_AND_SEND_PR, RoutingType::ROUTER_XY>(
            client_interface,
            w_router_noc_xy,
            src_addr,  // source address in sender’s memory
            w_dst_mesh_id,
            w_dst_device_id,
            dst_noc_addr,       // destination write address
            packet_size_bytes,  // number of bytes to write to remote destination
            0,
            w_depth,
            0,
            0);
    } else {
        client_interface++;
        // For sideband header, we will use header slot in second client interface.
        // Use AsyncWriteMode::ALL, since the header slot needs to be initialized.
        fabric_async_write_multicast<data_mode, AsyncWriteMode::ALL, RoutingType::ROUTER_XY>(
            client_interface,
            w_router_noc_xy,
            src_addr,  // source address in sender’s memory
            w_dst_mesh_id,
            w_dst_device_id,
            dst_noc_addr,       // destination write address
            packet_size_bytes,  // number of bytes to write to remote destination
            0,
            w_depth,
            0,
            0);
    }
    // Flush all pull requests
    client_interface = reinterpret_cast<volatile fabric_pull_client_interface_t*>(client_interface_addr);
    for (uint32_t i = 0; i < num_dirs; i++) {
        fabric_wait_for_pull_request_flushed(client_interface);
        client_interface++;
    }
}
