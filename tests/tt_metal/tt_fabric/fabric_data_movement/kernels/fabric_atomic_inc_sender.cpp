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
    constexpr uint32_t fabric_mode = get_compile_time_arg_val(1);
    constexpr uint32_t test_mode = get_compile_time_arg_val(2);
    constexpr uint32_t data_mode = get_compile_time_arg_val(3);

    uint32_t rt_args_idx = 0;
    uint32_t src_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_noc_offset = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t atomic_inc = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t wrap_boundary = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_mesh_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_device_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t router_noc_xy = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t outbound_eth_chan = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    uint64_t dst_noc_addr = get_noc_addr_helper(dst_noc_offset, dst_addr);
    uint32_t packet_size_bytes = PACKET_HEADER_SIZE_BYTES;

    uint32_t client_interface_addr = get_write_ptr(client_interface_cb);
    if constexpr (fabric_mode == fabric_mode::PULL) {
        volatile fabric_pull_client_interface_t* client_interface =
            (volatile fabric_pull_client_interface_t*)client_interface_addr;

        fabric_atomic_inc<ClientDataMode::PACKETIZED_DATA, AsyncWriteMode::ALL>(
            client_interface,
            router_noc_xy,
            src_addr,  // source address in sender’s memory
            dst_mesh_id,
            dst_device_id,
            dst_noc_addr,  // destination write address
            atomic_inc,
            wrap_boundary);

        fabric_wait_for_pull_request_flushed(client_interface);
    } else {
        volatile fabric_push_client_interface_t* client_interface =
            (volatile fabric_push_client_interface_t*)client_interface_addr;

        fabric_endpoint_init<RoutingType::ROUTING_TABLE>(client_interface, outbound_eth_chan);
        fabric_client_connect(client_interface, 0, dst_mesh_id, dst_device_id);
        fabric_atomic_inc<
            ClientDataMode::PACKETIZED_DATA,
            (AsyncWriteMode)(AsyncWriteMode::PUSH | AsyncWriteMode::ADD_HEADER)>(
            client_interface,
            router_noc_xy,
            src_addr,  // source address in sender’s memory
            dst_mesh_id,
            dst_device_id,
            dst_noc_addr,  // destination write address
            atomic_inc,
            wrap_boundary);
        fabric_client_disconnect(client_interface);
    }
}
