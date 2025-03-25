// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

using namespace tt::tt_fabric;

void kernel_main() {
    constexpr uint32_t client_interface_cb = get_compile_time_arg_val(0);
    constexpr uint32_t data_mode = get_compile_time_arg_val(1);

    uint32_t rt_args_idx = 0;
    uint32_t src_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_noc_offset = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t num_bytes = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_mesh_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_device_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t router_noc_xy = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    uint64_t dst_noc_addr = get_noc_addr_helper(dst_noc_offset, dst_addr);
    uint32_t packet_size_bytes = num_bytes + PACKET_HEADER_SIZE_BYTES;

    uint32_t client_interface_addr = get_write_ptr(client_interface_cb);
    volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface =
        reinterpret_cast<volatile tt_l1_ptr fabric_pull_client_interface_t*>(client_interface_addr);
    fabric_endpoint_init(client_interface, 0 /* unused */);

    fabric_async_write<(ClientDataMode)data_mode, AsyncWriteMode::ALL, RoutingType::ROUTER_XY>(
        client_interface,
        router_noc_xy,
        src_addr,  // source address in sender’s memory
        dst_mesh_id,
        dst_device_id,
        dst_noc_addr,      // destination write address
        packet_size_bytes  // number of bytes to write to remote destination
    );

    fabric_wait_for_pull_request_flushed(client_interface);
}
