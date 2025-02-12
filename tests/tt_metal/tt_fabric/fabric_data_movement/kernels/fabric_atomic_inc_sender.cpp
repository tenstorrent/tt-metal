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
    uint32_t atomic_inc = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t wrap_boundary = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_mesh_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_device_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t router_noc_xy = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    uint64_t dst_noc_addr = get_noc_addr_helper(dst_noc_offset, dst_addr);
    uint32_t packet_size_bytes = PACKET_HEADER_SIZE_BYTES;
    fabric_atomic_inc_add_header(
        src_addr,  // source address in sender’s memory
        dst_mesh_id,
        dst_device_id,
        dst_noc_addr,  // destination write address
        atomic_inc,
        wrap_boundary);

    // make sure fabric node gatekeeper is available.
    fabric_endpoint_init<false>(client_interface_addr, gk_interface_addr_l, gk_interface_addr_h);

    fabric_setup_pull_request(
        src_addr,          // source address in sender’s memory
        packet_size_bytes  // number of bytes to write to remote destination
    );

    fabric_send_pull_request<false>(router_noc_xy, dst_mesh_id, dst_device_id);
    fabric_wait_for_pull_request_flushed();
}
