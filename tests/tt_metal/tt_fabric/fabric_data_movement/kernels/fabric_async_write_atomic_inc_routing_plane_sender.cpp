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
    uint32_t dst_write_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_atomic_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t num_bytes = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t atomic_inc = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_mesh_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t dst_device_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    uint32_t routing_plane = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    uint64_t dst_write_noc_addr = get_noc_addr_helper(dst_noc_offset, dst_write_addr);
    uint64_t dst_atomic_noc_addr = get_noc_addr_helper(dst_noc_offset, dst_atomic_addr);
    uint32_t packet_size_bytes = num_bytes + PACKET_HEADER_SIZE_BYTES;

    // make sure fabric node gatekeeper is available.
    fabric_endpoint_init(client_interface_addr, gk_interface_addr_l, gk_interface_addr_h);

    fabric_async_write_atomic_inc(
        routing_plane,
        src_addr,  // source address in sender’s memory
        dst_mesh_id,
        dst_device_id,
        dst_write_noc_addr,   // destination write address
        dst_atomic_noc_addr,  // destination atomic address
        packet_size_bytes,    // number of bytes to write to remote destination
        atomic_inc            // atomic increment value
    );
    fabric_wait_for_pull_request_flushed();
}
