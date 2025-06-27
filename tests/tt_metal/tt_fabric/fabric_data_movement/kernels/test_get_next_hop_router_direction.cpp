// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

void kernel_main() {
    uint32_t src_mesh_id = get_arg_val<uint32_t>(0);
    uint32_t src_fabric_dev_id = get_arg_val<uint32_t>(1);
    uint32_t result_addr = get_arg_val<uint32_t>(2);
    uint32_t num_devices = get_arg_val<uint32_t>(3);

    volatile tt_l1_ptr uint32_t* result_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

    for (uint32_t dst_idx = 0; dst_idx < num_devices; dst_idx++) {
        uint32_t dst_mesh_id = get_arg_val<uint32_t>(4 + dst_idx * 2);
        uint32_t dst_fabric_dev_id = get_arg_val<uint32_t>(4 + dst_idx * 2 + 1);

        eth_chan_directions direction = get_next_hop_router_direction(dst_mesh_id, dst_fabric_dev_id);
        result_ptr[dst_idx] = static_cast<uint32_t>(direction);

        DPRINT << "Routing: [" << src_mesh_id << "/" << src_fabric_dev_id << "] -> [" << dst_mesh_id << "/"
               << dst_fabric_dev_id << "] direction:" << static_cast<uint32_t>(direction) << "\n";
    }
}
