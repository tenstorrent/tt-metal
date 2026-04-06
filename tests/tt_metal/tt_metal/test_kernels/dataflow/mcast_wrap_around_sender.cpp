// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender kernel for wrap-around multicast tests.
// Performs multicast write with coordinates that may wrap around the NoC torus.
// For noc1, coordinates are swapped since noc1 travels in opposite direction from noc0.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t local_l1_addr = get_arg_val<uint32_t>(0);    // Source data in L1
    uint32_t dst_addr = get_arg_val<uint32_t>(1);         // Destination address on receiver cores
    uint32_t dst_noc_x_start = get_arg_val<uint32_t>(2);  // Multicast start X (NOC coords)
    uint32_t dst_noc_y_start = get_arg_val<uint32_t>(3);  // Multicast start Y (NOC coords)
    uint32_t dst_noc_x_end = get_arg_val<uint32_t>(4);    // Multicast end X (NOC coords)
    uint32_t dst_noc_y_end = get_arg_val<uint32_t>(5);    // Multicast end Y (NOC coords)
    uint32_t num_dests = get_arg_val<uint32_t>(6);        // Number of destination cores
    uint32_t data_size_bytes = get_arg_val<uint32_t>(7);  // Size of data to multicast

    // noc1 travels in opposite direction from noc0 (left/up vs right/down).
    // Swap coordinates for noc1 to maintain correct multicast semantics.
    if (noc_index == 1) {
        uint32_t temp_x = dst_noc_x_start;
        uint32_t temp_y = dst_noc_y_start;
        dst_noc_x_start = dst_noc_x_end;
        dst_noc_y_start = dst_noc_y_end;
        dst_noc_x_end = temp_x;
        dst_noc_y_end = temp_y;
    }

    uint64_t dst_noc_multicast_addr =
        get_noc_multicast_addr(dst_noc_x_start, dst_noc_y_start, dst_noc_x_end, dst_noc_y_end, dst_addr);

    noc_async_write_multicast(local_l1_addr, dst_noc_multicast_addr, data_size_bytes, num_dests, false);

    noc_async_write_barrier();
}
