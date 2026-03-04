// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// Reader kernel for max-utilization workload.
// Only the top-left core of the grid performs a NOC0 multicast to all
// remaining cores, alternating between pattern A (0x5555) and pattern B (0xAAAA).
// Every multicast transfer is data_transfer_size bytes.
// All other cores exit immediately.
//
// Compile-time args:
//   0: num_loops           – number of workload repetitions (stress-test loop)
//   1: l1_tx_A_addr        – L1 address of pattern A (0x5555...) to send
//   2: l1_tx_B_addr        – L1 address of pattern B (0xAAAA...) to send
//   3: l1_rx_addr          – L1 destination address on all receiving cores
//   4: (unused)
//   5: (unused)
//   6: (unused)
//   7: transfer_size       – transfer size in bytes (data_transfer_size)
//
// Runtime args:
//   0: is_sender        – 1 if this is the top-left (multicast sender) core, 0 otherwise
//   1: mcast_start_x    – NOC0 physical x of top-left corner of multicast rectangle
//   2: mcast_start_y    – NOC0 physical y of top-left corner of multicast rectangle
//   3: mcast_end_x      – NOC0 physical x of bottom-right corner of multicast rectangle
//   4: mcast_end_y      – NOC0 physical y of bottom-right corner of multicast rectangle
//   5: num_dests        – number of destination cores (total grid cores minus self)

void kernel_main() {
    constexpr uint32_t num_loops = get_compile_time_arg_val(0);
    constexpr uint32_t l1_tx_A_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_tx_B_addr = get_compile_time_arg_val(2);
    constexpr uint32_t l1_rx_addr = get_compile_time_arg_val(3);
    constexpr uint32_t transfer_size = get_compile_time_arg_val(7);

    uint32_t is_sender = get_arg_val<uint32_t>(0);

    // if (is_sender) {
    //     uint32_t mcast_start_x = get_arg_val<uint32_t>(1);
    //     uint32_t mcast_start_y = get_arg_val<uint32_t>(2);
    //     uint32_t mcast_end_x = get_arg_val<uint32_t>(3);
    //     uint32_t mcast_end_y = get_arg_val<uint32_t>(4);
    //     uint32_t num_dests = get_arg_val<uint32_t>(5);
    //     uint64_t mcast_dst =
    //         get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, l1_rx_addr, 0);

    //     for (uint32_t iter = 0; iter < num_loops / 4; ++iter) {
    //         // Pattern A (0x5555...)
    //         noc_async_write_multicast(l1_tx_A_addr, mcast_dst, transfer_size, num_dests, false, 0);

    //         // Pattern B (0xAAAA...)
    //         noc_async_write_multicast(l1_tx_B_addr, mcast_dst, transfer_size, num_dests, false, 0);
    //     }
    //     noc_async_write_barrier(0);
    // }
}
