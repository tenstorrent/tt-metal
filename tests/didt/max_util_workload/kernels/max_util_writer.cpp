// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// Writer kernel for max-utilization workload.
// Decoupled from compute - only generates NOC traffic to neighbors.
// No CB dependencies, no DRAM writing.
// Uses NOC1 to send 8KB in a loop to cores to the left and up.
// Target physical coordinates are passed as runtime arguments from host.
//
// Compile-time args:
//   0: num_loops           – number of workload repetitions (stress-test loop)
//   1: l1_tx_A_addr        – L1 address to use as source for transfers
//   2: l1_tx_B_addr        – L1 address to use as source for transfers
//   3: l1_rx_left_addr     – L1 address to use as source for transfers
//   4: l1_rx_up_addr       – L1 address to use as source for transfers
//   5: l1_rx_right_addr    – L1 address to use as source for transfers
//   6: l1_rx_down_addr     – L1 address to use as source for transfers
//   5: transfer_size       - transfer size (8KB)
//
// Runtime args:
//   0: target_left_x       – physical x coordinate of core to the left
//   1: target_left_y       – physical y coordinate of core to the left
//   2: target_up_x         – physical x coordinate of core above
//   3: target_up_y         – physical y coordinate of core above

void kernel_main() {
    constexpr uint32_t num_loops = get_compile_time_arg_val(0);
    constexpr uint32_t l1_tx_A_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_tx_B_addr = get_compile_time_arg_val(2);
    constexpr uint32_t l1_rx_left_addr = get_compile_time_arg_val(3);
    constexpr uint32_t l1_rx_up_addr = get_compile_time_arg_val(4);
    constexpr uint32_t l1_rx_right_addr = get_compile_time_arg_val(5);
    constexpr uint32_t l1_rx_down_addr = get_compile_time_arg_val(6);
    constexpr uint32_t transfer_size = get_compile_time_arg_val(7);

    // Get target physical coordinates from runtime args
    uint32_t target_left_x = get_arg_val<uint32_t>(0);
    uint32_t target_left_y = get_arg_val<uint32_t>(1);
    uint32_t target_up_x = get_arg_val<uint32_t>(2);
    uint32_t target_up_y = get_arg_val<uint32_t>(3);

    for (uint32_t iter = 0; iter < num_loops; ++iter) {
        // // Send pattern A
        // // Send 8KB to core on the left (using NOC1)
        // uint64_t noc_addr_left = get_noc_addr(target_left_x, target_left_y, l1_rx_right_addr, 1);
        // noc_async_write(l1_tx_A_addr, noc_addr_left, transfer_size, 1);

        // // // Send 8KB to core up (using NOC1)
        // // uint64_t noc_addr_up = get_noc_addr(target_up_x, target_up_y, l1_rx_down_addr, 1);
        // // noc_async_write(l1_tx_A_addr, noc_addr_up, transfer_size, 1);

        // // Send pattern B
        // // Send 8KB to core on the left (using NOC1)
        // noc_async_write(l1_tx_B_addr, noc_addr_left, transfer_size, 1);

        // // // Send 8KB to core up (using NOC1)
        // // noc_async_write(l1_tx_B_addr, noc_addr_up, transfer_size, 1);
    }

    noc_async_write_barrier(1);
}
