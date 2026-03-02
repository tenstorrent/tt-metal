// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// Reader kernel for max-utilization workload.
// Decoupled from compute - only generates NOC traffic to neighbors.
// No CB dependencies, no DRAM reading.
// Uses NOC0 to send 8KB in a loop to cores to the right and down.
// Target physical coordinates are passed as runtime arguments from host.
//
// Compile-time args:
//   0: num_iterations      – number of workload repetitions (stress-test loop)
//   1: l1_tx_A_addr        – L1 address to use as source for transfers
//   2: l1_tx_B_addr        – L1 address to use as source for transfers
//   3: l1_rx_left_addr     – L1 address to use as source for transfers
//   4: l1_rx_up_addr       – L1 address to use as source for transfers
//   5: l1_rx_right_addr    – L1 address to use as source for transfers
//   6: l1_rx_down_addr     – L1 address to use as source for transfers
//   7: transfer_size       - transfer size (8KB)
//
// Runtime args:
//   0: target_right_x      – physical x coordinate of core to the right
//   1: target_right_y      – physical y coordinate of core to the right
//   2: target_down_x       – physical x coordinate of core below
//   3: target_down_y       – physical y coordinate of core below

void kernel_main() {
    constexpr uint32_t num_iterations = get_compile_time_arg_val(0);
    constexpr uint32_t l1_tx_A_addr = get_compile_time_arg_val(1);
    constexpr uint32_t l1_tx_B_addr = get_compile_time_arg_val(2);
    constexpr uint32_t l1_rx_left_addr = get_compile_time_arg_val(3);
    constexpr uint32_t l1_rx_up_addr = get_compile_time_arg_val(4);
    constexpr uint32_t l1_rx_right_addr = get_compile_time_arg_val(5);
    constexpr uint32_t l1_rx_down_addr = get_compile_time_arg_val(6);
    constexpr uint32_t transfer_size = get_compile_time_arg_val(7);

    // Get target physical coordinates from runtime args
    uint32_t target_right_x = get_arg_val<uint32_t>(0);
    uint32_t target_right_y = get_arg_val<uint32_t>(1);
    uint32_t target_down_x = get_arg_val<uint32_t>(2);
    uint32_t target_down_y = get_arg_val<uint32_t>(3);

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        // // Send pattern A
        // // Send 8KB to core on the right (using NOC0)
        // uint64_t noc_addr_right = get_noc_addr(target_right_x, target_right_y, l1_rx_left_addr, 0);
        // noc_async_write(l1_tx_A_addr, noc_addr_right, transfer_size, 0);

        // // // Send 8KB to core down (using NOC0)
        // // uint64_t noc_addr_down = get_noc_addr(target_down_x, target_down_y, l1_rx_up_addr, 0);
        // // noc_async_write(l1_tx_A_addr, noc_addr_down, transfer_size, 0);

        // // Send pattern B
        // // Send 8KB to core on the left (using NOC0)
        // noc_async_write(l1_tx_B_addr, noc_addr_right, transfer_size, 0);

        // // Send 8KB to core down (using NOC0)
        // // noc_async_write(l1_tx_B_addr, noc_addr_down, transfer_size, 0);
    }

    noc_async_write_barrier(0);
}
