// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h"

// Optional launch barrier for the max-utilization workload. The top-left
// worker waits until every other worker's compute kernel is ready, then
// releases the full grid with a multicast semaphore update.
//
// Compile-time args:
//   0: super_sync_sender_semaphore_id
//   1: super_sync_receiver_semaphore_id
//   2: l1_super_sync_addr
//
// Runtime args:
//   0: is_sender
//   1: sender NOC0 x
//   2: sender NOC0 y
//   3: multicast end NOC0 x
//   4: multicast end NOC0 y
//   5: number of destination workers
//   6: this worker's NOC0 x
//   7: this worker's NOC0 y

void kernel_main() {
    constexpr uint32_t super_sync_sender_semaphore_id = get_compile_time_arg_val(0);
    constexpr uint32_t super_sync_receiver_semaphore_id = get_compile_time_arg_val(1);
    constexpr uint32_t l1_super_sync_addr = get_compile_time_arg_val(2);

    uint32_t is_sender = get_arg_val<uint32_t>(0);
    uint32_t super_sync_core_x = get_arg_val<uint32_t>(1);
    uint32_t super_sync_core_y = get_arg_val<uint32_t>(2);
    uint32_t super_sync_mcast_end_x = get_arg_val<uint32_t>(3);
    uint32_t super_sync_mcast_end_y = get_arg_val<uint32_t>(4);
    uint32_t num_dests = get_arg_val<uint32_t>(5);
    uint32_t core_x = get_arg_val<uint32_t>(6);
    uint32_t core_y = get_arg_val<uint32_t>(7);

    // Farther cores wait less to compensate for multicast propagation delay.
    const uint32_t x_distance = core_x > super_sync_core_x ? core_x - super_sync_core_x : super_sync_core_x - core_x;
    const uint32_t y_distance = core_y > super_sync_core_y ? core_y - super_sync_core_y : super_sync_core_y - core_y;
    const uint32_t distance_from_super_sync_core = x_distance + y_distance;
    const uint32_t distance_compensation = distance_from_super_sync_core * 9;
    const uint32_t cycles_to_wait = distance_compensation < 220 ? 220 - distance_compensation : 0;
    const uint32_t super_sync_core_wait_cycles = 600;

    const uint64_t super_sync_sender_semaphore_addr = get_semaphore(super_sync_sender_semaphore_id);
    const uint64_t super_sync_receiver_semaphore_addr = get_semaphore(super_sync_receiver_semaphore_id);
    volatile tt_l1_ptr uint32_t* super_sync_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(super_sync_sender_semaphore_addr);
    const uint64_t super_sync_sender_semaphore_noc_addr =
        get_noc_addr(super_sync_core_x, super_sync_core_y, super_sync_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* super_sync_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(super_sync_receiver_semaphore_addr);
    const uint64_t super_sync_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        super_sync_core_x,
        super_sync_core_y,
        super_sync_mcast_end_x,
        super_sync_mcast_end_y,
        super_sync_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* l1_super_sync_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_super_sync_addr);

    if (is_sender) {
        (*super_sync_receiver_semaphore_addr_ptr) = VALID;
        noc_semaphore_wait(l1_super_sync_addr_ptr, 1);
        noc_semaphore_wait(super_sync_sender_semaphore_addr_ptr, num_dests);
        noc_semaphore_set(super_sync_sender_semaphore_addr_ptr, 0);
        noc_semaphore_set_multicast(
            super_sync_receiver_semaphore_addr, super_sync_receiver_semaphore_noc_addr, num_dests);
        ckernel::wait(super_sync_core_wait_cycles);
        noc_semaphore_set(l1_super_sync_addr_ptr, 0);
    } else {
        noc_semaphore_wait(l1_super_sync_addr_ptr, 1);
        noc_semaphore_set(super_sync_receiver_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(super_sync_sender_semaphore_noc_addr, 1);
        noc_semaphore_wait(super_sync_receiver_semaphore_addr_ptr, VALID);
        ckernel::wait(cycles_to_wait);
        noc_semaphore_set(l1_super_sync_addr_ptr, 0);
    }
}
