// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t physical_start_x = get_compile_time_arg_val(1);
    constexpr uint32_t physical_start_y = get_compile_time_arg_val(2);
    constexpr uint32_t physical_end_x = get_compile_time_arg_val(3);
    constexpr uint32_t physical_end_y = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores_c_dim = get_compile_time_arg_val(5);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t sender_valid_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(8);

    uint32_t l1_base_address = get_arg_val<uint32_t>(0);        // sender: source of in0 data
    uint32_t mcast_size_bytes = get_arg_val<uint32_t>(1);       // sender: bytes to multicast
    uint32_t in0_mcast_output_addr = get_arg_val<uint32_t>(2);  // all cores: multicast dest

    uint32_t sender_sem_addr = get_semaphore(sender_sem_id);
    uint32_t sender_valid_sem_addr = get_semaphore(sender_valid_sem_id);
    uint32_t receiver_sem_addr = get_semaphore(receiver_sem_id);

    volatile tt_l1_ptr uint32_t* sender_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_sem_addr);
    volatile tt_l1_ptr uint32_t* receiver_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem_addr);

    bool is_sender = (my_x[0] == physical_start_x);

    uint64_t dst_data_mcast_addr =
        get_noc_multicast_addr(physical_start_x, my_y[0], physical_end_x, my_y[0], in0_mcast_output_addr);

    uint64_t dst_receiver_sem_mcast_addr =
        get_noc_multicast_addr(physical_start_x, my_y[0], physical_end_x, my_y[0], receiver_sem_addr);

    uint64_t sender_sem_noc_addr = get_noc_addr(physical_start_x, my_y[0], sender_sem_addr);

    if (is_sender) {
        noc_semaphore_wait(sender_sem_ptr, num_cores_c_dim - 1);
        noc_semaphore_set(sender_sem_ptr, 0);

        noc_async_write_multicast_loopback_src(
            l1_base_address, dst_data_mcast_addr, mcast_size_bytes, num_cores_c_dim, true);

        noc_semaphore_set_multicast_loopback_src(
            sender_valid_sem_addr, dst_receiver_sem_mcast_addr, num_cores_c_dim, false);
    } else {
        noc_semaphore_inc(sender_sem_noc_addr, 1);
    }

    noc_semaphore_wait(receiver_sem_ptr, 1);
    noc_semaphore_set(receiver_sem_ptr, 0);
}
