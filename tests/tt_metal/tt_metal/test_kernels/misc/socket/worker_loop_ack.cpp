// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t config_sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t credits0_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t credits1_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t worker_local_data_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr uint32_t data_size = get_compile_time_arg_val(6);
    constexpr uint32_t receiver_core_x = get_compile_time_arg_val(7);
    constexpr uint32_t receiver_core_y = get_compile_time_arg_val(8);
    constexpr uint32_t output_data_addr = get_compile_time_arg_val(9);

    uint32_t remote_data_core_x = get_arg_val<uint32_t>(0);
    uint32_t remote_data_core_y = get_arg_val<uint32_t>(1);
    uint32_t output_data_core_x = get_arg_val<uint32_t>(2);
    uint32_t output_data_core_y = get_arg_val<uint32_t>(3);

    uint32_t socket_config_addr = get_write_ptr(config_cb_id);
    volatile tt_l1_ptr uint32_t* config_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(config_sem_id));

    volatile tt_l1_ptr uint32_t* curr_credits_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credits0_sem_id));
    volatile tt_l1_ptr uint32_t* next_credits_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credits1_sem_id));

    // Could just use a bit in the config instead of separate sem
    uint64_t credits_sem_noc_addr = get_noc_addr(receiver_core_x, receiver_core_y, 0);
    uint64_t remote_data_noc_addr = get_noc_addr(remote_data_core_x, remote_data_core_y, 0);
    uint64_t output_data_noc_addr = get_noc_addr(output_data_core_x, output_data_core_y, output_data_addr);

    noc_semaphore_wait(config_sem_addr, 1);

    constexpr uint32_t num_pages = data_size / page_size;
    // Better optimize to not init the receiver socket interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, page_size);

    // Loop can be optimized to not be page based, and read larger chunks/more in parallel at once.
    for (uint32_t i = 0; i < num_pages; ++i) {
        noc_semaphore_wait(curr_credits_sem_addr, 1);
        uint64_t read_addr = remote_data_noc_addr | receiver_socket.read_ptr;
        cb_reserve_back(worker_local_data_cb_id, 1);
        uint32_t write_addr = get_write_ptr(worker_local_data_cb_id);
        noc_async_read(read_addr, write_addr, page_size);
        // Just used for wrapping semantics. Add better impl for sockets
        socket_pop_pages(receiver_socket, 1);
        noc_async_read_barrier();
        cb_push_back(worker_local_data_cb_id, 1);
        noc_async_write(write_addr, output_data_noc_addr, page_size);
        output_data_noc_addr += page_size;
        // Just accumulate instead of writing to L1
        *curr_credits_sem_addr = 0;
        cb_pop_front(worker_local_data_cb_id, 1);
        noc_async_writes_flushed();
        noc_semaphore_inc(credits_sem_noc_addr | (uint32_t)curr_credits_sem_addr, 1);
        std::swap(curr_credits_sem_addr, next_credits_sem_addr);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
