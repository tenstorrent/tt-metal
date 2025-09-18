// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t config_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t credits_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t page_size = get_compile_time_arg_val(4);
    constexpr uint32_t data_size = get_compile_time_arg_val(5);
    constexpr uint32_t worker_core_start_x = get_compile_time_arg_val(6);
    constexpr uint32_t worker_core_start_y = get_compile_time_arg_val(7);
    constexpr uint32_t worker_core_end_x = get_compile_time_arg_val(8);
    constexpr uint32_t worker_core_end_y = get_compile_time_arg_val(9);
    constexpr uint32_t num_worker_cores = get_compile_time_arg_val(10);

    uint32_t worker_config_addr = get_write_ptr(config_cb_id);
    uint64_t worker_config_mcast_noc_addr = get_noc_multicast_addr(
        worker_core_start_x, worker_core_start_y, worker_core_end_x, worker_core_end_y, worker_config_addr);

    // Could just use a bit in the config instead of separate sem
    volatile tt_l1_ptr uint32_t* worker_config_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(config_sem_id));
    *worker_config_sem_ptr = 1;
    uint64_t worker_config_sem_mcast_noc_addr = get_noc_multicast_addr(
        worker_core_start_x,
        worker_core_start_y,
        worker_core_end_x,
        worker_core_end_y,
        (uint32_t)worker_config_sem_ptr);

    noc_async_write_multicast(
        socket_config_addr,
        worker_config_mcast_noc_addr,
        sizeof(SocketReceiverInterface) /* Not the correct sizes */,
        num_worker_cores,
        true);
    noc_semaphore_set_multicast((uint32_t)worker_config_sem_ptr, worker_config_sem_mcast_noc_addr, num_worker_cores);

    constexpr uint32_t num_pages = data_size / page_size;
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, page_size);

    volatile tt_l1_ptr uint32_t* credits_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credits_sem_id));
    uint64_t credits_sem_mcast_noc_addr = get_noc_multicast_addr(
        worker_core_start_x, worker_core_start_y, worker_core_end_x, worker_core_end_y, (uint32_t)credits_sem_ptr);

    // Loop can be optimized to not be page based, and send all available credits at once.
    for (uint32_t i = 0; i < num_pages; ++i) {
        // Should be safe even if we don't flush before modifying the value in l1 since it's an always increasing word
        // Otherwise insert a flush before modifying
        *credits_sem_ptr = i + 1;
        socket_wait_for_pages(receiver_socket, i + 1);
        // TODO: Can mcast inline write
        noc_semaphore_set_multicast((uint32_t)credits_sem_ptr, credits_sem_mcast_noc_addr, num_worker_cores);
    }
    noc_semaphore_wait(worker_config_sem_ptr, num_worker_cores + 1);
    socket_pop_pages(receiver_socket, num_pages);
    socket_notify_sender(receiver_socket);
    update_socket_config(receiver_socket);
    noc_async_write_barrier();
}
