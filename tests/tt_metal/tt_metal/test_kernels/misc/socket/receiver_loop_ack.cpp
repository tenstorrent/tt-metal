// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t config_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t credits0_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t credits1_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t page_size = get_compile_time_arg_val(5);
    constexpr uint32_t data_size = get_compile_time_arg_val(6);
    constexpr uint32_t worker_core_start_x = get_compile_time_arg_val(7);
    constexpr uint32_t worker_core_start_y = get_compile_time_arg_val(8);
    constexpr uint32_t worker_core_end_x = get_compile_time_arg_val(9);
    constexpr uint32_t worker_core_end_y = get_compile_time_arg_val(10);
    constexpr uint32_t num_worker_cores = get_compile_time_arg_val(11);

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

    uint32_t sending_credits_sem_addr = get_semaphore(credits0_sem_id);
    uint32_t waiting_credits_sem_addr = get_semaphore(credits1_sem_id);
    uint64_t credits_sem_mcast_noc_addr =
        get_noc_multicast_addr(worker_core_start_x, worker_core_start_y, worker_core_end_x, worker_core_end_y, 0);

    // Loop can be optimized to not be page based, and send all available credits at once.
    *reinterpret_cast<volatile uint32_t*>(sending_credits_sem_addr) = 1;
    socket_wait_for_pages(receiver_socket, 1);
    noc_semaphore_set_multicast(
        sending_credits_sem_addr, credits_sem_mcast_noc_addr | sending_credits_sem_addr, num_worker_cores);
    std::swap(sending_credits_sem_addr, waiting_credits_sem_addr);
    for (uint32_t i = 1; i < num_pages; ++i) {
        // Just accumulate instead of writing to L1
        *reinterpret_cast<volatile uint32_t*>(sending_credits_sem_addr) = 1;
        // Wait for 2 since we haven't popped the prev page yet
        // Can wait for one if we move the pop earlier
        socket_wait_for_pages(receiver_socket, 2);
        noc_semaphore_set_multicast(
            sending_credits_sem_addr, credits_sem_mcast_noc_addr | sending_credits_sem_addr, num_worker_cores);
        // Pop before waiting to avoid doing it after receiving acks from workers
        socket_pop_pages(receiver_socket, 1);
        noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(waiting_credits_sem_addr), num_worker_cores + 1);
        // We can accumulate acks for some number of loops before notifying sender to better optimize
        socket_notify_sender(receiver_socket);
        std::swap(sending_credits_sem_addr, waiting_credits_sem_addr);
    }
    socket_pop_pages(receiver_socket, 1);
    noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(waiting_credits_sem_addr), num_worker_cores + 1);
    socket_notify_sender(receiver_socket);
    update_socket_config(receiver_socket);
    noc_async_write_barrier();
}
