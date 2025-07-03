// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    constexpr uint32_t config0_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t config0_sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t config1_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t config1_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t credits0_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t credits1_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t in0_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t in1_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t receiver_core0_x = get_compile_time_arg_val(8);
    constexpr uint32_t receiver_core0_y = get_compile_time_arg_val(9);
    constexpr uint32_t receiver_core1_x = get_compile_time_arg_val(10);
    constexpr uint32_t receiver_core1_y = get_compile_time_arg_val(11);
    constexpr uint32_t output_data_core_x = get_compile_time_arg_val(12);
    constexpr uint32_t output_data_core_y = get_compile_time_arg_val(13);
    constexpr uint32_t output_data_addr = get_compile_time_arg_val(14);
    constexpr uint32_t page_size = get_compile_time_arg_val(15);
    constexpr uint32_t data_size = get_compile_time_arg_val(16);

    // Receivers write socket configs to these addresses
    uint32_t socket_config_addr0 = get_write_ptr(config0_cb_id);
    uint32_t socket_config_addr1 = get_write_ptr(config1_cb_id);
    // Receivers update this semaphore after configs are written
    volatile tt_l1_ptr uint32_t* config0_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(config0_sem_id));
    volatile tt_l1_ptr uint32_t* config1_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(config1_sem_id));
    // Receivers send data credits to this semaphore
    volatile tt_l1_ptr uint32_t* credits0_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credits0_sem_id));
    volatile tt_l1_ptr uint32_t* credits1_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credits1_sem_id));
    // This kernel notifies receivers that data has been read through these credits
    uint64_t credits0_sem_noc_addr = get_noc_addr(receiver_core0_x, receiver_core0_y, (uint32_t)credits0_sem_addr);
    uint64_t credits1_sem_noc_addr = get_noc_addr(receiver_core1_x, receiver_core1_y, (uint32_t)credits1_sem_addr);

    uint64_t output_data_noc_addr = get_noc_addr(output_data_core_x, output_data_core_y, output_data_addr);
    constexpr uint32_t num_pages = data_size / page_size;

    noc_semaphore_wait(config0_sem_addr, 1);
    noc_semaphore_wait(config1_sem_addr, 1);
    // Create Socket Interfaces from configs supplied by receivers
    SocketReceiverInterface receiver_socket0 = create_receiver_socket_interface(socket_config_addr0);
    SocketReceiverInterface receiver_socket1 = create_receiver_socket_interface(socket_config_addr1);

    set_receiver_socket_page_size(receiver_socket0, page_size);
    set_receiver_socket_page_size(receiver_socket1, page_size);

    for (uint32_t i = 0; i < num_pages; i++) {
        // Wait for receivers to notify that data is available to read
        noc_semaphore_wait_min(credits0_sem_addr, i + 1);
        noc_semaphore_wait_min(credits1_sem_addr, i + 1);

        uint64_t read_addr0 = get_noc_addr(receiver_core0_x, receiver_core0_y, receiver_socket0.read_ptr);
        uint64_t read_addr1 = get_noc_addr(receiver_core1_x, receiver_core1_y, receiver_socket1.read_ptr);

        uint32_t in0_local_addr = get_write_ptr(in0_cb_id);
        uint32_t in1_local_addr = get_write_ptr(in1_cb_id);
        cb_reserve_back(in0_cb_id, 1);
        cb_reserve_back(in1_cb_id, 1);

        noc_async_read(read_addr0, in0_local_addr, page_size);
        noc_async_read(read_addr1, in1_local_addr, page_size);

        noc_async_read_barrier();

        socket_pop_pages(receiver_socket0, 1);
        socket_pop_pages(receiver_socket1, 1);

        cb_push_back(in0_cb_id, 1);
        cb_push_back(in1_cb_id, 1);

        volatile tt_l1_ptr uint32_t* data0_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_local_addr);
        volatile tt_l1_ptr uint32_t* data1_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_local_addr);

        // Reduce into in0 CB (safe because we won't read into this slot until the write is issued)
        for (uint32_t i = 0; i < page_size / sizeof(uint32_t); i++) {
            data0_addr[i] = data0_addr[i] + data1_addr[i];
        }
        noc_async_write(in0_local_addr, output_data_noc_addr, page_size);
        output_data_noc_addr += page_size;
        cb_pop_front(in0_cb_id, 1);
        cb_pop_front(in1_cb_id, 1);
        noc_async_writes_flushed();
        noc_semaphore_inc(credits0_sem_noc_addr, 1);
        noc_semaphore_inc(credits1_sem_noc_addr, 1);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
