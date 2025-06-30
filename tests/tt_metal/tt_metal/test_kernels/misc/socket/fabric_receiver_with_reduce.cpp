// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t data_size = get_compile_time_arg_val(2);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t config_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t credits_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t reduction_core_x = get_compile_time_arg_val(6);
    constexpr uint32_t reduction_core_y = get_compile_time_arg_val(7);

    // Setup Socket Config on Worker
    uint32_t worker_config_addr = get_write_ptr(config_cb_id);
    uint64_t worker_config_unicast_noc_addr = get_noc_addr(reduction_core_x, reduction_core_y, worker_config_addr);
    uint64_t worker_config_sem_noc_addr =
        get_noc_addr(reduction_core_x, reduction_core_y, (uint32_t)get_semaphore(config_sem_id));

    noc_async_write(socket_config_addr, worker_config_unicast_noc_addr, sizeof(SocketReceiverInterface));
    noc_semaphore_inc(worker_config_sem_noc_addr, 1);

    size_t rt_args_idx = 0;
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    constexpr uint32_t fabric_packet_header_cb_id = 0;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));

    fabric_connection.open();
    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    uint32_t outstanding_data_size = data_size;
    set_receiver_socket_page_size(receiver_socket, page_size);

    volatile tt_l1_ptr uint32_t* credits_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credits_sem_id));

    uint64_t credits_sem_noc_addr = get_noc_addr(reduction_core_x, reduction_core_y, (uint32_t)credits_sem_addr);

    // Reads one page at a time and sends ack to sender, can be optimized to notify
    // receiver after reading larger chunks
    constexpr uint32_t num_pages = data_size / page_size;
    for (uint32_t i = 0; i < num_pages; ++i) {
        socket_wait_for_pages(receiver_socket, 1);
        noc_inline_dw_write(credits_sem_noc_addr, i + 1);
        noc_semaphore_wait(credits_sem_addr, i + 1);
        socket_pop_pages(receiver_socket, 1);
        fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
    }

    update_socket_config(receiver_socket);
    fabric_connection.close();
}
