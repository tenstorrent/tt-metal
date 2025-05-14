// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t src_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_data_cores = get_compile_time_arg_val(4);
    constexpr uint32_t src_core_x = get_compile_time_arg_val(5);
    constexpr uint32_t src_core_y = get_compile_time_arg_val(6);
    constexpr uint32_t scratch_cb_index = get_compile_time_arg_val(7);

    std::array<uint64_t, num_data_cores> data_noc_coord_addrs;
    for (uint32_t i = 0; i < num_data_cores; ++i) {
        data_noc_coord_addrs[i] = get_noc_addr(get_arg_val<uint32_t>(i * 2), get_arg_val<uint32_t>(i * 2 + 1), 0);
    }

    // Get Receiver Side Data Cores through RTAs populated by host Socket Queries

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    uint32_t outstanding_data_size = data_size;
    set_sender_socket_page_size(sender_socket, page_size);

    uint64_t receiver_noc_coord_addr = get_noc_addr(sender_socket.downstream_noc_x, sender_socket.downstream_noc_y, 0);

    uint64_t src_noc_addr = get_noc_addr(src_core_x, src_core_y, src_buffer_addr);

    uint32_t local_write_addr = get_write_ptr(scratch_cb_index);

    noc_async_read(src_noc_addr, local_write_addr, num_data_cores * data_size);
    noc_async_read_barrier();

    uint32_t data_addr = local_write_addr;

    // Sends 1 page at a time and does handshake with receiver, can be optimized
    // to notify receiver after writing larger chunks
    uint32_t i = 0;
    while (outstanding_data_size) {
        socket_reserve_pages(sender_socket, 1);
        for (uint32_t i = 0; i < num_data_cores; ++i) {
            // noc_async_read into local CB + barrier
            // fabric write from local CB to data cores + barrier
            // Issuing the write requires the wptr from the socket itself
            // The user can get the wptr directly from the sender_socket, or
            // we can add wrappers issue the write itself
            noc_async_write(data_addr, data_noc_coord_addrs[i] | sender_socket.write_ptr, page_size);
            data_addr += page_size;
        }
        outstanding_data_size -= page_size;
        socket_push_pages(sender_socket, 1);
        noc_async_write_barrier();
        socket_notify_receiver(sender_socket);
    }
    socket_barrier(sender_socket);
    // Write updated socket configs to the L1 config buffer (were cached on stack during kernel execution)
    update_socket_config(sender_socket);
}
