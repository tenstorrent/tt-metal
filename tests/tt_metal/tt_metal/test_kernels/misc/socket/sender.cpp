// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "dataflow_api.h"
#include "socket_api.h"

void kernel_main() {
    // Get this value from MeshSocket struct on host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    // Get Receiver Side Data Cores through RTAs populated by host Socket Queries

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    tt_l1_ptr uint32_t* socket_config_words = reinterpret_cast<tt_l1_ptr uint32_t*>(socket_config_addr);

    uint32_t pcie_xy_enc = socket_config_words[12];  // 8 words of MD + 4 words of ack
    uint32_t data_addr_hi = socket_config_words[13];

    uint32_t outstanding_data_size = data_size;
    set_sender_socket_page_size(sender_socket, page_size);

    uint32_t data_addr = local_l1_buffer_addr;
    // Sends 1 page at a time and does handshake with receiver, can be optimized
    // to notify receiver after writing larger chunks
    while (outstanding_data_size) {
        socket_reserve_pages(sender_socket, 1);
        // noc_async_read into local CB + barrier
        // fabric write from local CB to data cores + barrier
        // Issuing the write requires the wptr from the socket itself
        // The user can get the wptr directly from the sender_socket, or
        // we can add wrappers issue the write itself
        noc_write_init_state<0>(NOC_0, NOC_UNICAST_WRITE_VC);
        uint64_t pcie_data_addr =
            (static_cast<uint64_t>(data_addr_hi) << 32) | (static_cast<uint32_t>(sender_socket.write_ptr));
        for (uint32_t i = 0; i < sender_socket.num_downstreams; i++) {
            // sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, i);
            noc_wwrite_with_state<DM_DEDICATED_NOC, 0, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                NOC_0, data_addr, pcie_xy_enc, pcie_data_addr, page_size, 1);
        }
        data_addr += page_size;
        outstanding_data_size -= page_size;
        noc_async_write_barrier();
        socket_push_pages(sender_socket, 1);
        pcie_socket_notify_receiver(sender_socket);
    }
    socket_barrier(sender_socket);
    // Write updated socket configs to the L1 config buffer (were cached on stack during kernel execution)
    update_socket_config(sender_socket);
}
