// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2H (Device to Host) Socket Sender Kernel
// This kernel sends data from device L1 memory to host via PCIe.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

void kernel_main() {
    // Compile-time arguments from host
    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);

    // Create socket interface from config buffer
    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, page_size);

    // Read PCIe-specific config from the config buffer
    // Layout: 8 words MD + 4 words ack + downstream encoding area
    // [12] = pcie_xy_enc, [13] = data_addr_hi, [14] = bytes_sent_addr_hi
    // [2] = data_addr_lo (write_ptr)
    tt_l1_ptr uint32_t* socket_config_words = reinterpret_cast<tt_l1_ptr uint32_t*>(socket_config_addr);
    uint32_t pcie_xy_enc = socket_config_words[12];
    uint32_t data_addr_hi = socket_config_words[13];
    uint32_t data_addr_lo = socket_config_words[2];  // Initial write_ptr = data buffer start

    uint32_t outstanding_data_size = data_size;
    uint32_t src_addr = local_l1_buffer_addr;

    // Calculate page-aligned FIFO size
    uint32_t fifo_page_aligned_size =
        sender_socket.downstream_fifo_total_size - (sender_socket.downstream_fifo_total_size % page_size);

    // Track write pointer in host buffer
    uint32_t host_write_ptr = data_addr_lo;
    uint32_t host_fifo_start = data_addr_lo;

    // Initialize NOC for PCIe writes
    noc_write_init_state<0>(NOC_0, NOC_UNICAST_WRITE_VC);

    // Send data one page at a time
    while (outstanding_data_size) {
        // DeviceZoneScopedN("Loop");
        //  Wait for space in the receiver's FIFO
        {
            DeviceZoneScopedN("Res");
            socket_reserve_pages(sender_socket, 1);
        }

        // Build 64-bit PCIe destination address
        uint64_t pcie_dest_addr = (static_cast<uint64_t>(data_addr_hi) << 32) | static_cast<uint64_t>(host_write_ptr);

        // Write data to PCIe-mapped host memory using PCIe write primitive
        noc_wwrite_with_state<DM_DEDICATED_NOC, 0, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_0, src_addr, pcie_xy_enc, pcie_dest_addr, page_size, 1);

        src_addr += page_size;
        outstanding_data_size -= page_size;

        // Update host write pointer with wrap-around
        host_write_ptr += page_size;
        if (host_write_ptr >= host_fifo_start + fifo_page_aligned_size) {
            host_write_ptr = host_fifo_start;
        }

        {
            // DeviceZoneScopedN("Push");
            //  Update socket state
            socket_push_pages(sender_socket, 1);
        }

        {
            // DeviceZoneScopedN("Note");
            //  Notify host via PCIe
            pcie_socket_notify_receiver(sender_socket);
            // Barrier to ensure PCIe write is visible to host before next iteration
            noc_async_write_barrier();
        }
    }

    // Wait for all acks from receiver
    socket_barrier(sender_socket);

    // Write final socket state back to L1
    update_socket_config(sender_socket);
}
