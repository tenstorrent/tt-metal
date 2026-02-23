// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    DeviceZoneScopedN("kernel_main_total");

    constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t local_l1_buffer_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t data_size = get_compile_time_arg_val(3);
    constexpr uint32_t measurement_buffer_addr = get_compile_time_arg_val(4);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(5);

    SocketReceiverInterface receiver_socket;
    uint64_t pcie_data_addr;
    uint32_t pcie_xy_enc;
    constexpr uint32_t max_noc_burst_bytes = NOC_MAX_BURST_SIZE;
    uint64_t start_timestamp;

    {
        DeviceZoneScopedN("setup");
        receiver_socket = create_receiver_socket_interface(socket_config_addr);
        set_receiver_socket_page_size(receiver_socket, page_size);

        // Set up PCIe read addresses for host pinned RAM
        pcie_data_addr = (static_cast<uint64_t>(receiver_socket.h2d.data_addr_hi) << 32) |
                         (static_cast<uint64_t>(receiver_socket.h2d.data_addr_lo));
        pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;
        start_timestamp = get_timestamp();
    }

    uint32_t outstanding_data_size;
    uint64_t dst_noc_addr;
    uint64_t page_src_addr;
    uint32_t page_dst_addr;
    uint32_t page_bytes_remaining;

    for (uint32_t i = 0; i < num_iterations; i++) {
        {
            DeviceZoneScopedN("It setup");
            outstanding_data_size = data_size;
            dst_noc_addr = get_noc_addr(local_l1_buffer_addr);
            socket_wait_for_pages(receiver_socket, 1);
            page_src_addr = pcie_data_addr + receiver_socket.read_ptr - receiver_socket.fifo_addr;
            page_dst_addr = receiver_socket.read_ptr;
            page_bytes_remaining = page_size;
        }

        {
            DeviceZoneScopedN("It loop");
            while (page_bytes_remaining) {
                uint32_t chunk_bytes =
                    (page_bytes_remaining > max_noc_burst_bytes) ? max_noc_burst_bytes : page_bytes_remaining;
                noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                    NOC_INDEX, pcie_xy_enc, page_src_addr, page_dst_addr, chunk_bytes);
                page_src_addr += chunk_bytes;
                page_dst_addr += chunk_bytes;
                page_bytes_remaining -= chunk_bytes;
            }
        }

        {
            DeviceZoneScopedN("It barrier");
            noc_async_read_barrier();
            noc_async_write(receiver_socket.read_ptr, dst_noc_addr, page_size);
            dst_noc_addr += page_size;
            noc_async_write_barrier();
            socket_pop_pages(receiver_socket, 1);
            socket_notify_sender(receiver_socket);
        }
    }
    noc_async_write_barrier();

    uint64_t end_timestamp = get_timestamp();
    *reinterpret_cast<volatile uint64_t*>(measurement_buffer_addr) = end_timestamp - start_timestamp;

    update_socket_config(receiver_socket);
}
