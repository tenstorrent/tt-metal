// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent fixed-shape H2D receiver for H2DStreamService.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "../../../unified_kernels/termination.hpp"

constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_socket_pages = get_compile_time_arg_val(3);
constexpr uint32_t output_tensor_addr = get_compile_time_arg_val(4);
constexpr uint32_t output_tensor_page_size = get_compile_time_arg_val(5);
constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(6);
constexpr uint32_t scratch_buffer_cb_index = get_compile_time_arg_val(7);
// Worker-sync block (indices 8..15). Unused when worker_sync_enabled == 0.
constexpr uint32_t worker_sync_enabled = get_compile_time_arg_val(8);
constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(9);
constexpr uint32_t consumed_counter_addr = get_compile_time_arg_val(10);
constexpr uint32_t worker_mcast_noc_x_start = get_compile_time_arg_val(11);
constexpr uint32_t worker_mcast_noc_y_start = get_compile_time_arg_val(12);
constexpr uint32_t worker_mcast_noc_x_end = get_compile_time_arg_val(13);
constexpr uint32_t worker_mcast_noc_y_end = get_compile_time_arg_val(14);
constexpr uint32_t num_workers = get_compile_time_arg_val(15);
// Metadata multicast block (indices 16..18). Unused when metadata_enabled == 0.
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(16);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(17);
constexpr uint32_t metadata_l1_addr = get_compile_time_arg_val(18);
constexpr auto output_tensor_accessor_args = TensorAccessorArgs<19>();

// Reads one socket page from PCIe host RAM into L1; caller must barrier afterward.
inline void noc_read_page_chunked(uint32_t pcie_xy_enc, uint64_t src_pcie, uint32_t dst_l1, uint32_t size) {
    while (size) {
        uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            NOC_INDEX, pcie_xy_enc, src_pcie, dst_l1, chunk);
        src_pcie += chunk;
        dst_l1 += chunk;
        size -= chunk;
    }
}

void kernel_main() {
    auto output_tensor_accessor = TensorAccessor(output_tensor_accessor_args, output_tensor_addr);

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, socket_page_size);

    const uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;
    const uint64_t base_pinned =
        (static_cast<uint64_t>(receiver_socket.h2d.data_addr_hi) << 32) | receiver_socket.h2d.data_addr_lo;

    const uint32_t cb_l1_addr = get_write_ptr(scratch_buffer_cb_index);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    uint64_t worker_mcast_addr = 0;
    volatile tt_l1_ptr uint32_t* consumed_ptr = nullptr;
    uint32_t last_consumed = 0;
    if constexpr (worker_sync_enabled) {
        worker_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            data_ready_sem_addr);
        consumed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_counter_addr);
    }

    uint64_t metadata_mcast_addr = 0;
    if constexpr (metadata_enabled) {
        metadata_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            metadata_l1_addr);
    }

    bool terminated = false;
    while (!terminated) {
        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            if (!deepseek_b1_ops::socket_wait_for_pages_with_termination(receiver_socket, 1, termination_semaphore)) {
                terminated = true;
                break;
            }

            noc_read_page_chunked(
                pcie_xy_enc,
                base_pinned + receiver_socket.read_ptr - receiver_socket.fifo_addr,
                cb_l1_addr,
                socket_page_size);
            noc_async_read_barrier();

            const uint32_t base_page = chunk * pages_per_chunk;
            uint32_t src = cb_l1_addr;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                const uint64_t noc_dst = output_tensor_accessor.get_noc_addr(base_page + i);
                noc_async_write<output_tensor_page_size>(src, noc_dst, output_tensor_page_size);
                src += output_tensor_page_size;
            }

            noc_async_write_barrier();

            socket_pop_pages(receiver_socket, 1);
            socket_notify_sender(receiver_socket);
        }
        if constexpr (metadata_enabled) {
            if (!deepseek_b1_ops::socket_wait_for_pages_with_termination(
                    receiver_socket, 1, termination_semaphore)) {
                terminated = true;
                break;
            }
            noc_read_page_chunked(
                pcie_xy_enc,
                base_pinned + receiver_socket.read_ptr - receiver_socket.fifo_addr,
                cb_l1_addr,
                socket_page_size);
            noc_async_read_barrier();

            noc_async_write_multicast(
                cb_l1_addr,
                metadata_mcast_addr,
                metadata_size_bytes,
                /*num_dests=*/num_workers);
            noc_async_write_barrier();

            socket_pop_pages(receiver_socket, 1);
            socket_notify_sender(receiver_socket);
        }

        if (terminated) {
            break;
        }

        if constexpr (worker_sync_enabled) {
            noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);

            while (true) {
                invalidate_l1_cache();
                const uint32_t cur = *consumed_ptr;
                if ((cur - last_consumed) == num_workers) {
                    last_consumed = cur;
                    break;
                }
            }
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    update_socket_config(receiver_socket);
}
