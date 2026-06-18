// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent fixed-shape D2H sender for D2HStreamService.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "pcie_noc_utils.h"
#include "../../../unified_kernels/termination.hpp"

constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t termination_semaphore_addr = get_compile_time_arg_val(1);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_socket_pages = get_compile_time_arg_val(3);
constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(4);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(5);
constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(6);
constexpr uint32_t scratch_buffer_cb_index = get_compile_time_arg_val(7);
constexpr uint32_t worker_sync_enabled = get_compile_time_arg_val(8);
constexpr uint32_t transfer_done_sem_addr = get_compile_time_arg_val(9);
constexpr uint32_t write_ack_counter_addr = get_compile_time_arg_val(10);
constexpr uint32_t worker_mcast_noc_x_start = get_compile_time_arg_val(11);
constexpr uint32_t worker_mcast_noc_y_start = get_compile_time_arg_val(12);
constexpr uint32_t worker_mcast_noc_x_end = get_compile_time_arg_val(13);
constexpr uint32_t worker_mcast_noc_y_end = get_compile_time_arg_val(14);
constexpr uint32_t num_workers = get_compile_time_arg_val(15);
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(16);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(17);
constexpr uint32_t metadata_l1_addr = get_compile_time_arg_val(18);
constexpr auto input_tensor_accessor_args = TensorAccessorArgs<19>();

void kernel_main() {
    Noc noc;
    CircularBuffer scratch_cb(scratch_buffer_cb_index);

    auto input_tensor_accessor = TensorAccessor(input_tensor_accessor_args, input_tensor_addr);

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, socket_page_size);

    const uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
    const uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;

    const uint32_t cb_l1_addr = scratch_cb.get_write_ptr();

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    // Device 2.0 migration: legacy primitives retained — the PCIe wide-write-with-state path
    // (noc_write_init_state + noc_async_wide_write_any_len_with_state, used below to drain into the
    // host-pinned socket FIFO) has no Device 2.0 equivalent; Noc::inline_dw_write is single-DW only.
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    uint64_t worker_mcast_addr = 0;
    if constexpr (worker_sync_enabled) {
        worker_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            transfer_done_sem_addr);
    }

    volatile tt_l1_ptr uint32_t* write_ack_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_ack_counter_addr);
    uint32_t last_write_ack = 0;

    bool terminated = false;
    while (!terminated) {
        if constexpr (worker_sync_enabled) {
            // Device 2.0 migration: legacy primitive retained — transfer_done_sem_addr is a GlobalSemaphore
            // address, and Semaphore<> binds to per-program ids via get_semaphore<>(id) (no GlobalSemaphore
            // wrapper exists), so Semaphore::inc_multicast cannot target it.
            noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);
            noc.async_atomic_barrier();
        }

        while (true) {
            invalidate_l1_cache();
            const uint32_t cur = *write_ack_ptr;
            if ((cur - last_write_ack) == num_workers) {
                last_write_ack = cur;
                break;
            }
            if (termination_semaphore[0] == 1) {
                terminated = true;
                break;
            }
        }
        if (terminated) {
            break;
        }

        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            const uint32_t base_page = chunk * pages_per_chunk;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                noc.async_read<NocOptions::DEFAULT, input_tensor_page_size>(
                    input_tensor_accessor,
                    scratch_cb,
                    input_tensor_page_size,
                    {.page_id = base_page + i},
                    {.offset_bytes = i * input_tensor_page_size});
            }
            noc.async_read_barrier();

            if (!deepseek_b1_ops::socket_reserve_pages_with_termination(sender_socket, 1, termination_semaphore)) {
                terminated = true;
                break;
            }

            noc_async_wide_write_any_len_with_state(
                NOC_INDEX,
                cb_l1_addr,
                pcie_xy_enc,
                ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                    sender_socket.write_ptr,
                socket_page_size);
            noc.async_writes_flushed();

            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
        }

        if (terminated) {
            break;
        }

        if constexpr (metadata_enabled) {
            if (!deepseek_b1_ops::socket_reserve_pages_with_termination(sender_socket, 1, termination_semaphore)) {
                terminated = true;
                break;
            }

            noc_async_wide_write_any_len_with_state(
                NOC_INDEX,
                metadata_l1_addr,
                pcie_xy_enc,
                ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                    sender_socket.write_ptr,
                socket_page_size);
            noc.async_writes_flushed();

            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
        }
    }

    noc.async_write_barrier();
    noc.async_read_barrier();
    update_socket_config(sender_socket);
    socket_barrier(sender_socket);
}
