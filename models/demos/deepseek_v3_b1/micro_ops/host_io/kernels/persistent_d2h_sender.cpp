// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent fixed-shape D2H sender for D2HStreamService.
//
// Worker-sync protocol (mirror of H2D consumed-counter handshake):
//   1. Service core multicasts transfer_done_sem → workers may write backing DRAM.
//   2. Each worker writes its slice; when metadata is enabled the designated
//      metadata master worker fans the replicated metadata IN to this service
//      core's metadata L1 region before acking. Every worker (incl. master)
//      atomic-incs the service-core write_ack counter.
//   3. Service core waits for exactly num_workers acks, then streams the backing
//      tensor to the host FIFO and, if metadata is enabled, reads this core's
//      metadata L1 region and ships it as the trailing socket page.
//
// Host-only path (worker_sync_enabled == 0): host bumps write_ack via
// notify_backing_ready() before read_from_tensor().

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
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
    auto input_tensor_accessor = TensorAccessor(input_tensor_accessor_args, input_tensor_addr);

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, socket_page_size);

    const uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
    const uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;

    // Single-slot scratch CB; use the write pointer consistently across PCIe-in and
    // NoC-out since no producer/consumer split exists in this kernel.
    const uint32_t cb_l1_addr = get_write_ptr(scratch_buffer_cb_index);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

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
            // Phase 1: unlock backing DRAM for worker writes this iteration. The
            // workers' transfer_done global semaphore starts at 0 (= locked) so they
            // block until the service core releases the backing tensor memory region;
            // the very first thing this loop does each iteration is unlock it.
            noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);
            noc_async_atomic_barrier();
        }

        // Phase 2: wait for all producers to ack before sending.
        // In tests (host-only path), notify_backing_ready() is called so the host plays the producer,
        // writing the backing tensor and bumping write_ack just as a worker would.
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

        // Phase 3: read backing tensor and stream to host FIFO. This loop reads num_socket_pages chunks of
        // pages_per_chunk pages each from the input tensor and writes them to the scratch CB
        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            // DRAM -> L1 scratch CB
            const uint32_t base_page = chunk * pages_per_chunk;
            uint32_t dst = cb_l1_addr;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                const uint64_t noc_src = input_tensor_accessor.get_noc_addr(base_page + i);
                noc_async_read<input_tensor_page_size>(noc_src, dst, input_tensor_page_size);
                dst += input_tensor_page_size;
            }
            noc_async_read_barrier();

            // Wait for FIFO Space
            if (!deepseek_b1_ops::socket_reserve_pages_with_termination(sender_socket, 1, termination_semaphore)) {
                terminated = true;
                break;
            }

            // L1 scratch CB -> Host FIFO over PCIe
            noc_async_wide_write_any_len_with_state(
                NOC_INDEX,
                cb_l1_addr,
                pcie_xy_enc,
                ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                    sender_socket.write_ptr,
                socket_page_size);
            noc_async_writes_flushed();

            // Push to Host FIFO
            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
        }

        if (terminated) {
            break;
        }

        if constexpr (metadata_enabled) {
            // Ship the metadata as the trailing socket page. metadata_l1_addr is the
            // service-core staging region the metadata master worker fanned the
            // metadata in to: it is a full, zero-padded socket page on this same service core
            // and is NOC-accessible, so write it straight to the host FIFO — no
            // intermediate scratch-CB copy needed.
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
            noc_async_writes_flushed();

            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
        }
    }

    noc_async_write_barrier();
    noc_async_read_barrier();
    update_socket_config(sender_socket);
    socket_barrier(sender_socket);
}
