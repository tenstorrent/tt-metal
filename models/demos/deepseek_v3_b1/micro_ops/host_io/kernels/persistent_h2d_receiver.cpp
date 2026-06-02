// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent fixed-shape H2D receiver for H2DStreamService.
//
// Each iteration of the outer loop drains exactly ONE full tensor's worth of
// data (num_socket_pages chunks of socket_page_size bytes each), fanning out
// to pages_per_chunk DRAM/L1 tensor pages of output_tensor_page_size bytes
// per chunk. The same compile-time arg set therefore describes every
// forward_to_tensor call for the lifetime of the service.
//
// The outer loop exits cleanly when the host sets `termination_semaphore` to 1.
// The check happens inside the socket-wait polling loop (via
// `socket_wait_for_pages_with_termination`) so shutdown stays responsive even
// when no data is in flight. The service's destructor `barrier()`s every
// socket before signalling termination, which guarantees we never break out
// of the inner loop with chunks still pending and leave the backing tensor in
// a half-written state.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "../../../unified_kernels/termination.hpp"
#include "api/debug/dprint.h"

// CT-arg layout (must stay in sync with build_persistent_h2d_program in
// ttnn/core/tensor/socket_services.cpp). termination_semaphore_addr is placed
// at index 1 to match the convention used by the other persistent kernels in
// this directory (h2d_receiver.cpp, d2h_sender.cpp, ...).
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
// metadata_size_bytes is the un-padded user-specified size; the host pads the
// trailing socket page to socket_page_size, but the kernel only multicasts
// these many bytes so the worker reads exactly what was provided.
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(16);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(17);
constexpr uint32_t metadata_l1_addr = get_compile_time_arg_val(18);
constexpr auto output_tensor_accessor_args = TensorAccessorArgs<19>();

// H2D: read one socket page from PCIe host RAM into L1 in NOC_MAX_BURST_SIZE chunks.
// Caller must call noc_async_read_barrier() after this returns.
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

    // Hoist socket invariants out of the loop. read_ptr/fifo_addr are read each iteration
    // since they advance with socket_pop_pages.
    const uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;
    const uint64_t base_pinned =
        (static_cast<uint64_t>(receiver_socket.h2d.data_addr_hi) << 32) | receiver_socket.h2d.data_addr_lo;

    // Single-slot scratch CB; use the write pointer consistently across PCIe-in and NoC-out
    // since no producer/consumer split exists in this kernel.
    const uint32_t cb_l1_addr = get_write_ptr(scratch_buffer_cb_index);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    // Worker-sync state. Allocated unconditionally to keep the kernel-main body
    // readable; the `if constexpr` gates make the compiler dead-code-eliminate
    // everything when `worker_sync_enabled == 0`.
    uint64_t worker_mcast_addr = 0;
    volatile tt_l1_ptr uint32_t* consumed_ptr = nullptr;
    uint32_t last_consumed = 0;  // counter snapshot at the start of each iteration
    if constexpr (worker_sync_enabled) {
        worker_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            data_ready_sem_addr);
        consumed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_counter_addr);
    }

    // Metadata multicast destination — same multicast bbox as data_ready_sem
    // (worker_cores), different in-core L1 address. Hoisted out so the per-
    // transfer block doesn't recompute it.
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
        // Drain exactly one full tensor's worth of data: num_socket_pages chunks.
        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            // Polling wait with termination check so shutdown stays responsive
            // when the host stops sending data.
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

            // Fan out pages_per_chunk tensor pages from the scratch buffer to the device tensor.
            const uint32_t base_page = chunk * pages_per_chunk;
            uint32_t src = cb_l1_addr;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                const uint64_t noc_dst = output_tensor_accessor.get_noc_addr(base_page + i);
                noc_async_write<output_tensor_page_size>(src, noc_dst, output_tensor_page_size);
                src += output_tensor_page_size;
            }

            // Source-side wait only; the destination ack can drain concurrently with the next
            // iteration's PCIe read. The trailing barrier after the outer loop guarantees
            // per-transfer durability.
            noc_async_write_barrier();

            // Early host release: data is now in L1, free the pinned FIFO slot so the host can
            // refill it while we NoC-write this chunk to DRAM. `update_socket_config` is
            // deferred to the end of the kernel — the local socket struct carries the
            // updated read_ptr / bytes_acked across the loop without needing to be
            // written back to L1 on every chunk.
            socket_pop_pages(receiver_socket, 1);
            socket_notify_sender(receiver_socket);
        }
        // Optional inline-metadata multicast. Compile-time-gated; when disabled,
        // the entire block is dead-code-eliminated.
        //
        // The host pushes ONE trailing socket page per transfer carrying
        // metadata in its first `metadata_size_bytes`, zero-padded to
        // socket_page_size. We:
        //   1. wait for that trailing page (termination-aware),
        //   2. PCIe-read it into the scratch CB,
        //   3. multicast the first metadata_size_bytes to every worker core's
        //      local copy at metadata_l1_addr,
        //   4. pop the metadata page and notify the host.
        //
        // The barrier in step 3 must complete BEFORE the data_ready_sem
        // multicast below, so workers observing data_ready see consistent
        // (DRAM + L1) state.
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


        // Termination short-circuit: if the inner loop exited via
        // socket_wait_for_pages_with_termination (no transfer actually
        // completed), skip the worker-sync block entirely. Without this,
        // the kernel would multicast data_ready to workers and then poll
        // the consumed counter for acks that will never come (workers
        // aren't running during teardown), hanging wait_done in the dtor.
        // Contract preserved: every multicast still corresponds to exactly
        // one completed transfer.
        if (terminated) {
            break;
        }

        // Optional worker-sync handshake. Compile-time-gated: when disabled,
        // the entire block is removed by the compiler.
        if constexpr (worker_sync_enabled) {
            // Contract: every transfer that completes the inner loop above gets
            // a matching worker ack. The only termination point in this kernel
            // is the `socket_wait_for_pages_with_termination` poll at the start
            // of the next transfer — we don't poll termination inside the sync.
            //
            // 1. Multicast atomic-inc of the data_ready semaphore to every
            //    worker core. Each worker observes (cur - last_seen) >= 1 on
            //    its local copy and proceeds to consume the new tensor.
            noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);

            // 2. Wait for EXACTLY num_workers more increments on the consumed
            //    counter since the previous iteration. Unsigned modulo subtraction
            //    stays correct after the counter wraps at uint32 (workers don't
            //    get >2^31 iterations ahead of the service). Exact equality
            //    enforces the 1-ack-per-transfer protocol — anything else is a
            //    contract violation.
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

    // Flush the local socket state (advanced read_ptr / bytes_acked) back to
    // the L1 config buffer once, on the way out. We avoid doing this on every
    // pop in the hot path — the local socket struct in this kernel is the
    // source of truth across the loop, and nothing else reads the L1 config
    // until the next service run on this core.
    update_socket_config(receiver_socket);
}
