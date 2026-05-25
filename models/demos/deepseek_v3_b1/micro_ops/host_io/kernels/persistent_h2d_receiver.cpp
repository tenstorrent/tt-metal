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
constexpr auto output_tensor_accessor_args = TensorAccessorArgs<8>();

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

    bool terminated = false;
    DPRINT << "Number of socket pages: " << num_socket_pages << ENDL();
    DPRINT << "Socket Page Size: " << socket_page_size << ENDL();
    DPRINT << "Bytes Sent Address: " << receiver_socket.bytes_sent_addr << ENDL();
    DPRINT << "Bytes sent value: " << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.bytes_sent_addr) << ENDL();
    DPRINT << "Is H2D: " << receiver_socket.is_h2d << ENDL();
    DPRINT << "PCIe XY Enc: " << pcie_xy_enc << ENDL();
    DPRINT << "Socket Config Address: " << socket_config_addr << ENDL();
    DPRINT << "CB L1 Address: " << cb_l1_addr << ENDL();
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
            // refill it while we NoC-write this chunk to DRAM.
            socket_pop_pages(receiver_socket, 1);
            socket_notify_sender(receiver_socket);
            update_socket_config(receiver_socket);
        }

        // Per-transfer durability barrier. Reached even on mid-transfer termination so any
        // in-flight writes still land before the kernel exits.
        noc_async_write_barrier();
    }
}
