// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"

// CT-arg layout (must stay in sync with copy_tensor_over_socket in
// ttnn/core/tensor/tensor_ops.cpp).
constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_socket_pages = get_compile_time_arg_val(2);
constexpr uint32_t output_tensor_addr = get_compile_time_arg_val(3);
constexpr uint32_t output_tensor_page_size = get_compile_time_arg_val(4);
constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(5);
constexpr uint32_t scratch_buffer_cb_index = get_compile_time_arg_val(6);
constexpr auto output_tensor_accessor_args = TensorAccessorArgs<7>();

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

    for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
        socket_wait_for_pages(receiver_socket, 1);

        noc_read_page_chunked(
            pcie_xy_enc,
            base_pinned + receiver_socket.read_ptr - receiver_socket.fifo_addr,
            cb_l1_addr,
            socket_page_size);
        noc_async_read_barrier();

        // Early host release: data is now in L1, free the pinned FIFO slot so the host can
        // refill it while we NoC-write this chunk to DRAM.
        socket_pop_pages(receiver_socket, 1);
        socket_notify_sender(receiver_socket);
        update_socket_config(receiver_socket);

        // Fan out pages_per_chunk tensor pages from the scratch buffer to DRAM.
        const uint32_t base_page = chunk * pages_per_chunk;
        uint32_t src = cb_l1_addr;
        for (uint32_t i = 0; i < pages_per_chunk; ++i) {
            const uint64_t noc_dst = output_tensor_accessor.get_noc_addr(base_page + i);
            noc_async_write<output_tensor_page_size>(src, noc_dst, output_tensor_page_size);
            src += output_tensor_page_size;
        }

        // Source-side wait only; the destination ack can drain concurrently with the next
        // iteration's PCIe read. The trailing barrier after the loop guarantees durability.
        noc_async_writes_flushed();
    }

    noc_async_write_barrier();
}
