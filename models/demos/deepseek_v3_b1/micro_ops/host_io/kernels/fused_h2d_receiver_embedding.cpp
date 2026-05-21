// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/debug/dprint.h"

// Get this value from MeshSocket struct on host
constexpr uint32_t socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t transfer_size_bytes = get_compile_time_arg_val(1);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(2);
constexpr uint32_t output_tensor_addr = get_compile_time_arg_val(3);
constexpr uint32_t output_tensor_num_pages = get_compile_time_arg_val(4);
constexpr uint32_t output_tensor_page_size = get_compile_time_arg_val(5);
constexpr uint32_t scratch_buffer_cb_index = get_compile_time_arg_val(6);
constexpr auto output_tensor_accessor_args = TensorAccessorArgs<7>();

inline void noc_write_page_chunked(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    while (size) {
        uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, chunk, 1);
        src_l1 += chunk;
        dst_pcie += chunk;
        size -= chunk;
    }
}

// H2D: read one page from PCIe host RAM into L1 in NOC_MAX_BURST_SIZE chunks.
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

    uint32_t read_addr_hi = receiver_socket.h2d.data_addr_hi;
    uint32_t read_addr_lo = receiver_socket.h2d.data_addr_lo;
    uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;

    // Wait for pages in H2D socket
    DPRINT << "Socket Page Size: " << socket_page_size << ENDL();
    socket_wait_for_pages(receiver_socket, 1);

    // Pages available in H2D socket - read over PCIe
    noc_read_page_chunked(
        pcie_xy_enc,
        ((static_cast<uint64_t>(read_addr_hi) << 32) | read_addr_lo) + receiver_socket.read_ptr -
            receiver_socket.fifo_addr,
        get_write_ptr(scratch_buffer_cb_index),
        socket_page_size);

    noc_async_read_barrier();

    DPRINT << "Read Pages" << ENDL();
    volatile tt_l1_ptr uint32_t* data_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(scratch_buffer_cb_index));

    for (int i = 0; i < 16; i++) {
        DPRINT << "Data at index " << i << ": " << data_ptr[i] << ENDL();
    }
    auto read_ptr = get_read_ptr(scratch_buffer_cb_index);
    for (uint32_t page_index = 0; page_index < output_tensor_num_pages; ++page_index) {
        auto noc_write_addr = output_tensor_accessor.get_noc_addr(page_index);
        noc_async_write<output_tensor_page_size>(read_ptr, noc_write_addr, output_tensor_page_size);
        read_ptr += output_tensor_page_size;
    }

    noc_async_write_barrier();
    socket_pop_pages(receiver_socket, 1);
    socket_notify_sender(receiver_socket);
    update_socket_config(receiver_socket);

    noc_async_write_barrier();
}
