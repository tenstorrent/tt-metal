// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "../micro_ops/host_io/kernels/pcie_noc_utils.h"
#endif

namespace unified_kernels {

#if defined(COMPILE_FOR_BRISC)

// Send the front page of a CB to the host via a D2H (PCIe) socket.
//
// The caller must have already pushed data into cb_id (cb_push_back).
// After this call the page is popped from the CB and the socket is advanced.
//
//   socket_config_addr  – L1 address of the SocketSenderInterface config
//   cb_id               – circular buffer holding the outbound page
//   page_size_bytes     – number of bytes to write over PCIe
FORCE_INLINE void socket_send_d2h_from_cb(uint32_t socket_config_addr, uint32_t cb_id, uint32_t page_size_bytes) {
    SocketSenderInterface sender = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender, page_size_bytes);
    const uint32_t write_addr_hi = sender.d2h.data_addr_hi;
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;

    socket_reserve_pages(sender, 1);
    cb_wait_front(cb_id, 1);
    const uint32_t read_addr = get_read_ptr(cb_id);

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    noc_async_wide_write_any_len_with_state(
        NOC_INDEX,
        read_addr,
        pcie_xy_enc,
        ((static_cast<uint64_t>(write_addr_hi) << 32) | sender.downstream_fifo_addr) + sender.write_ptr,
        page_size_bytes);
    noc_async_writes_flushed();

    cb_pop_front(cb_id, 1);
    socket_push_pages(sender, 1);
    socket_notify_receiver(sender);
    update_socket_config(sender);
    noc_async_write_barrier();
}

// Send the front page of a CB to one or more remote devices via D2D sockets.
//
// The caller must have already pushed data into cb_id (cb_push_back).
// After this call the page is popped from the CB and the socket is advanced.
//
//   socket_config_addr  – L1 address of the SocketSenderInterface config
//   cb_id               – circular buffer holding the outbound page
//   page_size_bytes     – number of bytes per page
FORCE_INLINE void socket_send_d2d_from_cb(uint32_t socket_config_addr, uint32_t cb_id, uint32_t page_size_bytes) {
    SocketSenderInterface sender = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender, page_size_bytes);

    socket_reserve_pages(sender, 1);
    cb_wait_front(cb_id, 1);
    const uint32_t read_addr = get_read_ptr(cb_id);
    for (uint32_t i = 0; i < sender.num_downstreams; i++) {
        sender_downstream_encoding enc = get_downstream_encoding(sender, i);
        noc_async_write(
            read_addr,
            get_noc_addr(
                enc.d2d.downstream_noc_x, enc.d2d.downstream_noc_y, sender.write_ptr + sender.downstream_fifo_addr),
            page_size_bytes);
    }
    noc_async_write_barrier();

    cb_pop_front(cb_id, 1);
    socket_push_pages(sender, 1);
    socket_notify_receiver(sender);
    update_socket_config(sender);
}

// Send the front page of a CB via the appropriate socket type.
//
//   socket_mode: 1 = D2H (PCIe to host), 2 = D2D (device-to-device)
FORCE_INLINE void socket_send_from_cb(
    uint32_t socket_mode, uint32_t socket_config_addr, uint32_t cb_id, uint32_t page_size_bytes) {
    if (socket_mode == 1) {
        socket_send_d2h_from_cb(socket_config_addr, cb_id, page_size_bytes);
    } else if (socket_mode == 2) {
        socket_send_d2d_from_cb(socket_config_addr, cb_id, page_size_bytes);
    }
}

#endif  // COMPILE_FOR_BRISC

}  // namespace unified_kernels
