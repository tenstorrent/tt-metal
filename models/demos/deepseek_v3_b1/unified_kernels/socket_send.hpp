// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "../micro_ops/host_io/kernels/pcie_noc_utils.h"

namespace unified_kernels {

// Reusable socket send: reads `num_cb_pages` from `cb_id`, sends `total_send_bytes`
// as a single logical socket page via the socket at `socket_config_addr`.
//
// SocketMode=1: D2H (PCIe write via noc_async_wide_write_any_len_with_state)
// SocketMode=2: D2D (NOC write to downstream device(s))
template <uint32_t SocketMode>
FORCE_INLINE void socket_send_from_cb(
    uint32_t socket_config_addr, uint32_t cb_id, uint32_t num_cb_pages, uint32_t total_send_bytes) {
    static_assert(SocketMode == 1 || SocketMode == 2, "SocketMode must be 1 (D2H) or 2 (D2D)");

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, total_send_bytes);

    socket_reserve_pages(sender_socket, 1);
    DPRINT << ">soc" << ENDL();
    cb_wait_front(cb_id, num_cb_pages);
    const uint32_t read_addr = get_read_ptr(cb_id);

    if constexpr (SocketMode == 1) {
        const uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
        const uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        noc_async_wide_write_any_len_with_state(
            NOC_INDEX,
            read_addr,
            pcie_xy_enc,
            ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                sender_socket.write_ptr,
            total_send_bytes);
        noc_async_writes_flushed();
    } else {
        for (uint32_t i = 0; i < sender_socket.num_downstreams; i++) {
            sender_downstream_encoding downstream_enc = get_downstream_encoding(sender_socket, i);
            noc_async_write(
                read_addr,
                get_noc_addr(
                    downstream_enc.d2d.downstream_noc_x,
                    downstream_enc.d2d.downstream_noc_y,
                    sender_socket.write_ptr + sender_socket.downstream_fifo_addr),
                total_send_bytes);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(cb_id, num_cb_pages);
    socket_push_pages(sender_socket, 1);
    socket_notify_receiver(sender_socket);
    update_socket_config(sender_socket);
    DPRINT << "<soc" << ENDL();
    if constexpr (SocketMode == 1) {
        noc_async_write_barrier();
    }
}

}  // namespace unified_kernels

#endif  // COMPILE_FOR_BRISC
