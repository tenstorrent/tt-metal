// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer half of the persistent D2H stream service (paired with persistent_d2h_reader.cpp,
// running on the other data-movement RISC of the same service core). The writer owns the
// D2H socket: it consumes full socket pages the reader stages in the data CB, reserves
// socket FIFO pages, writes each page to host-pinned memory over PCIe, flushes, and
// notifies the host receiver. After all payload pages for a transfer, the writer sends
// one trailing metadata socket page (if enabled).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/socket_api.h"
#include "pcie_noc_utils.h"
#include "../../../unified_kernels/termination.hpp"

constexpr uint32_t socket_page_size = get_compile_time_arg_val(0);
constexpr uint32_t num_socket_pages = get_compile_time_arg_val(1);
constexpr uint32_t data_cbuf_index = get_compile_time_arg_val(2);
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(3);
constexpr uint32_t metadata_l1_addr = get_compile_time_arg_val(4);
constexpr uint32_t metadata_size_bytes = get_compile_time_arg_val(5);

void kernel_main() {
    Noc noc;
    CircularBuffer data_cbuf(data_cbuf_index);

    const uint32_t socket_config_addr = get_arg_val<uint32_t>(0);
    const uint32_t termination_semaphore_addr = get_arg_val<uint32_t>(1);

    SocketSenderInterface sender_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(sender_socket, socket_page_size);

    const uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
    const uint32_t pcie_xy_enc = sender_socket.d2h.pcie_xy_enc;

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    // Device 2.0 migration: legacy primitives retained — the PCIe wide-write-with-state path
    // (noc_write_init_state + noc_async_wide_write_any_len_with_state, used below to drain into the
    // host-pinned socket FIFO) has no Device 2.0 equivalent; Noc::inline_dw_write is single-DW only.
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    bool terminated = false;
    while (!terminated) {
        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            // Reserve FIFO space BEFORE waiting for CB data. This avoids a deadlock where the
            // writer holds a CB slot (blocking the reader) while waiting for the host to drain
            // the FIFO. socket_reserve_pages only polls bytes_acked — it doesn't advance
            // bytes_sent, so no data is visible to the host until socket_push_pages below.
            if (!deepseek_b1_ops::socket_reserve_pages_with_termination(sender_socket, 1, termination_semaphore)) {
                terminated = true;
                break;
            }

            while (!data_cbuf.pages_available_at_front(1)) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == 1) {
                    terminated = true;
                    break;
                }
            }
            if (terminated) {
                break;
            }

            const uint32_t cb_l1_addr = data_cbuf.get_read_ptr();

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

            data_cbuf.pop_front(1);
        }

        if (terminated) {
            break;
        }

        if constexpr (num_socket_pages == 0) {
            // Metadata-only: wait for the reader's per-transfer token page (gated on write_ack)
            // before pushing metadata, so we don't free-run stale records into the socket FIFO.
            while (!data_cbuf.pages_available_at_front(1)) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == 1) {
                    terminated = true;
                    break;
                }
            }
            if (terminated) {
                break;
            }
            data_cbuf.pop_front(1);
        }

        if constexpr (metadata_enabled) {
            if (!deepseek_b1_ops::socket_reserve_pages_with_termination(sender_socket, 1, termination_semaphore)) {
                terminated = true;
                break;
            }

            // Write only the meaningful metadata bytes, not the full socket page. The socket page
            // is the atomic unit for push/notify, but the PCIe write only needs to transfer the
            // actual metadata — the rest of the FIFO page is stale/zero and the host only copies
            // metadata_size_bytes out of it.
            noc_async_wide_write_any_len_with_state(
                NOC_INDEX,
                metadata_l1_addr,
                pcie_xy_enc,
                ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
                    sender_socket.write_ptr,
                metadata_size_bytes);
            noc.async_writes_flushed();

            socket_push_pages(sender_socket, 1);
            socket_notify_receiver(sender_socket);
        }
    }

    noc.async_write_barrier();
    update_socket_config(sender_socket);
    socket_barrier(sender_socket);
}
