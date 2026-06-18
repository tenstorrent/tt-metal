// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader half of the persistent H2D stream service (paired with persistent_h2d_writer.cpp,
// running on the other data-movement RISC of the same service core). The reader owns the
// socket: it pulls each socket page from host pinned memory over PCIe into a data-CB slot,
// acknowledges the page as soon as it is staged in L1 (it has no visibility into the
// writer's DRAM completion), and hands the slot to the writer. Acking after the read --
// rather than after the tensor write -- recycles the host FIFO slot a DRAM-write sooner.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "../../../unified_kernels/termination.hpp"

constexpr uint32_t socket_page_size = get_compile_time_arg_val(0);
constexpr uint32_t num_socket_pages = get_compile_time_arg_val(1);
constexpr uint32_t data_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(3);
constexpr uint32_t metadata_cb_index = get_compile_time_arg_val(4);

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
    const uint32_t socket_config_addr = get_arg_val<uint32_t>(0);
    const uint32_t termination_semaphore_addr = get_arg_val<uint32_t>(1);

    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, socket_page_size);

    const uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;
    const uint64_t base_pinned =
        (static_cast<uint64_t>(receiver_socket.h2d.data_addr_hi) << 32) | receiver_socket.h2d.data_addr_lo;

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    // Stage one socket page into `cb`: wait for the page and a free slot (both terminable),
    // read it into the slot, push it to the writer, then ack the socket page. The ack is still
    // before the DRAM write, but never before the writer can observe the staged page.
    // Returns false when termination is observed while waiting.
    auto stage_one_page = [&](uint32_t cb) -> bool {
        if (!deepseek_b1_ops::socket_wait_for_pages_with_termination(receiver_socket, 1, termination_semaphore)) {
            return false;
        }
        if (!deepseek_b1_ops::cb_reserve_for_pages_with_termination(cb, 1, termination_semaphore)) {
            return false;
        }
        const uint32_t dst = get_write_ptr(cb);
        noc_read_page_chunked(
            pcie_xy_enc, base_pinned + receiver_socket.read_ptr - receiver_socket.fifo_addr, dst, socket_page_size);
        noc_async_read_barrier();
        cb_push_back(cb, 1);
        socket_pop_pages(receiver_socket, 1);
        socket_notify_sender(receiver_socket);
        return true;
    };

    bool terminated = false;
    while (!terminated) {
        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            if (!stage_one_page(data_cb_index)) {
                terminated = true;
                break;
            }
        }
        if (terminated) {
            break;
        }
        if constexpr (metadata_enabled) {
            if (!stage_one_page(metadata_cb_index)) {
                break;
            }
        }
    }

    update_socket_config(receiver_socket);
}
