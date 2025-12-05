// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-time profiler NCRISC kernel (slow path)
// Drains the L1 ring buffer populated by the BRISC kernel and pushes each
// entry to the host via PCIe using the D2H socket. Runs on NOC 1 (NCRISC
// dedicated NOC). On architectures where NOC0 != NOC1 coordinates (WH), the
// host passes PCIE_NOC_X/Y so the kernel can compute the correct NOC1 encoding.

#include <cstdint>
#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/dev_msgs.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "api/debug/dprint.h"

// Real-time profiler page size - must match host-side kRealtimeProfilerPageSize
constexpr uint32_t realtime_profiler_page_size = RT_PROFILER_ENTRY_SIZE;  // 64 bytes

// Compile-time defines set by host:
// RING_BUFFER_ADDR  - L1 address of the shared ring buffer
// PCIE_NOC_X        - PCIe core X coordinate in NOC-0 space (WH only)
// PCIE_NOC_Y        - PCIe core Y coordinate in NOC-0 space (WH only)

volatile tt_l1_ptr realtime_profiler_msg_t* realtime_profiler_mailbox =
    reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(GET_MAILBOX_ADDRESS_DEV(realtime_profiler));

volatile RtProfilerRingBuffer* ring_buffer = reinterpret_cast<volatile RtProfilerRingBuffer*>(RING_BUFFER_ADDR);

// On WH, NCRISC uses NOC1 which requires a different PCIe XY encoding than
// what the D2H socket config provides (NOC0-based).  The host passes NOC0
// coordinates via RT_PROFILER_PCIE_NOC_X/Y kernel defines (WH only) so we
// can compute the NOC1 encoding at compile time.  On BH, no override is
// needed — the socket's encoding is already correct.
#ifdef RT_PROFILER_PCIE_NOC_X
constexpr uint64_t pcie_noc_xy_full =
    uint64_t(NOC_XY_PCIE_ENCODING(NOC_X_PHYS_COORD(RT_PROFILER_PCIE_NOC_X), NOC_Y_PHYS_COORD(RT_PROFILER_PCIE_NOC_Y)));
constexpr uint32_t pcie_xy_enc_noc1 = static_cast<uint32_t>(pcie_noc_xy_full >> 32);
#endif

// Push one ring buffer entry to the host via PCIe D2H socket
__attribute__((noinline)) void push_entry_to_host(
    SocketSenderInterface& sock,
    uint32_t slot_addr,
    uint32_t pcie_xy_enc,
    uint32_t data_addr_hi,
    uint32_t& host_write_ptr,
    uint32_t host_fifo_start,
    uint32_t fifo_page_aligned_size) {
    noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);
    socket_reserve_pages(sock, 1);

    uint64_t pcie_dest_addr = (static_cast<uint64_t>(data_addr_hi) << 32) | static_cast<uint64_t>(host_write_ptr);

    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        noc_index, slot_addr, pcie_xy_enc, pcie_dest_addr, realtime_profiler_page_size, 1);

    host_write_ptr += realtime_profiler_page_size;
    if (host_write_ptr >= host_fifo_start + fifo_page_aligned_size) {
        host_write_ptr = host_fifo_start;
    }

    socket_push_pages(sock, 1);
    socket_notify_receiver(sock);

    noc_async_write_barrier();
}

// Heartbeat markers written to ring_buffer->_pad[] so host can diagnose NCRISC progress.
// _pad[0] = stage (1=started, 2=config_wait, 3=socket_init, 4=main_loop, 5=pushing)
// _pad[1] = config_buffer_addr seen by NCRISC
// _pad[2] = pcie_xy_enc from socket config
// _pad[3] = fifo_addr_lo from socket config
// _pad[4] = loop iteration counter
// _pad[5] = push count
// _pad[6] = L1 address of realtime_profiler_mailbox (where NCRISC reads config_buffer_addr)
// _pad[7] = raw 32-bit value at that address
// _pad[8] = RING_BUFFER_ADDR define value

void kernel_main() {
    ring_buffer->_pad[0] = 1;  // stage: kernel started
    ring_buffer->_pad[6] = reinterpret_cast<uint32_t>(&realtime_profiler_mailbox->config_buffer_addr);
    ring_buffer->_pad[8] = RING_BUFFER_ADDR;

    // Wait for config_buffer_addr to be written by the host
    ring_buffer->_pad[0] = 2;  // stage: waiting for config
    uint32_t socket_config_addr = 0;
    uint32_t wait_iters = 0;
    while (socket_config_addr == 0) {
        invalidate_l1_cache();
        socket_config_addr = realtime_profiler_mailbox->config_buffer_addr;
        // Also write the raw value we see, so host can compare
        ring_buffer->_pad[7] = socket_config_addr;
        wait_iters++;
        ring_buffer->_pad[4] = wait_iters;
        if (ring_buffer->terminate) {
            return;
        }
    }
    ring_buffer->_pad[1] = socket_config_addr;

    // Initialize socket from the config buffer (fresh read every launch)
    ring_buffer->_pad[0] = 3;  // stage: initializing socket
    invalidate_l1_cache();
    SocketSenderInterface profiler_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(profiler_socket, realtime_profiler_page_size);

#ifdef RT_PROFILER_PCIE_NOC_X
    profiler_socket.d2h.pcie_xy_enc = pcie_xy_enc_noc1;
    uint32_t pcie_xy_enc = pcie_xy_enc_noc1;
#else
    uint32_t pcie_xy_enc = profiler_socket.d2h.pcie_xy_enc;
#endif
    uint32_t data_addr_hi = profiler_socket.d2h.data_addr_hi;
    uint32_t data_addr_lo = profiler_socket.downstream_fifo_addr;
    uint32_t fifo_page_aligned_size = profiler_socket.downstream_fifo_total_size -
                                      (profiler_socket.downstream_fifo_total_size % realtime_profiler_page_size);
    uint32_t host_write_ptr = data_addr_lo;
    uint32_t host_fifo_start = data_addr_lo;

    ring_buffer->_pad[2] = pcie_xy_enc;
    ring_buffer->_pad[3] = data_addr_lo;

    ring_buffer->_pad[0] = 4;  // stage: entering main loop
    uint32_t loop_count = 0;
    uint32_t push_count = 0;

    while (true) {
        invalidate_l1_cache();
        loop_count++;
        ring_buffer->_pad[4] = loop_count;

        if (rt_ring_empty(ring_buffer)) {
            if (ring_buffer->terminate) {
                return;
            }
            continue;
        }

        ring_buffer->_pad[0] = 5;  // stage: pushing entry
        uint32_t slot_addr = rt_ring_data_addr(ring_buffer, ring_buffer->read_index);
        push_entry_to_host(
            profiler_socket,
            slot_addr,
            pcie_xy_enc,
            data_addr_hi,
            host_write_ptr,
            host_fifo_start,
            fifo_page_aligned_size);
        ring_buffer->read_index++;
        push_count++;
        ring_buffer->_pad[5] = push_count;
        ring_buffer->_pad[0] = 4;  // stage: back to main loop
    }
}
