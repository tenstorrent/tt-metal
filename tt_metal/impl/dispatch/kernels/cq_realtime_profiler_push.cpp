// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-time profiler NCRISC kernel (slow path)
// Drains the L1 ring buffer populated by the BRISC kernel and pushes each
// entry to the host via PCIe using the D2H socket. Uses the host-provided
// PCIe encoding from the socket config (both NOCs share the same coordinate space).

#include <cstdint>
#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/dev_msgs.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "api/debug/dprint.h"

// Real-time profiler page size - must match host-side kRealtimeProfilerPageSize
constexpr uint32_t realtime_profiler_page_size = RT_PROFILER_ENTRY_SIZE;  // 64 bytes

// Compile-time defines:
// RING_BUFFER_ADDR  - L1 address of the shared ring buffer (set by host)

volatile tt_l1_ptr realtime_profiler_msg_t* realtime_profiler_mailbox =
    reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(GET_MAILBOX_ADDRESS_DEV(realtime_profiler));

volatile RtProfilerRingBuffer* ring_buffer = reinterpret_cast<volatile RtProfilerRingBuffer*>(RING_BUFFER_ADDR);

// D2H socket state
static SocketSenderInterface profiler_socket;
static bool socket_initialized = false;
static uint32_t pcie_xy_enc = 0;
static uint32_t data_addr_hi = 0;
static uint32_t host_write_ptr = 0;
static uint32_t host_fifo_start = 0;
static uint32_t fifo_page_aligned_size = 0;

FORCE_INLINE
bool init_socket() {
    if (socket_initialized) {
        return true;
    }

    invalidate_l1_cache();

    uint32_t socket_config_addr = realtime_profiler_mailbox->config_buffer_addr;
    if (socket_config_addr == 0) {
        return false;
    }

    profiler_socket = create_sender_socket_interface(socket_config_addr);
    set_sender_socket_page_size(profiler_socket, realtime_profiler_page_size);

    pcie_xy_enc = profiler_socket.d2h.pcie_xy_enc;
    data_addr_hi = profiler_socket.d2h.data_addr_hi;
    uint32_t data_addr_lo = profiler_socket.downstream_fifo_addr;

    fifo_page_aligned_size = profiler_socket.downstream_fifo_total_size -
                             (profiler_socket.downstream_fifo_total_size % realtime_profiler_page_size);

    host_write_ptr = data_addr_lo;
    host_fifo_start = data_addr_lo;

    socket_initialized = true;
    return true;
}

// Push one ring buffer entry to the host via PCIe D2H socket
__attribute__((noinline)) void push_entry_to_host(uint32_t slot_addr) {
    noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);
    socket_reserve_pages(profiler_socket, 1);

    uint64_t pcie_dest_addr = (static_cast<uint64_t>(data_addr_hi) << 32) | static_cast<uint64_t>(host_write_ptr);

    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        noc_index, slot_addr, pcie_xy_enc, pcie_dest_addr, realtime_profiler_page_size, 1);

    host_write_ptr += realtime_profiler_page_size;
    if (host_write_ptr >= host_fifo_start + fifo_page_aligned_size) {
        host_write_ptr = host_fifo_start;
    }

    socket_push_pages(profiler_socket, 1);
    socket_notify_receiver(profiler_socket);

    noc_async_write_barrier();
}

void kernel_main() {
    DPRINT << "REALTIME NCRISC: push kernel started" << ENDL();

    while (true) {
        invalidate_l1_cache();

        if (rt_ring_empty(ring_buffer)) {
            if (ring_buffer->terminate) {
                DPRINT << "REALTIME NCRISC: draining complete, exiting" << ENDL();
                return;
            }
            continue;
        }

        if (!init_socket()) {
            continue;
        }

        uint32_t slot_addr = rt_ring_data_addr(ring_buffer, ring_buffer->read_index);
        push_entry_to_host(slot_addr);
        ring_buffer->read_index++;
    }
}
