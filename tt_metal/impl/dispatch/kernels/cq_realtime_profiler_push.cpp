// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Real-time profiler NCRISC kernel (slow path)
// Drains the L1 ring buffer populated by the BRISC kernel and pushes each
// entry to the host via PCIe using the D2H socket. Runs on NOC 1 (NCRISC
// dedicated NOC). On architectures where NOC0 != NOC1 coordinates (WH), the
// host passes PCIE_NOC_X/Y so the kernel can compute the correct NOC1 encoding.

#include <algorithm>
#include <cstdint>
#include "internal/risc_attribs.h"
#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/realtime_profiler_msgs.h"
// Uncomment to compile in ncrisc_debug L1 heartbeats (RT_PROF_NCRISC_DBG_* in realtime_profiler_ring_buffer.hpp):
// #define RT_PROFILER_NCRISC_DEBUG
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "api/debug/dprint.h"

// Real-time profiler page size - must match host-side
// RealtimeProfilerRuntimeSizes::page_size (which is also RT_PROFILER_ENTRY_SIZE).
constexpr uint32_t realtime_profiler_page_size = RT_PROFILER_ENTRY_SIZE;  // 64 bytes

// Compile-time defines set by host:
// RING_BUFFER_ADDR  - L1 address of the shared ring buffer
// PCIE_NOC_X        - PCIe core X coordinate in NOC-0 space (WH only)
// PCIE_NOC_Y        - PCIe core Y coordinate in NOC-0 space (WH only)

// L1 region carved by DispatchMemMap (CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG) on this
// reserved RT-profiler tensix core. Mirrors cq_realtime_profiler.cpp; address via compile-time
// define REALTIME_PROFILER_MSG_ADDR set by host.
volatile tt_l1_ptr realtime_profiler_msg_t* rt_profiler_msg =
    reinterpret_cast<volatile tt_l1_ptr realtime_profiler_msg_t*>(REALTIME_PROFILER_MSG_ADDR);

volatile RtProfilerRingBuffer* ring_buffer = reinterpret_cast<volatile RtProfilerRingBuffer*>(RING_BUFFER_ADDR);

// WH NCRISC runs on NOC1, which needs a different PCIe XY encoding than the NOC0-based one
// stored in the D2H socket config; the host passes the NOC0 coordinates via
// RT_PROFILER_PCIE_NOC_X/Y on WH so we can re-encode at compile time. BH is already correct.
#ifdef RT_PROFILER_PCIE_NOC_X
constexpr uint64_t pcie_noc_xy_full =
    uint64_t(NOC_XY_PCIE_ENCODING(NOC_X_PHYS_COORD(RT_PROFILER_PCIE_NOC_X), NOC_Y_PHYS_COORD(RT_PROFILER_PCIE_NOC_Y)));
constexpr uint32_t pcie_xy_enc_noc1 = static_cast<uint32_t>(pcie_noc_xy_full >> 32);
#endif

// Push `num_pages` ring entries to the host via PCIe D2H socket
__attribute__((noinline)) void push_entries_to_host(
    SocketSenderInterface& socket,
    uint32_t src_index,
    uint32_t num_pages,
    uint32_t pcie_xy_enc,
    uint32_t data_addr_hi,
    uint32_t& host_write_ptr,
    uint32_t host_fifo_start,
    uint32_t fifo_page_aligned_size) {
    RT_PROF_NCRISC_DBG_INC(ring_buffer, socket_reserve_pages_enter_count);
    socket_reserve_pages(socket, num_pages);
    RT_PROF_NCRISC_DBG_INC(ring_buffer, socket_reserve_pages_exit_count);

    noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);
    constexpr uint32_t kMaxPagesPerWrite = NOC_MAX_BURST_SIZE / realtime_profiler_page_size;
    uint32_t remaining = num_pages;
    while (remaining > 0) {
        const uint32_t ring_slot = src_index & (RT_PROFILER_RING_CAPACITY - 1);
        const uint32_t pages_to_ring_wrap = RT_PROFILER_RING_CAPACITY - ring_slot;
        const uint32_t pages_to_fifo_wrap =
            (host_fifo_start + fifo_page_aligned_size - host_write_ptr) / realtime_profiler_page_size;
        const uint32_t run = std::min({remaining, pages_to_ring_wrap, pages_to_fifo_wrap, kMaxPagesPerWrite});

        const uint64_t pcie_dest_addr =
            (static_cast<uint64_t>(data_addr_hi) << 32) | static_cast<uint64_t>(host_write_ptr);
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            noc_index,
            rt_ring_data_addr(ring_buffer, src_index),
            pcie_xy_enc,
            pcie_dest_addr,
            run * realtime_profiler_page_size,
            1);

        src_index += run;
        host_write_ptr += run * realtime_profiler_page_size;
        if (host_write_ptr >= host_fifo_start + fifo_page_aligned_size) {
            host_write_ptr = host_fifo_start;
        }
        remaining -= run;
    }

    socket_push_pages(socket, num_pages);
    socket_notify_receiver(socket);
    noc_async_write_barrier();
    RT_PROF_NCRISC_DBG_INC(ring_buffer, push_write_barrier_exit_count);
}

void kernel_main() {
    RT_PROF_NCRISC_DBG_SET(ring_buffer, stage, RT_PROFILER_NCRISC_STAGE_STARTED);
    RT_PROF_NCRISC_DBG_SET(
        ring_buffer, config_buffer_addr_field_l1_ptr, reinterpret_cast<uint32_t>(&rt_profiler_msg->config_buffer_addr));
    RT_PROF_NCRISC_DBG_SET(ring_buffer, ring_buffer_addr_literal, RING_BUFFER_ADDR);

    // Wait for config_buffer_addr to be written by the host
    RT_PROF_NCRISC_DBG_SET(ring_buffer, stage, RT_PROFILER_NCRISC_STAGE_CONFIG_WAIT);
    uint32_t socket_config_addr = 0;
    uint32_t wait_iters = 0;
    while (socket_config_addr == 0) {
        invalidate_l1_cache();
        socket_config_addr = rt_profiler_msg->config_buffer_addr;
        RT_PROF_NCRISC_DBG_SET(ring_buffer, config_buffer_addr_raw, socket_config_addr);
        wait_iters++;
        RT_PROF_NCRISC_DBG_SET(ring_buffer, loop_iteration, wait_iters);
        if (ring_buffer->terminate) {
            return;
        }
    }
    RT_PROF_NCRISC_DBG_SET(ring_buffer, socket_config_addr, socket_config_addr);

    // Initialize socket from the config buffer (fresh read every launch)
    RT_PROF_NCRISC_DBG_SET(ring_buffer, stage, RT_PROFILER_NCRISC_STAGE_SOCKET_INIT);
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

    RT_PROF_NCRISC_DBG_SET(ring_buffer, pcie_xy_enc, pcie_xy_enc);
    RT_PROF_NCRISC_DBG_SET(ring_buffer, fifo_addr_lo, data_addr_lo);

    RT_PROF_NCRISC_DBG_SET(ring_buffer, stage, RT_PROFILER_NCRISC_STAGE_MAIN_LOOP);
    uint32_t loop_count = 0;
    uint32_t push_count = 0;

    while (true) {
        invalidate_l1_cache();
        loop_count++;
        RT_PROF_NCRISC_DBG_SET(ring_buffer, loop_iteration, loop_count);

        const uint32_t read_index = ring_buffer->read_index;
        const uint32_t write_index = ring_buffer->write_index;
        if (write_index == read_index) {
            if (ring_buffer->terminate) {
                return;
            }
            continue;
        }

        RT_PROF_NCRISC_DBG_SET(ring_buffer, stage, RT_PROFILER_NCRISC_STAGE_PUSHING);
        const uint32_t available = write_index - read_index;
        push_entries_to_host(
            profiler_socket,
            read_index,
            available,
            pcie_xy_enc,
            data_addr_hi,
            host_write_ptr,
            host_fifo_start,
            fifo_page_aligned_size);
        ring_buffer->read_index = read_index + available;
        push_count += available;
        RT_PROF_NCRISC_DBG_SET(ring_buffer, push_count, push_count);
        RT_PROF_NCRISC_DBG_SET(ring_buffer, stage, RT_PROFILER_NCRISC_STAGE_MAIN_LOOP);
    }
}
