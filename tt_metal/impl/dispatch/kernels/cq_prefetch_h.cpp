// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefetch kernel
//  - 3 flavors: _hd (host and dram), _h (host only), _d (DRAM only)
//  - fetches commands from host (if applicable), executes
//  - uses HostQ for host handshaking, ComDatQ for commands (from host),
//    double buffered ScratchBuf for out of band data (e.g., from DRAM)
//  - syncs w/ dispatcher via 2 semaphores, page_ready, page_done

#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_prefetch.hpp"
#include "debug/dprint.h"

constexpr uint32_t downstream_cb_base = get_compile_time_arg_val(0);
constexpr uint32_t downstream_cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t downstream_cb_pages = get_compile_time_arg_val(2);
constexpr uint32_t local_downstream_cb_sem = get_compile_time_arg_val(3);
constexpr uint32_t downstream_cb_sem = get_compile_time_arg_val(4);

constexpr uint32_t pcie_base = get_compile_time_arg_val(5);
constexpr uint32_t pcie_size = get_compile_time_arg_val(6);
constexpr uint32_t prefetch_q_base = get_compile_time_arg_val(7);
constexpr uint32_t prefetch_q_size = get_compile_time_arg_val(8);
constexpr uint32_t prefetch_q_rd_ptr_addr = get_compile_time_arg_val(9);

constexpr uint32_t cmddat_q_base = get_compile_time_arg_val(10);
constexpr uint32_t cmddat_q_size = get_compile_time_arg_val(11);

constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint32_t downstream_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t downstream_cb_page_size = 1 << downstream_cb_log_page_size;
constexpr uint32_t downstream_cb_end = downstream_cb_base + (1 << downstream_cb_log_page_size) * downstream_cb_pages;
constexpr uint32_t prefetch_q_end = prefetch_q_base + prefetch_q_size;
constexpr uint32_t cmddat_q_end = cmddat_q_base + cmddat_q_size;

static uint32_t pcie_read_ptr = pcie_base;
static uint32_t downstream_data_ptr = downstream_cb_base;

constexpr uint32_t prefetch_q_log_minsize = 4;

static_assert((downstream_cb_base & (downstream_cb_page_size - 1)) == 0);

static uint32_t process_relay_inline_all(uint32_t data_ptr, uint32_t fence) {

    uint32_t length = fence - data_ptr;

    // Downstream doesn't have FetchQ to tell it how much data to process
    // This packet header just contains the length
    volatile tt_l1_ptr uint32_t *dptr = (volatile tt_l1_ptr uint32_t *)data_ptr;
    *dptr = length;

    uint32_t npages = (length + downstream_cb_page_size - 1) >> downstream_cb_log_page_size;

    // Assume the dispatch buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    downstream_cb_acquire_pages<my_noc_xy, local_downstream_cb_sem>(npages);
    uint32_t dispatch_pages_left = (downstream_cb_end - downstream_data_ptr) / downstream_cb_page_size;
    if (dispatch_pages_left >= npages) {
        noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), length);
        downstream_data_ptr += npages * downstream_cb_page_size;
    } else {
        uint32_t tail_pages = npages - dispatch_pages_left;
        uint32_t available = dispatch_pages_left * downstream_cb_page_size;
        if (available > 0) {
            noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), available);
            data_ptr += available;
            length -= available;
        }

        noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_cb_base), length);
        downstream_data_ptr = downstream_cb_base + tail_pages * downstream_cb_page_size;
    }

    // XXXXX - painful syncing right now?  move this into get_cmds
    noc_async_writes_flushed();
    downstream_cb_release_pages<downstream_noc_xy, downstream_cb_sem>(npages);

    return data_ptr + length;
}

void kernel_main() {

    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;

    DPRINT << "prefetch_h" << ENDL();

    bool done = false;
    while (!done) {
        fetch_q_get_cmds<cmddat_q_base,
                         cmddat_q_size,
                         pcie_base,
                         pcie_size,
                         prefetch_q_base,
                         prefetch_q_end,
                         prefetch_q_log_minsize,
                         prefetch_q_rd_ptr_addr,
                         CQ_PREFETCH_DOWNSTREAM_PACKET_HEADER_SIZE>(fence, cmd_ptr, pcie_read_ptr);

        volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

        if (cmd->base.cmd_id == CQ_PREFETCH_CMD_TERMINATE) {
            DPRINT << "terminating\n";
            done = true;
        } else {
            cmd_ptr = process_relay_inline_all(cmd_ptr, fence);
        }
    }

    DPRINT << "prefetch_h out\n" << ENDL();
}
