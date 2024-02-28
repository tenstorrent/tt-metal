// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefetch kernel
//  - 3 flavors: _hd (host and dram), _h (host only), _d (DRAM only)
//  - fetches commands from host (if applicable), executes
//  - uses HostQ for host handshaking, ComDatQ for commands (from host),
//    ScratchBuf for out of band data (e.g., from DRAM)
//  - syncs w/ dispatcher via 2 semaphores, page_ready, page_done

#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"
#include "debug/dprint.h"

constexpr uint32_t dispatch_cb_base = get_compile_time_arg_val(0);
constexpr uint32_t dispatch_cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t dispatch_cb_pages = get_compile_time_arg_val(2);
constexpr uint32_t dispatch_cb_sem = get_compile_time_arg_val(3);
constexpr uint32_t pcie_base = get_compile_time_arg_val(4);
constexpr uint32_t pcie_size = get_compile_time_arg_val(5);
constexpr uint32_t host_q_base = get_compile_time_arg_val(6);
constexpr uint32_t host_q_size = get_compile_time_arg_val(7);
constexpr uint32_t host_q_rd_ptr_addr = get_compile_time_arg_val(8);;
constexpr uint32_t cmddat_q_base = get_compile_time_arg_val(9);
constexpr uint32_t cmddat_q_size = get_compile_time_arg_val(10);
constexpr uint32_t scratch_cb_base = get_compile_time_arg_val(11);
constexpr uint32_t scratch_cb_size = get_compile_time_arg_val(12);


constexpr uint32_t prefetch_noc_xy = uint32_t(NOC_XY_ENCODING(PREFETCH_NOC_X, PREFETCH_NOC_Y));
constexpr uint32_t dispatch_noc_xy = uint32_t(NOC_XY_ENCODING(DISPATCH_NOC_X, DISPATCH_NOC_Y));
constexpr uint32_t dispatch_cb_page_size = 1 << dispatch_cb_log_page_size;
constexpr uint32_t dispatch_cb_end = dispatch_cb_base + (1 << dispatch_cb_log_page_size) * dispatch_cb_pages;
constexpr uint32_t host_q_end = host_q_base + host_q_size;
constexpr uint32_t cmddat_q_end = cmddat_q_base + cmddat_q_size;

static uint32_t pcie_read_ptr = pcie_base;
static uint32_t dispatch_data_ptr = dispatch_cb_base;

FORCE_INLINE
void dispatch_cb_acquire_pages(uint32_t n) {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dispatch_cb_sem));
    DEBUG_STATUS('A', 'P', 'W');
    while (*sem_addr == 0);
    DEBUG_STATUS('A', 'P', 'D');
    noc_semaphore_inc(get_noc_addr_helper(prefetch_noc_xy, (uint32_t)sem_addr), -n);
}

FORCE_INLINE
void dispatch_cb_release_pages(uint32_t n) {
    noc_semaphore_inc(get_noc_addr_helper(dispatch_noc_xy, get_semaphore(dispatch_cb_sem)), n);
}

FORCE_INLINE
void read_from_pcie(volatile tt_l1_ptr uint32_t *& host_q_rd_ptr, uint32_t& pending_read_size, uint32_t fence, uint32_t size) {

    uint64_t host_src_addr = get_noc_addr_helper(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y), pcie_read_ptr);
    DPRINT << "read: " << HEX() << host_src_addr << ENDL();
    noc_async_read(host_src_addr, fence, size);
    pending_read_size = size;
    pcie_read_ptr += size;

    *host_q_rd_ptr = 0;

    // XXXXX WRONG: racy, can't free this up until we've finished processing
    // Wrap
    if ((uint32_t)++host_q_rd_ptr == host_q_end) host_q_rd_ptr = (volatile tt_l1_ptr uint32_t*)host_q_base;

    // Tell host we read
    *(volatile tt_l1_ptr uint32_t *) host_q_rd_ptr_addr = (uint32_t)host_q_rd_ptr;
}


// This routine can be called in 8 states based on the boolean values cmd_ready, host_q_ready, read_pending:
//  - !cmd_ready, !host_q_ready, !read_pending: stall on host_q, issue read, read barrier
//  - !cmd_ready, !host_q_ready,  read pending: read barrier (and re-evaluate host_q_ready)
//  - !cmd_ready,  host_q_ready, !read_pending: issue read, read barrier (XXXX +issue read after?)
//  - !cmd_ready,  host_q_ready,  read_pending: read barrier, issue read
//  -  cmd_ready, !host_q_ready, !read_pending: exit
//  -  cmd_ready, !host_q_ready,  read_pending: exit (no barrier yet)
//  -  cmd_ready,  host_q_ready, !read_pending: issue read
//  -  cmd_ready,  host_q_ready,  read_pending: exit (don't add latency to the in flight request)
//
// With WH tagging of reads:
// open question: should fetcher loop on host_q_ready issuing reads until !host_q_ready
//  - !cmd_ready, !host_q_ready, !read_pending: stall on host_q, issue read, read barrier
//  - !cmd_ready, !host_q_ready,  read pending: read barrier on oldest tag
//  - !cmd_ready,  host_q_ready, !read_pending: issue read, read barrier (XXXX +retry after?)
//  - !cmd_ready,  host_q_ready,  read_pending: issue read, read barrier on oldest tag
//  -  cmd_ready, !host_q_ready, !read_pending: exit
//  -  cmd_ready, !host_q_ready,  read_pending: exit (no barrier yet)
//  -  cmd_ready,  host_q_ready, !read_pending: issue and tag read
//  -  cmd_ready,  host_q_ready,  read_pending: issue and tag read
static void get_cmds(uint32_t& fence, bool cmd_ready) {

    static uint32_t pending_read_size = 0;
    static volatile tt_l1_ptr uint32_t* host_q_rd_ptr = (volatile tt_l1_ptr uint32_t*)host_q_base;

    uint32_t fetch_size = *host_q_rd_ptr;
    DPRINT << (uint32_t)host_q_rd_ptr_addr << ENDL();
    DPRINT << (uint32_t)host_q_rd_ptr << ENDL();

    if (fetch_size != 0 && pending_read_size == 0) {
        DPRINT << "read1" << ENDL();
        read_from_pcie(host_q_rd_ptr, pending_read_size, fence, fetch_size);
    }
    if (!cmd_ready) {
        if (pending_read_size != 0) {
            DPRINT << "barrier" << ENDL();
            noc_async_read_barrier();
            fence += pending_read_size;
            pending_read_size = 0;

            // After the stall, re-check the host
            fetch_size = *host_q_rd_ptr;
            if (fetch_size != 0) {
                DPRINT << "read2" << ENDL();
                read_from_pcie(host_q_rd_ptr, pending_read_size, fence, fetch_size);
            }
        } else {
            // By here, host_q_ready must be false
            // Nothing to fetch, nothing pending, nothing available, stall on host
            DEBUG_STATUS('H', 'Q', 'W');
            DPRINT << "stall" << ENDL();
            while ((fetch_size = *host_q_rd_ptr) == 0);
            DPRINT << "recurse" << ENDL();
            get_cmds(fence, cmd_ready);
            DEBUG_STATUS('H', 'Q', 'D');
        }
    }
}

static uint32_t process_debug_cmd(uint32_t cmd_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t checksum = 0;
    uint32_t *data = (uint32_t *)((uint32_t)cmd + (uint32_t)sizeof(CQPrefetchCmd));
    for (uint32_t i = 0; i < cmd->debug.size / sizeof(uint32_t); i++) {
        checksum += *data++;
    }

    if (checksum != cmd->debug.checksum) {
        DPRINT << "checksum" << ENDL();
        DEBUG_STATUS('!', 'C', 'H', 'K');
        while(1);
    }

    return cmd_ptr + cmd->debug.stride;
}

static uint32_t process_relay_inline_cmd(uint32_t cmd_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t length = cmd->relay_inline.length;

    cmd_ptr += sizeof(CQPrefetchCmd);
    uint32_t npages = (length + dispatch_cb_page_size - 1) >> dispatch_cb_log_page_size;

    // XXXX make this a subroutine sharable w/ dram writes?
    dispatch_cb_acquire_pages(npages);
    noc_async_write(cmd_ptr, get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr), length);
    dispatch_data_ptr += dispatch_cb_page_size;
    if (dispatch_data_ptr == dispatch_cb_end) {
        dispatch_data_ptr = dispatch_cb_base;
    }
    // XXXXX - painful syncing right now?
    noc_async_write_barrier();
    dispatch_cb_release_pages(npages);

    return cmd_ptr + cmd->relay_inline.length;
}

void kernel_main() {

    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;

    DPRINT << "prefetcher" << ENDL();

    bool done = false;
    while (!done) {
        DPRINT << "top: " << fence << " " << cmd_ptr << ENDL();
        get_cmds(fence, cmd_ptr != fence);
        DPRINT << "after: " << fence << " " << cmd_ptr << ENDL();

        volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

        switch (cmd->base.cmd_id) {
        case CQ_PREFETCH_CMD_RELAY_DRAM_PAGED:
            break;

        case CQ_PREFETCH_CMD_RELAY_INLINE:
            DPRINT << "inline" << ENDL();
            cmd_ptr = process_relay_inline_cmd(cmd_ptr);
            break;

        case CQ_PREFETCH_CMD_WRAP:
            cmd_ptr = cmddat_q_base;
            break;

        case CQ_PREFETCH_CMD_STALL:
            break;

        case CQ_PREFETCH_CMD_DEBUG:
            DPRINT << "debug" << ENDL();
            cmd_ptr = process_debug_cmd(cmd_ptr);
            break;

        case CQ_PREFETCH_CMD_TERMINATE:
            DPRINT << "terminating\n";
            done = true;
            break;

        default:
            DPRINT << "prefetcher invalid command:" << (uint32_t)cmd->base.cmd_id << " " << cmd_ptr << " " << fence << " " << host_q_end << ENDL();
            DEBUG_STATUS('!', 'C', 'M', 'D');
            while(1);
        }
    }

    DPRINT << "prefetch out\n" << ENDL();
}
