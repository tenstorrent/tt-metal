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

static uint32_t host_q_log_minsize = 4;

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
void read_from_pcie(volatile tt_l1_ptr uint16_t *& host_q_rd_ptr,
                    uint32_t& pending_read_size,
                    uint32_t& fence,
                    uint32_t cmd_ptr,
                    uint32_t size) {

    // Wrap cmddat_q
    if (fence + size > cmddat_q_base + cmddat_q_size) {
        // only wrap if there are no commands ready, otherwise we'll leave some on the floor
        if (cmd_ptr != fence) {
            return;
        }
        fence = cmddat_q_base;
    }

    uint64_t host_src_addr = get_noc_addr_helper(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y), pcie_read_ptr);
    noc_async_read(host_src_addr, fence, size);
    pending_read_size = size;
    pcie_read_ptr += size;

    *host_q_rd_ptr = 0;

    // Tell host we read
    *(volatile tt_l1_ptr uint32_t *) host_q_rd_ptr_addr = (uint32_t)host_q_rd_ptr;

    host_q_rd_ptr++;

    // Wrap host_q
    if ((uint32_t)host_q_rd_ptr == host_q_end) {
        host_q_rd_ptr = (volatile tt_l1_ptr uint16_t*)host_q_base;
    }
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
static void get_cmds(uint32_t& fence, uint32_t& cmd_ptr) {

    static uint32_t pending_read_size = 0;
    static volatile tt_l1_ptr uint16_t* host_q_rd_ptr = (volatile tt_l1_ptr uint16_t*)host_q_base;

    if (fence < cmd_ptr) {
        DPRINT << "wrap cmd ptr1\n";
        cmd_ptr = fence;
    }

    bool cmd_ready = (cmd_ptr != fence);
    uint32_t fetch_size = (uint32_t)*host_q_rd_ptr << host_q_log_minsize;
    DPRINT << (uint32_t)host_q_rd_ptr_addr << ENDL();
    DPRINT << (uint32_t)host_q_rd_ptr << ENDL();

    if (fetch_size != 0 && pending_read_size == 0) {
        DPRINT << "read1: " << (uint32_t)host_q_rd_ptr << " " << " " << fence << " " << fetch_size << ENDL();
        read_from_pcie(host_q_rd_ptr, pending_read_size, fence, cmd_ptr, fetch_size);
    }
    if (!cmd_ready) {
        if (pending_read_size != 0) {
            DPRINT << "barrier" << ENDL();
            noc_async_read_barrier();

            // wrap the cmddat_q
            if (fence < cmd_ptr) {
                cmd_ptr = fence;
            }
            // XXXXX hack for now: prefetching the next command if this is a wrap command
            // will cause the next command to be mis-located. there are multiple ways to fix this
            // hacking here for now, revisit later
            volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)fence;
            fence += pending_read_size;
            pending_read_size = 0;
            if (cmd->base.cmd_id != CQ_PREFETCH_CMD_WRAP) {
                // After the stall, re-check the host
                fetch_size = (uint32_t)*host_q_rd_ptr << host_q_log_minsize;
                if (fetch_size != 0) {
                    DPRINT << "read2: " << (uint32_t)host_q_rd_ptr << " " << fetch_size << ENDL();
                    read_from_pcie(host_q_rd_ptr, pending_read_size, fence, cmd_ptr, fetch_size);
                }
            } else {
                DPRINT << "prefetcher wrap" << ENDL();
            }
        } else {
            // By here, host_q_ready must be false
            // Nothing to fetch, nothing pending, nothing available, stall on host
            DEBUG_STATUS('H', 'Q', 'W');
            DPRINT << "prefetcher stall" << ENDL();
            while ((fetch_size = *host_q_rd_ptr) == 0);
            DPRINT << "recurse" << ENDL();
            get_cmds(fence, cmd_ptr);
            DEBUG_STATUS('H', 'Q', 'D');
        }
    }
}

static uint32_t process_debug_cmd(uint32_t cmd_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t checksum = 0;
    uint32_t *data = (uint32_t *)((uint32_t)cmd + (uint32_t)sizeof(CQPrefetchCmd));
    uint32_t size = cmd->debug.size;
    for (uint32_t i = 0; i < size / sizeof(uint32_t); i++) {
        checksum += *data++;
    }

    if (checksum != cmd->debug.checksum) {
        DPRINT << "checksum" << checksum << " " << cmd->debug.checksum << ENDL();
        DEBUG_STATUS('!', 'C', 'H', 'K');
        while(1);
    }

    return cmd_ptr + cmd->debug.stride;
}

static uint32_t process_relay_inline_cmd(uint32_t cmd_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t length = cmd->relay_inline.length;

    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);
    uint32_t npages = (length + dispatch_cb_page_size - 1) >> dispatch_cb_log_page_size;

    // Assume the dispatch buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    dispatch_cb_acquire_pages(npages);
    uint32_t dispatch_pages_left = (dispatch_cb_end - dispatch_data_ptr) / dispatch_cb_page_size;
    if (dispatch_pages_left >= npages) {
        noc_async_write(data_ptr, get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr), length);
        dispatch_data_ptr += npages * dispatch_cb_page_size;
    } else {
        uint32_t tail_pages = npages - dispatch_pages_left;
        uint32_t available = dispatch_pages_left * dispatch_cb_page_size;
        if (available > 0) {
            noc_async_write(data_ptr, get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr), available);
            data_ptr += available;
            length -= available;
        }

        noc_async_write(data_ptr, get_noc_addr_helper(dispatch_noc_xy, dispatch_cb_base), length);
        dispatch_data_ptr = dispatch_cb_base + tail_pages * dispatch_cb_page_size;
    }

    // XXXXX - painful syncing right now?  move this into get_cmds
    noc_async_writes_flushed();
    dispatch_cb_release_pages(npages);

    return cmd_ptr + cmd->relay_inline.stride;
}

void kernel_main() {

    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;

    DPRINT << "prefetcher" << ENDL();

    bool done = false;
    while (!done) {
        DPRINT << "top: " << fence << " " << cmd_ptr << ENDL();
        get_cmds(fence, cmd_ptr);
        DPRINT << "after: " << fence << " " << cmd_ptr << ENDL();

        volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

        switch (cmd->base.cmd_id) {
        case CQ_PREFETCH_CMD_RELAY_DRAM_PAGED:
            DPRINT << "relay dram page" << ENDL();
            break;

        case CQ_PREFETCH_CMD_RELAY_INLINE:
            DPRINT << "inline" << ENDL();
            cmd_ptr = process_relay_inline_cmd(cmd_ptr);
            break;

        case CQ_PREFETCH_CMD_WRAP:
            DPRINT << "dev wrap: " << (uint32_t)sizeof(CQPrefetchCmd) << ENDL();
            pcie_read_ptr = pcie_base;
            cmd_ptr += 32;
            break;

        case CQ_PREFETCH_CMD_STALL:
            DPRINT << "stall" << ENDL();
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
