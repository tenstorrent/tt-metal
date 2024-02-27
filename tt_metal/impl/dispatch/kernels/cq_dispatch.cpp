// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch kernel
//  - receives data in pages from prefetch kernel into the dispatch buffer ring buffer
//  - processes commands with embedded data from the dispatch buffer to write/sync/etc w/ destination
//  - sync w/ prefetcher is via 2 semaphores, page_ready, page_done
//  - page size must be a power of 2
//  - # blocks must evenly divide the dispatch buffer size
//  - dispatch buffer base must be page size aligned

#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"
#include "debug/dprint.h"

constexpr uint32_t dispatch_cb_base = get_compile_time_arg_val(0);
constexpr uint32_t dispatch_cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t dispatch_cb_pages = get_compile_time_arg_val(2);
constexpr uint32_t dispatch_cb_sem = get_compile_time_arg_val(3);
constexpr uint32_t dispatch_cb_blocks = get_compile_time_arg_val(4);

constexpr uint32_t prefetch_noc_xy = uint32_t(NOC_XY_ENCODING(PREFETCH_NOC_X, PREFETCH_NOC_Y));
constexpr uint32_t dispatch_noc_xy = uint32_t(NOC_XY_ENCODING(DISPATCH_NOC_X, DISPATCH_NOC_Y));
constexpr uint32_t dispatch_cb_page_size = 1 << dispatch_cb_log_page_size;
constexpr uint32_t dispatch_cb_size = dispatch_cb_page_size * dispatch_cb_pages;
constexpr uint32_t dispatch_cb_end = dispatch_cb_base + dispatch_cb_size;


// Break buffer into blocks, 1/n of the total (dividing equally)
// Do bookkeeping (release, etc) based on blocks
// Note: due to the current method of release pages, up to 1 block of pages
// may be unavailable to the prefetcher at any time
constexpr uint32_t dispatch_cb_pages_per_block = dispatch_cb_pages / dispatch_cb_blocks;

static uint32_t block_next_start_addr[dispatch_cb_blocks];
static uint32_t block_noc_writes_to_clear[dispatch_cb_blocks];
static int rd_block_idx;
static int wr_block_idx;

static uint32_t cb_fence; // walks through cb page by page
static uint32_t cmd_ptr;  // walks through pages in cb cmd by cmd


FORCE_INLINE
uint32_t dispatch_cb_acquire_pages() {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dispatch_cb_sem));

    DEBUG_STATUS('A', 'P', 'W');
    uint32_t available;
    while ((available = *sem_addr) == 0);
    DEBUG_STATUS('A', 'P', 'D');

    // Set a fence to limit how much is processed at once
    uint32_t limit = (block_next_start_addr[rd_block_idx] - cb_fence) >> dispatch_cb_log_page_size;
    if (available > limit) available = limit;

    noc_semaphore_inc(get_noc_addr_helper(dispatch_noc_xy, (uint32_t)sem_addr), -available);

    return available;
}

FORCE_INLINE
void dispatch_cb_block_release_pages() {

    uint32_t sem_addr = get_semaphore(dispatch_cb_sem);

    uint32_t noc_progress = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
    if (noc_progress >= block_noc_writes_to_clear[wr_block_idx]) { // XXXXX ugh, 32 bit wrap?
        noc_semaphore_inc(get_noc_addr_helper(prefetch_noc_xy, sem_addr), dispatch_cb_pages_per_block);
        wr_block_idx++;
        wr_block_idx &= (dispatch_cb_blocks - 1);

        // if >dispatch_cb_pages_per_block are in flight away from this core
        // then we can fall behind by a block and never catch up
        // checking twice ensures we "gain" on the front if possible
        if (noc_progress >= block_noc_writes_to_clear[wr_block_idx]) {
            noc_semaphore_inc(get_noc_addr_helper(prefetch_noc_xy, sem_addr), dispatch_cb_pages_per_block);
            wr_block_idx++;
            wr_block_idx &= (dispatch_cb_blocks - 1);
        }
    }
}

FORCE_INLINE
void move_to_next_block() {

    // This is subtle: in the free-running case, we don't want to clear the current block
    // if the noc catches up so we artificially inflate the clear value by 1 when we start
    // a block and adjust it down by 1 here as we complete a block
    uint32_t write_count = block_noc_writes_to_clear[rd_block_idx];
    block_noc_writes_to_clear[rd_block_idx] = write_count - 1;

    rd_block_idx++;
    if (rd_block_idx == dispatch_cb_blocks) {
        rd_block_idx = 0;
        cb_fence = dispatch_cb_base;
        cmd_ptr = dispatch_cb_base;
    }

    block_noc_writes_to_clear[rd_block_idx] = write_count; // this is plus 1
}

FORCE_INLINE
void get_dispatch_cb_page() {
    // Strided past the data that has arrived, get the next page
    if (cb_fence == block_next_start_addr[rd_block_idx]) {
        move_to_next_block();
    }

    // Wait for dispatcher to supply a page
    uint32_t n_pages = dispatch_cb_acquire_pages();
    cb_fence += n_pages * dispatch_cb_page_size;
}

// Note that for non-paged writes, the number of writes per page is always 1
// This means each noc_write frees up a page
FORCE_INLINE
void dispatch_write() {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;

    uint32_t dst_noc = cmd->write.dst_noc_addr;
    uint32_t dst_addr = cmd->write.dst_addr;
    uint32_t length = cmd->write.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);

    while (length != 0) {
        uint32_t xfer_size = (length > dispatch_cb_page_size) ? dispatch_cb_page_size : length;
        uint64_t dst = get_noc_addr_helper(dst_noc, dst_addr);

        // Get a page if needed
        if (data_ptr + xfer_size > cb_fence) {
            // Check for block completion
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    uint32_t orphan_size = dispatch_cb_end - data_ptr;
                    if (orphan_size != 0) {
                        noc_async_write(data_ptr, dst, orphan_size);
                        block_noc_writes_to_clear[rd_block_idx]++;
                        length -= orphan_size;
                        xfer_size -= orphan_size;
                        dst_addr += orphan_size;
                    }
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                    dst = get_noc_addr_helper(dst_noc, dst_addr);
                }

                move_to_next_block();
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = dispatch_cb_acquire_pages();
            cb_fence += n_pages * dispatch_cb_page_size;

            // Release pages for prefetcher
            // Since we gate how much we acquire to < 1/4 the buffer, this should be called enough
            dispatch_cb_block_release_pages();
        }

        noc_async_write(data_ptr, dst, xfer_size);
        block_noc_writes_to_clear[rd_block_idx]++; // XXXXX maybe just write the noc internal api counter

        length -= xfer_size;
        data_ptr += xfer_size;
        dst_addr += xfer_size;
    }
    cmd_ptr = data_ptr;
}

static uint32_t process_debug_cmd(uint32_t cmd_ptr) {

    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t checksum = 0;
    uint32_t *data = (uint32_t *)((uint32_t)cmd + (uint32_t)sizeof(CQDispatchCmd));
    uint32_t size = cmd->debug.size;
    DPRINT << "checksum: " << cmd->debug.size << ENDL();

    // Dispatch checksum only handles running checksum on a single page
    // Host code prevents larger from flowing through
    // This way this code doesn't have to fetch multiple pages and then run
    // a cmd within those pages (messing up the implementation of that command)
    for (uint32_t i = 0; i < size / sizeof(uint32_t); i++) {
        checksum += *data++;
    }

    if (checksum != cmd->debug.checksum) {
        DPRINT << "!checksum" << ENDL();
        DEBUG_STATUS('!', 'C', 'H', 'K');
        while(1);
    }

    return cmd_ptr + cmd->debug.stride;
}

void kernel_main() {
    DPRINT << "dispatcher start" << ENDL();;

    for (uint32_t i = 0; i < dispatch_cb_blocks; i++) {
        uint32_t next_block = i + 1;
        uint32_t offset = next_block * dispatch_cb_pages_per_block * dispatch_cb_page_size;
        block_next_start_addr[i] = dispatch_cb_base + offset;
    }

    cb_fence = dispatch_cb_base;
    rd_block_idx = 0;
    wr_block_idx = 0;
    block_noc_writes_to_clear[0] = noc_nonposted_writes_num_issued[noc_index] + 1;
    cmd_ptr = dispatch_cb_base;
    bool done = false;
    while (!done) {
        if (cmd_ptr == cb_fence) {
            get_dispatch_cb_page();
        }

    re_run_command:
        volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;

        switch (cmd->base.cmd_id) {
        case CQ_DISPATCH_CMD_WRITE:
            DEBUG_STATUS('D', 'W', 'B');
            dispatch_write();
            DEBUG_STATUS('D', 'W', 'D');
            break;

        case CQ_DISPATCH_CMD_WRITE_PAGED:
            break;

        case CQ_DISPATCH_CMD_WAIT:
            break;

        case CQ_DISPATCH_CMD_GO:
            break;

        case CQ_DISPATCH_CMD_SINK:
            break;

        case CQ_DISPATCH_CMD_DEBUG:
            cmd_ptr = process_debug_cmd(cmd_ptr);
            goto re_run_command;
            break;

        case CQ_DISPATCH_CMD_TERMINATE:
            done = true;
            break;

        default:
            DPRINT << "dispatcher invalid command:" << cmd_ptr << " " << cb_fence << " " << " " << dispatch_cb_base << " " << dispatch_cb_end << " " << rd_block_idx << " " << ENDL();
            DEBUG_STATUS('!', 'C', 'M', 'D');
            while(1);
        }

        // Move to next page
        cmd_ptr += (dispatch_cb_page_size - (cmd_ptr & (dispatch_cb_page_size - 1))) & (dispatch_cb_page_size - 1);

        // XXXXX move this inside while loop waiting for get_dispatch_cb_page above
        // XXXXX can potentially clear a partial block when stalled w/ some more bookkeeping
        dispatch_cb_block_release_pages();
    }

    dispatch_cb_block_release_pages();
    noc_async_write_barrier();

    DPRINT << "dispatch out" << ENDL();
}
