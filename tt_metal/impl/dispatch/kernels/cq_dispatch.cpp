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
#include "debug/assert.h"

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


#define L1_ALIGNMENT 16 // XXXXX is the defined elsewhere?


FORCE_INLINE
uint32_t dispatch_cb_acquire_pages() {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dispatch_cb_sem));

    static uint32_t available = 0;

    if (available == 0) {
        // Ensure last sem_inc has landed
        noc_async_write_barrier(); // XXXX TODO(pgk) can we do better on wormhole?

        DEBUG_STATUS('A', 'P', 'W');
        while ((available = *sem_addr) == 0);
        DEBUG_STATUS('A', 'P', 'D');
    }

    // Set a fence to limit how much is processed at once
    uint32_t limit = (block_next_start_addr[rd_block_idx] - cb_fence) >> dispatch_cb_log_page_size;
    uint32_t usable = (available > limit) ? limit : available;

    noc_semaphore_inc(get_noc_addr_helper(dispatch_noc_xy, (uint32_t)sem_addr), -usable);
    available -= usable;

    return usable;
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
void move_rd_to_next_block() {

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
        move_rd_to_next_block();
    }

    // Wait for dispatcher to supply a page
    uint32_t n_pages = dispatch_cb_acquire_pages();
    cb_fence += n_pages * dispatch_cb_page_size;
}

// Note that for non-paged writes, the number of writes per page is always 1
// This means each noc_write frees up a page
template<bool multicast>
FORCE_INLINE
void process_write_linear(uint32_t num_mcast_dests) {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;

    uint32_t dst_noc = cmd->write.noc_xy_addr;
    uint32_t dst_addr = cmd->write.addr;
    uint32_t length = cmd->write.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);
    DPRINT << "dispatch_write: " << length << " num_mcast_dests: " << num_mcast_dests << ENDL();
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
                        if constexpr (multicast){
                            noc_async_write_multicast(data_ptr, dst, orphan_size, num_mcast_dests);
                        } else {
                            noc_async_write(data_ptr, dst, orphan_size);
                        }
                        block_noc_writes_to_clear[rd_block_idx]++;
                        length -= orphan_size;
                        xfer_size -= orphan_size;
                        dst_addr += orphan_size;
                    }
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                    dst = get_noc_addr_helper(dst_noc, dst_addr);
                }

                move_rd_to_next_block();
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = dispatch_cb_acquire_pages();
            cb_fence += n_pages * dispatch_cb_page_size;

            // Release pages for prefetcher
            // Since we gate how much we acquire to < 1/4 the buffer, this should be called enough
            dispatch_cb_block_release_pages();
        }

        if constexpr (multicast){
            noc_async_write_multicast(data_ptr, dst, xfer_size, num_mcast_dests);
        } else {
            noc_async_write(data_ptr, dst, xfer_size);
        }
        block_noc_writes_to_clear[rd_block_idx]++; // XXXXX maybe just write the noc internal api counter

        length -= xfer_size;
        data_ptr += xfer_size;
        dst_addr += xfer_size;
    }
    cmd_ptr = data_ptr;
}

FORCE_INLINE
void process_write() {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;
    uint32_t num_mcast_dests = cmd->write.num_mcast_dests;
    if (num_mcast_dests == 0) {
        process_write_linear<false>(0);
    } else {
        process_write_linear<true>(num_mcast_dests);
    }
}

template<bool is_dram>
FORCE_INLINE
void process_write_paged() {
    volatile tt_l1_ptr CQDispatchCmd *cmd = (volatile tt_l1_ptr CQDispatchCmd *)cmd_ptr;

    uint32_t page_id = cmd->write_paged.start_page;
    uint32_t base_addr = cmd->write_paged.base_addr;
    uint32_t page_size = cmd->write_paged.page_size;
    uint32_t pages = cmd->write_paged.pages;
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd);
    uint32_t write_length = pages * page_size;
    InterleavedAddrGen<is_dram> addr_gen;
    addr_gen.bank_base_address = base_addr;
    addr_gen.page_size = page_size;
    uint64_t dst_addr_offset = 0; // Offset into page.

    DPRINT << "process_write_paged - pages: " << pages << " page_size: " << page_size << " dispatch_cb_page_size: " << dispatch_cb_page_size;
    DPRINT << " start_page: " << page_id << " base_addr: " << HEX() << base_addr << DEC() << ENDL();

    while (write_length != 0) {

        uint32_t xfer_size = page_size > dispatch_cb_page_size ? dispatch_cb_page_size : page_size;
        uint64_t dst = addr_gen.get_noc_addr(page_id, dst_addr_offset); // XXXX replace this w/ walking the banks to save mul on GS

        // Get a Dispatch page if needed
        if (data_ptr + xfer_size > cb_fence) {
            // Check for block completion
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    uint32_t orphan_size = dispatch_cb_end - data_ptr;
                    if (orphan_size != 0) {
                        noc_async_write(data_ptr, dst, orphan_size);
                        block_noc_writes_to_clear[rd_block_idx]++;
                        write_length -= orphan_size;
                        xfer_size -= orphan_size;
                        dst_addr_offset += orphan_size;
                    }
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                    dst = addr_gen.get_noc_addr(page_id, dst_addr_offset);
                }
                move_rd_to_next_block();
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

        // If paged write is not completed for a page (dispatch_cb_page_size < page_size) then add offset, otherwise incr page_id.
        if (dst_addr_offset + xfer_size < page_size) {
            dst_addr_offset += xfer_size;
        } else {
            page_id++;
            dst_addr_offset = 0;
        }

        write_length -= xfer_size;
        data_ptr += xfer_size;
    }

    cmd_ptr = data_ptr;
}

// Packed write command
// Layout looks like:
//   - CQDispatchCmd struct
//   - count CQDispatchWritePackedSubCmd structs (max 1020)
//   - pad to L1 alignment
//   - count data packets of size size, each L1 aligned
//
// Note that there are multiple size restrictions on this cmd:
//  - all sub_cmds fit in one page
//  - size fits in one page
//
// Since all subcmds all appear in the first page and given the size restrictions
// this command can't be too many pages.  All pages are released at the end
template<bool mcast, typename WritePackedSubCmd>
FORCE_INLINE
void process_write_packed() {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t count = cmd->write_packed.count;
    uint32_t xfer_size = cmd->write_packed.size;
    uint32_t dst_addr = cmd->write_packed.addr;

    ASSERT(xfer_size < dispatch_cb_page_size);

    volatile WritePackedSubCmd tt_l1_ptr *sub_cmd_ptr =
        (volatile WritePackedSubCmd tt_l1_ptr *)(cmd_ptr + sizeof(CQDispatchCmd));
    uint32_t data_ptr = cmd_ptr + sizeof(CQDispatchCmd) + count * sizeof(WritePackedSubCmd);
    data_ptr = (data_ptr + L1_ALIGNMENT - 1) & ~(L1_ALIGNMENT - 1);
    uint32_t stride = (xfer_size + L1_ALIGNMENT - 1) & ~(L1_ALIGNMENT - 1);

    DPRINT << "dispatch_write_packed: " << xfer_size << " " << stride << " " << data_ptr << " " << count << ENDL();
    while (count != 0) {
        uint32_t dst_noc = sub_cmd_ptr->noc_xy_addr;
        uint32_t num_dests = mcast ?
            ((volatile CQDispatchWritePackedMulticastSubCmd tt_l1_ptr *)sub_cmd_ptr)->num_mcast_dests :
            0;
        sub_cmd_ptr++;
        uint64_t dst = get_noc_addr_helper(dst_noc, dst_addr);

        // Get a page if needed
        if (data_ptr + xfer_size > cb_fence) {
            DPRINT << data_ptr << " " << cb_fence << ENDL();
            // Check for block completion
            uint32_t remainder_xfer_size = 0;
            uint32_t remainder_dst_addr;
            uint32_t orphan_size;
            if (cb_fence == block_next_start_addr[rd_block_idx]) {
                // Check for dispatch_cb wrap
                if (rd_block_idx == dispatch_cb_blocks - 1) {
                    orphan_size = dispatch_cb_end - data_ptr;
                    if (orphan_size != 0) {
                        if (mcast) {
                            noc_async_write_multicast(data_ptr, dst, remainder_xfer_size, num_dests);
                        } else {
                            noc_async_write(data_ptr, dst, orphan_size);
                        }
                        block_noc_writes_to_clear[rd_block_idx]++;
                        remainder_xfer_size = xfer_size - orphan_size;
                        remainder_dst_addr = dst_addr + orphan_size;
                    }
                    cb_fence = dispatch_cb_base;
                    data_ptr = dispatch_cb_base;
                }

                move_rd_to_next_block();
            }

            // Wait for dispatcher to supply a page (this won't go beyond the buffer end)
            uint32_t n_pages = dispatch_cb_acquire_pages();
            cb_fence += n_pages * dispatch_cb_page_size;

            // This is done here so the common case doesn't have to restore the pointers
            if (remainder_xfer_size != 0) {
                uint64_t dst = get_noc_addr_helper(dst_noc, remainder_dst_addr);
                if (mcast) {
                    noc_async_write_multicast(data_ptr, dst, remainder_xfer_size, num_dests);
                } else {
                    noc_async_write(data_ptr, dst, remainder_xfer_size);
                }
                block_noc_writes_to_clear[rd_block_idx]++;

                count--;
                data_ptr += stride - orphan_size;

                continue;
            }
        }

        noc_async_write(data_ptr, dst, xfer_size);
        block_noc_writes_to_clear[rd_block_idx]++; // XXXXX maybe just write the noc internal api counter

        count--;
        data_ptr += stride;
    }

    // Release pages for prefetcher
    // Since we gate how much we acquire to < 1/4 the buffer, this should be called enough
    dispatch_cb_block_release_pages();

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

static void process_wait() {
    volatile CQDispatchCmd tt_l1_ptr *cmd = (volatile CQDispatchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t addr = cmd->wait.addr;
    uint32_t count = cmd->wait.count;

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    while (*sem_addr < count); // XXXXX use a wrapping compare
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
        case CQ_DISPATCH_CMD_WRITE_HOST:
            DEBUG_STATUS('D', 'W', 'B');
            DPRINT << "cmd_write\n";
            process_write();
            DEBUG_STATUS('D', 'W', 'D');
            break;

        case CQ_DISPATCH_CMD_WRITE_PAGED:
            DPRINT << "cmd_write_paged is_dram: " << (uint32_t) cmd->write_paged.is_dram << ENDL();
            if (cmd->write_paged.is_dram) {
                process_write_paged<true>();
            } else {
                process_write_paged<false>();
            }
            break;

        case CQ_DISPATCH_CMD_WRITE_PACKED:
            DPRINT << "cmd_write_packed" << ENDL();
            if (cmd->write_packed.is_multicast) {
                process_write_packed<true, CQDispatchWritePackedMulticastSubCmd>();
            } else {
                process_write_packed<false, CQDispatchWritePackedUnicastSubCmd>();
            }
            break;

        case CQ_DISPATCH_CMD_WAIT:
            DPRINT << "cmd_wait" << ENDL();
            process_wait();
            break;

        case CQ_DISPATCH_CMD_GO:
            DPRINT << "cmd_go" << ENDL();
            break;

        case CQ_DISPATCH_CMD_SINK:
            DPRINT << "cmd_sink" << ENDL();
            break;

        case CQ_DISPATCH_CMD_DEBUG:
            DPRINT << "cmd_debug" << ENDL();
            cmd_ptr = process_debug_cmd(cmd_ptr);
            goto re_run_command;
            break;

        case CQ_DISPATCH_CMD_TERMINATE:
            DPRINT << "dispatch terminate\n";
            done = true;
            break;

        default:
            DPRINT << "dispatcher invalid command:" << cmd_ptr << " " << cb_fence << " " << " " << dispatch_cb_base << " " << dispatch_cb_end << " " << rd_block_idx << " " << "xx" << ENDL();
            DPRINT << HEX() << *(uint32_t*)cmd_ptr << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr+1) << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr+2) << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr+3) << ENDL();
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
