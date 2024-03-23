// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Common prefetch code for use by _hd, _h, _d prefetch variants

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"

#define CQ_PREFETCH_DOWNSTREAM_PACKET_HEADER_SIZE L1_NOC_ALIGNMENT

extern const uint32_t scratch_db_top[2];


template<uint32_t my_noc_xy,
         uint32_t my_sem_id,
         uint32_t downstream_noc_xy,
         uint32_t downstream_sem_id,
         uint32_t cb_base,
         uint32_t cb_end,
         uint32_t cb_log_page_size,
         uint32_t cb_page_size>
FORCE_INLINE
void write_downstream(uint32_t& data_ptr,
                      uint32_t& downstream_data_ptr,
                      uint32_t length) {

    uint32_t npages = (length + cb_page_size - 1) >> cb_log_page_size;

    // Assume the downstream buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    downstream_cb_acquire_pages<my_noc_xy, my_sem_id>(npages);
    uint32_t downstream_pages_left = (cb_end - downstream_data_ptr) / cb_page_size;
    if (downstream_pages_left >= npages) {
        noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), length);
        downstream_data_ptr += npages * cb_page_size;
    } else {
        uint32_t tail_pages = npages - downstream_pages_left;
        uint32_t available = downstream_pages_left * cb_page_size;
        if (available > 0) {
            noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), available);
            data_ptr += available;
            length -= available;
        }

        noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, cb_base), length);
        downstream_data_ptr = cb_base + tail_pages * cb_page_size;
    }

    // XXXXX - painful syncing right now?  move this into get_cmds
    noc_async_writes_flushed();
    downstream_cb_release_pages<downstream_noc_xy, downstream_sem_id>(npages);
}

template<uint32_t cmddat_q_base,
         uint32_t cmddat_q_size,
         uint32_t pcie_base,
         uint32_t pcie_size,
         uint32_t prefetch_q_base,
         uint32_t prefetch_q_end,
         uint32_t prefetch_q_rd_ptr_addr,
         uint32_t preamble_size>
FORCE_INLINE
void read_from_pcie(volatile tt_l1_ptr uint16_t *& prefetch_q_rd_ptr,
                    uint32_t& pending_read_size,
                    uint32_t& fence,
                    uint32_t& pcie_read_ptr,
                    uint32_t cmd_ptr,
                    uint32_t size) {

    // Wrap cmddat_q
    if (fence + size + preamble_size > cmddat_q_base + cmddat_q_size) {
        // only wrap if there are no commands ready, otherwise we'll leave some on the floor
        // TODO: does this matter for perf?
        if (cmd_ptr != fence) {
            return;
        }
        fence = cmddat_q_base;
    }
    fence += preamble_size;

    // Wrap pcie/hugepage
    if (pcie_read_ptr + size > pcie_size) {
        pcie_read_ptr = pcie_base;
    }

    uint64_t host_src_addr = get_noc_addr_helper(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y), pcie_read_ptr);
    noc_async_read(host_src_addr, fence, size);
    pending_read_size = size;
    pcie_read_ptr += size;

    *prefetch_q_rd_ptr = 0;

    // Tell host we read
    *(volatile tt_l1_ptr uint32_t *) prefetch_q_rd_ptr_addr = (uint32_t)prefetch_q_rd_ptr;

    prefetch_q_rd_ptr++;

    // Wrap prefetch_q
    if ((uint32_t)prefetch_q_rd_ptr == prefetch_q_end) {
        prefetch_q_rd_ptr = (volatile tt_l1_ptr uint16_t*)prefetch_q_base;
    }
}

// This routine can be called in 8 states based on the boolean values cmd_ready, prefetch_q_ready, read_pending:
//  - !cmd_ready, !prefetch_q_ready, !read_pending: stall on prefetch_q, issue read, read barrier
//  - !cmd_ready, !prefetch_q_ready,  read pending: read barrier (and re-evaluate prefetch_q_ready)
//  - !cmd_ready,  prefetch_q_ready, !read_pending: issue read, read barrier (XXXX +issue read after?)
//  - !cmd_ready,  prefetch_q_ready,  read_pending: read barrier, issue read
//  -  cmd_ready, !prefetch_q_ready, !read_pending: exit
//  -  cmd_ready, !prefetch_q_ready,  read_pending: exit (no barrier yet)
//  -  cmd_ready,  prefetch_q_ready, !read_pending: issue read
//  -  cmd_ready,  prefetch_q_ready,  read_pending: exit (don't add latency to the in flight request)
//
// With WH tagging of reads:
// open question: should fetcher loop on prefetch_q_ready issuing reads until !prefetch_q_ready
//  - !cmd_ready, !prefetch_q_ready, !read_pending: stall on prefetch_q, issue read, read barrier
//  - !cmd_ready, !prefetch_q_ready,  read pending: read barrier on oldest tag
//  - !cmd_ready,  prefetch_q_ready, !read_pending: issue read, read barrier (XXXX +retry after?)
//  - !cmd_ready,  prefetch_q_ready,  read_pending: issue read, read barrier on oldest tag
//  -  cmd_ready, !prefetch_q_ready, !read_pending: exit
//  -  cmd_ready, !prefetch_q_ready,  read_pending: exit (no barrier yet)
//  -  cmd_ready,  prefetch_q_ready, !read_pending: issue and tag read
//  -  cmd_ready,  prefetch_q_ready,  read_pending: issue and tag read
template<uint32_t cmddat_q_base,
         uint32_t cmddat_q_size,
         uint32_t pcie_base,
         uint32_t pcie_size,
         uint32_t prefetch_q_base,
         uint32_t prefetch_q_end,
         uint32_t prefetch_q_log_minsize,
         uint32_t prefetch_q_rd_ptr_addr,
         uint32_t preamble_size>
inline void fetch_q_get_cmds(uint32_t& fence, uint32_t& cmd_ptr, uint32_t& pcie_read_ptr) {

    static uint32_t pending_read_size = 0;
    static volatile tt_l1_ptr uint16_t* prefetch_q_rd_ptr = (volatile tt_l1_ptr uint16_t*)prefetch_q_base;

    if (fence < cmd_ptr) {
        DPRINT << "wrap cmd ptr1 " << fence << " " << cmd_ptr << ENDL();
        cmd_ptr = fence;
    }

    bool cmd_ready = (cmd_ptr != fence);
    uint32_t fetch_size = (uint32_t)*prefetch_q_rd_ptr << prefetch_q_log_minsize;

    if (fetch_size != 0 && pending_read_size == 0) {
        DPRINT << "read1: " << (uint32_t)prefetch_q_rd_ptr << " " << " " << fence << " " << fetch_size << ENDL();
        read_from_pcie<cmddat_q_base,
                       cmddat_q_size,
                       pcie_base,
                       pcie_size,
                       prefetch_q_base,
                       prefetch_q_end,
                       prefetch_q_rd_ptr_addr,
                       preamble_size>
            (prefetch_q_rd_ptr, pending_read_size, fence, pcie_read_ptr, cmd_ptr, fetch_size);
    }
    if (!cmd_ready) {
        if (pending_read_size != 0) {
            DPRINT << "barrier" << ENDL();
            noc_async_read_barrier();

            // wrap the cmddat_q
            if (fence < cmd_ptr) {
                cmd_ptr = fence;
            }

            fence += pending_read_size;
            pending_read_size = 0;
            // After the stall, re-check the host
            fetch_size = (uint32_t)*prefetch_q_rd_ptr << prefetch_q_log_minsize;
            if (fetch_size != 0) {
                DPRINT << "read2: " << (uint32_t)prefetch_q_rd_ptr << " " << fetch_size << ENDL();
                read_from_pcie<cmddat_q_base,
                               cmddat_q_size,
                               pcie_base,
                               pcie_size,
                               prefetch_q_base,
                               prefetch_q_end,
                               prefetch_q_rd_ptr_addr,
                               preamble_size>
                    (prefetch_q_rd_ptr, pending_read_size, fence, pcie_read_ptr, cmd_ptr, fetch_size);
            }
        } else {
            // By here, prefetch_q_ready must be false
            // Nothing to fetch, nothing pending, nothing available, stall on host
            DEBUG_STATUS('H', 'Q', 'W');
            DPRINT << "prefetcher stall" << ENDL();
            while ((fetch_size = *prefetch_q_rd_ptr) == 0);
            DPRINT << "recurse" << ENDL();
            fetch_q_get_cmds<cmddat_q_base,
                             cmddat_q_size,
                             pcie_base,
                             pcie_size,
                             prefetch_q_base,
                             prefetch_q_end,
                             prefetch_q_log_minsize,
                             prefetch_q_rd_ptr_addr,
                             preamble_size>(fence, cmd_ptr, pcie_read_ptr);
            DEBUG_STATUS('H', 'Q', 'D');
        }
    }
}

uint32_t process_debug_cmd(uint32_t cmd_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t checksum = 0;
    uint32_t *data = (uint32_t *)((uint32_t)cmd + (uint32_t)sizeof(CQPrefetchCmd));
    uint32_t size = cmd->debug.size;
    for (uint32_t i = 0; i < size / sizeof(uint32_t); i++) {
        checksum += *data++;
    }

    if (checksum != cmd->debug.checksum) {
        DEBUG_STATUS('!', 'C', 'H', 'K');
        ASSERT(0);
    }

    return cmd_ptr + cmd->debug.stride;
}

template<uint32_t my_noc_xy,
         uint32_t my_dispatch_sem_id,
         uint32_t dispatch_noc_xy,
         uint32_t dispatch_sem_id,
         uint32_t cb_base,
         uint32_t cb_end,
         uint32_t cb_log_page_size,
         uint32_t cb_page_size>
static uint32_t process_relay_inline_cmd(uint32_t cmd_ptr,
                                         uint32_t& dispatch_data_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t length = cmd->relay_inline.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    write_downstream<my_noc_xy,
                     my_dispatch_sem_id,
                     dispatch_noc_xy,
                     dispatch_sem_id,
                     cb_base,
                     cb_end,
                     cb_log_page_size,
                     cb_page_size>(data_ptr, dispatch_data_ptr, length);

    return cmd_ptr + cmd->relay_inline.stride;
}

// This version of inline sends inline data to the dispatcher but doesn't flush the page to the dispatcher
// This is used to assemble dispatcher commands when data comes out of band, eg, reading from DRAM
// That means this command is stateful, incorrect use will be...bad
// NOTE: this routine assumes we're sending a command header and that is LESS THAN A PAGE
template<uint32_t my_noc_xy,
         uint32_t my_dispatch_sem_id,
         uint32_t dispatch_noc_xy,
         uint32_t cb_base,
         uint32_t cb_end>
static uint32_t process_relay_inline_noflush_cmd(uint32_t cmd_ptr,
                                                 uint32_t& dispatch_data_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t length = sizeof(CQDispatchCmd);
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    downstream_cb_acquire_pages<my_noc_xy, my_dispatch_sem_id>(1);
    if (dispatch_data_ptr == cb_end) {
        dispatch_data_ptr = cb_base;
    }
    noc_async_write(data_ptr, get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr), length);
    dispatch_data_ptr += length;

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

template<uint32_t extra_space,
         bool test_for_nonzero,
         uint32_t my_noc_xy,
         uint32_t local_dispatch_cb_sem_id,
         uint32_t dispatch_noc_xy,
         uint32_t dispatch_cb_base,
         uint32_t dispatch_cb_end,
         uint32_t dispatch_cb_page_size>
static uint32_t write_pages_to_dispatcher(uint32_t& dispatch_data_ptr,
                                          uint32_t& scratch_write_addr,
                                          uint32_t& amt_to_write) {

    uint32_t page_residual_space = dispatch_cb_page_size - (dispatch_data_ptr & (dispatch_cb_page_size - 1));
    uint32_t npages = (amt_to_write - page_residual_space + dispatch_cb_page_size + extra_space - 1) / dispatch_cb_page_size;

    // Grabbing all pages at once is ok if scratch_size < 3 * dispatch_cb_block_size
    if (!test_for_nonzero || npages != 0) {
        downstream_cb_acquire_pages<my_noc_xy, local_dispatch_cb_sem_id>(npages);
    }

    uint64_t noc_addr = get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr);
    if (dispatch_data_ptr + amt_to_write > dispatch_cb_end) {  // wrap
        uint32_t last_chunk_size = dispatch_cb_end - dispatch_data_ptr;
        noc_async_write(scratch_write_addr, noc_addr, last_chunk_size);
        dispatch_data_ptr = dispatch_cb_base;
        scratch_write_addr += last_chunk_size;
        amt_to_write -= last_chunk_size;
        noc_addr = get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr);
    }

    noc_async_write(scratch_write_addr, noc_addr, amt_to_write);
    dispatch_data_ptr += amt_to_write;

    return npages;
}

// This fn prefetches data from DRAM memory and writes data to the dispatch core.
// Reading from DRAM has the following characteristics:
//  - latency is moderately high ~400 cycles on WH
//  - DRAM bw is ~maximized when page size reaches 2K
//  - for kernel dispatch, it is expected that page sizes will often be <2K
//  - for buffer writing, page sizes will vary
//  - writing to dispatcher works best with 4K pages (2K pages cover overhead, 4K gives perf cushion)
//  - writing a 4K page takes ~32*4=128 cycles
//  - writing 4 4K pages is 512 cycles, close to parity w/ the latency of DRAM
//  - to hide the latency (~12% overhead), assume we need to read ~32 pages=128K, double buffered
//  - in other words, we'll never achieve high efficiency and always be (somewhat) latency bound
// Algorithm does:
//  - read a batch from DRAM
//  - loop: read a batch from DRAM while sending to dispatcher
//  - send a batch to dispatcher
// The size of the first read should be based on latency.  With small page sizes
// bandwidth will be low and we'll be DRAM bound (send to dispatcher is ~free).
// With larger pages we'll get closer to a bandwidth match
// The dispatch buffer is a ring buffer.
template<bool is_dram,
         uint32_t my_noc_xy,
         uint32_t local_dispatch_cb_sem_id,
         uint32_t dispatch_noc_xy,
         uint32_t dispatch_cb_sem_id,
         uint32_t dispatch_cb_base,
         uint32_t dispatch_cb_end,
         uint32_t dispatch_cb_page_size,
         uint32_t scratch_db_half_size>
uint32_t process_relay_paged_cmd(uint32_t cmd_ptr,
                                 uint32_t& dispatch_data_ptr) {

    // This ensures that a previous cmd using the scratch buf has finished
    noc_async_writes_flushed();

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t page_id = cmd->relay_paged.start_page;
    uint32_t base_addr = cmd->relay_paged.base_addr;
    uint32_t page_size = cmd->relay_paged.page_size;
    uint32_t pages = cmd->relay_paged.pages;
    uint32_t read_length = pages * page_size;

    InterleavedAddrGen<is_dram> addr_gen;
    addr_gen.bank_base_address = base_addr;
    addr_gen.page_size = page_size;

    // First step - read into DB0
    uint32_t scratch_read_addr = scratch_db_top[0];
    uint32_t amt_to_read = (scratch_db_half_size > read_length) ? read_length : scratch_db_half_size;
    uint32_t amt_read = 0;
    while (amt_to_read >= page_size) {
        uint64_t noc_addr = addr_gen.get_noc_addr(page_id); // XXXX replace this w/ walking the banks to save mul on GS
        noc_async_read(noc_addr, scratch_read_addr, page_size);
        scratch_read_addr += page_size;
        page_id++;
        amt_to_read -= page_size;
        amt_read += page_size;
    }
    noc_async_read_barrier();

    // Second step - read into DB[x], write from DB[x], toggle x, iterate
    // Writes are fast, reads are slow
    uint32_t db_toggle = 0;
    uint32_t scratch_write_addr;
    read_length -= amt_read;
    while (read_length != 0) {
        // This ensures that writes from prior iteration are done
        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_writes_flushed();

        db_toggle ^= 1;
        scratch_read_addr = scratch_db_top[db_toggle];
        scratch_write_addr = scratch_db_top[db_toggle ^ 1];

        uint32_t amt_to_write = amt_read;
        amt_to_read = (scratch_db_half_size > read_length) ? read_length : scratch_db_half_size;
        amt_read = 0;
        while (amt_to_read >= page_size) {
            uint64_t noc_addr = addr_gen.get_noc_addr(page_id); // XXXX replace this w/ walking the banks to save mul on GS
            noc_async_read(noc_addr, scratch_read_addr, page_size);
            scratch_read_addr += page_size;
            page_id++;
            amt_to_read -= page_size;
            amt_read += page_size;
        }

        // Third step - write from DB
        uint32_t npages = write_pages_to_dispatcher<
            0,
            false,
            my_noc_xy,
            local_dispatch_cb_sem_id,
            dispatch_noc_xy,
            dispatch_cb_base,
            dispatch_cb_end,
            dispatch_cb_page_size>(dispatch_data_ptr, scratch_write_addr, amt_to_write);
        downstream_cb_release_pages<dispatch_noc_xy, dispatch_cb_sem_id>(npages);

        read_length -= amt_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    scratch_write_addr = scratch_db_top[db_toggle];
    uint32_t amt_to_write = amt_read;
    uint32_t npages = write_pages_to_dispatcher<
        CQ_DISPATCH_CMD_SIZE,
        true,
        my_noc_xy,
        local_dispatch_cb_sem_id,
        dispatch_noc_xy,
        dispatch_cb_base,
        dispatch_cb_end,
        dispatch_cb_page_size>(dispatch_data_ptr, scratch_write_addr, amt_to_write);

    uint32_t pad_to_page = dispatch_cb_page_size - (dispatch_data_ptr & (dispatch_cb_page_size - 1));
    dispatch_data_ptr += pad_to_page;

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH
    downstream_cb_release_pages<dispatch_noc_xy, dispatch_cb_sem_id>(npages + 1);

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

template<uint32_t my_noc_xy,
         uint32_t local_dispatch_cb_sem_id,
         uint32_t dispatch_noc_xy,
         uint32_t dispatch_cb_sem_id,
         uint32_t dispatch_cb_base,
         uint32_t dispatch_cb_end,
         uint32_t dispatch_cb_page_size,
         uint32_t scratch_db_half_size>
uint32_t process_relay_linear_cmd(uint32_t cmd_ptr,
                                  uint32_t& dispatch_data_ptr) {

    // This ensures that a previous cmd using the scratch buf has finished
    noc_async_writes_flushed();

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t noc_xy_addr = cmd->relay_linear.noc_xy_addr;
    uint32_t read_addr = cmd->relay_linear.addr;
    uint32_t length = cmd->relay_linear.length;
    uint32_t read_length = length;

    // First step - read into DB0
    uint32_t scratch_read_addr = scratch_db_top[0];
    uint32_t amt_to_read = (scratch_db_half_size > read_length) ? read_length : scratch_db_half_size;
    uint64_t noc_addr = get_noc_addr_helper(noc_xy_addr, read_addr);
    noc_async_read(noc_addr, scratch_read_addr, amt_to_read);
    read_addr += amt_to_read;
    noc_async_read_barrier();

    // Second step - read into DB[x], write from DB[x], toggle x, iterate
    // Writes are fast, reads are slow
    uint32_t db_toggle = 0;
    uint32_t scratch_write_addr;
    read_length -= amt_to_read;
    while (read_length != 0) {
        // This ensures that writes from prior iteration are done
        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_writes_flushed();

        db_toggle ^= 1;
        scratch_read_addr = scratch_db_top[db_toggle];
        scratch_write_addr = scratch_db_top[db_toggle ^ 1];

        uint32_t amt_to_write = amt_to_read;
        amt_to_read = (scratch_db_half_size > read_length) ? read_length : scratch_db_half_size;
        noc_addr = get_noc_addr_helper(noc_xy_addr, read_addr);
        noc_async_read(noc_addr, scratch_read_addr, amt_to_read);
        read_addr += amt_to_read;

        // Third step - write from DB
        uint32_t npages = write_pages_to_dispatcher<
            0,
            false,
            my_noc_xy,
            local_dispatch_cb_sem_id,
            dispatch_noc_xy,
            dispatch_cb_base,
            dispatch_cb_end,
            dispatch_cb_page_size>(dispatch_data_ptr, scratch_write_addr, amt_to_write);

        downstream_cb_release_pages<dispatch_noc_xy, dispatch_cb_sem_id>(npages);

        read_length -= amt_to_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    scratch_write_addr = scratch_db_top[db_toggle];
    uint32_t amt_to_write = amt_to_read;
    uint32_t npages = write_pages_to_dispatcher<
        CQ_DISPATCH_CMD_SIZE,
        true,
        my_noc_xy,
        local_dispatch_cb_sem_id,
        dispatch_noc_xy,
        dispatch_cb_base,
        dispatch_cb_end,
        dispatch_cb_page_size>(dispatch_data_ptr, scratch_write_addr, amt_to_write);

    uint32_t pad_to_page = dispatch_cb_page_size - (dispatch_data_ptr & (dispatch_cb_page_size - 1));
    dispatch_data_ptr += pad_to_page;

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH
    downstream_cb_release_pages<dispatch_noc_xy, dispatch_cb_sem_id>(npages + 1);

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

template<uint32_t dispatch_sync_sem_id>
uint32_t process_stall(uint32_t cmd_ptr) {

    static uint32_t count = 0;

    count++;

    DEBUG_STATUS('P', 'S', 'W');
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dispatch_sync_sem_id));
    while (*sem_addr != count);
    DEBUG_STATUS('P', 'S', 'D');

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}
