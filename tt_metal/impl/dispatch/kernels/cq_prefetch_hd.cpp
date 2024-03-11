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
#include "debug/dprint.h"

constexpr uint32_t dispatch_cb_base = get_compile_time_arg_val(0);
constexpr uint32_t dispatch_cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t dispatch_cb_pages = get_compile_time_arg_val(2);
constexpr uint32_t dispatch_cb_sem = get_compile_time_arg_val(3);
constexpr uint32_t pcie_base = get_compile_time_arg_val(4);
constexpr uint32_t pcie_size = get_compile_time_arg_val(5);
constexpr uint32_t prefetch_q_base = get_compile_time_arg_val(6);
constexpr uint32_t prefetch_q_size = get_compile_time_arg_val(7);
constexpr uint32_t prefetch_q_rd_ptr_addr = get_compile_time_arg_val(8);;
constexpr uint32_t cmddat_q_base = get_compile_time_arg_val(9);
constexpr uint32_t cmddat_q_size = get_compile_time_arg_val(10);
constexpr uint32_t scratch_db_base = get_compile_time_arg_val(11);
constexpr uint32_t scratch_db_size = get_compile_time_arg_val(12);

constexpr uint32_t prefetch_noc_xy = uint32_t(NOC_XY_ENCODING(PREFETCH_NOC_X, PREFETCH_NOC_Y));
constexpr uint32_t dispatch_noc_xy = uint32_t(NOC_XY_ENCODING(DISPATCH_NOC_X, DISPATCH_NOC_Y));
constexpr uint32_t dispatch_cb_page_size = 1 << dispatch_cb_log_page_size;
constexpr uint32_t dispatch_cb_end = dispatch_cb_base + (1 << dispatch_cb_log_page_size) * dispatch_cb_pages;
constexpr uint32_t prefetch_q_end = prefetch_q_base + prefetch_q_size;
constexpr uint32_t cmddat_q_end = cmddat_q_base + cmddat_q_size;

constexpr uint32_t scratch_db_half_size = scratch_db_size / 2;
constexpr uint32_t scratch_db_base0 = scratch_db_base;
constexpr uint32_t scratch_db_base1 = scratch_db_base + scratch_db_half_size;

static uint32_t pcie_read_ptr = pcie_base;
static uint32_t dispatch_data_ptr = dispatch_cb_base;

static uint32_t prefetch_q_log_minsize = 4;

static uint32_t scratch_db_top[2] = {scratch_db_base0, scratch_db_base1};

static_assert((dispatch_cb_base & (dispatch_cb_page_size - 1)) == 0);


FORCE_INLINE
void dispatch_cb_acquire_pages(uint32_t n) {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dispatch_cb_sem));
    DEBUG_STATUS('A', 'P', 'W');

    // Ensure last sem_inc has landed
    noc_async_write_barrier(); // XXXX TODO(pgk) can we do better on wormhole?

    while (*sem_addr < n);
    DEBUG_STATUS('A', 'P', 'D');
    noc_semaphore_inc(get_noc_addr_helper(prefetch_noc_xy, (uint32_t)sem_addr), -n);
}

FORCE_INLINE
void dispatch_cb_release_pages(uint32_t n) {
    noc_semaphore_inc(get_noc_addr_helper(dispatch_noc_xy, get_semaphore(dispatch_cb_sem)), n);
}

FORCE_INLINE
void read_from_pcie(volatile tt_l1_ptr uint16_t *& prefetch_q_rd_ptr,
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
static void get_cmds(uint32_t& fence, uint32_t& cmd_ptr) {

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
        read_from_pcie(prefetch_q_rd_ptr, pending_read_size, fence, cmd_ptr, fetch_size);
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
                read_from_pcie(prefetch_q_rd_ptr, pending_read_size, fence, cmd_ptr, fetch_size);
            }
        } else {
            // By here, prefetch_q_ready must be false
            // Nothing to fetch, nothing pending, nothing available, stall on host
            DEBUG_STATUS('H', 'Q', 'W');
            DPRINT << "prefetcher stall" << ENDL();
            while ((fetch_size = *prefetch_q_rd_ptr) == 0);
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

// This version of inline sends inline data to the dispatcher but doesn't flush the page to the dispatcher
// This is used to assemble dispatcher commands when data comes out of band, eg, reading from DRAM
// That means this command is stateful, incorrect use will be...bad
// This routine assumes we're sending a command header and that is less than a page
static uint32_t process_relay_inline_noflush_cmd(uint32_t cmd_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t length = sizeof(CQDispatchCmd);
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    // Assume the dispatch buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    dispatch_cb_acquire_pages(1);
    if (dispatch_data_ptr == dispatch_cb_end) {
        dispatch_data_ptr = dispatch_cb_base;
    }
    noc_async_write(data_ptr, get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr), length);
    dispatch_data_ptr += length;

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
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
template<bool is_dram>
uint32_t process_relay_paged_cmd(uint32_t cmd_ptr) {

    // This ensures that a previous cmd using the scratch buf has finished
    noc_async_writes_flushed();

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t page_id = cmd->relay_paged.start_page;
    uint32_t base_addr = cmd->relay_paged.base_addr;
    uint32_t page_size = cmd->relay_paged.page_size;
    uint32_t pages = cmd->relay_paged.pages;
    uint32_t read_length = pages * page_size;
    uint32_t write_length = pages * page_size;

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
        uint32_t page_residual_space = dispatch_cb_page_size - (dispatch_data_ptr & (dispatch_cb_page_size - 1));
        uint32_t npages = (amt_to_write - page_residual_space + dispatch_cb_page_size - 1) / dispatch_cb_page_size;
        // Grabbing all pages at once is ok if scratch_size < 3 * dispatch_cb_block_size
        dispatch_cb_acquire_pages(npages);
        uint64_t noc_addr = get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr);

        if (dispatch_data_ptr + amt_to_write > dispatch_cb_end) {  // wrap
            uint32_t last_chunk_size = dispatch_cb_end - dispatch_data_ptr;
            noc_async_write(scratch_write_addr, noc_addr, last_chunk_size);
            dispatch_data_ptr = dispatch_cb_base;
            scratch_write_addr += last_chunk_size;
            write_length -= last_chunk_size;
            amt_to_write -= last_chunk_size;
            noc_addr = get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr);
        }

        noc_async_write(scratch_write_addr, noc_addr, amt_to_write);
        dispatch_data_ptr += amt_to_write;

        dispatch_cb_release_pages(npages);

        read_length -= amt_read;
        write_length -= amt_to_write;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    scratch_write_addr = scratch_db_top[db_toggle];
    int32_t amt_to_write = amt_read;
    uint32_t page_residual_space = dispatch_cb_page_size - (dispatch_data_ptr & (dispatch_cb_page_size - 1));
    uint32_t npages = (amt_to_write - page_residual_space + dispatch_cb_page_size + CQ_DISPATCH_CMD_SIZE - 1) / dispatch_cb_page_size;
    if (npages > 0) {
        dispatch_cb_acquire_pages(npages);
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

    uint32_t pad_to_page = dispatch_cb_page_size - (dispatch_data_ptr & (dispatch_cb_page_size - 1));
    dispatch_data_ptr += pad_to_page;

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH
    dispatch_cb_release_pages(npages + 1);

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

void kernel_main() {

    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;

    DPRINT << "prefetcher" << ENDL();

    bool done = false;
    while (!done) {
        get_cmds(fence, cmd_ptr);

        volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

        switch (cmd->base.cmd_id) {
        case CQ_PREFETCH_CMD_RELAY_PAGED:
            DPRINT << "relay dram page: " << fence << " " << cmd_ptr << ENDL();
            if (cmd->relay_paged.is_dram) {
                cmd_ptr = process_relay_paged_cmd<true>(cmd_ptr);
            } else {
                cmd_ptr = process_relay_paged_cmd<false>(cmd_ptr);
            }
            break;

        case CQ_PREFETCH_CMD_RELAY_INLINE:
            DPRINT << "inline" << ENDL();
            cmd_ptr = process_relay_inline_cmd(cmd_ptr);
            break;

        case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
            DPRINT << "inline no flush" << ENDL();
            cmd_ptr = process_relay_inline_noflush_cmd(cmd_ptr);
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
            DPRINT << "prefetcher invalid command:" << (uint32_t)cmd->base.cmd_id << " " << cmd_ptr << " " << fence << " " << prefetch_q_end << ENDL();
            DPRINT << HEX() << *(uint32_t*)cmd_ptr << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr+1) << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr+2) << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr+3) << ENDL();
            DPRINT << HEX() << *((uint32_t*)cmd_ptr+4) << ENDL();
            DEBUG_STATUS('!', 'C', 'M', 'D');
            while(1);
        }
    }

    DPRINT << "prefetch out\n" << ENDL();
}
