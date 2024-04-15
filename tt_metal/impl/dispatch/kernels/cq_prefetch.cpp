// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefetch kernel
//  - 3 flavors: _hd (host and dram), _h (host only), _d (DRAM only)
//  - fetches commands from host (if applicable), executes
//  - uses HostQ for host handshaking, ComDatQ for commands (from host),
//    double buffered ScratchBuf for out of band data (e.g., from DRAM)
//  - syncs w/ dispatcher via 2 semaphores, page_ready, page_done

#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/impl/dispatch/dispatch_address_map.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"
#include "debug/dprint.h"

constexpr uint32_t downstream_cb_base = get_compile_time_arg_val(0);
constexpr uint32_t downstream_cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t downstream_cb_pages = get_compile_time_arg_val(2);
constexpr uint32_t my_downstream_cb_sem_id = get_compile_time_arg_val(3);
constexpr uint32_t downstream_cb_sem_id = get_compile_time_arg_val(4);

// unused for prefetch_d
constexpr uint32_t pcie_base = get_compile_time_arg_val(5);
constexpr uint32_t pcie_size = get_compile_time_arg_val(6);
constexpr uint32_t prefetch_q_base = get_compile_time_arg_val(7);
constexpr uint32_t prefetch_q_size = get_compile_time_arg_val(8);
constexpr uint32_t prefetch_q_rd_ptr_addr = get_compile_time_arg_val(9);

constexpr uint32_t cmddat_q_base = get_compile_time_arg_val(10);
constexpr uint32_t cmddat_q_size = get_compile_time_arg_val(11);

// unused for prefetch_h
constexpr uint32_t scratch_db_base = get_compile_time_arg_val(12);
constexpr uint32_t scratch_db_size = get_compile_time_arg_val(13);
constexpr uint32_t downstream_sync_sem_id = get_compile_time_arg_val(14);

// prefetch_d specific
constexpr uint32_t cmddat_q_pages = get_compile_time_arg_val(15);
constexpr uint32_t my_upstream_cb_sem_id = get_compile_time_arg_val(16);
constexpr uint32_t upstream_cb_sem_id = get_compile_time_arg_val(17);
constexpr uint32_t cmddat_q_log_page_size = get_compile_time_arg_val(18);
constexpr uint32_t cmddat_q_blocks = get_compile_time_arg_val(19);

constexpr uint32_t dispatch_h_exec_buf_sem_id = get_compile_time_arg_val(20);

constexpr uint32_t is_d_variant = get_compile_time_arg_val(21);
constexpr uint32_t is_h_variant = get_compile_time_arg_val(22);

constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t downstream_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t downstream_cb_page_size = 1 << downstream_cb_log_page_size;
constexpr uint32_t downstream_cb_end = downstream_cb_base + (1 << downstream_cb_log_page_size) * downstream_cb_pages;
constexpr uint32_t prefetch_q_end = prefetch_q_base + prefetch_q_size;
constexpr uint32_t cmddat_q_page_size = 1 << cmddat_q_log_page_size;
constexpr uint32_t cmddat_q_end = cmddat_q_base + cmddat_q_size;

constexpr uint32_t scratch_db_half_size = scratch_db_size / 2;
constexpr uint32_t scratch_db_base0 = scratch_db_base;
constexpr uint32_t scratch_db_base1 = scratch_db_base + scratch_db_half_size;

static uint32_t pcie_read_ptr = pcie_base;
static uint32_t downstream_data_ptr = downstream_cb_base;

constexpr uint32_t prefetch_q_log_minsize = 4;

const uint32_t scratch_db_top[2] = {scratch_db_base0, scratch_db_base1};

constexpr uint32_t cmddat_q_pages_per_block = cmddat_q_pages / cmddat_q_blocks;

static uint32_t block_next_start_addr[cmddat_q_blocks];
static uint32_t block_noc_writes_to_clear[cmddat_q_blocks];
static uint32_t rd_block_idx;

static struct PrefetchExecBufState {
    uint32_t page_id;
    uint32_t base_addr;
    uint32_t log_page_size;
    uint32_t pages;
    uint32_t length;
} exec_buf_state;

static_assert((downstream_cb_base & (downstream_cb_page_size - 1)) == 0);

template<bool cmddat_wrap_enable,
         bool exec_buf>
bool process_cmd(uint32_t& cmd_ptr,
                 uint32_t& downstream_data_ptr,
                 uint32_t& stride);

FORCE_INLINE
void write_downstream(uint32_t& data_ptr,
                      uint32_t& downstream_data_ptr,
                      uint32_t length) {

    uint32_t remaining = downstream_cb_end - downstream_data_ptr;
    if (length > remaining) {
        if (remaining > 0) {
            noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), remaining);
            data_ptr += remaining;
            length -= remaining;
        }
        downstream_data_ptr = downstream_cb_base;
    }

    noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), length);
    downstream_data_ptr += length;
}

template<uint32_t preamble_size>
FORCE_INLINE
void read_from_pcie(volatile tt_l1_ptr uint32_t *& prefetch_q_rd_ptr,
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

    // Wrap pcie/hugepage
    if (pcie_read_ptr + size > pcie_base + pcie_size) {
        pcie_read_ptr = pcie_base;
    }

    uint64_t host_src_addr = get_noc_addr_helper(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y), pcie_read_ptr);
    DPRINT << "read_from_pcie: " << fence + preamble_size << " " << pcie_read_ptr << ENDL();
    noc_async_read(host_src_addr, fence + preamble_size, size);
    pending_read_size = size + preamble_size;
    pcie_read_ptr += size;

    *prefetch_q_rd_ptr = 0;

    // Tell host we read
    *(volatile tt_l1_ptr uint32_t *) prefetch_q_rd_ptr_addr = (uint32_t)prefetch_q_rd_ptr;

    prefetch_q_rd_ptr++;

    // Wrap prefetch_q
    if ((uint32_t)prefetch_q_rd_ptr == prefetch_q_end) {
        prefetch_q_rd_ptr = (volatile tt_l1_ptr uint32_t*)prefetch_q_base;
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
template<uint32_t preamble_size>
void fetch_q_get_cmds(uint32_t& fence, uint32_t& cmd_ptr, uint32_t& pcie_read_ptr) {

    static uint32_t pending_read_size = 0;
    static volatile tt_l1_ptr uint32_t* prefetch_q_rd_ptr = (volatile tt_l1_ptr uint32_t*)prefetch_q_base;

    DPRINT << "fetch_q_get_cmds: " << cmd_ptr << " " << fence << ENDL();
    if (fence < cmd_ptr) {
        DPRINT << "fetch_q_get_cmds wrap cmd" << ENDL();
        cmd_ptr = fence;
    }

    bool cmd_ready = (cmd_ptr != fence);
    uint32_t fetch_size = (uint32_t)*prefetch_q_rd_ptr << prefetch_q_log_minsize;

    if (fetch_size != 0 && pending_read_size == 0) {
        read_from_pcie<preamble_size>
            (prefetch_q_rd_ptr, pending_read_size, fence, pcie_read_ptr, cmd_ptr, fetch_size);
    }
    if (!cmd_ready) {
        if (pending_read_size != 0) {
            DPRINT << "fetch_q_get_cmds barrier" << ENDL();
            noc_async_read_barrier();

            // wrap the cmddat_q
            if (fence < cmd_ptr) {
                cmd_ptr = fence;
            }

            fence += pending_read_size;
            pending_read_size = 0;

            // Ugly hack for now.  Snoops the command, don't fetch the next if we are doing an exec_buf
            volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
            if (cmd->base.cmd_id != CQ_PREFETCH_CMD_EXEC_BUF) {
                // After the stall, re-check the host
                fetch_size = (uint32_t)*prefetch_q_rd_ptr << prefetch_q_log_minsize;
                if (fetch_size != 0) {
                    read_from_pcie<preamble_size>
                        (prefetch_q_rd_ptr, pending_read_size, fence, pcie_read_ptr, cmd_ptr, fetch_size);
                }
            }
        } else {
            // By here, prefetch_q_ready must be false
            // Nothing to fetch, nothing pending, nothing available, stall on host
            DEBUG_STATUS('H', 'Q', 'W');
            DPRINT << "prefetcher stall" << ENDL();
            while ((fetch_size = *prefetch_q_rd_ptr) == 0);
            DPRINT << "recurse" << ENDL();
            fetch_q_get_cmds<preamble_size>(fence, cmd_ptr, pcie_read_ptr);
            DEBUG_STATUS('H', 'Q', 'D');
        }
    }
}

uint32_t process_debug_cmd(uint32_t cmd_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t checksum = 0;
    uint32_t data_start = (uint32_t)cmd + sizeof(CQPrefetchCmd);
    uint32_t *data = (uint32_t *)data_start;
    uint32_t size = cmd->debug.size;

    uint32_t front_size = (size <= cmddat_q_end - data_start) ? size : cmddat_q_end - data_start;
    for (uint32_t i = 0; i < front_size / sizeof(uint32_t); i++) {
        checksum += *data++;
    }
    uint32_t back_size = size - front_size;
    if (back_size > 0) {
        data = (uint32_t *)cmddat_q_base;
        for (uint32_t i = 0; i < back_size / sizeof(uint32_t); i++) {
            checksum += *data++;
        }
    }

    if (checksum != cmd->debug.checksum) {
        DEBUG_STATUS('!', 'C', 'H', 'K');
        ASSERT(0);
    }

    return cmd->debug.stride;
}

template<bool cmddat_wrap_enable>
static uint32_t process_relay_inline_cmd(uint32_t cmd_ptr,
                                         uint32_t& downstream_data_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t length = cmd->relay_inline.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    uint32_t npages = (length + downstream_cb_page_size - 1) >> downstream_cb_log_page_size;

    // Assume the downstream buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(npages);

    uint32_t remaining = cmddat_q_end - data_ptr;
    if (cmddat_wrap_enable && length > remaining) {
        // wrap cmddat
        write_downstream(data_ptr, downstream_data_ptr, remaining);
        length -= remaining;
        data_ptr = cmddat_q_base;
    }

    write_downstream(data_ptr, downstream_data_ptr, length);

    // Round to nearest page
    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);

    // XXXXX - painful syncing right now?  move this into get_cmds
    noc_async_writes_flushed();
    cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(npages);

    return cmd->relay_inline.stride;
}

// This version of inline sends inline data to the dispatcher but doesn't flush the page to the dispatcher
// This is used to assemble dispatcher commands when data comes out of band, eg, reading from DRAM
// That means this command is stateful, incorrect use will be...bad
// NOTE: this routine assumes we're sending a command header and that is LESS THAN A PAGE
static uint32_t process_relay_inline_noflush_cmd(uint32_t cmd_ptr,
                                                 uint32_t& dispatch_data_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t length = sizeof(CQDispatchCmd);
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(1);
    if (dispatch_data_ptr == downstream_cb_end) {
        dispatch_data_ptr = downstream_cb_base;
    }
    noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, dispatch_data_ptr), length);
    dispatch_data_ptr += length;

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

template<uint32_t extra_space,
         bool test_for_nonzero>
static uint32_t write_pages_to_dispatcher(uint32_t& downstream_data_ptr,
                                          uint32_t& scratch_write_addr,
                                          uint32_t& amt_to_write) {

    uint32_t page_residual_space = downstream_cb_page_size - (downstream_data_ptr & (downstream_cb_page_size - 1));
    uint32_t npages = (amt_to_write - page_residual_space + downstream_cb_page_size + extra_space - 1) / downstream_cb_page_size;

    // Grabbing all pages at once is ok if scratch_size < 3 * downstream_cb_block_size
    if (!test_for_nonzero || npages != 0) {
        cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(npages);
    }

    uint64_t noc_addr = get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr);
    if (downstream_data_ptr == downstream_cb_end) {
        downstream_data_ptr = downstream_cb_base;
    } else if (downstream_data_ptr + amt_to_write > downstream_cb_end) {  // wrap
        uint32_t last_chunk_size = downstream_cb_end - downstream_data_ptr;
        noc_async_write(scratch_write_addr, noc_addr, last_chunk_size);
        downstream_data_ptr = downstream_cb_base;
        scratch_write_addr += last_chunk_size;
        amt_to_write -= last_chunk_size;
        noc_addr = get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr);
    }

    noc_async_write(scratch_write_addr, noc_addr, amt_to_write);
    downstream_data_ptr += amt_to_write;

    return npages;
}

// This isn't the right way to handle large pages, but expedient for now
// In the future, break them down into smaller pages...
template<bool is_dram>
uint32_t process_relay_paged_cmd_large(uint32_t cmd_ptr,
                                       uint32_t& downstream__data_ptr,
                                       uint32_t page_id,
                                       uint32_t base_addr,
                                       uint32_t page_size,
                                       uint32_t pages,
                                       uint32_t length_adjust) {

#if ENABLE_PREFETCH_DPRINTS
    DPRINT << "relay_paged_cmd_large: " << page_size << " " << pages << " " << length_adjust << ENDL();
#endif

    InterleavedAddrGen<is_dram> addr_gen;
    addr_gen.bank_base_address = base_addr;
    addr_gen.page_size = page_size;

    // First step - read into DB0
    uint32_t scratch_read_addr = scratch_db_top[0];
    uint64_t noc_addr = addr_gen.get_noc_addr(page_id);
    noc_async_read(noc_addr, scratch_read_addr, scratch_db_half_size);
    uint32_t amt_read = scratch_db_half_size;
    uint32_t page_length = page_size - amt_read;
    uint32_t page_offset = amt_read;

    // Second step - read into DB[x], write from DB[x], toggle x, iterate
    // Writes are fast, reads are slow
    uint32_t db_toggle = 0;
    uint32_t scratch_write_addr;
    uint32_t read_length = pages * page_size - amt_read;
    uint32_t write_length = pages * page_size - length_adjust;

    noc_async_read_barrier();
    while (read_length != 0) {
        // This ensures that writes from prior iteration are done
        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_writes_flushed();

        db_toggle ^= 1;
        scratch_read_addr = scratch_db_top[db_toggle];
        scratch_write_addr = scratch_db_top[db_toggle ^ 1];

        uint32_t amt_to_write = amt_read;
        uint64_t noc_addr = addr_gen.get_noc_addr(page_id, page_offset);
        if (page_length <= scratch_db_half_size) {
            noc_async_read(noc_addr, scratch_read_addr, page_length);
            page_id++;
            page_offset = 0;
            amt_read = page_length;
            page_length = page_size;

            if (amt_read < scratch_db_half_size &&
                read_length > amt_read) {
                noc_addr = addr_gen.get_noc_addr(page_id, 0);
                uint32_t amt_to_read = scratch_db_half_size - amt_read;
                noc_async_read(noc_addr, scratch_read_addr + amt_read, amt_to_read);
                page_length -= amt_to_read;
                amt_read = scratch_db_half_size;
                page_offset = amt_to_read;
            }
        } else {
            noc_async_read(noc_addr, scratch_read_addr, scratch_db_half_size);
            page_length -= scratch_db_half_size;
            page_offset += scratch_db_half_size;
            amt_read = scratch_db_half_size;
        }

        // Third step - write from DB
        if (write_length < amt_to_write) {
            amt_to_write = write_length;
        }

        write_length -= amt_to_write;
        uint32_t npages = write_pages_to_dispatcher<0, false>
            (downstream_data_ptr, scratch_write_addr, amt_to_write);
        cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(npages);

        read_length -= amt_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    if (write_length > 0) {
        scratch_write_addr = scratch_db_top[db_toggle];
        uint32_t amt_to_write = write_length;
        ASSERT((amt_to_write & 0x1f) == 0);

        uint32_t npages = write_pages_to_dispatcher<CQ_DISPATCH_CMD_SIZE, true>
            (downstream_data_ptr, scratch_write_addr, amt_to_write);

        // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH with 16 bytes written
        cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(npages + 1);
    } else {
        cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(1);
    }

    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
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
uint32_t process_relay_paged_cmd(uint32_t cmd_ptr,
                                 uint32_t& downstream__data_ptr,
                                 uint32_t page_id) {

    // This ensures that a previous cmd using the scratch buf has finished
    noc_async_writes_flushed();

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    uint32_t base_addr = cmd->relay_paged.base_addr;
    uint32_t page_size = cmd->relay_paged.page_size;
    uint32_t pages = cmd->relay_paged.pages;

    if (page_size > scratch_db_half_size) {
        return process_relay_paged_cmd_large<is_dram>(cmd_ptr, downstream_data_ptr, page_id, base_addr, page_size, pages, cmd->relay_paged.length_adjust);
    }

    InterleavedAddrGen<is_dram> addr_gen;
    addr_gen.bank_base_address = base_addr;
    addr_gen.page_size = page_size;

    // First step - read into DB0
    uint32_t read_length = pages * page_size;
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
        uint32_t npages = write_pages_to_dispatcher<0, false>
            (downstream_data_ptr, scratch_write_addr, amt_to_write);
        cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(npages);

        read_length -= amt_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    // Note that we may write less than full pages despite reading full pages based on length_adjust
    // Expectation is that the gain from reading less is small to 0, revisit as needed
    ASSERT(cmd->relay_paged.length_adjust < page_size);
    scratch_write_addr = scratch_db_top[db_toggle];
    uint32_t amt_to_write = amt_read - cmd->relay_paged.length_adjust;
    ASSERT((amt_to_write & 0x1f) == 0);
    uint32_t npages = write_pages_to_dispatcher<CQ_DISPATCH_CMD_SIZE, true>
        (downstream_data_ptr, scratch_write_addr, amt_to_write);

    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH with 16 bytes written
    cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(npages + 1);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_relay_linear_cmd(uint32_t cmd_ptr,
                                  uint32_t& downstream_data_ptr) {

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
        uint32_t npages = write_pages_to_dispatcher<0, false>(downstream_data_ptr, scratch_write_addr, amt_to_write);

        cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(npages);

        read_length -= amt_to_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    scratch_write_addr = scratch_db_top[db_toggle];
    uint32_t amt_to_write = amt_to_read;
    uint32_t npages = write_pages_to_dispatcher<CQ_DISPATCH_CMD_SIZE, true>
        (downstream_data_ptr, scratch_write_addr, amt_to_write);

    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH
    cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(npages + 1);

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_stall(uint32_t cmd_ptr) {

    static uint32_t count = 0;

    count++;

    DEBUG_STATUS('P', 'S', 'W');
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(downstream_sync_sem_id));
    while (*sem_addr != count);
    DEBUG_STATUS('P', 'S', 'D');

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

void paged_read_into_cmddat_q(uint32_t read_ptr) {

    uint32_t page_id = exec_buf_state.page_id;
    uint32_t base_addr = exec_buf_state.base_addr;
    uint32_t log_page_size = exec_buf_state.log_page_size;
    uint32_t page_size = 1 << log_page_size;
    uint32_t pages = exec_buf_state.pages;

    uint32_t pages_at_once = (pages > NUM_DRAM_BANKS) ? NUM_DRAM_BANKS : pages; // XXXX tune
    uint32_t read_length = pages_at_once << log_page_size;

    InterleavedAddrGen<true> addr_gen;
    addr_gen.bank_base_address = base_addr;
    addr_gen.page_size = page_size;

    while (pages_at_once != 0) {
        uint64_t noc_addr = addr_gen.get_noc_addr(page_id); // XXXX replace this w/ walking the banks to save mul on GS
        noc_async_read(noc_addr, read_ptr, page_size);
        read_ptr += page_size;
        page_id++;
        pages_at_once--;
        pages--;
    }

    exec_buf_state.page_id = page_id;
    exec_buf_state.pages = pages;
    exec_buf_state.length += read_length;
}

static uint32_t process_relay_inline_exec_buf_cmd(uint32_t& cmd_ptr,
                                                  uint32_t& downstream_data_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t length = cmd->relay_inline.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    uint32_t npages = (length + downstream_cb_page_size - 1) >> downstream_cb_log_page_size;

    // Assume the downstream buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(npages);

    uint32_t stride = cmd->relay_inline.stride;
    uint32_t remaining_stride = exec_buf_state.length;
    uint32_t remaining = exec_buf_state.length - sizeof(CQPrefetchCmd);
    while (length > remaining) {
        // wrap cmddat
        write_downstream(data_ptr, downstream_data_ptr, remaining);
        length -= remaining;
        stride -= remaining_stride;
        exec_buf_state.length = 0;
        data_ptr = cmddat_q_base;
        cmd_ptr = cmddat_q_base;

        // fetch more
        noc_async_writes_flushed(); // XXXXX no no no no
        paged_read_into_cmddat_q(cmd_ptr);
        noc_async_read_barrier(); // XXXXX no no no no
        remaining = exec_buf_state.length;
        remaining_stride = exec_buf_state.length;
    }

    write_downstream(data_ptr, downstream_data_ptr, length);

    // Round to nearest page
    downstream_data_ptr = round_up_pow2(downstream_data_ptr, downstream_cb_page_size);

    // XXXXX - painful syncing right now?  move this into get_cmds
    noc_async_writes_flushed();
    cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(npages);

    return stride;
}

uint32_t process_exec_buf_cmd(uint32_t cmd_ptr_outer,
                              uint32_t& downstream_data_ptr) {

    // dispatch on eth cores is memory constrained, so exec_buf re-uses the cmddat_q
    // prefetch_h stalls upon issuing an exec_buf to prevent conflicting use of the cmddat_q,
    // the exec_buf contains the release commands
    // this takes away all credits from prefetch_h so prefetch_h doesn't begin until this cmd is done
    if (!is_h_variant) {
        cb_release_pages<upstream_noc_xy, upstream_cb_sem_id>(-(cmddat_q_pages - 1));
    }

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr_outer;

    exec_buf_state.page_id = 0;
    exec_buf_state.base_addr = cmd->exec_buf.base_addr;
    exec_buf_state.log_page_size = cmd->exec_buf.log_page_size;
    exec_buf_state.pages = cmd->exec_buf.pages;
    exec_buf_state.length = 0;

    DPRINT << exec_buf_state.page_id << " " << exec_buf_state.base_addr << " " << " " << exec_buf_state.log_page_size << " " << exec_buf_state.pages << ENDL();

    bool done = false;
    while (!done) {
        uint32_t cmd_ptr = cmddat_q_base;

        paged_read_into_cmddat_q(cmd_ptr);
        noc_async_read_barrier(); // XXXXX no no no no

        while (exec_buf_state.length > 0) {
            uint32_t stride;
            done = process_cmd<false, true>(cmd_ptr, downstream_data_ptr, stride);

            if (done) {
                break;
            }

            exec_buf_state.length -= stride;
            cmd_ptr += stride;
        }
    }

    // release the pages acquired above to free up prefetch_h
    if (!is_h_variant) {
        cb_release_pages<upstream_noc_xy, upstream_cb_sem_id>(cmddat_q_pages - 1);
    }

    return CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

template<bool cmddat_wrap_enable,
         bool exec_buf>
bool process_cmd(uint32_t& cmd_ptr,
                 uint32_t& downstream_data_ptr,
                 uint32_t& stride) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;
    bool done = false;

    switch (cmd->base.cmd_id) {
    case CQ_PREFETCH_CMD_RELAY_LINEAR:
        DPRINT << "relay linear: " << cmd_ptr << ENDL();
        stride = process_relay_linear_cmd(cmd_ptr, downstream_data_ptr);
        break;

    case CQ_PREFETCH_CMD_RELAY_PAGED:
        DPRINT << "relay dram page: " << cmd_ptr << ENDL();
        {
            uint32_t packed_page_flags = cmd->relay_paged.packed_page_flags;
            uint32_t is_dram = packed_page_flags & (1 << CQ_PREFETCH_RELAY_PAGED_IS_DRAM_SHIFT);
            uint32_t start_page =
                (packed_page_flags >> CQ_PREFETCH_RELAY_PAGED_START_PAGE_SHIFT) &
                CQ_PREFETCH_RELAY_PAGED_START_PAGE_MASK;
            if (is_dram) {
                stride = process_relay_paged_cmd<true>(cmd_ptr, downstream_data_ptr, start_page);
            } else {
                stride = process_relay_paged_cmd<false>(cmd_ptr, downstream_data_ptr, start_page);
            }
        }
        break;

    case CQ_PREFETCH_CMD_RELAY_INLINE:
        DPRINT << "relay inline" << ENDL();
        if (exec_buf) {
            stride = process_relay_inline_exec_buf_cmd(cmd_ptr, downstream_data_ptr);
        } else {
            stride = process_relay_inline_cmd<cmddat_wrap_enable>(cmd_ptr, downstream_data_ptr);
        }
        break;

    case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
        DPRINT << "inline no flush" << ENDL();
        stride = process_relay_inline_noflush_cmd(cmd_ptr, downstream_data_ptr);
        break;

    case CQ_PREFETCH_CMD_EXEC_BUF:
        DPRINT << "exec buf: " << cmd_ptr << ENDL();
        ASSERT(!exec_buf);
        stride = process_exec_buf_cmd(cmd_ptr, downstream_data_ptr);
        break;

    case CQ_PREFETCH_CMD_EXEC_BUF_END:
        DPRINT << "exec buf end: " << cmd_ptr << ENDL();
        ASSERT(exec_buf);
        done = true;
        break;

    case CQ_PREFETCH_CMD_STALL:
        DPRINT << "stall" << ENDL();
        stride = process_stall(cmd_ptr);
        break;

    case CQ_PREFETCH_CMD_DEBUG:
        DPRINT << "debug" << ENDL();
        // Splitting debug cmds not implemented for exec_bufs (yet)
        if (exec_buf) {
            ASSERT(0);
        }
        stride = process_debug_cmd(cmd_ptr);
        break;

    case CQ_PREFETCH_CMD_TERMINATE:
        DPRINT << "terminating\n";
        ASSERT(!exec_buf);
        done = true;
        break;

    default:
        DPRINT << "prefetch invalid command:" << (uint32_t)cmd->base.cmd_id << " " << cmd_ptr << " " << cmddat_q_base << ENDL();
        DPRINT << HEX() << *(uint32_t*)cmd_ptr << ENDL();
        DPRINT << HEX() << *((uint32_t*)cmd_ptr+1) << ENDL();
        DPRINT << HEX() << *((uint32_t*)cmd_ptr+2) << ENDL();
        DPRINT << HEX() << *((uint32_t*)cmd_ptr+3) << ENDL();
        DPRINT << HEX() << *((uint32_t*)cmd_ptr+4) << ENDL();
        DEBUG_STATUS('!', 'C', 'M', 'D');
        ASSERT(0);
    }

    return done;
}

static uint32_t process_relay_inline_all(uint32_t data_ptr, uint32_t fence) {

    uint32_t length = fence - data_ptr;

    // Downstream doesn't have FetchQ to tell it how much data to process
    // This packet header just contains the length
    volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader *dptr =
        (volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader *)data_ptr;
    dptr->length = length;

    uint32_t npages = (length + downstream_cb_page_size - 1) >> downstream_cb_log_page_size;

    // Assume the dispatch buffer is big relative to cmddat command size that we can
    // grab what we need in one chunk
    cb_acquire_pages<my_noc_xy, my_downstream_cb_sem_id>(npages);
    uint32_t downstream_pages_left = (downstream_cb_end - downstream_data_ptr) >> downstream_cb_log_page_size;
    if (downstream_pages_left >= npages) {
        noc_async_write(data_ptr, get_noc_addr_helper(downstream_noc_xy, downstream_data_ptr), length);
        downstream_data_ptr += npages * downstream_cb_page_size;
    } else {
        uint32_t tail_pages = npages - downstream_pages_left;
        uint32_t available = downstream_pages_left * downstream_cb_page_size;
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
    cb_release_pages<downstream_noc_xy, downstream_cb_sem_id>(npages);

    return fence;
}

// Gets cmds from upstream prefetch_h
// Note the prefetch_h uses the HostQ and grabs whole commands
// Shared command processor assumes whole commands are present, really
// just matters for the inline command which could be re-implemented
// This grabs whole (possibly sets of if multiple in a page) commands
inline uint32_t relay_cb_get_cmds(uint32_t& fence, uint32_t& data_ptr) {

    DPRINT << "get_commands: " << data_ptr << " " << fence << " " << cmddat_q_base << " " << cmddat_q_end << ENDL();
    if (data_ptr == fence) {
        get_cb_page<
            cmddat_q_base,
            cmddat_q_blocks,
            cmddat_q_log_page_size,
            my_noc_xy,
            my_upstream_cb_sem_id>(data_ptr,
                                   fence,
                                   block_noc_writes_to_clear,
                                   block_next_start_addr,
                                   rd_block_idx);
    }

    volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader *cmd_ptr =
        (volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader *)data_ptr;
    uint32_t length = cmd_ptr->length;

    uint32_t pages_ready = (fence - data_ptr) >> cmddat_q_log_page_size;
    uint32_t pages_needed = (length + cmddat_q_page_size - 1) >> cmddat_q_log_page_size;
    int32_t pages_pending = pages_needed - pages_ready;
    int32_t npages = 0;

    // TODO
    // Ugly: get_cb_page was written to process 1 page at a time, we need multiple
    // If it wraps, it resets the data_ptr to the top of the buffer, hand it a dummy for now
    uint32_t dummy_data_ptr = data_ptr;
    while (npages < pages_pending) {
        npages += get_cb_page<
            cmddat_q_base,
            cmddat_q_blocks,
            cmddat_q_log_page_size,
            my_noc_xy,
            my_upstream_cb_sem_id>(dummy_data_ptr,
                                   fence,
                                   block_noc_writes_to_clear,
                                   block_next_start_addr,
                                   rd_block_idx);
    }

    data_ptr += sizeof(CQPrefetchHToPrefetchDHeader);
    DPRINT << "done get cmds\n";

    return length - sizeof(CQPrefetchHToPrefetchDHeader);
}

// prefetch_h stalls sending commands to prefetch_d until notified by dispatch_d that the exec_buf is done
// future optimization: this routine could pull from fetch_q while stalled. to do so we'd need to parse
// commands in kernel_main_h
void process_exec_buf_cmd_h() {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dispatch_h_exec_buf_sem_id));

    DEBUG_STATUS('E', 'B', 'C', 'W');
    while (*sem_addr == 0);
    DEBUG_STATUS('E', 'B', 'C', 'D');
    *sem_addr = 0;
}

void kernel_main_h() {

    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;

    bool done = false;
    while (!done) {
        fetch_q_get_cmds<sizeof(CQPrefetchHToPrefetchDHeader)>(fence, cmd_ptr, pcie_read_ptr);

        volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)(cmd_ptr + sizeof(CQPrefetchHToPrefetchDHeader));
        cmd_ptr = process_relay_inline_all(cmd_ptr, fence);

        // Note: one fetch_q entry can contain multiple commands
        // The code below assumes these commands arrive individually, packing them would require parsing all cmds
        if (cmd->base.cmd_id == CQ_PREFETCH_CMD_EXEC_BUF) {
            DPRINT << "exec buf\n";
            process_exec_buf_cmd_h();
        } else if (cmd->base.cmd_id == CQ_PREFETCH_CMD_TERMINATE) {
            DPRINT << "terminating\n";
            done = true;
        }
    }
}

void kernel_main_d() {

    for (uint32_t i = 0; i < cmddat_q_blocks; i++) {
        uint32_t next_block = i + 1;
        uint32_t offset = next_block * cmddat_q_pages_per_block * cmddat_q_page_size;
        block_next_start_addr[i] = cmddat_q_base + offset;
    }

    rd_block_idx = 0;
    block_noc_writes_to_clear[0] = noc_nonposted_writes_num_issued[noc_index] + 1;

    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;

    bool done = false;
    while (!done) {
        // cmds come in packed batches based on HostQ reads in prefetch_h
        // once a packed batch ends, we need to jump to the next page
        uint32_t length = relay_cb_get_cmds(fence, cmd_ptr);

        uint32_t amt_processed = 0;
        while (length > amt_processed) {
            uint32_t stride;
            done = process_cmd<true, false>(cmd_ptr, downstream_data_ptr, stride);
            amt_processed += stride;

            // This is ugly: relay_inline_cmd code can wrap and this can wrap
            // They peacefully coexist because we won't wrap there and here at once
            if (cmd_ptr + stride >= cmddat_q_end) {
                stride -= cmddat_q_end - cmd_ptr;
                cmd_ptr = cmddat_q_base;
            }
            cmd_ptr += stride;
        }

        // XXXXX should free in blocks...
        uint32_t total_length = length + sizeof(CQPrefetchHToPrefetchDHeader);
        uint32_t pages_to_free = (total_length + cmddat_q_page_size - 1) >> cmddat_q_log_page_size;
        cb_release_pages<upstream_noc_xy, upstream_cb_sem_id>(pages_to_free);

        // Move to next page
        cmd_ptr = round_up_pow2(cmd_ptr, cmddat_q_page_size);
    }

    // Set upstream semaphore MSB to signal completion and path teardown
    // in case prefetch_d is connected to a depacketizing stage.
    // This should be replaced with a signal similar to what packetized components
    // use.
    DPRINT << "prefetch_d done" << ENDL();
    noc_semaphore_inc(get_noc_addr_helper(upstream_noc_xy, get_semaphore(upstream_cb_sem_id)), 0x80000000);
}

void kernel_main_hd() {

    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;

    bool done = false;
    while (!done) {
        constexpr uint32_t preamble_size = 0;
        fetch_q_get_cmds<preamble_size>(fence, cmd_ptr, pcie_read_ptr);

        volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

        uint32_t stride;
        done = process_cmd<false, false>(cmd_ptr, downstream_data_ptr, stride);
        cmd_ptr += stride;
    }
}

void kernel_main() {
    DPRINT << "prefetcher_" << is_h_variant << is_d_variant << ": start" << ENDL();
    if (is_h_variant and is_d_variant) {
        kernel_main_hd();
    } else if (is_h_variant) {
        kernel_main_h();
    } else if (is_d_variant) {
        kernel_main_d();
    } else {
        ASSERT(0);
    }
    DPRINT << "prefetcher_" << is_h_variant << is_d_variant << ": out" << ENDL();
}
