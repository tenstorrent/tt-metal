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

constexpr uint32_t dispatch_cb_base = get_compile_time_arg_val(0);
constexpr uint32_t dispatch_cb_log_page_size = get_compile_time_arg_val(1);
constexpr uint32_t dispatch_cb_pages = get_compile_time_arg_val(2);
constexpr uint32_t dispatch_cb_sem = get_compile_time_arg_val(3);
constexpr uint32_t pcie_base = get_compile_time_arg_val(4);
constexpr uint32_t pcie_size = get_compile_time_arg_val(5);
constexpr uint32_t prefetch_q_base = get_compile_time_arg_val(6);
constexpr uint32_t prefetch_q_size = get_compile_time_arg_val(7);
constexpr uint32_t prefetch_q_rd_ptr_addr = get_compile_time_arg_val(8);
constexpr uint32_t cmddat_q_base = get_compile_time_arg_val(9);
constexpr uint32_t cmddat_q_size = get_compile_time_arg_val(10);
constexpr uint32_t scratch_db_base = get_compile_time_arg_val(11);
constexpr uint32_t scratch_db_size = get_compile_time_arg_val(12);
constexpr uint32_t dispatch_sync_sem = get_compile_time_arg_val(13);

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

constexpr uint32_t prefetch_q_log_minsize = 4;

static const uint32_t scratch_db_top[2] = {scratch_db_base0, scratch_db_base1};

static_assert((dispatch_cb_base & (dispatch_cb_page_size - 1)) == 0);


static uint32_t process_debug_cmd(uint32_t cmd_ptr) {

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

static uint32_t process_relay_inline_cmd(uint32_t cmd_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t length = cmd->relay_inline.length;
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    write_downstream<prefetch_noc_xy,
                     dispatch_cb_sem, // XXXX fixme, same by coincidence
                     dispatch_noc_xy,
                     dispatch_cb_sem,
                     dispatch_cb_base,
                     dispatch_cb_end,
                     dispatch_cb_log_page_size,
                     dispatch_cb_page_size>(data_ptr, dispatch_data_ptr, length);

    return cmd_ptr + cmd->relay_inline.stride;
}

// This version of inline sends inline data to the dispatcher but doesn't flush the page to the dispatcher
// This is used to assemble dispatcher commands when data comes out of band, eg, reading from DRAM
// That means this command is stateful, incorrect use will be...bad
// NOTE: this routine assumes we're sending a command header and that is LESS THAN A PAGE
static uint32_t process_relay_inline_noflush_cmd(uint32_t cmd_ptr) {

    volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

    uint32_t length = sizeof(CQDispatchCmd);
    uint32_t data_ptr = cmd_ptr + sizeof(CQPrefetchCmd);

    downstream_cb_acquire_pages<prefetch_noc_xy, dispatch_cb_sem>(1);
    if (dispatch_data_ptr == dispatch_cb_end) {
        dispatch_data_ptr = dispatch_cb_base;
    }
    noc_async_write(data_ptr, get_noc_addr_helper(dispatch_noc_xy, dispatch_data_ptr), length);
    dispatch_data_ptr += length;

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

template<uint32_t extra_space, bool test_for_nonzero>
static uint32_t write_pages_to_dispatcher(uint32_t dispatch_noc_xy,
                                          uint32_t& dispatch_data_ptr,
                                          uint32_t& scratch_write_addr,
                                          uint32_t& amt_to_write) {

    uint32_t page_residual_space = dispatch_cb_page_size - (dispatch_data_ptr & (dispatch_cb_page_size - 1));
    uint32_t npages = (amt_to_write - page_residual_space + dispatch_cb_page_size + extra_space - 1) / dispatch_cb_page_size;

    // Grabbing all pages at once is ok if scratch_size < 3 * dispatch_cb_block_size
    if (!test_for_nonzero || npages != 0) {
        downstream_cb_acquire_pages<prefetch_noc_xy, dispatch_cb_sem>(npages);
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
        uint32_t npages = write_pages_to_dispatcher<0, false>(dispatch_noc_xy, dispatch_data_ptr, scratch_write_addr, amt_to_write);
        downstream_cb_release_pages<dispatch_noc_xy, dispatch_cb_sem>(npages);

        read_length -= amt_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    scratch_write_addr = scratch_db_top[db_toggle];
    uint32_t amt_to_write = amt_read;
    uint32_t npages = write_pages_to_dispatcher<CQ_DISPATCH_CMD_SIZE, true>(dispatch_noc_xy, dispatch_data_ptr, scratch_write_addr, amt_to_write);

    uint32_t pad_to_page = dispatch_cb_page_size - (dispatch_data_ptr & (dispatch_cb_page_size - 1));
    dispatch_data_ptr += pad_to_page;

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH
    downstream_cb_release_pages<dispatch_noc_xy, dispatch_cb_sem>(npages + 1);

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_relay_linear_cmd(uint32_t cmd_ptr) {

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
        uint32_t npages = write_pages_to_dispatcher<0, false>(dispatch_noc_xy, dispatch_data_ptr, scratch_write_addr, amt_to_write);

        downstream_cb_release_pages<dispatch_noc_xy, dispatch_cb_sem>(npages);

        read_length -= amt_to_read;

        // TODO(pgk); we can do better on WH w/ tagging
        noc_async_read_barrier();
    }

    // Third step - write from DB
    scratch_write_addr = scratch_db_top[db_toggle];
    uint32_t amt_to_write = amt_to_read;
    uint32_t npages = write_pages_to_dispatcher<CQ_DISPATCH_CMD_SIZE, true>(dispatch_noc_xy, dispatch_data_ptr, scratch_write_addr, amt_to_write);

    uint32_t pad_to_page = dispatch_cb_page_size - (dispatch_data_ptr & (dispatch_cb_page_size - 1));
    dispatch_data_ptr += pad_to_page;

    // One page was acquired w/ the cmd in CMD_RELAY_INLINE_NOFLUSH
    downstream_cb_release_pages<dispatch_noc_xy, dispatch_cb_sem>(npages + 1);

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

uint32_t process_stall(uint32_t cmd_ptr) {

    static uint32_t count = 0;

    count++;

    DEBUG_STATUS('P', 'S', 'W');
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(dispatch_sync_sem));
    while (*sem_addr != count);
    DEBUG_STATUS('P', 'S', 'D');

    return cmd_ptr + CQ_PREFETCH_CMD_BARE_MIN_SIZE;
}

void kernel_main() {

    uint32_t cmd_ptr = cmddat_q_base;
    uint32_t fence = cmddat_q_base;

    DPRINT << "prefetcher" << ENDL();

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
                         0>(fence, cmd_ptr, pcie_read_ptr);

        volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

        switch (cmd->base.cmd_id) {
        case CQ_PREFETCH_CMD_RELAY_LINEAR:
            DPRINT << "relay linear: " << fence << " " << cmd_ptr << ENDL();
            cmd_ptr = process_relay_linear_cmd(cmd_ptr);
            break;

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
            cmd_ptr = process_stall(cmd_ptr);
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
            ASSERT(0);
        }
    }

    DPRINT << "prefetch out\n" << ENDL();
}
