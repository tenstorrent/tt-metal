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
constexpr uint32_t local_downstream_cb_sem_id = get_compile_time_arg_val(3);
constexpr uint32_t downstream_cb_sem_id = get_compile_time_arg_val(4);

constexpr uint32_t cmddat_cb_base = get_compile_time_arg_val(5);
constexpr uint32_t cmddat_cb_log_page_size = get_compile_time_arg_val(6);
constexpr uint32_t cmddat_cb_pages = get_compile_time_arg_val(7);
constexpr uint32_t local_upstream_cb_sem_id = get_compile_time_arg_val(8);
constexpr uint32_t upstream_cb_sem_id = get_compile_time_arg_val(9);
constexpr uint32_t cmddat_cb_blocks = get_compile_time_arg_val(10);

constexpr uint32_t scratch_db_base = get_compile_time_arg_val(11);
constexpr uint32_t scratch_db_size = get_compile_time_arg_val(12);

constexpr uint32_t dispatch_sync_sem_id = get_compile_time_arg_val(13);

constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t downstream_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t downstream_cb_page_size = 1 << downstream_cb_log_page_size;
constexpr uint32_t downstream_cb_end = downstream_cb_base + (1 << downstream_cb_log_page_size) * downstream_cb_pages;
constexpr uint32_t cmddat_cb_page_size = 1 << cmddat_cb_log_page_size;
constexpr uint32_t cmddat_cb_size = cmddat_cb_page_size * cmddat_cb_pages;
constexpr uint32_t cmddat_cb_end = cmddat_cb_base + cmddat_cb_size;

constexpr uint32_t scratch_db_half_size = scratch_db_size / 2;
constexpr uint32_t scratch_db_base0 = scratch_db_base;
constexpr uint32_t scratch_db_base1 = scratch_db_base + scratch_db_half_size;

static uint32_t downstream_data_ptr = downstream_cb_base;

const uint32_t scratch_db_top[2] = {scratch_db_base0, scratch_db_base1};

constexpr uint32_t cmddat_cb_pages_per_block = cmddat_cb_pages / cmddat_cb_blocks;

static uint32_t block_next_start_addr[cmddat_cb_blocks];
static uint32_t block_noc_writes_to_clear[cmddat_cb_blocks];
static uint32_t rd_block_idx;

static_assert((downstream_cb_base & (downstream_cb_page_size - 1)) == 0);


// Gets cmds from upstream prefetch_h
// Note the prefetch_h uses the HostQ and grabs whole commands
// Shared command processor assumes whole commands are present, really
// just matters for the inline command which could be re-implemented
// This grabs whole (possibly sets of if multiple in a page) commands
inline uint32_t relay_cb_get_cmds(uint32_t& fence, uint32_t& data_ptr) {

    DPRINT << "get_commands: " << data_ptr << " " << fence << " " << cmddat_cb_base << " " << cmddat_cb_end << ENDL();
    if (data_ptr == fence) {
        upstream_get_cb_page<
            cmddat_cb_base,
            cmddat_cb_blocks,
            cmddat_cb_log_page_size,
            my_noc_xy,
            local_upstream_cb_sem_id>(data_ptr,
                                      fence,
                                      block_noc_writes_to_clear,
                                      block_next_start_addr,
                                      rd_block_idx);
    }

    volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader *cmd_ptr =
        (volatile tt_l1_ptr CQPrefetchHToPrefetchDHeader *)data_ptr;
    uint32_t length = cmd_ptr->length;

    uint32_t pages_ready = (fence - data_ptr) >> cmddat_cb_log_page_size;
    uint32_t pages_needed = (length + cmddat_cb_page_size - 1) >> cmddat_cb_log_page_size;
    int32_t pages_pending = pages_needed - pages_ready;
    int32_t npages = 0;

    // TODO
    // Ugly: upstream_get_cb_page was written to process 1 page at a time, we need multiple
    // If it wraps, it resets the data_ptr to the top of the buffer, hand it a dummy for now
    uint32_t dummy_data_ptr = data_ptr;
    while (npages < pages_pending) {
        npages += upstream_get_cb_page<
            cmddat_cb_base,
            cmddat_cb_blocks,
            cmddat_cb_log_page_size,
            my_noc_xy,
            local_upstream_cb_sem_id>(dummy_data_ptr,
                                      fence,
                                      block_noc_writes_to_clear,
                                      block_next_start_addr,
                                      rd_block_idx);
    }

    data_ptr += sizeof(CQPrefetchHToPrefetchDHeader);

    return length - sizeof(CQPrefetchHToPrefetchDHeader);
}

void kernel_main() {

    for (uint32_t i = 0; i < cmddat_cb_blocks; i++) {
        uint32_t next_block = i + 1;
        uint32_t offset = next_block * cmddat_cb_pages_per_block * cmddat_cb_page_size;
        block_next_start_addr[i] = cmddat_cb_base + offset;
    }

    rd_block_idx = 0;
    block_noc_writes_to_clear[0] = noc_nonposted_writes_num_issued[noc_index] + 1;

    uint32_t cmd_ptr = cmddat_cb_base;
    uint32_t fence = cmddat_cb_base;

    DPRINT << "prefetch_d" << ENDL();

    bool done = false;
    while (!done) {
        // cmds come in packed batches based on HostQ reads in prefetch_h
        // once a packed batch ends, we need to jump to the next page
        uint32_t length = relay_cb_get_cmds(fence, cmd_ptr);

        uint32_t amt_processed = 0;
        while (length > amt_processed) {
            uint32_t stride;
            done = process_cmd<
                true,
                my_noc_xy,
                local_downstream_cb_sem_id,
                downstream_noc_xy,
                downstream_cb_sem_id,
                cmddat_cb_base,
                cmddat_cb_end,
                downstream_cb_base,
                downstream_cb_end,
                downstream_cb_log_page_size,
                downstream_cb_page_size,
                dispatch_sync_sem_id,
                scratch_db_half_size>(cmd_ptr, downstream_data_ptr, stride);

            amt_processed += stride;
            if (cmd_ptr + stride >= cmddat_cb_end) {
                stride -= cmddat_cb_end - cmd_ptr;
                cmd_ptr = cmddat_cb_base;
            }
            cmd_ptr += stride;
        }

        // XXXXX should free in blocks...
        uint32_t total_length = length + sizeof(CQPrefetchHToPrefetchDHeader);
        uint32_t pages_to_free = (total_length + cmddat_cb_page_size - 1) >> cmddat_cb_log_page_size;
        upstream_cb_release_pages<upstream_noc_xy, upstream_cb_sem_id>(pages_to_free);

        // Move to next page
        cmd_ptr += (cmddat_cb_page_size - (cmd_ptr & (cmddat_cb_page_size - 1))) & (cmddat_cb_page_size - 1);
    }

    DPRINT << "prefetch_d out\n" << ENDL();
}
