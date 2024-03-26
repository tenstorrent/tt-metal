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
constexpr uint32_t local_dispatch_cb_sem_id = get_compile_time_arg_val(3);
constexpr uint32_t dispatch_cb_sem_id = get_compile_time_arg_val(4);

constexpr uint32_t pcie_base = get_compile_time_arg_val(5);
constexpr uint32_t pcie_size = get_compile_time_arg_val(6);
constexpr uint32_t prefetch_q_base = get_compile_time_arg_val(7);
constexpr uint32_t prefetch_q_size = get_compile_time_arg_val(8);
constexpr uint32_t prefetch_q_rd_ptr_addr = get_compile_time_arg_val(9);

constexpr uint32_t cmddat_q_base = get_compile_time_arg_val(10);
constexpr uint32_t cmddat_q_size = get_compile_time_arg_val(11);

constexpr uint32_t scratch_db_base = get_compile_time_arg_val(12);
constexpr uint32_t scratch_db_size = get_compile_time_arg_val(13);

constexpr uint32_t dispatch_sync_sem_id = get_compile_time_arg_val(14);

constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint32_t dispatch_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
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

const uint32_t scratch_db_top[2] = {scratch_db_base0, scratch_db_base1};

static_assert((dispatch_cb_base & (dispatch_cb_page_size - 1)) == 0);


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

        uint32_t stride;
        done = process_cmd<
            false,
            my_noc_xy,
            local_dispatch_cb_sem_id,
            dispatch_noc_xy,
            dispatch_cb_sem_id,
            cmddat_q_base,
            cmddat_q_end,
            dispatch_cb_base,
            dispatch_cb_end,
            dispatch_cb_log_page_size,
            dispatch_cb_page_size,
            dispatch_sync_sem_id,
            scratch_db_half_size>(cmd_ptr, dispatch_data_ptr, stride);

        cmd_ptr += stride;
    }

    DPRINT << "prefetch out\n" << ENDL();
}
