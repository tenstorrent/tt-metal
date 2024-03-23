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
constexpr uint32_t cmddat_cb_size = get_compile_time_arg_val(7);
constexpr uint32_t local_upstream_cb_sem_id = get_compile_time_arg_val(8);
constexpr uint32_t upstream_cb_sem_id = get_compile_time_arg_val(9);

constexpr uint32_t scratch_db_base = get_compile_time_arg_val(10);
constexpr uint32_t scratch_db_size = get_compile_time_arg_val(11);

constexpr uint32_t dispatch_sync_sem_id = get_compile_time_arg_val(11);

constexpr uint32_t my_noc_xy = uint32_t(NOC_XY_ENCODING(MY_NOC_X, MY_NOC_Y));
constexpr uint32_t upstream_noc_xy = uint32_t(NOC_XY_ENCODING(UPSTREAM_NOC_X, UPSTREAM_NOC_Y));
constexpr uint32_t downstream_noc_xy = uint32_t(NOC_XY_ENCODING(DOWNSTREAM_NOC_X, DOWNSTREAM_NOC_Y));
constexpr uint32_t downstream_cb_page_size = 1 << downstream_cb_log_page_size;
constexpr uint32_t downstream_cb_end = downstream_cb_base + (1 << downstream_cb_log_page_size) * downstream_cb_pages;
constexpr uint32_t cmddat_cb_end = cmddat_cb_base + cmddat_cb_size;

constexpr uint32_t scratch_db_half_size = scratch_db_size / 2;
constexpr uint32_t scratch_db_base0 = scratch_db_base;
constexpr uint32_t scratch_db_base1 = scratch_db_base + scratch_db_half_size;

static uint32_t downstream_data_ptr = downstream_cb_base;

const uint32_t scratch_db_top[2] = {scratch_db_base0, scratch_db_base1};

static_assert((downstream_cb_base & (downstream_cb_page_size - 1)) == 0);


void kernel_main() {

    uint32_t cmd_ptr = cmddat_cb_base;
    uint32_t fence = cmddat_cb_base;

    DPRINT << "prefetcher" << ENDL();

    bool done = false;
    while (!done) {
        // Hang...need to write this code!!!
        while(1);

        volatile CQPrefetchCmd tt_l1_ptr *cmd = (volatile CQPrefetchCmd tt_l1_ptr *)cmd_ptr;

        switch (cmd->base.cmd_id) {
        case CQ_PREFETCH_CMD_RELAY_LINEAR:
            DPRINT << "relay linear: " << fence << " " << cmd_ptr << ENDL();
            cmd_ptr = process_relay_linear_cmd<
                my_noc_xy,
                local_downstream_cb_sem_id,
                downstream_noc_xy,
                downstream_cb_sem_id,
                downstream_cb_base,
                downstream_cb_end,
                downstream_cb_page_size,
                scratch_db_half_size>(cmd_ptr, downstream_data_ptr);
            break;

        case CQ_PREFETCH_CMD_RELAY_PAGED:
            DPRINT << "relay dram page: " << fence << " " << cmd_ptr << ENDL();
            if (cmd->relay_paged.is_dram) {
                cmd_ptr = process_relay_paged_cmd<
                    true,
                    my_noc_xy,
                    local_downstream_cb_sem_id,
                    downstream_noc_xy,
                    downstream_cb_sem_id,
                    downstream_cb_base,
                    downstream_cb_end,
                    downstream_cb_page_size,
                    scratch_db_half_size>(cmd_ptr, downstream_data_ptr);
            } else {
                cmd_ptr = process_relay_paged_cmd<
                    false,
                    my_noc_xy,
                    local_downstream_cb_sem_id,
                    downstream_noc_xy,
                    downstream_cb_sem_id,
                    downstream_cb_base,
                    downstream_cb_end,
                    downstream_cb_page_size,
                    scratch_db_half_size>(cmd_ptr, downstream_data_ptr);
            }
            break;

        case CQ_PREFETCH_CMD_RELAY_INLINE:
            DPRINT << "inline" << ENDL();
            cmd_ptr = process_relay_inline_cmd<
                my_noc_xy,
                local_downstream_cb_sem_id,
                downstream_noc_xy,
                downstream_cb_sem_id,
                downstream_cb_base,
                downstream_cb_end,
                downstream_cb_log_page_size,
                downstream_cb_page_size>(cmd_ptr, downstream_data_ptr);
            break;

        case CQ_PREFETCH_CMD_RELAY_INLINE_NOFLUSH:
            DPRINT << "inline no flush" << ENDL();
            cmd_ptr = process_relay_inline_noflush_cmd<
                my_noc_xy,
                local_downstream_cb_sem_id,
                downstream_noc_xy,
                downstream_cb_base,
                downstream_cb_end>(cmd_ptr, downstream_data_ptr);
            break;

        case CQ_PREFETCH_CMD_STALL:
            DPRINT << "stall" << ENDL();
            cmd_ptr = process_stall<dispatch_sync_sem_id>(cmd_ptr);
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
            DPRINT << "prefetcher invalid command:" << (uint32_t)cmd->base.cmd_id << " " << cmd_ptr << " " << fence << " " << ENDL();
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
