// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Diagnostic only: synthetic SwiGLU root-count broadcast, not the full MoE op.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "internal/firmware_common.h"

void kernel_main() {
    const bool root = get_arg_val<uint32_t>(0);
    const uint32_t diag_addr = get_arg_val<uint32_t>(1);
    const uint32_t sem_id = get_arg_val<uint32_t>(2);
    constexpr bool final_ack_barrier = get_compile_time_arg_val(0);
    constexpr uint32_t poll_limit = get_compile_time_arg_val(1);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(diag_addr);
    for (uint32_t i = 0; i < 128; ++i) {
        d[i] = 0;
    }
    d[0] = 0x41434b41;  // ACKA
    d[1] = noc_index;
    d[2] = noc_nonposted_writes_num_issued[noc_index];
    d[3] = noc_nonposted_writes_acked[noc_index];
    d[4] = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
    d[5] = NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
    const uint32_t flag_addr = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    Noc noc;
    if (root) {
        const auto lo = get_virtual_coord_from_worker_logical_coord(0, 0);
        const auto hi = get_virtual_coord_from_worker_logical_coord(7, 7);
        d[12] = lo.x;
        d[13] = lo.y;
        d[14] = hi.x;
        d[15] = hi.y;
        const uint32_t scratch = get_write_ptr(0);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t i = 0; i < 768; ++i) {
            p[i] = i;
        }
        // Three linked 1024-byte nonposted broadcasts reproduce indices/counts/starts.
        for (uint32_t i = 0; i < 3; ++i) {
            const uint32_t addr = scratch + i * 1024;
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(addr),
                MulticastEndpoint{},
                1024,
                63,
                {},
                {.noc_x_start = lo.x, .noc_y_start = lo.y, .noc_x_end = hi.x, .noc_y_end = hi.y, .addr = addr},
                true);
        }
        noc.async_writes_flushed();
        *flag = 1;
        noc_semaphore_set_multicast(flag_addr, get_noc_multicast_addr(lo.x, lo.y, hi.x, hi.y, flag_addr), 63, false);
        noc.async_writes_flushed();
        *flag = 0;
        d[6] = noc_nonposted_writes_num_issued[noc_index];
        d[7] = noc_nonposted_writes_acked[noc_index];
        d[8] = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
        d[9] = NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
        // The A/B experimental difference is exactly this sender-only final barrier.
        if constexpr (final_ack_barrier) {
            noc.async_write_barrier();
        }
        d[10] = NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
        d[11] = 1;
    } else {
        uint32_t poll = 0;
        while (*flag != 1 && poll < poll_limit) {
            invalidate_l1_cache();
            ++poll;
        }
        if (*flag != 1) {
            d[11] = 3;
            d[16] = poll;
            return;
        }
        *flag = 0;
        d[11] = 1;
        d[16] = poll;
    }
    asm volatile("fence" ::: "memory");
}
