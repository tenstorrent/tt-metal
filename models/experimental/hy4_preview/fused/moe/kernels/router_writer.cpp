// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router core control (BRISC, NOC1) of the Hy4 decode MoE program (fused/moe/program.py): wait (bounded) for the
// hub's x (program semaphore SEM_X), start the weight streamer and compute, then send this core's FP32 score tile
// (its 32 experts for the 8 rows) to tile `column` of the hub's score buffer and count it in the hub's semaphore
// SEM_SCORES.
//
// With ROUTER_CANDIDATES this core also cuts its 32 experts to the 8 best of every row (score + bias) and sends the
// lists to the hub, which then only merges.
//
// runtime args: [column, scores_addr, bias_addr, candidates_addr]
// compile-time args: [cb_x, cb_ctl, cb_out, x_tiles, hub_x, hub_y]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "moe_dm.hpp"

void kernel_main() {
    const uint32_t column = get_arg_val<uint32_t>(0);
    const uint32_t scores_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t x_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t hub_x = get_compile_time_arg_val(4);
    constexpr uint32_t hub_y = get_compile_time_arg_val(5);

    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl));
    const uint32_t released =
        hy4moe::wait_ge(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hy4moe::SEM_X)), 1) ? 1 : 0;
    ctl[0] = released;    // the streamer's item count
    ctl[1] = 0xFFFFFFFF;  // no plain slots
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hy4moe::SEM_LOCAL)), 1);
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, released);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, released);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, released);
    if (released) {
        cb_push_back(cb_x, x_tiles);  // written by the hub's multicast
        cb_wait_front(cb_out, 1);
#ifdef ROUTER_CANDIDATES
        // this column's K best experts of every row, by score + bias (Q30; the bias of the 32 experts read from the
        // hub's table while compute ran): best first, ties to the lower id
        {
            constexpr uint32_t K = 8;
            const uint32_t bias_addr = get_arg_val<uint32_t>(2);
            const uint32_t cand_addr = get_arg_val<uint32_t>(3);
            const uint32_t scratch = get_write_ptr(cb_ctl);  // [8 control | 128 bias | candidates]
            noc_async_read(get_noc_addr(hub_x, hub_y, bias_addr + column * 128), scratch + 8, 128);
            noc_async_read_barrier();
            invalidate_l1_cache();
            const int32_t* bias = reinterpret_cast<const int32_t*>(scratch + 8);
            const uint32_t* tile = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_out));
            hy4moe::Candidate* cand = reinterpret_cast<hy4moe::Candidate*>(scratch + 136);
            for (uint32_t r = 0; r < ROWS; ++r) {
                hy4moe::Candidate* best = cand + r * K;
                for (uint32_t j = 0; j < K; ++j) {
                    best[j].key = INT32_MIN;
                    best[j].score = 0;
                    best[j].id = 0;
                }
                for (uint32_t face = 0; face < 2; ++face) {
                    const uint32_t* w = tile + (face * ROWS + r) * 16;
                    for (uint32_t j = 0; j < 16; ++j) {
                        const int32_t s = hy4moe::q30(w[j]);
                        hy4moe::insert_candidate<K>(best, s + bias[face * 16 + j], s, column * 32 + face * 16 + j);
                    }
                }
            }
            constexpr uint32_t cand_bytes = ROWS * K * sizeof(hy4moe::Candidate);
            if (hy4moe::write_buf_free()) {
                noc_async_write(scratch + 136, get_noc_addr(hub_x, hub_y, cand_addr + column * cand_bytes), cand_bytes);
            }
        }
#endif
        if (hy4moe::write_buf_free()) {  // (bounded NOC waits)
            noc_async_write(
                get_read_ptr(cb_out),
                get_noc_addr(hub_x, hub_y, scores_addr + column * hy4moe::ROW_TILE_BYTES_F32),
                hy4moe::ROW_TILE_BYTES_F32);
            hy4moe::writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    // counted either way: the hub does not wait for a core that gave up
    if (hy4moe::atomic_buf_free()) {
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, get_semaphore(hy4moe::SEM_SCORES)), 1);
        hy4moe::atomics_flushed();
    }
}
