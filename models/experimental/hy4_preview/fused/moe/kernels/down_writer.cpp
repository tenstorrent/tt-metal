// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down core control and output of the Hy4 decode MoE program (BRISC, NOC1) (fused/moe/program.py).
//
// 1. Wait (bounded) for the hub's release. U goes to the weight streamer (control word + local semaphore) and to the
//    compute threads (mailboxes).
// 2. Union slot i's activation is complete when its counter in the meta page reached P (the finishers multicast the
//    tiles into the activation buffer, then increment it); the slot is then handed to compute.
// 3. This core's `cols` output tiles ([ROWS, 32] BF16 of the chip's partial sum) go to tiles col0 .. of the output
//    buffer in the hub's L1; then the hub's semaphore SEM_OUT is incremented. With an empty union the tiles are
//    zeros.
//
// runtime args: [col0, out_addr, status_addr]
// compile-time args: [cb_meta, cb_ctl, cb_act, cb_out, P, cols, hub_x, hub_y]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "moe_dm.hpp"

void kernel_main() {
    const uint32_t col0 = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t status_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_meta = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t P = get_compile_time_arg_val(4);
    constexpr uint32_t cols = get_compile_time_arg_val(5);
    constexpr uint32_t hub_x = get_compile_time_arg_val(6);
    constexpr uint32_t hub_y = get_compile_time_arg_val(7);

    const uint32_t meta_l1 = get_write_ptr(cb_meta);
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl));
    volatile tt_l1_ptr uint32_t* ctr_act =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1 + hy4moe::META_CTR_ACT);

    uint32_t U = 0;
    const bool released =
        hy4moe::wait_ge(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hy4moe::SEM_GO)), 1);
    if (released) {
        U = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1)[0];
        if (U > hy4moe::MAX_SLOTS) {
            U = 0;
        }
    }
    const uint32_t n_plain =
        released ? reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1)[hy4moe::META_N_PLAIN / 2] : 0;
    const uint32_t first_plain = U > n_plain ? U - n_plain : 0;
    ctl[0] = U;
    ctl[1] = first_plain;
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hy4moe::SEM_LOCAL)), 1);
    for (const auto thread :
         {ckernel::ThreadId::UnpackThreadId, ckernel::ThreadId::MathThreadId, ckernel::ThreadId::PackThreadId}) {
        ckernel::mailbox_write(thread, U);
        ckernel::mailbox_write(thread, first_plain);
    }

    bool timed_out = !released;
#ifdef HY4_TIMING
    const uint32_t t0 = hy4moe::now();
    uint32_t t_first = t0;
#endif
    for (uint32_t i = 0; i < U; ++i) {
        if (!timed_out) {  // after one timeout nothing is waited for any more: the program only has to end
            timed_out = !hy4moe::wait_ge(&ctr_act[i], P);
        }
        cb_push_back(cb_act, P);
#ifdef HY4_TIMING
        if (i == 0) {
            t_first = hy4moe::now();
        }
#endif
    }

    const uint64_t out_noc = get_noc_addr(hub_x, hub_y, out_addr + col0 * hy4moe::ROW_TILE_BYTES);
    uint32_t src = get_write_ptr(cb_out);
    if (U > 0) {
        cb_wait_front(cb_out, cols);
        src = get_read_ptr(cb_out);
    } else {
        volatile tt_l1_ptr uint32_t* zw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
        for (uint32_t i = 0; i < cols * hy4moe::ROW_TILE_BYTES / 4; ++i) {
            zw[i] = 0;
        }
    }
    // (bounded NOC waits: a core that cannot send reports it and ends)
    bool send_failed = !hy4moe::write_buf_free();
    if (!send_failed) {
        noc_async_write(src, out_noc, cols * hy4moe::ROW_TILE_BYTES);
        send_failed = !hy4moe::writes_flushed() || !hy4moe::atomic_buf_free();
    }
    if (U > 0) {
        cb_pop_front(cb_out, cols);
    }
    // the consumer of the output buffer (the cross-chip reduce's sender) counts the down cores here
    if (!send_failed) {
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, get_semaphore(hy4moe::SEM_OUT)), 1);
        send_failed = !hy4moe::atomics_flushed();
    }
    timed_out = timed_out || send_failed;
#ifdef HY4_TIMING
    hy4moe::report_timing(hub_x, hub_y, status_addr, t_first - t0, hy4moe::now() - t0, meta_l1 + hy4moe::META_SCRATCH);
#endif
    if (timed_out) {
        hy4moe::report(
            hub_x,
            hub_y,
            status_addr,
            hy4moe::ST_DOWN,
            send_failed ? 3 : (released ? 2 : 1),
            meta_l1 + hy4moe::META_SCRATCH);
    }
}
