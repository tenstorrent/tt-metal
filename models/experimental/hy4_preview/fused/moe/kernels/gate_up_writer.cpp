// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Gate|up core control and partial-sum forwarding of the Hy4 decode MoE program (BRISC, NOC1)
// (fused/moe/program.py).
//
// 1. Wait (bounded) for the hub's release: x and the meta page are in this core's L1. U goes to the weight streamer
//    (control word + local semaphore) and the item count to the compute threads (mailboxes).
// 2. Per item v (v % modulus == residue): the piece g = (v % pb) * banks + bank is K quarter q = g % S of pair
//    p = g / S of union slot v / pb; its two FP32 partial tiles go to the slot's finisher (core slot % NF, entry
//    ((slot / NF) * P + p) * S + q of its partial buffer), then the counter of unit u = slot * P + p on the finisher
//    is incremented.
//
// runtime args: [residue, modulus, logical_bank]
// compile-time args: [cb_x, cb_meta, cb_ctl, cb_part, cb_out, x_tiles, pb, banks, S, P, NF, hub_x, hub_y, status_addr,
//                     (finisher noc x, y) * NF]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "moe_dm.hpp"

void kernel_main() {
    const uint32_t residue = get_arg_val<uint32_t>(0);
    const uint32_t modulus = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(2);
    constexpr uint32_t cb_part = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t x_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t pb = get_compile_time_arg_val(6);
    constexpr uint32_t banks = get_compile_time_arg_val(7);
    constexpr uint32_t S = get_compile_time_arg_val(8);
    constexpr uint32_t P = get_compile_time_arg_val(9);
    constexpr uint32_t NF = get_compile_time_arg_val(10);
    constexpr uint32_t hub_x = get_compile_time_arg_val(11);
    constexpr uint32_t hub_y = get_compile_time_arg_val(12);
    constexpr uint32_t status_addr = get_compile_time_arg_val(13);
    constexpr uint32_t fin_ct = 14;
    constexpr uint32_t part_bytes = 2 * hy4moe::ROW_TILE_BYTES_F32;

    const uint32_t meta_l1 = get_write_ptr(cb_meta);
    const uint32_t part_l1 = get_write_ptr(cb_part);  // the same address on every core
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl));

    uint32_t U = 0;
    const bool released =
        hy4moe::wait_ge(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hy4moe::SEM_GO)), 1);
    if (released) {
        U = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1)[0];
        if (U > hy4moe::MAX_SLOTS) {
            U = 0;
        }
    }
    const uint32_t total = U * pb;
    const uint32_t n_items = total > residue ? (total - residue + modulus - 1) / modulus : 0;
    const uint32_t n_plain =
        released ? reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1)[hy4moe::META_N_PLAIN / 2] : 0;
    const uint32_t first_plain = U > n_plain ? U - n_plain : 0;
    ctl[0] = U;
    ctl[1] = first_plain;
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hy4moe::SEM_LOCAL)), 1);
    for (const auto thread :
         {ckernel::ThreadId::UnpackThreadId, ckernel::ThreadId::MathThreadId, ckernel::ThreadId::PackThreadId}) {
        ckernel::mailbox_write(thread, n_items);
        ckernel::mailbox_write(thread, first_plain);
    }
    if (!released) {
        hy4moe::report(hub_x, hub_y, status_addr, hy4moe::ST_GATE_UP, 1, meta_l1 + hy4moe::META_SCRATCH);
        return;
    }
    cb_push_back(cb_x, x_tiles);  // written by the hub's multicast
    bool send_failed = false;
#ifdef HY4_TIMING
    const uint32_t t0 = hy4moe::now();
    uint32_t t_first = t0;
#endif

    for (uint32_t v = residue; v < total; v += modulus) {
        const uint32_t slot = v / pb;
        const uint32_t g = (v - slot * pb) * banks + bank;
        const uint32_t p = g / S;
        const uint32_t q = g - p * S;
        const uint32_t u = slot * P + p;
        const uint32_t f = slot % NF;  // a finisher owns whole slots
        const uint32_t entry = ((slot / NF) * P + p) * S + q;
        const uint32_t fx = kernel_compile_time_args[fin_ct + 2 * f];
        const uint32_t fy = kernel_compile_time_args[fin_ct + 2 * f + 1];
        cb_wait_front(cb_out, 2);
        if (!send_failed) {  // (bounded: a core that cannot send stops sending; the finishers give up on their own)
            if (hy4moe::write_buf_free()) {
                noc_async_write(get_read_ptr(cb_out), get_noc_addr(fx, fy, part_l1 + entry * part_bytes), part_bytes);
                send_failed = !hy4moe::writes_flushed() || !hy4moe::atomic_buf_free();
            } else {
                send_failed = true;
            }
            if (!send_failed) {
                noc_semaphore_inc(get_noc_addr(fx, fy, meta_l1 + hy4moe::META_CTR_PART + 4 * u), 1);
            }
        }
        cb_pop_front(cb_out, 2);
#ifdef HY4_TIMING
        if (v == residue) {
            t_first = hy4moe::now();
        }
#endif
    }
    send_failed = !hy4moe::atomics_flushed() || send_failed;
#ifdef HY4_TIMING
    hy4moe::report_timing(hub_x, hub_y, status_addr, t_first - t0, hy4moe::now() - t0, meta_l1 + hy4moe::META_SCRATCH);
#endif
    if (send_failed) {
        hy4moe::report(hub_x, hub_y, status_addr, hy4moe::ST_GATE_UP, 3, meta_l1 + hy4moe::META_SCRATCH);
    }
}
