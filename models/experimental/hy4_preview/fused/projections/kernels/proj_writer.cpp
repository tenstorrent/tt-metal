// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Control and output of one projection unit of the Hy4 decode projection programs (BRISC, NOC1)
// (fused/projections/program.py).
//
// 1. Wait (bounded) for the hub's release (the go word): the left operand buffer and the meta page are in this
//    core's L1. The block count goes to the compute threads (mailboxes); the weight streamer runs on its own from
//    the start of the kernel.
// 2. The unit's constant page for the SFPU (proj_sfpu.hpp) is built from the meta page's parameters.
// 3. A unit of a second stage reads its left operand from the activation buffer, which first-stage units fill by
//    multicast: it waits (bounded) until its group's counter in the meta page reached `wait_need`.
// 4. The unit's output tiles go either to the output buffer in the hub's L1 (then word `done_word` of the hub's go
//    tensor is incremented: word 1 unless the unit was given another done group; the units of a group have all
//    written when the word equals their number), or by multicast to every core's activation buffer (then the group's
//    counter is incremented by multicast). A statistic tile (FIN_STAT) goes to the hub's statistic buffer.
//
// Every wait on another core and on the NOC is bounded (proj_dm.hpp): a unit that times out reports in the hub's
// status words and carries on with what it has, so the program always ends.
//
// runtime args: [out_tile0, out_addr, status_addr, wait_idx, wait_need, ctr_idx, stat_addr, go_addr, done_word]
// compile-time args: [cb_in, cb_meta, cb_ctl (unused), cb_sc, cb_out, cb_stat, cb_act, in_push, n_blocks, fin, ct,
// dst_act,
//                     hub_x, hub_y, mc_x0, mc_y0, mc_x1, mc_y1, mc_dests]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "proj_dm.hpp"

void kernel_main() {
#ifdef HY4_EMPTY  // measurement only: the launch floor of the program
    return;
#endif
    const uint32_t out_tile0 = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t status_addr = get_arg_val<uint32_t>(2);
    const uint32_t wait_idx = get_arg_val<uint32_t>(3);
    const uint32_t wait_need = get_arg_val<uint32_t>(4);
    const uint32_t ctr_idx = get_arg_val<uint32_t>(5);
    const uint32_t stat_addr = get_arg_val<uint32_t>(6);
    const uint32_t go_addr = get_arg_val<uint32_t>(7);
    const uint32_t done_word = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_stat = get_compile_time_arg_val(5);
    constexpr uint32_t cb_act = get_compile_time_arg_val(6);
    constexpr uint32_t in_push = get_compile_time_arg_val(7);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t fin = get_compile_time_arg_val(9);
    constexpr uint32_t ct = get_compile_time_arg_val(10);
    constexpr uint32_t dst_act = get_compile_time_arg_val(11);
    constexpr uint32_t hub_x = get_compile_time_arg_val(12);
    constexpr uint32_t hub_y = get_compile_time_arg_val(13);
    constexpr uint32_t mc_x0 = get_compile_time_arg_val(14);
    constexpr uint32_t mc_y0 = get_compile_time_arg_val(15);
    constexpr uint32_t mc_x1 = get_compile_time_arg_val(16);
    constexpr uint32_t mc_y1 = get_compile_time_arg_val(17);
    constexpr uint32_t mc_dests = get_compile_time_arg_val(18);
    constexpr bool has_const = hy4proj::fin_has_const(fin);
    constexpr uint32_t out_bytes = ct * hy4proj::ROW_TILE_BYTES;

    const uint32_t meta_l1 = get_write_ptr(cb_meta);
    const uint32_t scratch = meta_l1 + hy4proj::META_SCRATCH;
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_addr);
    volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1 + hy4proj::META_CTR);

    const bool released = hy4proj::wait_ge(go, 1);
    if (released) {
        go[0] = 0;  // consumed: the word is zero again for the next run
    }
    for (const auto thread :
         {ckernel::ThreadId::UnpackThreadId, ckernel::ThreadId::MathThreadId, ckernel::ThreadId::PackThreadId}) {
        ckernel::mailbox_write(thread, released ? n_blocks : 0);
    }
    if (!released) {
        hy4proj::report(hub_x, hub_y, status_addr, hy4proj::ST_UNIT, 1, scratch);
        return;
    }
#ifdef HY4_TIMING
    const uint32_t t0 = hy4proj::now();
#endif

    if constexpr (has_const) {
        // FP32 [32, 32] page: face 0 row r = scale[r] on all 16 columns; row 8 + r = the lane sums of token row r on
        // the even columns
        volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_sc));
        volatile tt_l1_ptr uint32_t* scale =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1 + hy4proj::PARAM_SCALE);
        volatile tt_l1_ptr uint32_t* lanes =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1 + hy4proj::PARAM_LANES);
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uint32_t s = scale[r];
            for (uint32_t c = 0; c < 16; ++c) {
                page[r * 16 + c] = s;
            }
#if ROWS == 8
            for (uint32_t l = 0; l < hy4proj::LANES; ++l) {
                page[(8 + r) * 16 + 2 * l] = lanes[r * hy4proj::LANES + l];
                page[(8 + r) * 16 + 2 * l + 1] = 0;
            }
#else
            // (16 token rows: the scales fill face 0; the lane sums of token row r are row r of face 2, which starts
            // 512 words into the FP32 page)
            for (uint32_t l = 0; l < hy4proj::LANES; ++l) {
                page[512 + r * 16 + 2 * l] = lanes[r * hy4proj::LANES + l];
                page[512 + r * 16 + 2 * l + 1] = 0;
            }
#endif
        }
        cb_push_back(cb_sc, 1);
    }

    bool timed_out = false;
    if (wait_idx != hy4proj::NO_WAIT) {
        timed_out = !hy4proj::wait_ge(&ctr[wait_idx], wait_need);
    }
    cb_push_back(cb_in, in_push);  // written by the hub's multicast, or by the first-stage units'

    cb_wait_front(cb_out, ct);
    bool send_failed = !hy4proj::write_buf_free();
    if constexpr (dst_act) {
        const uint32_t act_l1 = get_write_ptr(cb_act);  // the same address on every core
        if (!send_failed) {
            send_failed = !hy4proj::grid_mcast_write(
                              get_read_ptr(cb_out),
                              mc_x0,
                              mc_y0,
                              mc_x1,
                              mc_y1,
                              act_l1 + out_tile0 * hy4proj::ROW_TILE_BYTES,
                              out_bytes,
                              mc_dests) ||
                          !hy4proj::writes_flushed() ||
                          !hy4proj::grid_mcast_inc(
                              mc_x0, mc_y0, mc_x1, mc_y1, meta_l1 + hy4proj::META_CTR + 4 * ctr_idx, mc_dests);
        }
    } else {
        if (!send_failed) {
            noc_async_write(
                get_read_ptr(cb_out),
                get_noc_addr(hub_x, hub_y, out_addr + out_tile0 * hy4proj::ROW_TILE_BYTES),
                out_bytes);
            send_failed = !hy4proj::writes_flushed() || !hy4proj::atomic_buf_free();
        }
        // the consumer of the output buffer (the cross-chip step's sender) counts the units here: a word of the
        // hub's go tensor (1, or 1 + the unit's done group), which the hub zeroes at its start
        if (!send_failed) {
            noc_semaphore_inc(get_noc_addr(hub_x, hub_y, go_addr + 4 * done_word), 1);
        }
    }
    cb_pop_front(cb_out, ct);

    if constexpr (fin == hy4proj::FIN_STAT) {
        cb_wait_front(cb_stat, 1);
        if (!send_failed && hy4proj::write_buf_free()) {
            noc_async_write(get_read_ptr(cb_stat), get_noc_addr(hub_x, hub_y, stat_addr), hy4proj::ROW_TILE_BYTES_F32);
            send_failed = !hy4proj::writes_flushed();
        } else {
            send_failed = true;
        }
        cb_pop_front(cb_stat, 1);
    }
    send_failed = !hy4proj::atomics_flushed() || send_failed;
#ifdef HY4_TIMING
    hy4proj::report_timing(hub_x, hub_y, status_addr, 0, hy4proj::now() - t0, scratch);
#endif
    if (timed_out || send_failed) {
        hy4proj::report(hub_x, hub_y, status_addr, hy4proj::ST_UNIT, send_failed ? 3 : 2, scratch);
    }
}
