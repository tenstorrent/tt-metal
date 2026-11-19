// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Finisher control of the Hy4 decode MoE program (BRISC, NOC1) (fused/moe/program.py). This core finishes the union
// slots f, f + NF, ... in order, each as its P units (pairs of gate / up tile columns).
//
// 1. Wait (bounded) for the hub's release; the unit count and the first plain slot go to the compute threads
//    (mailboxes).
// 2. Unit u = slot * P + pair is ready when its counter in the meta page reached S (the gate|up cores wrote its S
//    partials into the partial buffer and incremented it). Then its partials, and with a slot's first unit the
//    slot's scale tile (row r = w[r][slot], FP32), are handed to compute.
// 3. A slot's P activation tiles are multicast together to every core of the chip (tiles slot * P .. of the
//    activation buffer, the same address everywhere), then the slot's activation counter is incremented by P by
//    multicast: the down cores start on a slot when it reached P.
//
// runtime args: [f]
// compile-time args: [cb_meta, cb_part, cb_sc, cb_act_out, cb_act, S, P, NF, mc_x0, mc_y0, mc_x1, mc_y1, mc_dests,
//                     hub_x, hub_y, status_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "moe_dm.hpp"

void kernel_main() {
    const uint32_t f = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_meta = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_act_out = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act = get_compile_time_arg_val(4);
    constexpr uint32_t S = get_compile_time_arg_val(5);
    constexpr uint32_t P = get_compile_time_arg_val(6);
    constexpr uint32_t NF = get_compile_time_arg_val(7);
    constexpr uint32_t mc_x0 = get_compile_time_arg_val(8);
    constexpr uint32_t mc_y0 = get_compile_time_arg_val(9);
    constexpr uint32_t mc_x1 = get_compile_time_arg_val(10);
    constexpr uint32_t mc_y1 = get_compile_time_arg_val(11);
    constexpr uint32_t mc_dests = get_compile_time_arg_val(12);
    constexpr uint32_t hub_x = get_compile_time_arg_val(13);
    constexpr uint32_t hub_y = get_compile_time_arg_val(14);
    constexpr uint32_t status_addr = get_compile_time_arg_val(15);

    const uint32_t meta_l1 = get_write_ptr(cb_meta);
    const uint32_t act_l1 = get_write_ptr(cb_act);  // the same address on every core
    volatile tt_l1_ptr uint16_t* meta16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1);
    volatile tt_l1_ptr uint16_t* w = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1 + hy4moe::META_W);
    volatile tt_l1_ptr uint32_t* ctr_part =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1 + hy4moe::META_CTR_PART);

    uint32_t U = 0;
    const bool released =
        hy4moe::wait_ge(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hy4moe::SEM_GO)), 1);
    if (released) {
        U = meta16[0];
        if (U > hy4moe::MAX_SLOTS) {
            U = 0;
        }
    }
    const uint32_t n_slots = U > f ? (U - f + NF - 1) / NF : 0;  // slots f, f + NF, ...
    const uint32_t n_units = n_slots * P;
    // compute works out its units from U; the last n_plain slots get no clamp
    const uint32_t n_plain = released ? meta16[hy4moe::META_N_PLAIN / 2] : 0;
    const uint32_t first_plain = U > n_plain ? U - n_plain : 0;
    for (const auto thread :
         {ckernel::ThreadId::UnpackThreadId, ckernel::ThreadId::MathThreadId, ckernel::ThreadId::PackThreadId}) {
        ckernel::mailbox_write(thread, n_units);
        ckernel::mailbox_write(thread, first_plain);
    }
    bool timed_out = !released;

    uint32_t pushed = 0;
    uint32_t sent = 0;
    uint32_t spins = 0;
    bool send_failed = false;
#ifdef HY4_TIMING
    const uint32_t t0 = hy4moe::now();
    uint32_t t_first = t0;
#endif
    // Neither hand-over to compute may block here: compute needs this loop to take its activation tiles (cb_act_out)
    // before it can free a scale tile (cb_sc), so both are only attempted when they cannot wait.
    // Unit n of this core is pair n % P of slot f + (n / P) * NF. A slot's P activation tiles go out together.
    while (sent < n_units) {
        if (sent < pushed && cb_pages_available_at_front(cb_act_out, P)) {
            const uint32_t slot = f + (sent / P) * NF;
            // (bounded: a finisher that cannot send stops sending; the down cores give up on their own. The
            // increment follows the tiles on the same NOC and virtual channel, so it lands after them.)
            if (!send_failed) {
                send_failed = !hy4moe::grid_mcast_write(
                                  get_read_ptr(cb_act_out),
                                  mc_x0,
                                  mc_y0,
                                  mc_x1,
                                  mc_y1,
                                  act_l1 + slot * P * hy4moe::ROW_TILE_BYTES,
                                  P * hy4moe::ROW_TILE_BYTES,
                                  mc_dests) ||
                              !hy4moe::grid_mcast_inc(
                                  mc_x0, mc_y0, mc_x1, mc_y1, meta_l1 + hy4moe::META_CTR_ACT + 4 * slot, P, mc_dests);
            }
            cb_pop_front(cb_act_out, P);
#ifdef HY4_TIMING
            if (sent == 0) {
                t_first = hy4moe::now();
            }
#endif
            sent += P;
            continue;
        }
        if (pushed < n_units && (pushed % P != 0 || cb_pages_reservable_at_back(cb_sc, 1))) {
            const uint32_t slot = f + (pushed / P) * NF;
            const uint32_t u = slot * P + pushed % P;
            invalidate_l1_cache();
            // after one timeout nothing is waited for any more: the program only has to end
            if (ctr_part[u] >= S || timed_out || spins >= HY4_SPIN_LIMIT) {
                timed_out = timed_out || ctr_part[u] < S;
                spins = 0;
                if (pushed % P == 0) {  // the slot's scale tile: row r = w[r][slot]
                    volatile tt_l1_ptr uint32_t* t =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_sc));
                    for (uint32_t r = 0; r < ROWS; ++r) {
                        const uint32_t v = static_cast<uint32_t>(w[r * hy4moe::MAX_SLOTS + slot])
                                           << 16;  // BF16 -> FP32
                        for (uint32_t c = 0; c < 16; ++c) {
                            t[r * 16 + c] = v;           // face 0, row r
                            t[(ROWS + r) * 16 + c] = v;  // face 1, row r
                        }
                    }
                    cb_push_back(cb_sc, 1);
                }
                cb_push_back(cb_part, 2 * S);
                ++pushed;
                continue;
            }
            ++spins;  // waiting for the unit's partials
        }
    }
    send_failed = !hy4moe::atomics_flushed() || send_failed;
#ifdef HY4_TIMING
    hy4moe::report_timing(hub_x, hub_y, status_addr, t_first - t0, hy4moe::now() - t0, meta_l1 + hy4moe::META_SCRATCH);
#endif
    if (timed_out || send_failed) {
        hy4moe::report(
            hub_x,
            hub_y,
            status_addr,
            hy4moe::ST_FINISH,
            send_failed ? 3 : (released ? 2 : 1),
            meta_l1 + hy4moe::META_SCRATCH);
    }
}
