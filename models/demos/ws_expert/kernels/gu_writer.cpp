// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert gate/up sender (BRISC, NOC0), per piece in the compute's order:
//   SEND: the fp32 partial (CB 4, GW tiles) -> the previous core's partial ring (2 slots), after its grant; then an
//         atomic increment of its ARRIVED semaphore.
//   else: the row's NPAIR h tiles (CB 16) multicast into slot (q % NS) of every down core's h ring at this pair
//         group's offset, once all down cores freed that slot (CONS >= q + 1, published by the down leader); then the
//         row's flag word for this pair group set to q + 1 on every down core.
// q = e * R + qpos: the row's place in the down cores' order.
// RT: 0 npiece, 1 J (pair group), 2 prev x, 3 prev y (NOC coords of the previous core in the grid column),
//     then per piece: flags (1 PRE, 2 SEND), qpos
// CT: 0 NPAIR, 1 R, 2 NS, 3 KTD (h tiles a row = pairs in all), 4 (unused), 5 GC (pair groups), 6 NEXP,
//     7 hring addr, 8 flags addr, 9 pin addr, 10-13 down rect NOC0 (x0, y0, x1, y1), 14 ND (down cores),
//     15-16 (unused), 17 h tile bytes
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t pout_cb = tt::CBIndex::c_4;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t SEM_CONS = 0, SEM_GRANT = 1, SEM_ARR = 2, SEM_SCR = 3;

void kernel_main() {
    constexpr uint32_t npair = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t NS = get_compile_time_arg_val(2);
    constexpr uint32_t KTD = get_compile_time_arg_val(3);
    constexpr uint32_t GC = get_compile_time_arg_val(5);
    constexpr uint32_t NEXP = get_compile_time_arg_val(6);
    constexpr uint32_t hring = get_compile_time_arg_val(7);
    constexpr uint32_t flags = get_compile_time_arg_val(8);
    constexpr uint32_t pin = get_compile_time_arg_val(9);
    constexpr uint32_t dx0 = get_compile_time_arg_val(10), dy0 = get_compile_time_arg_val(11);
    constexpr uint32_t dx1 = get_compile_time_arg_val(12), dy1 = get_compile_time_arg_val(13);
    constexpr uint32_t ND = get_compile_time_arg_val(14);
    constexpr uint32_t htile = get_compile_time_arg_val(17);
    constexpr uint32_t stamps_addr = get_compile_time_arg_val(18);
    volatile tt_l1_ptr uint32_t* prof = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stamps_addr) + 32;
    uint32_t t_h = 0, t_cons = 0, t_bar = 0, t_part = 0;
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    constexpr uint32_t gw = 2 * npair;
    constexpr uint32_t pbytes = gw * 4096;  // fp32 partial

    const uint32_t npiece = get_arg_val<uint32_t>(0);
    const uint32_t J = get_arg_val<uint32_t>(1);
    const uint32_t px = get_arg_val<uint32_t>(2), py = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* cons = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_CONS));
    volatile tt_l1_ptr uint32_t* grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GRANT));
    const uint32_t scr_addr = get_semaphore(SEM_SCR);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    const uint64_t prev_arr = get_noc_addr(px, py, get_semaphore(SEM_ARR));
    uint32_t nsent = 0;
    for (uint32_t e = 0; e < NEXP; ++e) {
        for (uint32_t p = 0; p < npiece; ++p) {
            const uint32_t fl = get_arg_val<uint32_t>(4 + 2 * p);
            const uint32_t qpos = get_arg_val<uint32_t>(5 + 2 * p);
#ifdef WS_NO_SEND
            if (fl & 2) {
                cb_wait_front(pout_cb, gw);
                noc_semaphore_wait_min(grant, nsent + 1);
                noc_semaphore_inc(prev_arr, 1);
                cb_pop_front(pout_cb, gw);
                ++nsent;
            } else {
                cb_wait_front(h_cb, npair);
                cb_pop_front(h_cb, npair);
            }
            continue;
#endif
            if (fl & 2) {
                uint32_t a = wall();
                cb_wait_front(pout_cb, gw);
                noc_semaphore_wait_min(grant, nsent + 1);
                t_part += wall() - a;
                noc_async_write(get_read_ptr(pout_cb), get_noc_addr(px, py, pin + (nsent & 1) * pbytes), pbytes);
                noc_async_write_barrier();
                noc_semaphore_inc(prev_arr, 1);
                cb_pop_front(pout_cb, gw);
                ++nsent;
            } else {
                const uint32_t q = e * R + qpos;
                const uint32_t slot = q % NS;
                uint32_t a = wall();
                cb_wait_front(h_cb, npair);
                uint32_t b = wall();
                noc_semaphore_wait_min(cons, q + 1);
                uint32_t c = wall();
                t_h += b - a;
                t_cons += c - b;
                noc_async_write_multicast(
                    get_read_ptr(h_cb),
                    get_noc_multicast_addr(dx0, dy0, dx1, dy1, hring + (slot * KTD + J * npair) * htile),
                    npair * htile,
                    ND);
                noc_async_write_barrier();
                *scr = q + 1;
                noc_semaphore_set_multicast(
                    scr_addr, get_noc_multicast_addr(dx0, dy0, dx1, dy1, flags + (slot * GC + J) * 16), ND);
                noc_async_write_barrier();
                t_bar += wall() - c;
                cb_pop_front(h_cb, npair);
            }
        }
    }
    noc_async_atomic_barrier();
    prof[0] = t_h;
    prof[1] = t_cons;
    prof[2] = t_bar;
    prof[3] = t_part;
}
