// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// X40 TP32 24-core variant (tp24), BRISC of core (bank b = me, group i = grp).  The 8 cores (d, grp) d = 0..7 form a
// GROUP that behaves exactly like the 8 bank cores of tp_writer.cpp, but for the experts a % 3 == grp of the active
// list only (q = index within the group, expert = active[3 q + grp]).  The act of an expert goes to the 8 cores of its
// group only (they run the down for their own experts).  After the down, each core holds a partial output (24 MT
// tiles): the cores of groups 1, 2 send theirs to core (b, 0) (landing cb_lnd, semaphore 2), which sums the three in
// compute (cb_fin) and writes the output tiles 24 b ... X40 TP32 routed experts, BRISC of bank core b (NC = 8 bank
// cores).  Per call:
//   route -> plan (distinct experts, na) -> meta page for compute; score tiles of the experts this core reduces (a = me
//   + 8 j) x K slice (tiles 24 me .. 24 me + 23) -> cb_x partials P[a] (4 tiles) -> reducer core a % 8, slot ((a / 8) *
//   8 + me); semaphore 0 counts them reducer: all 8 partials arrived -> cb_pr pushed (compute sums, swiglu, score) ->
//   act tiles (cb_actl, 2 per expert) -> every core's cb_act slot a; semaphore 1 counts them cb_act pushed (2 na tiles)
//   -> compute down -> cb_o (6 x 4 tiles) -> output tiles 24 me + 4 s + n of out (std tiles, rows 0..R-1)
// CT: 0 cb_x, 1 cb_p, 2 cb_pr, 3 cb_sc, 4 cb_actl, 5 cb_act, 6 cb_o, 7 cb_meta, 8 cb_rt, 9 T, 10 R, 11 NMAX, then
// TensorAccessorArgs(x), (route), (out) RT: 0 me, 1 first output tile, 2 grp.  Common: 0 x addr, 1 route addr, 2 out
// addr, 3.. physical (x, y) of the 24 cores (index 3 d + grp).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "tp_route.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_p = get_compile_time_arg_val(1);
    constexpr uint32_t cb_pr = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(3);
    constexpr uint32_t cb_actl = get_compile_time_arg_val(4);
    constexpr uint32_t cb_act = get_compile_time_arg_val(5);
    constexpr uint32_t cb_o = get_compile_time_arg_val(6);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(7);
    constexpr uint32_t cb_rt = get_compile_time_arg_val(8);
    constexpr uint32_t T = get_compile_time_arg_val(9);
    constexpr uint32_t R = get_compile_time_arg_val(10);
    constexpr uint32_t NMAX = get_compile_time_arg_val(11);
    constexpr uint32_t NC = 8;
    constexpr uint32_t MT = (T > 8) ? 2 : 1;  // row tiles of 8 rows
    constexpr uint32_t KB = 24;
    constexpr uint32_t TB = 64 * R;  // bytes of one in0 tile [R, 32] bf16
    constexpr uint32_t FB = 32 * R;  // bytes of one face [R, 16]
    constexpr uint32_t NJMAX = (NMAX + NC - 1) / NC;
    constexpr uint32_t PNB = 512;  // bytes of an [8, 32] bf16 tile
    constexpr uint32_t cb_lnd = get_compile_time_arg_val(12);
    constexpr uint32_t cb_fin = get_compile_time_arg_val(13);
    constexpr auto x_args = TensorAccessorArgs<14>();
    constexpr auto r_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t SHARED = get_compile_time_arg_val(
        o_args.next_compile_time_args_offset());  // B19: 1 = a shared-expert core of bank b NoC-writes a third partial
                                                  // (slot 2) into the leader's landing buffer

    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t ocol0 = get_arg_val<uint32_t>(1);
    const uint32_t grp = get_arg_val<uint32_t>(2);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t route_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(2);
    auto nx = [&](uint32_t d) { return get_common_arg_val<uint32_t>(3 + 2 * (3 * d + grp)); };  // group member d
    auto ny = [&](uint32_t d) { return get_common_arg_val<uint32_t>(4 + 2 * (3 * d + grp)); };
    auto lx = [&]() { return get_common_arg_val<uint32_t>(3 + 2 * (3 * me)); };  // leader (b, 0) of this bank
    auto ly = [&]() { return get_common_arg_val<uint32_t>(4 + 2 * (3 * me)); };

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, 2048);
    const auto r_acc = TensorAccessor(r_args, route_addr, T * 64);
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    experimental::CB cbx(cb_x), cbp(cb_p), cbpr(cb_pr), cbsc(cb_sc), cbactl(cb_actl), cbact(cb_act), cbo(cb_o),
        cbmeta(cb_meta), cblnd(cb_lnd), cbfin(cb_fin);
    const uint32_t sem_p_addr = get_semaphore(0);
    const uint32_t sem_a_addr = get_semaphore(1);
    const uint32_t sem_l_addr = get_semaphore(2);
    volatile tt_l1_ptr uint32_t* sem_l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l_addr);
    volatile tt_l1_ptr uint32_t* sem_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_p_addr);
    volatile tt_l1_ptr uint32_t* sem_a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_a_addr);

    // the landing buffers are reserved up front: their addresses (identical on every core) are the NoC destinations of
    // the exchange
    cbpr.reserve_back(NJMAX * NC * 4 * MT);
    cbact.reserve_back(2 * NMAX * MT);
    const uint32_t pr_base = cbpr.get_write_ptr();
    const uint32_t act_base = cbact.get_write_ptr();
    if (grp == 0) {
        cblnd.reserve_back((SHARED ? 72 : 48) * MT);
    }
    const uint32_t lnd_base = cblnd.get_write_ptr();  // identical on every core (same CB layout)
    if constexpr (SHARED) {
        if (grp == 0) {  // tell the shared-expert core of this bank where to land (its semaphore 3)
            volatile tt_l1_ptr uint32_t* hs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
            *hs = lnd_base;
            const uint32_t sx = get_common_arg_val<uint32_t>(3 + 2 * 24 + 2 * me);
            const uint32_t sy = get_common_arg_val<uint32_t>(4 + 2 * 24 + 2 * me);
            noc_async_write(get_semaphore(3), get_noc_addr(sx, sy, get_semaphore(3)), 4);
            noc_async_write_barrier();
        }
    }

#ifdef XPUSH
    {  // B47b: wait for the 4 group sources (see tp24_reader.cpp); RT 3..: flag core x y, 4 flag addrs, counter addr
        DeviceZoneScopedN("XW_WAIT");
        const uint32_t fx = get_arg_val<uint32_t>(3), fy = get_arg_val<uint32_t>(4);
        volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(9));
        const uint32_t target = *cnt + XP_NPT;
        *cnt = target;
        const uint32_t scr = cbmeta.get_write_ptr();
        volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
        for (uint32_t i = 0; i < 4; ++i) {
            const uint64_t fn = get_noc_addr(fx, fy, get_arg_val<uint32_t>(5 + i));
            do {
                noc_async_read(fn, scr, 4);
                noc_async_read_barrier();
            } while ((int32_t)(sp[0] - target) < 0);
        }
    }
#endif
    // route, plan
    uint16_t active[128];
    const uint32_t rt_l1 =
        cbmeta.get_write_ptr() + 64;  // meta page = 64 B counters + route staging (page is 64 + 64 T bytes)
    noc_async_read(r_acc.get_noc_addr(0), rt_l1, T * 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* route = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt_l1);
    const uint32_t na =
        tp_build_plan_grp(route, T, active, grp);  // experts of this group: e % 3 == grp, ascending (row-independent)
    const uint32_t nj = (na > me) ? (na - me + NC - 1) / NC : 0;
    {
        cbmeta.reserve_back(1);
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbmeta.get_write_ptr());
        m[0] = na;
        m[1] = nj;
        cbmeta.push_back(1);
    }
    // score tiles (16x16 view of an [8, 32] tile: face f element index f * 128 + r * 16 + c) of the experts this core
    // reduces
    if (nj > 0) {
        cbsc.reserve_back(nj * MT);
        for (uint32_t jm = 0; jm < nj * MT; ++jm) {
            const uint32_t j = jm / MT, m = jm % MT;
            volatile tt_l1_ptr uint16_t* tile =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbsc.get_write_ptr() + jm * PNB);
            const uint32_t e = active[me + NC * j];
            for (uint32_t r = 0; r < 8; ++r) {
                uint16_t s = 0;
                const uint32_t rg = 8 * m + r;
                if (rg < T) {
                    for (uint32_t k = 0; k < 8; ++k) {
                        if ((route[rg * 16 + k] & 255u) == e) {
                            s = (uint16_t)(route[rg * 16 + 8 + k] & 0xFFFFu);
                        }
                    }
                }
                for (uint32_t c = 0; c < 16; ++c) {
                    tile[r * 16 + c] = s;
                    tile[128 + r * 16 + c] = s;
                }
            }
        }
        cbsc.push_back(nj * MT);
    }

    // x slice
    cbx.reserve_back(KB * MT);
#ifdef XPUSH
    {  // B47b: x slice of bank me from the push landing: [src chip i][bank][j][f][row-in-src (XP_T) x 32 B], pages of
       // 4096 B
        constexpr uint32_t PB = 32 * XP_T;  // bytes of one piece (XP_T rows x 32 B)
        constexpr uint32_t SRCB = 12288 * XP_T, BLKB = 1536 * XP_T;
        const auto xl = TensorAccessor(x_args, x_addr, 4096);
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t i = (8 * m) / XP_T; i < (8 * m + R) / XP_T; ++i) {
                for (uint32_t j = 0; j < KB; ++j) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        const uint32_t off = i * SRCB + me * BLKB + (j * 2 + f) * PB;
                        noc.async_read(
                            xl,
                            cbx,
                            PB,
                            {.page_id = off >> 12, .offset_bytes = off & 4095},
                            {.offset_bytes = (m * KB + j) * TB + f * FB + (i * XP_T - 8 * m) * 32});
                    }
                }
            }
        }
        noc.async_read_barrier();
        cbx.push_back(KB * MT);
    }
    if (false)
#endif
    {
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t j = 0; j < KB; ++j) {
                const uint32_t i = KB * me + j;
                // rows 8m .. 8m + R - 1 of the std tile: face 0 at 32 B per row, face 1 (cols 16..31) at 512 + same
                noc.async_read(
                    x_acc, cbx, FB, {.page_id = i, .offset_bytes = 32 * 8 * m}, {.offset_bytes = (m * KB + j) * TB});
                noc.async_read(
                    x_acc,
                    cbx,
                    FB,
                    {.page_id = i, .offset_bytes = 512 + 32 * 8 * m},
                    {.offset_bytes = (m * KB + j) * TB + FB});
            }
        }
        noc.async_read_barrier();
        cbx.push_back(KB * MT);
    }

    // per group of 8 experts: partials -> reducers (one write barrier, then the semaphores); the core that reduces
    // expert j = g (a = me + 8 g) then waits for its 8 partials, hands them to compute (cb_pr, 32 tiles per expert,
    // never popped before the end) and sends the act of that expert to every core while the next group streams
    const uint32_t ng = (na + 7) / 8;
    for (uint32_t g = 0; g < ng; ++g) {
        const uint32_t g0 = 8 * g;
        const uint32_t n = (na - g0 < 8) ? (na - g0) : 8;
        cbp.wait_front(4 * MT * n);
        const uint32_t src0 = cbp.get_read_ptr();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t a = g0 + i;
            const uint32_t r = a & 7u, j = a >> 3;
            const uint32_t dst = pr_base + ((j * NC + me) * 4 * MT) * PNB;
            noc_async_write(src0 + i * 4 * MT * PNB, get_noc_addr(nx(r), ny(r), dst), 4 * MT * PNB);
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t r = (g0 + i) & 7u;
            noc_semaphore_inc(get_noc_addr(nx(r), ny(r), sem_p_addr), 1);
        }
        noc_async_atomic_barrier();
        cbp.pop_front(4 * MT * n);
        if (g < nj) {
            noc_semaphore_wait_min(sem_p, NC * (g + 1));
            cbpr.push_back(32 * MT);
            cbactl.wait_front(2 * MT);
            const uint32_t src = cbactl.get_read_ptr();
            const uint32_t a = me + NC * g;
            for (uint32_t d = 0; d < NC; ++d) {
                for (uint32_t c = 0; c < 2 * MT; ++c) {  // c = m * 2 + k ; act tile (expert a, m, k) = a * 2 MT + c
                    const uint32_t dst = act_base + (a * 2 * MT + c) * TB;
                    if constexpr (R == 8) {
                        noc_async_write(src + c * PNB, get_noc_addr(nx(d), ny(d), dst), 512);
                    } else {  // [8, 32] source tile (face 1 at +256) -> [R, 32] in0 tile (face 1 at + 32 R)
                        noc_async_write(src + c * PNB, get_noc_addr(nx(d), ny(d), dst), FB);
                        noc_async_write(src + c * PNB + 256, get_noc_addr(nx(d), ny(d), dst + FB), FB);
                    }
                }
            }
            noc_async_write_barrier();
            for (uint32_t d = 0; d < NC; ++d) {
                noc_semaphore_inc(get_noc_addr(nx(d), ny(d), sem_a_addr), 1);
            }
            noc_async_atomic_barrier();
            cbactl.pop_front(2 * MT);
        }
    }
    *sem_p = 0;
    noc_semaphore_wait(sem_a, na);
    *sem_a = 0;
    cbact.push_back(2 * na * MT);

    if (grp != 0) {  // hand the partial output to the bank leader
        cbo.wait_front(24 * MT);
        noc_async_write(
            cbo.get_read_ptr(), get_noc_addr(lx(), ly(), lnd_base + (grp - 1) * 24 * MT * PNB), 24 * MT * PNB);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lx(), ly(), sem_l_addr), 1);
        noc_async_atomic_barrier();
        cbo.pop_front(24 * MT);
        return;
    }
    noc_semaphore_wait(sem_l, 2 + SHARED);
    *sem_l = 0;
    cblnd.push_back((SHARED ? 72 : 48) * MT);
    // output (leader): compute sums cb_o + the two landed partials into cb_fin
    cbfin.wait_front(24 * MT);
    for (uint32_t s = 0; s < 6; ++s) {
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t w = 0; w < 4; ++w) {
                const uint32_t pg = ocol0 + s * 4 + w;
                const uint32_t t_off = ((s * MT + m) * 4 + w) * PNB;
                noc.async_write(cbfin, o_acc, FB, {.offset_bytes = t_off}, {.page_id = pg, .offset_bytes = 32 * 8 * m});
                noc.async_write(
                    cbfin, o_acc, FB, {.offset_bytes = t_off + 256}, {.page_id = pg, .offset_bytes = 512 + 32 * 8 * m});
            }
        }
    }
    noc.async_write_barrier();
    cbfin.pop_front(24 * MT);
    noc_async_atomic_barrier();
}
