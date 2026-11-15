// SPDX-License-Identifier: Apache-2.0
// tp32pf device plan builder, ONE core, TWO RISCs (ROLE 0 = NCRISC/NoC0 reader config, ROLE 1 = NCRISC1/NoC1 writer
// config; same source). Same function as plan_build.cpp (output identical to hy4_tp32pf_plan.build_plan): the tokens
// are split in two halves, both RISCs run the histogram and the scatter of their half (the second half starts every
// expert's cursor after the first half's rows, so the order is the stable (token, slot) order); ROLE 0 does the prefix,
// the pad rows, the LPT and the output writes. Hand-shake through four L1 flags. CT: 0 cb_arr, 1 TT, 2 NC, 3 CAP, 4
// NPAGES, 5 ROLE, TensorAccessorArgs(route), (plan), (gidx), (dest), (scr). Common: 0 route, 1 plan, 2 gidx, 3 dest, 4
// scr
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_arr = get_compile_time_arg_val(0);
    constexpr uint32_t TT = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t CAP = get_compile_time_arg_val(3);
    constexpr uint32_t NPAGES = get_compile_time_arg_val(4);
    constexpr uint32_t ROLE = get_compile_time_arg_val(5);
    constexpr auto r_args = TensorAccessorArgs<6>();
    constexpr auto p_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr uint32_t NE = 256, ROWS = CAP * 32, HALF = TT / 2;
    const auto ra = TensorAccessor(r_args, get_common_arg_val<uint32_t>(0), 2048);

    if (ROLE == 0) {
        cb_reserve_back(cb_arr, NPAGES);
    }
    const uint32_t base = get_write_ptr(cb_arr);
    uint32_t* route = reinterpret_cast<uint32_t*>(base);
    uint32_t* gidx = route + TT * 16;
    uint32_t* dest = gidx + ROWS;
    uint16_t* scr = reinterpret_cast<uint16_t*>(dest + ROWS);
    uint32_t* plan = reinterpret_cast<uint32_t*>(scr + ROWS);
    uint32_t* cntA = plan + NC * 512;
    uint32_t* cntB = cntA + NE;
    uint32_t* start = cntB + NE;
    uint32_t* curA = start + NE;
    uint32_t* curB = curA + NE;
    uint32_t* mts = curB + NE;
    uint32_t* real = mts + NE;
    uint32_t* load = real + NE;
    uint32_t* nu = load + NC;
    volatile uint32_t* flags =
        nu + NC;  // [0] zeroed/ready, [1] pass 1 of B done, [2] cursors ready, [3] pass 2 of B done
    const uint32_t pages_half = TT / 64;

    if (ROLE == 0) {
        flags[1] = 0;
        flags[2] = 0;
        flags[3] = 0;
        for (uint32_t e = 0; e < NE; ++e) {
            cntA[e] = 0;
            cntB[e] = 0;
        }
        for (uint32_t c = 0; c < NC; ++c) {
            nu[c] = 0;
        }
        flags[0] = 0x5A5A5A5Au;
        for (uint32_t pg = 0; pg < pages_half; ++pg) {
            noc_async_read(ra.get_noc_addr(pg), base + pg * 2048, 2048);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < HALF; ++t) {
            const uint32_t* r = route + t * 16;
            for (uint32_t j = 0; j < 8; ++j) {
                cntA[r[j]] += 1;
            }
        }
        while (flags[1] != 0x5A5A5A5Au) {
        }
        uint32_t s = 0, maxm = 0;
        for (uint32_t e = 0; e < NE; ++e) {
            const uint32_t ca = cntA[e], c = ca + cntB[e];
            const uint32_t m = (c + 31) >> 5;
            const uint32_t mp = (m + 1) & ~1u;
            start[e] = s;
            curA[e] = s * 32;
            curB[e] = s * 32 + ca;
            mts[e] = mp;
            real[e] = m;
            for (uint32_t row = s * 32 + c; row < (s + mp) * 32; ++row) {
                gidx[row] = 0;
                dest[row] = 0xFFFFFFFFu;
                scr[row] = 0;
            }
            s += mp;
            if (mp > maxm) {
                maxm = mp;
            }
        }
        flags[2] = 0x5A5A5A5Au;
        for (uint32_t t = 0; t < HALF; ++t) {
            const uint32_t* r = route + t * 16;
            const uint32_t dbase = ((t >> 5) << 16) | (t & 31);
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t e = r[j];
                const uint32_t row = curA[e];
                curA[e] = row + 1;
                gidx[row] = t;
                dest[row] = dbase | (j << 8);
                scr[row] = (uint16_t)(r[8 + j] & 0xFFFF);
            }
        }
        // LPT (heap over (load, core), largest unit first, ties by expert id) while ROLE 1 finishes its scatter
        uint32_t* heap = load;
        for (uint32_t c = 0; c < NC; ++c) {
            heap[c] = c;
        }
        for (uint32_t m = maxm; m >= 2; m -= 2) {
            for (uint32_t e = 0; e < NE; ++e) {
                if (mts[e] != m) {
                    continue;
                }
                const uint32_t key = heap[0];
                const uint32_t bc = key & 127, bl = key >> 7;
                const uint32_t k = nu[bc];
                uint32_t* pc = plan + bc * 512;
                pc[1 + 3 * k] = e;
                pc[2 + 3 * k] = start[e];
                pc[3 + 3 * k] = m;
                pc[304 + k] = real[e];
                nu[bc] = k + 1;
                const uint32_t nk = ((bl + m * 10 + 8) << 7) | bc;
                uint32_t i = 0;
                while (true) {
                    uint32_t l = 2 * i + 1;
                    if (l >= NC) {
                        break;
                    }
                    uint32_t r = l + 1;
                    uint32_t sm = (r < NC && heap[r] < heap[l]) ? r : l;
                    if (heap[sm] < nk) {
                        heap[i] = heap[sm];
                        i = sm;
                    } else {
                        break;
                    }
                }
                heap[i] = nk;
            }
        }
        for (uint32_t c = 0; c < NC; ++c) {
            plan[c * 512] = nu[c];
        }
        while (flags[3] != 0x5A5A5A5Au) {
        }
        const auto pa = TensorAccessor(p_args, get_common_arg_val<uint32_t>(1), 2048);
        const auto ga = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2), 2048);
        const auto da = TensorAccessor(d_args, get_common_arg_val<uint32_t>(3), 2048);
        const auto sa = TensorAccessor(s_args, get_common_arg_val<uint32_t>(4), 2048);
        for (uint32_t c = 0; c < NC; ++c) {
            noc_async_write(reinterpret_cast<uint32_t>(plan) + c * 2048, pa.get_noc_addr(c), 2048);
        }
        for (uint32_t pg = 0; pg < ROWS / 512; ++pg) {
            noc_async_write(reinterpret_cast<uint32_t>(gidx) + pg * 2048, ga.get_noc_addr(pg), 2048);
            noc_async_write(reinterpret_cast<uint32_t>(dest) + pg * 2048, da.get_noc_addr(pg), 2048);
        }
        for (uint32_t pg = 0; pg < ROWS / 1024; ++pg) {
            noc_async_write(reinterpret_cast<uint32_t>(scr) + pg * 2048, sa.get_noc_addr(pg), 2048);
        }
        noc_async_write_barrier();
        // clear the hand-shake flags: L1 survives to the next launch (a stale magic would let ROLE 1 start before ROLE
        // 0 zeroed the counters)
        flags[0] = 0;
        flags[1] = 0;
        flags[2] = 0;
        flags[3] = 0;
    } else {
        while (flags[0] != 0x5A5A5A5Au) {
        }
        for (uint32_t pg = pages_half; pg < TT / 32; ++pg) {
            noc_async_read(ra.get_noc_addr(pg), base + pg * 2048, 2048);
        }
        noc_async_read_barrier();
        for (uint32_t t = HALF; t < TT; ++t) {
            const uint32_t* r = route + t * 16;
            for (uint32_t j = 0; j < 8; ++j) {
                cntB[r[j]] += 1;
            }
        }
        flags[1] = 0x5A5A5A5Au;
        while (flags[2] != 0x5A5A5A5Au) {
        }
        for (uint32_t t = HALF; t < TT; ++t) {
            const uint32_t* r = route + t * 16;
            const uint32_t dbase = ((t >> 5) << 16) | (t & 31);
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t e = r[j];
                const uint32_t row = curB[e];
                curB[e] = row + 1;
                gidx[row] = t;
                dest[row] = dbase | (j << 8);
                scr[row] = (uint16_t)(r[8 + j] & 0xFFFF);
            }
        }
        flags[3] = 0x5A5A5A5Au;
    }
}
