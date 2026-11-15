// SPDX-License-Identifier: Apache-2.0
// tp32pf device plan builder, ONE core: route words (per token 16 x int32: 8 expert ids, 8 words whose low 16 bits are
// the bf16 score bits; [Tt / 32 pages x 512 words]) -> counting sort by expert (stable in (token, slot)), every expert
// padded to an even number of 32-row M-tiles, then the LPT assignment of the experts (units) to NC cores (same
// algorithm as hy4_tp32pf_plan.build_plan):
//   plan [NC pages of 512 words]: [nu, (e, mt0, mtc) x nu, ..., NREAL_OFF (304) + u = real M-tiles of unit u]
//   gidx [CAP / 16 pages of 512 words]: token of every padded row (pad 0)
//   dest [CAP / 16 pages]: 0xFFFFFFFF pad, else (token / 32) << 16 | slot << 8 | token % 32
//   scr  [CAP / 32 pages of 1024 u16]: bf16 score bits of every padded row (pad 0)
// CT: 0 cb_arr, 1 TT, 2 NC, 3 CAP, 4 NPAGES(cb_arr), TensorAccessorArgs(route), (plan), (gidx), (dest), (scr).
// Common: 0 route, 1 plan, 2 gidx, 3 dest, 4 scr
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_arr = get_compile_time_arg_val(0);
    constexpr uint32_t TT = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t CAP = get_compile_time_arg_val(3);
    constexpr uint32_t NPAGES = get_compile_time_arg_val(4);
    constexpr uint32_t MODE = get_compile_time_arg_val(5);
    constexpr auto r_args = TensorAccessorArgs<6>();
    constexpr auto p_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr uint32_t NE = 256, ROWS = CAP * 32;
    const auto ra = TensorAccessor(r_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto pa = TensorAccessor(p_args, get_common_arg_val<uint32_t>(1), 2048);
    const auto ga = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2), 2048);
    const auto da = TensorAccessor(d_args, get_common_arg_val<uint32_t>(3), 2048);
    const auto sa = TensorAccessor(s_args, get_common_arg_val<uint32_t>(4), 2048);

    cb_reserve_back(cb_arr, NPAGES);
    const uint32_t base = get_write_ptr(cb_arr);
    uint32_t* route = reinterpret_cast<uint32_t*>(base);
    uint32_t* gidx = route + TT * 16;
    uint32_t* dest = gidx + ROWS;
    uint16_t* scr = reinterpret_cast<uint16_t*>(dest + ROWS);
    uint32_t* plan = reinterpret_cast<uint32_t*>(scr + ROWS);
    uint32_t* cnt = plan + NC * 512;
    uint32_t* start = cnt + NE;
    uint32_t* cur = start + NE;
    uint32_t* mts = cur + NE;
    uint32_t* real = mts + NE;
    uint32_t* load = real + NE;
    uint32_t* nu = load + NC;  // load[] doubles as the LPT heap

    for (uint32_t pg = 0; pg < TT / 32; ++pg) {
        noc_async_read(ra.get_noc_addr(pg), base + pg * 2048, 2048);
    }
    for (uint32_t e = 0; e < NE; ++e) {
        cnt[e] = 0;
    }
    for (uint32_t c = 0; c < NC; ++c) {
        nu[c] = 0;
    }
    noc_async_read_barrier();
    // pass 1: histogram
    for (uint32_t t = 0; t < ((MODE & 4) ? 0 : TT); ++t) {
        const uint32_t* r = route + t * 16;
        for (uint32_t j = 0; j < 8; ++j) {
            cnt[r[j]] += 1;
        }
    }
    uint32_t s = 0, maxm = 0;
    for (uint32_t e = 0; e < NE; ++e) {
        const uint32_t m = (cnt[e] + 31) >> 5;
        const uint32_t mp = (m + 1) & ~1u;
        start[e] = s;
        cur[e] = s * 32;
        mts[e] = mp;
        real[e] = m;
        s += mp;
        if (mp > maxm) {
            maxm = mp;
        }
        // pad rows of this expert's block (the rest of its last real M-tile and the padding M-tile): token 0, no
        // destination, score 0
        for (uint32_t row = start[e] * 32 + cnt[e]; row < (start[e] + mp) * 32; ++row) {
            gidx[row] = 0;
            dest[row] = 0xFFFFFFFFu;
            scr[row] = 0;
        }
    }
    // rows beyond the last expert block up to the cap are never read (K1/K2 only touch M-tiles of units)
    // pass 2: scatter (stable in (token, slot))
    for (uint32_t t = 0; t < ((MODE & 2) ? 0 : TT); ++t) {
        const uint32_t* r = route + t * 16;
        const uint32_t dbase = ((t >> 5) << 16) | (t & 31);
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t e = r[j];
            const uint32_t row = cur[e];
            cur[e] = row + 1;
            gidx[row] = t;
            dest[row] = dbase | (j << 8);
            scr[row] = (uint16_t)(r[8 + j] & 0xFFFF);
        }
    }
    // LPT over the units (experts), largest first, ties by expert id; load = 10 * M-tiles + 8. Min-heap over (load,
    // core): key = load * 128 + core.
    uint32_t* heap = load;  // reuse: heap[i] = key
    for (uint32_t c = 0; c < NC; ++c) {
        heap[c] = c;  // load 0
    }
    for (uint32_t m = ((MODE & 1) ? 0 : maxm); m >= 2; m -= 2) {
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
            // replace the root by the updated key and sift down
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
    for (uint32_t c = 0; c < NC; ++c) {
        noc_async_write(base + (plan - route) * 4 + c * 2048, pa.get_noc_addr(c), 2048);
    }
    for (uint32_t pg = 0; pg < ROWS / 512; ++pg) {
        noc_async_write(reinterpret_cast<uint32_t>(gidx) + pg * 2048, ga.get_noc_addr(pg), 2048);
        noc_async_write(reinterpret_cast<uint32_t>(dest) + pg * 2048, da.get_noc_addr(pg), 2048);
    }
    for (uint32_t pg = 0; pg < ROWS / 1024; ++pg) {
        noc_async_write(reinterpret_cast<uint32_t>(scr) + pg * 2048, sa.get_noc_addr(pg), 2048);
    }
    noc_async_write_barrier();
}
