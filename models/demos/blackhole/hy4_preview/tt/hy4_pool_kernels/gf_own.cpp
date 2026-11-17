// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B35 (HY4_B35_OWN=1): cs_gather + id_filter + the id all-gather as ONE multi-core program, bit-identical to the stock
// chain. The candidate at gathered position p comes from ring column p / KL and its token lives on that column (layout
// invariant of cs_cand2 LAYOUT=1), so the column filter of id_filter keeps exactly the positions of this chip's own
// block [c*KL, (c+1)*KL): only the chip's OWN local ids are needed (no all-gather of the ids).
//   pos uint32 RM [S,1,1,K] top-k positions into the 8*KL gathered candidates   lids uint32 RM [S,1,1,KL] this chip's
//   local ids   colinfo (64 B page: 0 = column, 1 / 2+s = valid tokens)   out uint32 RM [1,1,OWN*S,K]
// Work split: row s -> NQ cores (x = q, y = s). Core (q, s) filters pos[q*KQ, (q+1)*KQ) into a compact list, ships it
// to the leader (q = 0) slot q (+ its count) and bumps the leader's semaphore; the leader concatenates the lists in
// order (same order as the sequential scan), fills the sentinel tail and writes the OWN replicas.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t OWN = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t PERROW = get_compile_time_arg_val(4);
    constexpr uint32_t KL = get_compile_time_arg_val(5);
    constexpr uint32_t NQ = get_compile_time_arg_val(6);
    constexpr auto p_args = TensorAccessorArgs<7>();
    constexpr auto g_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t RB = K * 4;
    constexpr uint32_t KQ = K / NQ;
    constexpr uint32_t SLOTB = KQ * 4;
    const uint32_t s = get_arg_val<uint32_t>(0), q = get_arg_val<uint32_t>(1);
    const uint32_t lx = get_arg_val<uint32_t>(2), ly = get_arg_val<uint32_t>(3);
    const uint32_t a_p = get_common_arg_val<uint32_t>(0), a_g = get_common_arg_val<uint32_t>(1),
                   a_c = get_common_arg_val<uint32_t>(2), a_o = get_common_arg_val<uint32_t>(3);
    Noc noc;
    const auto p_acc = TensorAccessor(p_args, a_p, RB);
    const auto g_acc = TensorAccessor(g_args, a_g, KL * 4);
    const auto c_acc = TensorAccessor(c_args, a_c, 64);
    const auto o_acc = TensorAccessor(o_args, a_o, RB);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    // L1 (identical on every core): [pos chunk SLOTB][slots NQ * SLOTB][out RB][lids KL*4][colinfo 64][counts NQ * 16]
    const uint32_t pos_b = base;
    const uint32_t slots_b = pos_b + SLOTB;
    const uint32_t out_b = slots_b + NQ * SLOTB;
    const uint32_t lid_b = out_b + RB;
    const uint32_t ci_b = lid_b + KL * 4;
    const uint32_t cnt_b = ci_b + 64;
    noc.async_read(p_acc, cb, SLOTB, {.page_id = s, .offset_bytes = q * SLOTB}, {.offset_bytes = 0});
    noc.async_read(g_acc, cb, KL * 4, {.page_id = s, .offset_bytes = 0}, {.offset_bytes = lid_b - base});
    noc.async_read(c_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = ci_b - base});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const uint32_t* __restrict pp = reinterpret_cast<const uint32_t*>(pos_b);
    const uint32_t* __restrict cc = reinterpret_cast<const uint32_t*>(lid_b);
    const uint32_t* __restrict ci = reinterpret_cast<const uint32_t*>(ci_b);
    uint32_t* __restrict lst = reinterpret_cast<uint32_t*>(slots_b + q * SLOTB);
    const uint32_t c = ci[0], nvalid = PERROW ? ci[2 + s] : ci[1];
    const uint32_t lo = c * KL;
    uint32_t n = 0;
#pragma GCC unroll 4
    for (uint32_t i = 0; i < KQ; ++i) {
        const uint32_t qq = pp[i] - lo;
        if (qq < KL) {
            const uint32_t t = cc[qq];
            if (t < nvalid && ((t >> 6) & 7u) == c) {
                lst[n++] = ((t >> 9) << 6) | (t & 63u);
            }
        }
    }
    const uint32_t sem_addr = get_semaphore(0);
    if (q != 0) {
        const uint64_t lead = get_noc_addr(lx, ly, 0);
        // list into the leader's slot q, count into the leader's count word q, then the semaphore
        volatile tt_l1_ptr uint32_t* mycnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_b + q * 16);
        mycnt[0] = n;
        if (n) {
            noc_async_write(slots_b + q * SLOTB, get_noc_addr(lx, ly, slots_b + q * SLOTB), n * 4);
        }
        noc_async_write(cnt_b + q * 16, get_noc_addr(lx, ly, cnt_b + q * 16), 16);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lx, ly, sem_addr), 1);
        noc_async_atomic_barrier();
        (void)lead;
        return;
    }
    // leader
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, NQ - 1);
    invalidate_l1_cache();
    uint32_t* __restrict op = reinterpret_cast<uint32_t*>(out_b);
    const uint32_t* __restrict cnts = reinterpret_cast<const uint32_t*>(cnt_b);
    uint32_t tot = 0;
    for (uint32_t j = 0; j < NQ; ++j) {
        const uint32_t nj = (j == 0) ? n : cnts[j * 4];
        const uint32_t* __restrict src = reinterpret_cast<const uint32_t*>(slots_b + j * SLOTB);
        for (uint32_t i = 0; i < nj; ++i) {
            op[tot + i] = src[i];
        }
        tot += nj;
    }
#pragma GCC unroll 8
    for (uint32_t i = tot; i < K; ++i) {
        op[i] = 0xFFFFFFFFu;
    }
    for (uint32_t j = 0; j < OWN; ++j) {
        noc.async_write(cb, o_acc, RB, {.offset_bytes = out_b - base}, {.page_id = j * S + s, .offset_bytes = 0});
    }
    noc.async_write_barrier();
}
