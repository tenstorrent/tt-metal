// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused mHC collapse + RMSNorm, reader (NCRISC).  NC cores (a physically contiguous worker rectangle); core k owns
// column tiles [k*GPC, (k+1)*GPC).  Token groups of 8 tokens are stacked into one x tile (row 4*tl + i = stream i of
// token tl) so that
//   h tile (rows = tokens) = sum_g A_g @ X_(g, j),   A_g[t, 4*tl + i] = pre[t, i]
// is ONE matmul per (group, column tile).  Phases:
//  1. issue the pre reads and the weight reads, zero the A tiles / unused x rows, issue the x reads, ONE barrier, build
//  A
//  2. exchange: wait for this core's partial sum of squares (compute), write its T values (+ flag word) into slot[k] of
//  every other
//     core's L1 (unicast), wait for all flags, transpose the slots into the tile R[t, k] (compute does R @ ones, + eps,
//     rsqrt)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_pre = get_compile_time_arg_val(3);   // scratch: T * 64 B
    constexpr uint32_t cb_p = get_compile_time_arg_val(4);     // partial sums (from compute)
    constexpr uint32_t cb_slot = get_compile_time_arg_val(5);  // NC slots
    constexpr uint32_t cb_tot = get_compile_time_arg_val(6);
    constexpr uint32_t T = get_compile_time_arg_val(7);
    constexpr uint32_t NT = get_compile_time_arg_val(8);
    constexpr uint32_t G = get_compile_time_arg_val(9);  // token groups of 8
    constexpr uint32_t GPC = get_compile_time_arg_val(10);
    constexpr uint32_t NC = get_compile_time_arg_val(11);
    constexpr uint32_t MX0 = get_compile_time_arg_val(12);
    constexpr uint32_t MY0 = get_compile_time_arg_val(13);
    constexpr uint32_t MW = get_compile_time_arg_val(14);  // rectangle width
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto p_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;
    constexpr uint32_t SW = 36;  // slot words: T data words (<= 32), pad, flag
    constexpr uint32_t FLAG = 0xC0DE1234u;

    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t j0 = k * GPC;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    const auto w_acc = TensorAccessor(w_args, w_addr, TILE);
    experimental::CB cba(cb_a), cbx(cb_x), cbw(cb_w), cbpre(cb_pre), cbp(cb_p), cbslot(cb_slot), cbtot(cb_tot);

    cbslot.reserve_back(1);
    const uint32_t slot_base = cbslot.get_write_ptr();
    volatile tt_l1_ptr uint32_t* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_base);
    // flags are reset at the END of every run (a start-of-run reset could race with an early sender)
    cbtot.reserve_back(1);
    cbpre.reserve_back(1);
    cbw.reserve_back(GPC);
    cba.reserve_back(G);
    cbx.reserve_back(GPC * G);
    for (uint32_t t = 0; t < T; ++t) {
        noc.async_read(p_acc, cbpre, 64, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = t * 64});
    }
    for (uint32_t g = 0; g < GPC; ++g) {
        noc.async_read(w_acc, cbw, TILE, {.page_id = j0 + g, .offset_bytes = 0}, {.offset_bytes = g * TILE});
    }
    noc.async_write_zeros(cba, G * TILE, {.offset_bytes = 0});
    noc.async_write_zeros(cbtot, TILE, {.offset_bytes = 0});
    if constexpr (T % 8 != 0) {  // last group: rows 4*(T%8) .. 31 unused (T == 4: faces 2 and 3)
        for (uint32_t gi = 0; gi < GPC; ++gi) {
            noc.async_write_zeros(cbx, 2048, {.offset_bytes = (gi * G + G - 1) * TILE + 2048});
        }
    }
    noc.write_zeros_l1_barrier();
    for (uint32_t gi = 0; gi < GPC; ++gi) {
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t dst = (gi * G + (t >> 3)) * TILE + RO(4 * (t & 7));
            const uint32_t page = t * NT + j0 + gi;
            noc.async_read(x_acc, cbx, 256, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = dst});
            noc.async_read(x_acc, cbx, 256, {.page_id = page, .offset_bytes = 1024}, {.offset_bytes = dst + 1024});
        }
    }
    noc.async_read_barrier();
    {
        volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbpre.get_write_ptr());
        volatile tt_l1_ptr uint32_t* A = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
        for (uint32_t t = 0; t < T; ++t) {
            volatile tt_l1_ptr uint32_t* At = A + (t >> 3) * (TILE / 4) + RW(t);
            const uint32_t c0 = 4 * (t & 7);
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t c = c0 + i;
                At[((c >> 4) << 8) + (c & 15)] = sc[t * 16 + i];
            }
        }
    }
    cba.push_back(G);
    cbw.push_back(GPC);
    cbx.push_back(GPC * G);

    // ---- exchange of the partial sums of squares ----
    cbp.wait_front(1);
    volatile tt_l1_ptr uint32_t* pp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbp.get_read_ptr());
    for (uint32_t t = 0; t < T; ++t) {
        slots[k * SW + t] = pp[RW(t)];
    }
    slots[k * SW + SW - 1] = FLAG;
    cbp.pop_front(1);
    for (uint32_t d = 1; d < NC; ++d) {
        const uint32_t kk = (k + d) % NC;
        const uint64_t dst = get_noc_addr(MX0 + (kk % MW), MY0 + kk / MW, slot_base + k * SW * 4);
        noc_async_write(slot_base + k * SW * 4, dst, SW * 4);
    }
    for (uint32_t kk = 0; kk < NC; ++kk) {
        // bounded spin: a lost/late flag must not wedge the device (the result is then wrong and the test catches it)
        for (uint32_t spin = 0; spin < 20000000u; ++spin) {
            invalidate_l1_cache();
            if (slots[kk * SW + SW - 1] == FLAG) {
                break;
            }
        }
    }
    volatile tt_l1_ptr uint32_t* R = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbtot.get_write_ptr());
    for (uint32_t t = 0; t < T; ++t) {
        volatile tt_l1_ptr uint32_t* Rt = R + RW(t);
        volatile tt_l1_ptr uint32_t* st = slots + t;
#pragma GCC unroll 16
        for (uint32_t kk = 0; kk < 16; ++kk) {
            Rt[kk] = st[kk * SW];
        }
#pragma GCC unroll 16
        for (uint32_t kk = 16; kk < NC; ++kk) {
            Rt[256 + kk - 16] = st[kk * SW];
        }
    }
    cbtot.push_back(1);
    for (uint32_t kk = 0; kk < NC; ++kk) {
        slots[kk * SW + SW - 1] = 0;
    }
    noc_async_write_barrier();
}
