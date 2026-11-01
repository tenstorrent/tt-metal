// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC lane-parallel fused "expand + next front" (DSV41_MHC_LANEPAR), projection-core reader.  The attention-side expand
// of a layer
//     x2_j = post_j * y + sum_i comb[i, j] x_i
// is computed HERE, on the projection cores of the ffn-side front, for exactly the column tiles the core projects: the
// new stream tile never takes the DRAM round trip (expand write + front re-read) and the expand program is gone.  Core
// r owns the column tiles [r*TPJ, (r+1)*TPJ) of all four streams and ALL token groups.  Per (group g, column tile j)
// the B tile of mhc_expand5_reader is built
//     rows 4*tl + i : x[t, stream i, 32 cols] (2 x 256 B reads),   rows 16 + tl : y[t, 32 cols] (bf16 token rows ->
//     fp32)
// and multiplied by the group's A tile (A[4*tl + j][4*tl + i] = comb[t][i][j], A[4*tl + j][16 + tl] = post[t][j]; built
// here from the 256 B comb / post pieces): the compute kernel packs the product (rows 4*tl + j = new stream j of token
// tl: the X tile of the projection) into cb_x. This reader then (a) writes the two valid 256 B pieces of every new tile
// into the pooled stream buffer (the residual of the ffn-side expand) and (b) forwards the tile quads to the
// collapse+norm cores (as mhc_proj5_reader).  Order of the DRAM reads: B(group 0) first (small), then the weights, then
// B of the later groups.  Arithmetic: the same fp32 tile matmul as mhc_expand5 (K slots not used are exact zeros),
// bit-identical to expand + front.

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);  // new x tiles (written by the compute kernel)
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t TG = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);   // 32-col tiles per stream row (D/32)
    constexpr uint32_t TPJ = get_compile_time_arg_val(4);  // column tiles per core
    constexpr uint32_t NGRP = get_compile_time_arg_val(5);
    constexpr uint32_t cb_fx = get_compile_time_arg_val(6);
    constexpr uint32_t SEM2 = get_compile_time_arg_val(7);
    constexpr uint32_t GPC = get_compile_time_arg_val(8);
    constexpr uint32_t GG = get_compile_time_arg_val(9);
    constexpr uint32_t MX0 = get_compile_time_arg_val(10);
    constexpr uint32_t MY0 = get_compile_time_arg_val(11);
    constexpr uint32_t MW = get_compile_time_arg_val(12);
    constexpr uint32_t cb_b = get_compile_time_arg_val(13);
    constexpr uint32_t cb_a = get_compile_time_arg_val(14);
    constexpr uint32_t cb_ys = get_compile_time_arg_val(15);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(16);
    constexpr auto x_args = TensorAccessorArgs<17>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t y_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(5);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto w_acc = TensorAccessor(w_args, w_addr, TILE);
    const auto y_acc = TensorAccessor(y_args, y_addr, 2048);  // bf16 tile pages
    const auto c_acc = TensorAccessor(c_args, c_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    const auto o_acc = TensorAccessor(o_args, o_addr, TILE);
    experimental::CB cbx(cb_x), cbw(cb_w), cbb(cb_b), cba(cb_a), cbys(cb_ys), cbcs(cb_cs);

    DeviceZoneScopedN("PR_ALL");
    cbb.reserve_back(NGRP * TPJ);
    cba.reserve_back(NGRP);
    cbys.reserve_back(1);
    cbcs.reserve_back(1);
    cbw.reserve_back(4 * TPJ);

    // zero what no read overwrites: B rows of absent tokens (faces 0/1 tails), the whole y half (faces 2/3: the y rows
    // are written after the reads), and the A tiles
    for (uint32_t g = 0; g < NGRP; ++g) {
        const uint32_t nv = get_arg_val<uint32_t>(1 + g);
        for (uint32_t j = 0; j < TPJ; ++j) {
            const uint32_t tb = (g * TPJ + j) * TILE;
            if (nv < 4) {
                noc.async_write_zeros(cbb, 1024 - nv * 256, {.offset_bytes = tb + nv * 256});
                noc.async_write_zeros(cbb, 1024 - nv * 256, {.offset_bytes = tb + 1024 + nv * 256});
            }
            noc.async_write_zeros(cbb, 2048, {.offset_bytes = tb + 2048});
        }
    }
    noc.async_write_zeros(cba, NGRP * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();

    // B tile reads of group g (x pieces, raw bf16 y rows, comb / post pieces), transaction id 2 + g
    auto issue_b = [&](uint32_t g) {
        const uint32_t nv = get_arg_val<uint32_t>(1 + g);
        const uint32_t t0 = g * TG;
        const uint32_t base = ((t0 >> 4) << 10) + ((t0 & 15) << 5);
        for (uint32_t j = 0; j < TPJ; ++j) {
            const uint32_t jg = r * TPJ + j;
            const uint32_t bt = (g * TPJ + j) * TILE;
            for (uint32_t tl = 0; tl < nv; ++tl) {
                const uint32_t page = (t0 + tl) * NT + jg;
                noc.async_read<NocOptions::TXN_ID>(
                    x_acc,
                    cbb,
                    256,
                    {.page_id = page, .offset_bytes = 0},
                    {.offset_bytes = bt + tl * 256},
                    {.trid = 2 + g});
                noc.async_read<NocOptions::TXN_ID>(
                    x_acc,
                    cbb,
                    256,
                    {.page_id = page, .offset_bytes = 1024},
                    {.offset_bytes = bt + 1024 + tl * 256},
                    {.trid = 2 + g});
            }
            const uint32_t slot = (g * TPJ + j) * 2;
            noc.async_read<NocOptions::TXN_ID>(
                y_acc, cbys, 128, {.page_id = jg, .offset_bytes = base}, {.offset_bytes = slot * 128}, {.trid = 2 + g});
            noc.async_read<NocOptions::TXN_ID>(
                y_acc,
                cbys,
                128,
                {.page_id = jg, .offset_bytes = base + 512},
                {.offset_bytes = (slot + 1) * 128},
                {.trid = 2 + g});
        }
        for (uint32_t tl = 0; tl < nv; ++tl) {
            const uint32_t so = (g * 4 + tl) * 512;
            noc.async_read<NocOptions::TXN_ID>(
                c_acc, cbcs, 256, {.page_id = t0 + tl, .offset_bytes = 0}, {.offset_bytes = so}, {.trid = 2 + g});
            noc.async_read<NocOptions::TXN_ID>(
                p_acc, cbcs, 256, {.page_id = t0 + tl, .offset_bytes = 0}, {.offset_bytes = so + 256}, {.trid = 2 + g});
        }
    };
    issue_b(0);
    for (uint32_t n = 0; n < 4 * TPJ; ++n) {
        noc.async_read<NocOptions::TXN_ID>(
            w_acc, cbw, TILE, {.page_id = r * 4 * TPJ + n, .offset_bytes = 0}, {.offset_bytes = n * TILE}, {.trid = 1});
    }
    {
        DeviceZoneScopedN("PR_Wissued");
    }
    for (uint32_t g = 1; g < NGRP; ++g) {
        issue_b(g);
    }
    {
        DeviceZoneScopedN("PR_Xissued");
    }

    // per group: y rows (bf16 -> fp32) into the B tiles, the A tile, push both to the compute kernel
    volatile tt_l1_ptr uint32_t* B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* A = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
    volatile tt_l1_ptr uint32_t* S = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbys.get_write_ptr());
    volatile tt_l1_ptr uint32_t* SC = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbcs.get_write_ptr());
    for (uint32_t g = 0; g < NGRP; ++g) {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 2 + g});
        const uint32_t nv = get_arg_val<uint32_t>(1 + g);
        for (uint32_t j = 0; j < TPJ; ++j) {
            volatile tt_l1_ptr uint32_t* Bt = B + (g * TPJ + j) * (TILE / 4);
            for (uint32_t h = 0; h < 2; ++h) {
                volatile tt_l1_ptr uint32_t* src = S + ((g * TPJ + j) * 2 + h) * 32;  // 128 B slot = 4 rows x 8 words
                volatile tt_l1_ptr uint32_t* dst = Bt + 512 + 256 * h;
                for (uint32_t m = 0; m < nv; ++m) {
                    for (uint32_t e = 0; e < 8; ++e) {
                        const uint32_t w = src[m * 8 + e];
                        dst[m * 16 + 2 * e] = w << 16;
                        dst[m * 16 + 2 * e + 1] = w & 0xFFFF0000u;
                    }
                }
            }
        }
        volatile tt_l1_ptr uint32_t* At = A + g * (TILE / 4);
        for (uint32_t tl = 0; tl < nv; ++tl) {
            volatile tt_l1_ptr uint32_t* sc = SC + (g * 4 + tl) * 128;
            volatile tt_l1_ptr uint32_t* sp = sc + 64;
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t rr = 4 * tl + j;
                for (uint32_t i = 0; i < 4; ++i) {
                    At[rr * 16 + 4 * tl + i] = sc[i * 16 + j];
                }
                At[256 + rr * 16 + tl] = sp[j * 16];
            }
        }
        cba.push_back(1);
        cbb.push_back(TPJ);
    }
    {
        DeviceZoneScopedN("PR_Bdone");
    }

    // new tiles -> pooled stream buffer (valid pieces) and the collapse+norm cores
    experimental::CB cbf(cb_fx);
    const uint32_t xb =
        cbx.get_write_ptr();  // never advanced on this core: the base of the tiles the compute kernel pushes
    const uint32_t fxb = cbf.get_write_ptr();
    for (uint32_t g = 0; g < NGRP; ++g) {
        cbx.wait_front((g + 1) * TPJ);
        const uint32_t nv = get_arg_val<uint32_t>(1 + g);
        const uint32_t t0 = g * TG;
        for (uint32_t j = 0; j < TPJ; ++j) {
            const uint32_t jg = r * TPJ + j;
            const uint32_t k = jg / GPC;
            const uint32_t gi = jg - k * GPC;
            const uint32_t cx = MX0 + (k % MW);
            const uint32_t cy = MY0 + (k / MW);
            const uint32_t qg = g * (TG >> 2);  // quad 0 of the group (nv <= 4)
            const uint32_t src = xb + (g * TPJ + j) * TILE;
            const uint32_t dst = fxb + (gi * GG + (qg >> 1)) * TILE + (qg & 1) * 2048;
            noc_async_write(src, get_noc_addr(cx, cy, dst), 1024);
            noc_async_write(src + 1024, get_noc_addr(cx, cy, dst + 1024), 1024);
            for (uint32_t tl = 0; tl < nv; ++tl) {
                noc.async_write(
                    cbx,
                    o_acc,
                    256,
                    {.offset_bytes = (g * TPJ + j) * TILE + tl * 256},
                    {.page_id = (t0 + tl) * NT + jg, .offset_bytes = 0});
                noc.async_write(
                    cbx,
                    o_acc,
                    256,
                    {.offset_bytes = (g * TPJ + j) * TILE + 1024 + tl * 256},
                    {.page_id = (t0 + tl) * NT + jg, .offset_bytes = 1024});
            }
        }
        noc_async_writes_flushed();  // the forwarded data has left this core before the arrival counts follow it (same
                                     // route, in order)
        if (nv > 0) {
            for (uint32_t j = 0; j < TPJ; ++j) {
                const uint32_t jg = r * TPJ + j;
                const uint32_t k = jg / GPC;
                const uint32_t cx = MX0 + (k % MW);
                const uint32_t cy = MY0 + (k / MW);
                noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(SEM2)), 1);
            }
        }
    }
    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1});
    cbw.push_back(4 * TPJ);
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
