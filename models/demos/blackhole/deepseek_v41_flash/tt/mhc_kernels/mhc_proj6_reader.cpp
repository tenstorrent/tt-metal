// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC projection v5 (= v3 + forwarding of the gathered x rows to the collapse+norm cores, DSV41_MHC_FRONT_FAST),
// reader. (v3 text:) mHC projection v3 (NGRP token groups per core, weights read once), reader.  Core (g, r) owns the
// column-tile range [r*TPJ, (r+1)*TPJ) of ALL four streams and the token group g (TG tokens).  It gathers, for every
// column tile j of its range, one tile Xj with row 4*tl + i = x[tok0
// + tl, stream i, 32 columns] (rows 0..3 of the stream-row tile of token tok0 + tl, copied with two 256 B reads) and
// the 4*TPJ weight tiles.

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// row r of a 32x32 tile: byte offset of its first face half (fp32)
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t TG = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);   // 32-col tiles per stream row (D/32)
    constexpr uint32_t TPJ = get_compile_time_arg_val(4);  // column tiles per core
    constexpr uint32_t NGRP = get_compile_time_arg_val(5);
    constexpr uint32_t cb_fx =
        get_compile_time_arg_val(6);  // collapse+norm x tiles (same L1 address on the collapse+norm cores and here)
    constexpr uint32_t SEM2 = get_compile_time_arg_val(7);  // arrival counter of the collapse+norm cores
    constexpr uint32_t GPC = get_compile_time_arg_val(8);   // column tiles per collapse+norm core
    constexpr uint32_t GG = get_compile_time_arg_val(9);    // collapse+norm token groups of 8
    constexpr uint32_t MX0 =
        get_compile_time_arg_val(10);  // physical NOC x / y of collapse+norm core 0 and the rectangle width
    constexpr uint32_t MY0 = get_compile_time_arg_val(11);
    constexpr uint32_t MW = get_compile_time_arg_val(12);
    constexpr auto x_args = TensorAccessorArgs<13>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t r = get_arg_val<uint32_t>(0);
    // valid tokens of group g (token g*TG + tl): runtime args 1 + g (G = NGRP up to 4)
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto w_acc = TensorAccessor(w_args, w_addr, TILE);
    experimental::CB cbx(cb_x), cbw(cb_w);

    DeviceZoneScopedN("PR_ALL");
    cbx.reserve_back(NGRP * TPJ);
    cbw.reserve_back(4 * TPJ);
    // weights first (transaction id 1), then the x rows of group g (transaction id 2 + g): the compute of group g
    // starts when its rows have landed, not when all groups have
    // LANEPAR: the x rows (small) are issued FIRST (all groups at once), the weight tiles right behind them: the x rows
    // land and are forwarded to the collapse+norm cores while the weights stream; nothing waits for the weight barrier
    // before the x reads are issued zero only what the x reads below do not overwrite: the rows of absent tokens (faces
    // 2 / 3 of a 4-token group, the tail of a partial group)
    for (uint32_t g = 0; g < NGRP; ++g) {
        const uint32_t nv = get_arg_val<uint32_t>(1 + g);
        for (uint32_t j = 0; j < TPJ; ++j) {
            const uint32_t tb = (g * TPJ + j) * TILE;
            for (uint32_t q = 0; q < 2; ++q) {  // the two token quads of a tile (faces 0/1 and 2/3)
                const int32_t left = (int32_t)nv - 4 * (int32_t)q;
                const uint32_t nvq = (q * 4 < TG && left > 0) ? (left < 4 ? (uint32_t)left : 4u) : 0u;
                if (nvq == 0) {
                    noc.async_write_zeros(cbx, 2048, {.offset_bytes = tb + q * 2048});
                } else if (nvq < 4) {
                    noc.async_write_zeros(cbx, 1024 - nvq * 256, {.offset_bytes = tb + q * 2048 + nvq * 256});
                    noc.async_write_zeros(cbx, 1024 - nvq * 256, {.offset_bytes = tb + q * 2048 + 1024 + nvq * 256});
                }
            }
        }
    }
    // the weights go first (all cores): the DRAM serves the x rows of the groups behind them, so the compute of group g
    // can start when the rows of group g have landed
    noc.write_zeros_l1_barrier();
    {
        DeviceZoneScopedN("PR_zerodone");
    }
    // cbx.push_back advances the CB write pointer by TPJ tiles per group; the destination offsets of a group issued
    // after `pushed` pushes are corrected by that amount
    auto issue_group = [&](uint32_t g, uint32_t pushed) {
        const uint32_t adj = pushed * TPJ * TILE;
        const uint32_t nv = get_arg_val<uint32_t>(1 + g);
        for (uint32_t j = 0; j < TPJ; ++j) {
            for (uint32_t tl = 0; tl < nv; ++tl) {
                const uint32_t page = (g * TG + tl) * NT + r * TPJ + j;
                const uint32_t dst = (g * TPJ + j) * TILE;
                noc.async_read<NocOptions::TXN_ID>(
                    x_acc,
                    cbx,
                    256,
                    {.page_id = page, .offset_bytes = 0},
                    {.offset_bytes = dst + RO(4 * tl) - adj},
                    {.trid = 2 + g});
                noc.async_read<NocOptions::TXN_ID>(
                    x_acc,
                    cbx,
                    256,
                    {.page_id = page, .offset_bytes = 1024},
                    {.offset_bytes = dst + 1024 + RO(4 * tl) - adj},
                    {.trid = 2 + g});
            }
        }
    };
    for (uint32_t g = 0; g < NGRP; ++g) {
        issue_group(g, 0);
    }
    {
        DeviceZoneScopedN("PR_Xissued");
    }
    for (uint32_t n = 0; n < 4 * TPJ; ++n) {
        noc.async_read<NocOptions::TXN_ID>(
            w_acc, cbw, TILE, {.page_id = r * 4 * TPJ + n, .offset_bytes = 0}, {.offset_bytes = n * TILE}, {.trid = 1});
    }
    {
        DeviceZoneScopedN("PR_Wissued");
    }

    // ---- DSV41_MHC_FRONT_FAST: hand the gathered rows of every group to the collapse+norm cores as soon as they have
    // landed (their x layout is the same: row 4*(t%8)+i of tile (gi, t/8)), so they need no DRAM reads of x.  A token
    // quad (4 tokens = rows 0..15 of an x tile) is the faces 0 and 1 (or 2 and 3) of a tile: two 1 KB blocks per
    // (group, column tile, quad).
    experimental::CB cbf(cb_fx);
    const uint32_t xb = cbx.get_write_ptr();
    const uint32_t fxb = cbf.get_write_ptr();
    for (uint32_t g = 0; g < NGRP; ++g) {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 2 + g});
        cbx.push_back(TPJ);
        const uint32_t nv = get_arg_val<uint32_t>(1 + g);
        const uint32_t nq = (nv + 3) >> 2;
        for (uint32_t j = 0; j < TPJ; ++j) {
            const uint32_t jg = r * TPJ + j;
            const uint32_t k = jg / GPC;
            const uint32_t gi = jg - k * GPC;
            const uint32_t cx = MX0 + (k % MW);
            const uint32_t cy = MY0 + (k / MW);
            for (uint32_t q = 0; q < nq; ++q) {
                const uint32_t qg = g * (TG >> 2) + q;
                const uint32_t src = xb + (g * TPJ + j) * TILE + q * 2048;
                const uint32_t dst = fxb + (gi * GG + (qg >> 1)) * TILE + (qg & 1) * 2048;
                noc_async_write(src, get_noc_addr(cx, cy, dst), 1024);
                noc_async_write(src + 1024, get_noc_addr(cx, cy, dst + 1024), 1024);
            }
        }
        noc_async_writes_flushed();  // the data has left this core before the arrival counts follow it (same route, in
                                     // order)
        for (uint32_t j = 0; j < TPJ; ++j) {
            const uint32_t jg = r * TPJ + j;
            const uint32_t k = jg / GPC;
            const uint32_t cx = MX0 + (k % MW);
            const uint32_t cy = MY0 + (k / MW);
            for (uint32_t q = 0; q < nq; ++q) {
                noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(SEM2)), 1);
            }
        }
    }
    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1});
    cbw.push_back(4 * TPJ);
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
