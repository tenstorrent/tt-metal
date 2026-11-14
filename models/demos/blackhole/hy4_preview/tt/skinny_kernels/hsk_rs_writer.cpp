// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4-Preview router K-split + SELECT BRISC (hsk_rk_writer.cpp with the router_select fused: no rank / score tensors)
// Hy4-Preview router K-split BRISC (NC = 8 DRAM-bank cores, core b): x K slice (tiles [KL b, KL b + KL), std 32x32 tiles, faces 0 / 1 rows 0..15) -> cb_x tiny
// [16, 32] tiles; bias tile b -> cb_bias; the fp32 partial tiles from cb_pout -> slot b of core n's cb_red (tile n -> core n), semaphore 0 counts the NC arrivals;
// cb_score / cb_rank (fp32 [8, 32], rows 0..7) -> page b of the fp32 std-tile outputs (faces 0 / 1: rows 0..7).
// Select (exact fp32 top-K of rank[t, :], lowest index wins ties; w = bf16(score_i / (sum score + eps) * scale), the formulas of the router_select program): core b scans its 32 experts of every token row t < T
// (local top-K), sends the K (key, index, score) records of row t to core t (slot b), core t merges the NC x K candidates and writes row t of the outputs (weights bf16 [T,1,1,K], indices uint16 [T,1,1,K], 16 B pages).
// CT args: 0 cb_x, 1 cb_bias, 2 cb_score, 3 cb_rank, 4 NC, 5 KL, 6 cb_pout, 7 cb_red, 8 PW, 9 XR, 10 cb_sel (recv: NC slots x 128 B), 11 cb_stg (T slots x 128 B), 12 T, 13 eps bits, 14 scale bits,
//   then TensorAccessorArgs(x), (bias), (w out), (idx out).
// RT args: 0 core index b. Common: 0 x addr, 1 bias addr, 2 w addr, 3 idx addr, 4.. physical (x, y) of the NC cores. Semaphores: 0 partial arrivals, 1 select records.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_score = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rank = get_compile_time_arg_val(3);
    constexpr uint32_t NC = get_compile_time_arg_val(4);
    constexpr uint32_t KL = get_compile_time_arg_val(5);
    constexpr uint32_t cb_pout = get_compile_time_arg_val(6);
    constexpr uint32_t cb_red = get_compile_time_arg_val(7);
    constexpr uint32_t PW = get_compile_time_arg_val(8);
    constexpr uint32_t XR = get_compile_time_arg_val(9);  // rows of the x tiles (cb_x): 8 or 16
    constexpr uint32_t XTB = 64 * XR, XFB = 32 * XR;
    constexpr uint32_t cb_sel = get_compile_time_arg_val(10);
    constexpr uint32_t cb_stg = get_compile_time_arg_val(11);
    constexpr uint32_t TT = get_compile_time_arg_val(12);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(13);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(14);
    constexpr uint32_t K = 8;
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto b_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t r_addr = get_common_arg_val<uint32_t>(3);
    auto nx = [&](uint32_t d) { return get_common_arg_val<uint32_t>(4 + 2 * d); };
    auto ny = [&](uint32_t d) { return get_common_arg_val<uint32_t>(5 + 2 * d); };

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, 2048);
    const auto b_acc = TensorAccessor(b_args, b_addr, 4096);
    const auto s_acc = TensorAccessor(s_args, s_addr, K * 2);  // weights out [T,1,1,K] bf16 pages
    const auto r_acc = TensorAccessor(r_args, r_addr, K * 2);  // indices out [T,1,1,K] uint16 pages
    experimental::CB cbx(cb_x), cbb(cb_bias), cbs(cb_score), cbr(cb_rank), cbp(cb_pout), cbred(cb_red), cbsel(cb_sel), cbstg(cb_stg);
    volatile tt_l1_ptr uint32_t* sem_sel = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint32_t sem_sel_addr = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* sem_red = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    const uint32_t sem_red_addr = get_semaphore(0);

    cbred.reserve_back(NC);
    const uint32_t red_base = cbred.get_write_ptr();
    {
        DeviceZoneScopedN("B_XLOAD");
        cbx.reserve_back(KL);
        cbb.reserve_back(1);
        for (uint32_t j = 0; j < KL; ++j) {
            const uint32_t i = me * KL + j;
            if constexpr (XR == 16) {  // faces 0 and 1 of a std tile are the first 1024 B of the page = the [16, 32] tile
                noc.async_read(x_acc, cbx, 1024, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = j * 1024});
            } else {
                noc.async_read(x_acc, cbx, XFB, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = j * XTB});
                noc.async_read(x_acc, cbx, XFB, {.page_id = i, .offset_bytes = 512}, {.offset_bytes = j * XTB + XFB});
            }
        }
        // bias: std fp32 tile page -> fp32 tiny [8, 32] tile (face 0 rows 0..7 at +0, face 1 rows 0..7 at +512), the same format as the partial / score tiles
        noc.async_read(b_acc, cbb, 512, {.page_id = me, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read(b_acc, cbb, 512, {.page_id = me, .offset_bytes = 1024}, {.offset_bytes = 512});
        noc.async_read_barrier();
        cbx.push_back(KL);
        cbb.push_back(1);
    }
    for (uint32_t pass = 0; pass < NC / PW; ++pass) {
        DeviceZoneScopedN("B_PSEND");
        cbp.wait_front(PW);
        const uint32_t src = cbp.get_read_ptr();
        for (uint32_t j = 0; j < PW; ++j) {
            const uint32_t n = pass * PW + j;
            noc_async_write(src + j * 1024, get_noc_addr(nx(n), ny(n), red_base + me * 1024), 1024);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < PW; ++j) {
            const uint32_t n = pass * PW + j;
            noc_semaphore_inc(get_noc_addr(nx(n), ny(n), sem_red_addr), 1);
        }
        noc_async_atomic_barrier();
        cbp.pop_front(PW);
    }
    {
        DeviceZoneScopedN("B_PWAIT");
        noc_semaphore_wait(sem_red, NC);
        *sem_red = 0;
        cbred.push_back(NC);
    }
    {
        DeviceZoneScopedN("B_SEL");
        cbs.wait_front(1);
        cbr.wait_front(1);
        cbsel.reserve_back(1);
        cbstg.reserve_back(1);
        const uint32_t sel_base = cbsel.get_write_ptr();
        volatile tt_l1_ptr uint32_t* rk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbr.get_read_ptr());
        volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_read_ptr());
        volatile tt_l1_ptr uint32_t* stg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbstg.get_write_ptr());
        {
        DeviceZoneScopedN("S_LOCAL");
        for (uint32_t t = 0; t < TT; ++t) {
            // fp32 [8, 32] tiny tile: face 0 = 8 rows x 16 x 4 B at +0, face 1 at +512: element (row t, col c) at word (c < 16 ? 16 t + c : 128 + 16 t + c - 16)
            uint32_t bk[K], bi[K];
            for (uint32_t j = 0; j < K; ++j) {
                bk[j] = 0;
                bi[j] = 0;
            }
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t w = (c < 16) ? 16 * t + c : 128 + 16 * t + c - 16;
                const uint32_t b = rk[w];
                const uint32_t key = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
                if (key > bk[K - 1]) {
                    uint32_t p = K - 1;
                    while (p > 0 && key > bk[p - 1]) {
                        bk[p] = bk[p - 1];
                        bi[p] = bi[p - 1];
                        --p;
                    }
                    bk[p] = key;
                    bi[p] = c;
                }
            }
            volatile tt_l1_ptr uint32_t* rec = stg + t * 32;  // 128 B record: keys[8], global idx[8], score bits[8]
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t c = bi[j];
                const uint32_t w = (c < 16) ? 16 * t + c : 128 + 16 * t + c - 16;
                rec[j] = bk[j];
                rec[8 + j] = 32 * me + c;
                rec[16 + j] = sc[w];
            }
        }
        }
        {
        DeviceZoneScopedN("S_SEND");
        for (uint32_t t = 0; t < TT; ++t) {
            noc_async_write(cbstg.get_write_ptr() + t * 128, get_noc_addr(nx(t), ny(t), sel_base + me * 128), 96);
        }
        noc_async_write_barrier();
        for (uint32_t t = 0; t < TT; ++t) {
            noc_semaphore_inc(get_noc_addr(nx(t), ny(t), sem_sel_addr), 1);
        }
        noc_async_atomic_barrier();
        }
        if (me < TT) {
            {
            DeviceZoneScopedN("S_WAIT");
            noc_semaphore_wait(sem_sel, NC);
            }
            DeviceZoneScopedN("S_MERGE");
            *sem_sel = 0;
            volatile tt_l1_ptr uint32_t* rv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sel_base);
            uint32_t bk[K], bi[K], bs[K];
            for (uint32_t j = 0; j < K; ++j) {
                bk[j] = 0;
                bi[j] = 0;
                bs[j] = 0;
            }
            for (uint32_t b = 0; b < NC; ++b) {  // ascending core = ascending expert index: strictly-greater insertion keeps the lowest index on ties
                for (uint32_t j = 0; j < K; ++j) {
                    const uint32_t key = rv[b * 32 + j];
                    if (key > bk[K - 1]) {
                        uint32_t p = K - 1;
                        while (p > 0 && key > bk[p - 1]) {
                            bk[p] = bk[p - 1];
                            bi[p] = bi[p - 1];
                            bs[p] = bs[p - 1];
                            --p;
                        }
                        bk[p] = key;
                        bi[p] = rv[b * 32 + 8 + j];
                        bs[p] = rv[b * 32 + 16 + j];
                    }
                }
            }
            union {
                uint32_t u;
                float f;
            } cv;
            float den = 0.0f;
            float sv[K];
            for (uint32_t j = 0; j < K; ++j) {
                cv.u = bs[j];
                sv[j] = cv.f;
                den += sv[j];
            }
            cv.u = EPS_BITS;
            const float inv_den = cv.f + den;
            cv.u = SCALE_BITS;
            const float qf = cv.f / inv_den;  // ONE division: w = bf16(s * (scale / (sum + eps))) (within 1 fp32 ulp of the router_select (s / (sum + eps)) * scale before the bf16 rounding)
            volatile tt_l1_ptr uint16_t* ow = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbstg.get_write_ptr() + 1024);  // scratch after the records
            volatile tt_l1_ptr uint16_t* oi = ow + 32;
            for (uint32_t j = 0; j < K; ++j) {
                cv.f = sv[j] * qf;
                const uint32_t b = cv.u;
                ow[j] = (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
                oi[j] = (uint16_t)bi[j];
            }
            noc.async_write(cbstg, s_acc, K * 2, {.offset_bytes = 1024}, {.page_id = me, .offset_bytes = 0});
            noc.async_write(cbstg, r_acc, K * 2, {.offset_bytes = 1024 + 64}, {.page_id = me, .offset_bytes = 0});
            noc.async_write_barrier();
        }
        cbs.pop_front(1);
        cbr.pop_front(1);
    }
}
