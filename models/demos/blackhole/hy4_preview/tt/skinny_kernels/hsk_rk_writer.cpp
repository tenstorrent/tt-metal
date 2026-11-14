// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4-Preview router K-split BRISC (NC = 8 DRAM-bank cores, core b): x K slice (tiles [KL b, KL b + KL), std 32x32 tiles, faces 0 / 1 rows 0..15) -> cb_x tiny
// [16, 32] tiles; bias tile b -> cb_bias; the fp32 partial tiles from cb_pout -> slot b of core n's cb_red (tile n -> core n), semaphore 0 counts the NC arrivals;
// cb_score / cb_rank (fp32 [8, 32], rows 0..7) -> page b of the fp32 std-tile outputs (faces 0 / 1: rows 0..7).
// CT args: 0 cb_x, 1 cb_bias, 2 cb_score, 3 cb_rank, 4 NC, 5 KL, 6 cb_pout, 7 cb_red, 8 PW, then TensorAccessorArgs(x), (bias), (score), (rank).
// RT args: 0 core index b. Common: 0 x addr, 1 bias addr, 2 score addr, 3 rank addr, 4.. physical (x, y) of the NC cores.

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
    constexpr auto x_args = TensorAccessorArgs<10>();
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
    const auto s_acc = TensorAccessor(s_args, s_addr, 4096);
    const auto r_acc = TensorAccessor(r_args, r_addr, 4096);
    experimental::CB cbx(cb_x), cbb(cb_bias), cbs(cb_score), cbr(cb_rank), cbp(cb_pout), cbred(cb_red);
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
        DeviceZoneScopedN("B_OUT");
        cbs.wait_front(1);
        cbr.wait_front(1);
        // fp32 [8, 32] tiny tile: face 0 = rows 0..7 x 16 x 4 B = 512 B, face 1 at +512; std fp32 tile: face 0 at 0, face 1 at 1024
        noc.async_write(cbs, s_acc, 512, {.offset_bytes = 0}, {.page_id = me, .offset_bytes = 0});
        noc.async_write(cbs, s_acc, 512, {.offset_bytes = 512}, {.page_id = me, .offset_bytes = 1024});
        noc.async_write(cbr, r_acc, 512, {.offset_bytes = 0}, {.page_id = me, .offset_bytes = 0});
        noc.async_write(cbr, r_acc, 512, {.offset_bytes = 512}, {.page_id = me, .offset_bytes = 1024});
        noc.async_write_barrier();
        cbs.pop_front(1);
        cbr.pop_front(1);
    }
}
