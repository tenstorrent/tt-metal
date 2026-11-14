// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4-Preview router matmul BRISC (NC = 8 DRAM-bank cores, core b): x (std 32x32 DRAM / L1 tiles, rows 0..T-1) -> cb_x tiny [R, 32] tiles (every core
// converts KT / NC tiles and sends the slice to the others), bias tile b -> cb_bias, then cb_score / cb_rank (fp32 tiny [8, 32] tiles, rows 0..R-1) ->
// page b of the fp32 std-tile outputs (faces 0 / 1: rows 0..R-1).
// CT args: 0 cb_x, 1 cb_bias, 2 cb_score, 3 cb_rank, 4 NC, 5 KT, 6 R, then TensorAccessorArgs(x), (bias), (score), (rank).
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
    constexpr uint32_t KT = get_compile_time_arg_val(5);
    constexpr uint32_t R = get_compile_time_arg_val(6);
    constexpr uint32_t TB = 64 * R;  // bytes of one in0 tile [R, 32] bf16
    constexpr uint32_t FB = 32 * R;  // bytes of one face [R, 16] bf16
    constexpr auto x_args = TensorAccessorArgs<7>();
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
    experimental::CB cbx(cb_x), cbb(cb_bias), cbs(cb_score), cbr(cb_rank);

    {
        DeviceZoneScopedN("B_XLOAD");
        cbx.reserve_back(KT);
        cbb.reserve_back(1);
        constexpr uint32_t XPC = KT / NC;
        for (uint32_t j = 0; j < XPC; ++j) {
            const uint32_t i = me * XPC + j;
            if constexpr (R == 32) {  // std 32x32 tile
                noc.async_read(x_acc, cbx, 2048, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * 2048});
            } else {
                noc.async_read(x_acc, cbx, FB, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * TB});
                noc.async_read(x_acc, cbx, FB, {.page_id = i, .offset_bytes = 512}, {.offset_bytes = i * TB + FB});
            }
        }
        noc.async_read(b_acc, cbb, 4096, {.page_id = me, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbb.push_back(1);
        const uint32_t xb = cbx.get_write_ptr();
        for (uint32_t kk = 1; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            const uint32_t off = me * XPC * TB;
            noc_async_write(xb + off, get_noc_addr(nx(d), ny(d), xb + off), XPC * TB);
        }
        noc_async_write_barrier();
        for (uint32_t kk = 1; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            noc_semaphore_inc(get_noc_addr(nx(d), ny(d), get_semaphore(2)), 1);
        }
        noc_async_atomic_barrier();
        volatile tt_l1_ptr uint32_t* sem_x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        noc_semaphore_wait(sem_x, NC - 1);
        *sem_x = 0;
        cbx.push_back(KT);
    }
    {
        DeviceZoneScopedN("B_OUT");
        cbs.wait_front(1);
        cbr.wait_front(1);
        // fp32 [8, 32] tiny tile: face 0 = rows 0..7 x 16 x 4 B = 512 B, face 1 at +512; std fp32 tile: face 0 at 0, face 1 at 1024
        noc.async_write(cbs, s_acc, 64 * R, {.offset_bytes = 0}, {.page_id = me, .offset_bytes = 0});
        noc.async_write(cbs, s_acc, 64 * R, {.offset_bytes = 512}, {.page_id = me, .offset_bytes = 1024});
        noc.async_write(cbr, r_acc, 64 * R, {.offset_bytes = 0}, {.page_id = me, .offset_bytes = 0});
        noc.async_write(cbr, r_acc, 64 * R, {.offset_bytes = 512}, {.page_id = me, .offset_bytes = 1024});
        noc.async_write_barrier();
        cbs.pop_front(1);
        cbr.pop_front(1);
    }
}
