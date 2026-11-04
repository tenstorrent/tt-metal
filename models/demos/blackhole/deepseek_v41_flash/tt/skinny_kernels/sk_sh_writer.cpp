// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// shared-expert chain BRISC (the NC = 8 DRAM-bank cores of the chip, core b):
//   x: std 32x32 DRAM / L1 tiles (faces 0 / 1 = rows 0..7) or the compact buffer (16 KiB pages, tiny 8x32 tiles, 160
//   per row tile) -> cb_x zero tile -> cb_zero act_b (cb_actl) -> slot b of every core's cb_act; semaphore 1 partials
//   (cb_p = [gate, up]) -> slot b of every core's cb_prg / cb_pru; semaphore 0 ; act_8 (cb_act8, computed on every
//   core) -> own cb_act slot 8 output columns (cb_o, PNC tiles) -> std tiles of the fold tensor (faces 0 / 1: rows
//   0..7), page = first output tile of this core + j
// CT args: 0 cb_x, 1 cb_zero, 2 cb_actl, 3 cb_p, 4 cb_prg, 5 cb_pru, 6 cb_act, 7 cb_act8, 8 cb_o, 9 NC, 10 KT, 11 xc
// (compact x), 12 PNC, 13 SBW, 14 KA, then TensorAccessorArgs(x), TensorAccessorArgs(out) RT args: 0 core index b, 1
// first output column tile.  Common: 0 x addr, 1 out addr, 2.. physical (x, y) of the NC cores.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(1);
    constexpr uint32_t cb_actl = get_compile_time_arg_val(2);
    constexpr uint32_t cb_p = get_compile_time_arg_val(3);
    constexpr uint32_t cb_prg = get_compile_time_arg_val(4);
    constexpr uint32_t cb_pru = get_compile_time_arg_val(5);
    constexpr uint32_t cb_act = get_compile_time_arg_val(6);
    constexpr uint32_t cb_act8 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_o = get_compile_time_arg_val(8);
    constexpr uint32_t NC = get_compile_time_arg_val(9);
    constexpr uint32_t KT = get_compile_time_arg_val(10);
    constexpr bool XC = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t PNC = get_compile_time_arg_val(12);
    constexpr uint32_t SBW = get_compile_time_arg_val(13);
    constexpr uint32_t KA = get_compile_time_arg_val(14);
    constexpr bool XS = get_compile_time_arg_val(15) !=
                        0;  // x split: every core converts 1/NC of the std x tiles and sends the slice to the others
    constexpr uint32_t XPB = get_compile_time_arg_val(16);  // page bytes of the compact x buffer
    constexpr uint32_t MT = get_compile_time_arg_val(17);   // row tiles (1: T <= 8, 2: T = 9..16)
    constexpr auto x_args = TensorAccessorArgs<18>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t ocol0 = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(1);
    auto nx = [&](uint32_t d) { return get_common_arg_val<uint32_t>(2 + 2 * d); };
    auto ny = [&](uint32_t d) { return get_common_arg_val<uint32_t>(3 + 2 * d); };

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, XC ? XPB : 2048);
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    experimental::CB cbx(cb_x), cbz(cb_zero), cbactl(cb_actl), cbp(cb_p), cbprg(cb_prg), cbpru(cb_pru), cbact(cb_act),
        cbact8(cb_act8), cbo(cb_o);
    volatile tt_l1_ptr uint32_t* sem_part = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* sem_act = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint32_t sem_part_addr = get_semaphore(0), sem_act_addr = get_semaphore(1);

    cbprg.reserve_back(NC * MT);
    cbpru.reserve_back(NC * MT);
    cbact.reserve_back(KA * MT);
    {
        DeviceZoneScopedN("B_XLOAD");
        cbx.reserve_back(KT * MT);
        if constexpr (XC) {
            for (uint32_t pg = 0; pg < MT * KT * 512 / XPB; ++pg) {
                noc.async_read(x_acc, cbx, XPB, {.page_id = pg, .offset_bytes = 0}, {.offset_bytes = pg * XPB});
            }
        } else {
            constexpr uint32_t XPC = KT / NC;
            for (uint32_t j = 0; j < (XS ? XPC : KT); ++j) {
                const uint32_t i = XS ? me * XPC + j : j;
                noc.async_read(x_acc, cbx, 256, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * 512});
                noc.async_read(x_acc, cbx, 256, {.page_id = i, .offset_bytes = 512}, {.offset_bytes = i * 512 + 256});
                if constexpr (MT == 2) {  // rows 8..15
                    noc.async_read(
                        x_acc, cbx, 256, {.page_id = i, .offset_bytes = 256}, {.offset_bytes = (KT + i) * 512});
                    noc.async_read(
                        x_acc, cbx, 256, {.page_id = i, .offset_bytes = 768}, {.offset_bytes = (KT + i) * 512 + 256});
                }
            }
        }
        cbz.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbz.get_write_ptr());
            for (uint32_t i = 0; i < 128; ++i) {
                z[i] = 0;
            }
        }
        cbz.push_back(1);
        noc.async_read_barrier();
        if constexpr (XS && !XC) {
            constexpr uint32_t XPC2 = KT / NC;
            const uint32_t xb = cbx.get_write_ptr();
            for (uint32_t kk = 1; kk < NC; ++kk) {
                const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
                for (uint32_t m = 0; m < MT; ++m) {
                    const uint32_t off = (m * KT + me * XPC2) * 512;
                    noc_async_write(xb + off, get_noc_addr(nx(d), ny(d), xb + off), XPC2 * 512);
                }
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
        }
        cbx.push_back(KT * MT);
    }
    const uint32_t act_base = cbact.get_write_ptr();
    {
        DeviceZoneScopedN("B_PART");
        cbp.wait_front(2 * MT);
        const uint32_t src = cbp.get_read_ptr();
        const uint32_t g_base = cbprg.get_write_ptr();
        const uint32_t u_base = cbpru.get_write_ptr();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            for (uint32_t m = 0; m < MT; ++m) {
                noc_async_write(src + 2 * m * 512, get_noc_addr(nx(d), ny(d), g_base + (m * NC + me) * 512), 512);
                noc_async_write(src + (2 * m + 1) * 512, get_noc_addr(nx(d), ny(d), u_base + (m * NC + me) * 512), 512);
            }
        }
        noc_async_write_barrier();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            noc_semaphore_inc(get_noc_addr(nx(d), ny(d), sem_part_addr), 1);
        }
        noc_async_atomic_barrier();
        cbp.pop_front(2 * MT);
    }
    {
        DeviceZoneScopedN("B_ACT");
        cbactl.wait_front(MT);
        const uint32_t src = cbactl.get_read_ptr();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            for (uint32_t m = 0; m < MT; ++m) {
                noc_async_write(src + m * 512, get_noc_addr(nx(d), ny(d), act_base + (m * KA + me) * 512), 512);
            }
        }
        noc_async_write_barrier();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            noc_semaphore_inc(get_noc_addr(nx(d), ny(d), sem_act_addr), 1);
        }
        noc_async_atomic_barrier();
        cbactl.pop_front(MT);
    }
    {
        DeviceZoneScopedN("B_PWAIT");
        noc_semaphore_wait(sem_part, NC);
        *sem_part = 0;
        cbprg.push_back(NC * MT);
        cbpru.push_back(NC * MT);
    }
    {
        DeviceZoneScopedN("B_ACT8");
        cbact8.wait_front(MT);
        for (uint32_t m = 0; m < MT; ++m) {
            noc_async_write(
                cbact8.get_read_ptr() + m * 512, get_noc_addr(nx(me), ny(me), act_base + (m * KA + KA - 1) * 512), 512);
        }
        noc_async_write_barrier();
        cbact8.pop_front(MT);
        noc_semaphore_wait(sem_act, NC);
        *sem_act = 0;
        cbact.push_back(KA * MT);
    }
    {
        DeviceZoneScopedN("B_OUT");
        constexpr uint32_t NSB = PNC / SBW;
        for (uint32_t s = 0; s < NSB; ++s) {
            cbo.wait_front((s + 1) * SBW * MT);
            for (uint32_t w = 0; w < SBW; ++w) {
                const uint32_t pg = ocol0 + s * SBW + w;
                for (uint32_t m = 0; m < MT; ++m) {
                    const uint32_t t_off = ((s * MT + m) * SBW + w) * 512;
                    noc.async_write(cbo, o_acc, 256, {.offset_bytes = t_off}, {.page_id = pg, .offset_bytes = m * 256});
                    noc.async_write(
                        cbo, o_acc, 256, {.offset_bytes = t_off + 256}, {.page_id = pg, .offset_bytes = 512 + m * 256});
                }
            }
        }
        noc.async_write_barrier();
        cbo.pop_front(PNC * MT);
    }
}
