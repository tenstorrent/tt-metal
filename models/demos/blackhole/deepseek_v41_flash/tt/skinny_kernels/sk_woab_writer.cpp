// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// attention tail chain (wo_a -> wo_b) BRISC (data movement of the NC cores = the DRAM-bank cores of the chip):
//   x: std 32x32 tiles in DRAM (faces 0/1 carry rows 0..7, MT = 2: rows 8..15 in the second halves) -> tiny 8x32 tiles
//   in cb_x (MT groups of KT1 tiles) mm1 output (PN1 columns x MT tiles) -> every core's cb_yg (all-gather, group m at
//   tile m * KT2), semaphore 0 mm2 output (PN2 columns x MT tiles, sub-blocks of SBW2 columns) -> DRAM std tiles of
//   `part` (faces 0 / 1: rows 0..7, MT = 2: 0..15)
// CT args: 0 cb_x, 1 cb_y, 2 cb_yg, 3 cb_o, 4 NC, 5 PN1, 6 KT1, 7 KSKIP, 8 KT2, 9 PN2, 10 MT, 11 SBW2, 12 xc (compact
// x: KT1 = 160 tiles per row tile), then TensorAccessorArgs(x), TensorAccessorArgs(part) RT args: 0 first y tile (wo_a
// output column tile) of this core, 1 first output column tile of this core, 2 core index. Common: 0 x addr, 1 part
// addr, 2.. physical (x, y) of the NC cores.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_yg = get_compile_time_arg_val(2);
    constexpr uint32_t cb_o = get_compile_time_arg_val(3);
    constexpr uint32_t NC = get_compile_time_arg_val(4);
    constexpr uint32_t PN1 = get_compile_time_arg_val(5);
    constexpr uint32_t KT1 = get_compile_time_arg_val(6);
    constexpr uint32_t KSKIP = get_compile_time_arg_val(7);
    constexpr uint32_t KT2 = get_compile_time_arg_val(8);
    constexpr uint32_t PN2 = get_compile_time_arg_val(9);
    constexpr uint32_t MT = get_compile_time_arg_val(10);
    constexpr uint32_t SBW2 = get_compile_time_arg_val(11);
    constexpr bool XC = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t SBW1 = get_compile_time_arg_val(13);
    constexpr uint32_t XPB = get_compile_time_arg_val(14);  // page bytes of the compact x buffer
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto p_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t ycol0 = get_arg_val<uint32_t>(0);
    const uint32_t ocol0 = get_arg_val<uint32_t>(1);
    const uint32_t me = get_arg_val<uint32_t>(2);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(1);
    auto nx = [&](uint32_t d) { return get_common_arg_val<uint32_t>(2 + 2 * d); };
    auto ny = [&](uint32_t d) { return get_common_arg_val<uint32_t>(3 + 2 * d); };

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, XC ? XPB : 2048);
    const auto p_acc = TensorAccessor(p_args, p_addr, 2048);
    experimental::CB cbx(cb_x), cby(cb_y), cbyg(cb_yg), cbo(cb_o);
    volatile tt_l1_ptr uint32_t* sem_y = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    const uint32_t sem_y_addr = get_semaphore(0);

    {
        DeviceZoneScopedN("B_XLOAD");
        cbx.reserve_back(KT1 * MT);
        if constexpr (XC) {
            // compact x (rope_concat compact=True): row tile m = 5 pages of 16 KiB = KT1 (160) tiny tiles, contiguous
            for (uint32_t m = 0; m < MT; ++m) {
                for (uint32_t pg = 0; pg < 81920 / XPB; ++pg) {
                    noc.async_read(
                        x_acc,
                        cbx,
                        XPB,
                        {.page_id = m * (81920 / XPB) + pg, .offset_bytes = 0},
                        {.offset_bytes = m * KT1 * 512 + pg * XPB});
                }
            }
        }
        for (uint32_t i = 0; i < (XC ? 0 : KT1); ++i) {
            noc.async_read(x_acc, cbx, 256, {.page_id = KSKIP + i, .offset_bytes = 0}, {.offset_bytes = i * 512});
            noc.async_read(
                x_acc, cbx, 256, {.page_id = KSKIP + i, .offset_bytes = 512}, {.offset_bytes = i * 512 + 256});
            if constexpr (MT == 2) {
                noc.async_read(
                    x_acc, cbx, 256, {.page_id = KSKIP + i, .offset_bytes = 256}, {.offset_bytes = (KT1 + i) * 512});
                noc.async_read(
                    x_acc,
                    cbx,
                    256,
                    {.page_id = KSKIP + i, .offset_bytes = 768},
                    {.offset_bytes = (KT1 + i) * 512 + 256});
            }
        }
        cbyg.reserve_back(KT2 * MT);
        noc.async_read_barrier();
        cbx.push_back(KT1 * MT);
    }

    {
        DeviceZoneScopedN("B_YGATHER");
        // layout of cb_y: per sub-block s of SBW1 columns: [m][SBW1]
        cby.wait_front(PN1 * MT);
        const uint32_t y_ptr = cby.get_read_ptr();
        const uint32_t yg_base = cbyg.get_write_ptr();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            for (uint32_t s = 0; s < PN1 / SBW1; ++s) {
                for (uint32_t m = 0; m < MT; ++m) {
                    noc_async_write(
                        y_ptr + (s * MT + m) * SBW1 * 512,
                        get_noc_addr(nx(d), ny(d), yg_base + (m * KT2 + ycol0 + s * SBW1) * 512),
                        SBW1 * 512);
                }
            }
        }
        noc_async_write_barrier();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            noc_semaphore_inc(get_noc_addr(nx(d), ny(d), sem_y_addr), 1);
        }
        noc_async_atomic_barrier();
        noc_semaphore_wait(sem_y, NC);
        *sem_y = 0;
        cby.pop_front(PN1 * MT);
        cbyg.push_back(KT2 * MT);
    }

    {
        DeviceZoneScopedN("B_OUT");
        // cb_o sub-block s holds [m][SBW2] tiles
        constexpr uint32_t NSB = PN2 / SBW2;
        for (uint32_t s = 0; s < NSB; ++s) {
            cbo.wait_front((s + 1) * MT * SBW2);
            for (uint32_t w = 0; w < SBW2; ++w) {
                const uint32_t pg = ocol0 + s * SBW2 + w;
                for (uint32_t m = 0; m < MT; ++m) {
                    const uint32_t t_off = ((s * MT + m) * SBW2 + w) * 512;
                    noc.async_write(cbo, p_acc, 256, {.offset_bytes = t_off}, {.page_id = pg, .offset_bytes = m * 256});
                    noc.async_write(
                        cbo, p_acc, 256, {.offset_bytes = t_off + 256}, {.page_id = pg, .offset_bytes = 512 + m * 256});
                }
            }
        }
        noc.async_write_barrier();
        cbo.pop_front(PN2 * MT);
    }
}
