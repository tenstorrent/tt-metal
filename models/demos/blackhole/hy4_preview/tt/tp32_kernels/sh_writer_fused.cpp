// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// B19 FUSED variant: the output tiles are NoC-written into the landing buffer (slot 2) of the tp24 LEADER core of bank
// b (no DRAM output): the leader sums them with its own partial and the two group partials. Extra RT args: 2 leader x,
// 3 leader y (physical). Handshake: the leader stores its landing base address into this core's semaphore 3 (nonzero)
// at its start. Hy4-Preview shared-expert chain BRISC (NC = 8 DRAM-bank cores, core b). Copy of the DSV4
// sk_sh_writer.cpp without the pair-8 partial exchange.
//   x: std 32x32 DRAM / L1 tiles (faces 0 / 1 = rows 0..7) -> cb_x (every core converts KT / NC tiles and sends the
//   slice to the others, XS) zero tile -> cb_zero; act_b (cb_actl) -> slot b of every core's cb_act (semaphore 1); cb_o
//   (PNC tiles) -> std tiles of the output (faces 0 / 1: rows 0..7), page = first output tile of this core + j
// CT args: 0 cb_x, 1 cb_zero, 2 cb_actl, 3 cb_act, 4 cb_o, 5 NC, 6 KT, 7 PNC, 8 SBW, 9 KA, 10 XS, 11 MT, 12 R (rows of
// the in0 tiny tiles: 1, 2, 4 or 8; cb_x / cb_act pages are 64 R bytes), then TensorAccessorArgs(x),
// TensorAccessorArgs(out). RT args: 0 core index b, 1 first output column tile. Common: 0 x addr, 1 out addr, 2..
// physical (x, y) of the NC cores.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(1);
    constexpr uint32_t cb_actl = get_compile_time_arg_val(2);
    constexpr uint32_t cb_act = get_compile_time_arg_val(3);
    constexpr uint32_t cb_o = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);
    constexpr uint32_t KT = get_compile_time_arg_val(6);
    constexpr uint32_t PNC = get_compile_time_arg_val(7);
    constexpr uint32_t SBW = get_compile_time_arg_val(8);
    constexpr uint32_t KA = get_compile_time_arg_val(9);
    constexpr bool XS = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t MT = get_compile_time_arg_val(11);
    constexpr uint32_t R = get_compile_time_arg_val(12);
    constexpr uint32_t TB = 64 * R;                        // bytes of one in0 tile [R, 32] bf16
    constexpr uint32_t FB = 32 * R;                        // bytes of one face [R, 16]
    constexpr uint32_t RA = get_compile_time_arg_val(13);  // rows of the act tiles (cb_act)
    constexpr uint32_t TBA = 64 * RA;
    constexpr auto x_args = TensorAccessorArgs<14>();

    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t ocol0 = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t lx = get_arg_val<uint32_t>(2);
    const uint32_t ly = get_arg_val<uint32_t>(3);
    auto nx = [&](uint32_t d) { return get_common_arg_val<uint32_t>(2 + 2 * d); };
    auto ny = [&](uint32_t d) { return get_common_arg_val<uint32_t>(3 + 2 * d); };

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, 2048);
    experimental::CB cbx(cb_x), cbz(cb_zero), cbactl(cb_actl), cbact(cb_act), cbo(cb_o);
    volatile tt_l1_ptr uint32_t* sem_act = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint32_t sem_act_addr = get_semaphore(1);

    cbact.reserve_back(KA * MT);
    {
        DeviceZoneScopedN("B_XLOAD");
        cbx.reserve_back(KT * MT);
        constexpr uint32_t XPC = KT / NC;
        {
            DeviceZoneScopedN("B_XISSUE");
            for (uint32_t j = 0; j < (XS ? XPC : KT); ++j) {
                const uint32_t i = XS ? me * XPC + j : j;
                noc.async_read(x_acc, cbx, FB, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * TB});
                noc.async_read(x_acc, cbx, FB, {.page_id = i, .offset_bytes = 512}, {.offset_bytes = i * TB + FB});
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
        {
            DeviceZoneScopedN("B_XRB");
            noc.async_read_barrier();
        }
        if constexpr (XS) {
            const uint32_t xb = cbx.get_write_ptr();
            {
                DeviceZoneScopedN("B_XSEND");
                for (uint32_t kk = 1; kk < NC; ++kk) {
                    const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
                    for (uint32_t m = 0; m < MT; ++m) {
                        const uint32_t off = (m * KT + me * XPC) * TB;
                        noc_async_write(xb + off, get_noc_addr(nx(d), ny(d), xb + off), XPC * TB);
                    }
                }
                noc_async_write_barrier();
            }
            {
                DeviceZoneScopedN("B_XSEM");
                for (uint32_t kk = 1; kk < NC; ++kk) {
                    const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
                    noc_semaphore_inc(get_noc_addr(nx(d), ny(d), get_semaphore(2)), 1);
                }
                noc_async_atomic_barrier();
            }
            {
                DeviceZoneScopedN("B_XWAIT");
                volatile tt_l1_ptr uint32_t* sem_x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
                noc_semaphore_wait(sem_x, NC - 1);
                *sem_x = 0;
            }
        }
        cbx.push_back(KT * MT);
    }
    const uint32_t act_base = cbact.get_write_ptr();
    {
        DeviceZoneScopedN("B_ACT");
        cbactl.wait_front(MT);
        const uint32_t src = cbactl.get_read_ptr();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            for (uint32_t m = 0; m < MT; ++m) {
                const uint32_t dst = act_base + (m * KA + me) * TBA;
                if constexpr (RA == 8) {
                    noc_async_write(src + m * 512, get_noc_addr(nx(d), ny(d), dst), 512);
                } else {  // [8, 32] source tile (face 1 at +256) -> [RA, 32] in0 tile (face 1 at + 32 RA)
                    noc_async_write(src + m * 512, get_noc_addr(nx(d), ny(d), dst), 32 * RA);
                    noc_async_write(src + m * 512 + 256, get_noc_addr(nx(d), ny(d), dst + 32 * RA), 32 * RA);
                }
            }
        }
        noc_async_write_barrier();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (me + kk >= NC) ? me + kk - NC : me + kk;
            noc_semaphore_inc(get_noc_addr(nx(d), ny(d), sem_act_addr), 1);
        }
        noc_async_atomic_barrier();
        cbactl.pop_front(MT);
        noc_semaphore_wait(sem_act, NC);
        *sem_act = 0;
        cbact.push_back(KA * MT);
    }
    {
        DeviceZoneScopedN("B_OUT");
        volatile tt_l1_ptr uint32_t* sem_h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
        constexpr uint32_t NSB = PNC / SBW;
        constexpr uint32_t SLOT = 24 * MT;  // tiles per landing slot of the leader (tp24 layout: tile (s4 * MT + m) * 4
                                            // + w4 for output column j = 4 s4 + w4)
        noc_semaphore_wait_min(sem_h, 1);
        const uint32_t lnd = *sem_h;
        *sem_h = 0;
        for (uint32_t s = 0; s < NSB; ++s) {
            cbo.wait_front((s + 1) * SBW * MT);
        }
        const uint32_t src = cbo.get_read_ptr();
        for (uint32_t s = 0; s < NSB; ++s) {
            for (uint32_t w = 0; w < SBW; ++w) {
                const uint32_t j = s * SBW + w;
                for (uint32_t m = 0; m < MT; ++m) {
                    const uint32_t t_src = ((s * MT + m) * SBW + w);
                    const uint32_t t_dst = (((j >> 2) * MT + m) << 2) + (j & 3u);
                    noc_async_write(src + t_src * 512, get_noc_addr(lx, ly, lnd + (2 * SLOT + t_dst) * 512), 512);
                }
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lx, ly, get_semaphore(2)), 1);
        noc_async_atomic_barrier();
        cbo.pop_front(PNC * MT);
    }
}
