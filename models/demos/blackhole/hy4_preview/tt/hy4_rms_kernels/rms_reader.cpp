// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B47c (HY4_B47C_RMS=1): fused [residual add +] RMSNorm of the Hy4 MTP block, reader (NCRISC). Adapted from the DSV4
// mhc_cn exchange. NC cores (a physically contiguous 4x8 worker rectangle); core k owns column tiles [k*GPC, (k+1)*GPC)
// of the [1,1,T,D] tensors (one tile row, rows 0..T-1 valid, T <= 16 so only faces 0/1 carry data). Phases: read x (and
// residual) rows + weight tiles, zero the padding; exchange the T per-core partial sums of squares through L1 slots
// (unicast writes + flag word), transpose them into the tile R[t, k].
#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_p = get_compile_time_arg_val(3);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(5);
    constexpr uint32_t T = get_compile_time_arg_val(6);
    constexpr uint32_t GPC = get_compile_time_arg_val(7);
    constexpr uint32_t NC = get_compile_time_arg_val(8);
    constexpr uint32_t MX0 = get_compile_time_arg_val(9);
    constexpr uint32_t MY0 = get_compile_time_arg_val(10);
    constexpr uint32_t MW = get_compile_time_arg_val(11);
    constexpr uint32_t XF32 = get_compile_time_arg_val(12);   // x tiles fp32 (else bf16)
    constexpr uint32_t HAS_R = get_compile_time_arg_val(13);  // bf16 residual tiles
    constexpr auto x_args = TensorAccessorArgs<14>();
    constexpr auto r_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t XP = XF32 ? 4096 : 2048;  // x tile bytes
    constexpr uint32_t XFB = XF32 ? 1024 : 512;  // face stride
    constexpr uint32_t XRB = XF32 ? 64 : 32;     // row bytes inside a face
    constexpr uint32_t SW = 36;
    constexpr uint32_t FLAG = 0xC0DE1234u;

    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t j0 = k * GPC;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t r_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, XP);
    const auto r_acc = TensorAccessor(r_args, r_addr, 2048);
    const auto w_acc = TensorAccessor(w_args, w_addr, 4096);
    experimental::CB cbx(cb_x), cbr(cb_r), cbw(cb_w), cbp(cb_p), cbslot(cb_slot), cbtot(cb_tot);

    cbslot.reserve_back(1);
    const uint32_t slot_base = cbslot.get_write_ptr();
    volatile tt_l1_ptr uint32_t* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_base);
    cbtot.reserve_back(1);
    cbw.reserve_back(GPC);
    cbx.reserve_back(GPC);
    noc.async_write_zeros(cbtot, 4096, {.offset_bytes = 0});
    noc.async_write_zeros(cbx, GPC * XP, {.offset_bytes = 0});
    if constexpr (HAS_R) {
        cbr.reserve_back(GPC);
        noc.async_write_zeros(cbr, GPC * 2048, {.offset_bytes = 0});
    }
    noc.write_zeros_l1_barrier();
    for (uint32_t g = 0; g < GPC; ++g) {
        noc.async_read(w_acc, cbw, 4096, {.page_id = j0 + g, .offset_bytes = 0}, {.offset_bytes = g * 4096});
        noc.async_read(x_acc, cbx, T * XRB, {.page_id = j0 + g, .offset_bytes = 0}, {.offset_bytes = g * XP});
        noc.async_read(x_acc, cbx, T * XRB, {.page_id = j0 + g, .offset_bytes = XFB}, {.offset_bytes = g * XP + XFB});
        if constexpr (HAS_R) {
            noc.async_read(r_acc, cbr, T * 32, {.page_id = j0 + g, .offset_bytes = 0}, {.offset_bytes = g * 2048});
            noc.async_read(
                r_acc, cbr, T * 32, {.page_id = j0 + g, .offset_bytes = 512}, {.offset_bytes = g * 2048 + 512});
        }
    }
    noc.async_read_barrier();
    cbw.push_back(GPC);
    cbx.push_back(GPC);
    if constexpr (HAS_R) {
        cbr.push_back(GPC);
    }

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
