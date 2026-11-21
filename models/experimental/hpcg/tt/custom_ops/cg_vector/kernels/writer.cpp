// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused CG vector kernels, writer (NCRISC). DOT: this core's partial tile -> PART[core]. AXPY: w tiles over b.
// XR: x and r tiles in place, partial r.r tile -> PART[core]. ZERO: zero tiles over both outputs, nothing read (so a
// NaN or inf in them does not survive, as it would x - x).
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t o1_addr = get_common_arg_val<uint32_t>(0), o2_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t part_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t base = get_common_arg_val<uint32_t>(3), rem = get_common_arg_val<uint32_t>(4);
    const uint32_t core = get_absolute_logical_y() * get_common_arg_val<uint32_t>(5) + get_absolute_logical_x();
    const uint32_t nt = base + (core < rem), t0 = core * base + (core < rem ? core : rem);
    const uint32_t mode = get_common_arg_val<uint32_t>(7);  // 0 dot, 1 axpy, 2 xr, 3 zero (one binary for all four)
    const bool dot = mode == 0, xr = mode == 2;
    constexpr auto a1 = TensorAccessorArgs<0>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto ap = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto O1 = TensorAccessor(a1, o1_addr, 4096);
    const auto O2 = TensorAccessor(a2, o2_addr, 4096);
    const auto PART = TensorAccessor(ap, part_addr, 4096);
    constexpr auto ad = TensorAccessorArgs<ap.next_compile_time_args_offset()>();
    const auto Dv = TensorAccessor(ad, get_common_arg_val<uint32_t>(6), 4096);  // XR: Ap
    (void)O1;
    (void)O2;
    (void)PART;
    (void)Dv;
    if (mode == 3) {  // CB_OUT's L1 is free in this mode (no compute): one zeroed page, written over every tile
        volatile tt_l1_ptr uint32_t* z = (volatile tt_l1_ptr uint32_t*)get_write_ptr(CB_OUT);
        for (uint32_t i = 0; i < 1024; ++i) {
            z[i] = 0;
        }
        for (uint32_t t = t0; t < t0 + nt; ++t) {
            noc_async_write(get_write_ptr(CB_OUT), O1.get_noc_addr(t), 4096);
            noc_async_write(get_write_ptr(CB_OUT), O2.get_noc_addr(t), 4096);
        }
        noc_async_write_barrier();
        return;
    }
    if (!dot) {
        for (uint32_t t = t0; t < t0 + nt; ++t) {
            cb_wait_front(CB_OUT, 1);
            noc_async_write(get_read_ptr(CB_OUT), O1.get_noc_addr(t), 4096);
            if (xr) {
                cb_wait_front(CB_OUT2, 1);
                noc_async_write(get_read_ptr(CB_OUT2), O2.get_noc_addr(t), 4096);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, 1);
            if (xr) {
                cb_pop_front(CB_OUT2, 1);
            }
        }
    }
    for (uint32_t t = t0; dot && t < t0 + nt; t += VB) {  // DOT: b tiles (O2 = b), batched like the reader's
        const uint32_t n = (t0 + nt - t) < VB ? (t0 + nt - t) : VB;
        cb_reserve_back(CB_B, n);
        const uint32_t pb = get_write_ptr(CB_B);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(O2.get_noc_addr(t + i), pb + i * 4096, 4096);
        }
        noc_async_read_barrier();
        cb_push_back(CB_B, n);
    }
    if (dot || xr) {
        const uint32_t cbp = dot ? CB_OUT : CB_OUT3;
        // this core's value is element (0, 0) = word 0 of the packed tile: it goes to its own 64-byte slot of PART (the
        // rest of the slot zeroed), so the finish step reads core-count * 64 bytes instead of one tile per core
        cb_wait_front(cbp, 1);
        volatile tt_l1_ptr uint32_t* w = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cbp);
        for (uint32_t i = 1; i < 16; ++i) {
            w[i] = 0;
        }
        noc_async_write(get_read_ptr(cbp), PART.get_noc_addr((core * 64) / 4096) + (core * 64) % 4096, 64);
        noc_async_write_barrier();
        cb_pop_front(cbp, 1);
    }
}
