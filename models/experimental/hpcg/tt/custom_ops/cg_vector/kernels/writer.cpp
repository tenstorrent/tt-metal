// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused CG vector kernels, writer (NCRISC). DOT: this core's partial tile -> PART[core]. AXPY: w tiles over b.
// XR: x and r tiles in place, partial r.r tile -> PART[core].
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t o1_addr = get_arg_val<uint32_t>(0), o2_addr = get_arg_val<uint32_t>(1);
    const uint32_t part_addr = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3), nt = get_arg_val<uint32_t>(4), core = get_arg_val<uint32_t>(5);
    constexpr auto a1 = TensorAccessorArgs<0>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto ap = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto O1 = TensorAccessor(a1, o1_addr, 4096);
    const auto O2 = TensorAccessor(a2, o2_addr, 4096);
    const auto PART = TensorAccessor(ap, part_addr, 4096);
    constexpr auto ad = TensorAccessorArgs<ap.next_compile_time_args_offset()>();
    const auto Dv = TensorAccessor(ad, get_arg_val<uint32_t>(6), 4096);  // XR: Ap
    (void)O1;
    (void)O2;
    (void)PART;
    (void)Dv;
#if defined(MODE_AXPY) || defined(MODE_XR)
    for (uint32_t t = t0; t < t0 + nt; ++t) {
        cb_wait_front(CB_OUT, 1);
        noc_async_write(get_read_ptr(CB_OUT), O1.get_noc_addr(t), 4096);
#ifdef MODE_XR
        cb_wait_front(CB_OUT2, 1);
        noc_async_write(get_read_ptr(CB_OUT2), O2.get_noc_addr(t), 4096);
#endif
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 1);
#ifdef MODE_XR
        cb_pop_front(CB_OUT2, 1);
#endif
    }
#endif
#ifdef MODE_DOT
    for (uint32_t t = t0; t < t0 + nt; t += VB) {  // DOT: b tiles (O2 = b), batched like the reader's
        const uint32_t n = (t0 + nt - t) < VB ? (t0 + nt - t) : VB;
        cb_reserve_back(CB_B, n);
        const uint32_t pb = get_write_ptr(CB_B);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(O2.get_noc_addr(t + i), pb + i * 4096, 4096);
        }
        noc_async_read_barrier();
        cb_push_back(CB_B, n);
    }
#endif
#if defined(MODE_DOT) || defined(MODE_XR)
#ifdef MODE_DOT
    constexpr uint32_t cbp = CB_OUT;
#else
    constexpr uint32_t cbp = CB_OUT3;
#endif
    cb_wait_front(cbp, 1);
    noc_async_write(get_read_ptr(cbp), PART.get_noc_addr(core), 4096);
    noc_async_write_barrier();
    cb_pop_front(cbp, 1);
#endif
}
