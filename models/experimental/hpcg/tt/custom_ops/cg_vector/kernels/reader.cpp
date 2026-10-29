// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused CG vector kernels, reader (BRISC). Each core owns tiles [t0, t0 + nt) of the vectors.
// The mode is a runtime argument (MODE_DOT 0 / MODE_AXPY 1 / MODE_XR 2), so one binary serves all three:
// MODE_DOT : A = a, B = b                         (partial a.b per core)
// MODE_AXPY: A = a, B = b, S = s1 / s2            (w = a + S*b, written over b: p = z + beta*p)
// MODE_XR  : A = x, B = p, C = r, D = Ap, S = s1 / s2   (x += S*p, r -= S*Ap, partial r.r; S = alpha)
// MODE_ZERO 3: nothing read (the writer stores zeros)
// The scalar S is divided here in IEEE fp32 software arithmetic (correctly rounded) and broadcast into a tile.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0), b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(2), d_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t s1_addr = get_common_arg_val<uint32_t>(4), s2_addr = get_common_arg_val<uint32_t>(5);
    // tiles split over the cores: core i takes base (+ 1 for the first rem cores) tiles; i from the logical position
    const uint32_t base = get_common_arg_val<uint32_t>(6), rem = get_common_arg_val<uint32_t>(7);
    const uint32_t i = get_absolute_logical_y() * get_common_arg_val<uint32_t>(8) + get_absolute_logical_x();
    const uint32_t nt = base + (i < rem), t0 = i * base + (i < rem ? i : rem);
    const uint32_t mode = get_common_arg_val<uint32_t>(9);
    if (mode == 3) {
        return;
    }
    const bool dot = mode == 0, xr = mode == 2;
    constexpr auto aa = TensorAccessorArgs<0>();
    constexpr auto ab = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    constexpr auto ac = TensorAccessorArgs<ab.next_compile_time_args_offset()>();
    constexpr auto ad = TensorAccessorArgs<ac.next_compile_time_args_offset()>();
    constexpr auto as1 = TensorAccessorArgs<ad.next_compile_time_args_offset()>();
    constexpr auto as2 = TensorAccessorArgs<as1.next_compile_time_args_offset()>();
    const auto A = TensorAccessor(aa, a_addr, 4096);
    const auto B = TensorAccessor(ab, b_addr, 4096);
    const auto C = TensorAccessor(ac, c_addr, 4096);
    const auto D = TensorAccessor(ad, d_addr, 4096);
    const auto S1 = TensorAccessor(as1, s1_addr, 4096);
    const auto S2 = TensorAccessor(as2, s2_addr, 4096);
    (void)B;
    (void)C;
    (void)D;
    (void)S1;
    (void)S2;
    if (!dot) {  // S = s1 / s2 (element (0, 0) of two 1x1 tiles = word 0 of their pages)
        cb_reserve_back(CB_S, 1);
        const uint32_t sb = get_write_ptr(CB_S);
        noc_async_read(S1.get_noc_addr(0), sb, 64);
        noc_async_read(S2.get_noc_addr(0), sb + 64, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        volatile float* f = (volatile float*)sb;
        const float s = f[0] / f[16];
        volatile float* o = (volatile float*)sb;
        for (uint32_t i = 0; i < 1024; ++i) {
            o[i] = s;
        }
        cb_push_back(CB_S, 1);
    }
    // VB tiles per batch: all of a batch's reads are in flight together (one wait per batch, not per tile)
    for (uint32_t t = t0; t < t0 + nt; t += VB) {
        const uint32_t n = (t0 + nt - t) < VB ? (t0 + nt - t) : VB;
        cb_reserve_back(CB_A, n);
        if (!dot) {
            cb_reserve_back(CB_B, n);
        }
        if (xr) {
            cb_reserve_back(CB_C, n);
            cb_reserve_back(CB_D, n);
        }
        const uint32_t pa = get_write_ptr(CB_A), pb = get_write_ptr(CB_B);
        const uint32_t pc = get_write_ptr(CB_C), pd = get_write_ptr(CB_D);

        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(A.get_noc_addr(t + i), pa + i * 4096, 4096);
            // DOT: the writer reads b (one data-movement processor alone tops out near 420 GB/s chip-wide);
            // AXPY / XR keep all reads here: their writer also writes, and sharing measured slower
            if (!dot) {
                noc_async_read(B.get_noc_addr(t + i), pb + i * 4096, 4096);
            }
            if (xr) {
                noc_async_read(C.get_noc_addr(t + i), pc + i * 4096, 4096);
                noc_async_read(D.get_noc_addr(t + i), pd + i * 4096, 4096);
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_A, n);
        if (!dot) {
            cb_push_back(CB_B, n);
        }
        if (xr) {
            cb_push_back(CB_C, n);
            cb_push_back(CB_D, n);
        }
    }
}
