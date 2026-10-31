// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash QKV heads split + RoPE, writer (BRISC) (qkv_rope_reader.cpp): the heads into three heads-layout
// tensors [1, n, rows, 64] (page (h' RT + r) 2 + d): heads 0 .. NA - 1 into A, the next NB into B, the rest into C
// (the verify: A = Q and K, B / C = V heads 0 / 1; the drafter: A = Q, B = K, C = V).
// common runtime args: [a_addr, b_addr, c_addr]; compile-time args: [cb_out, HPC, NA, RT (row tiles), NB, accessors
// (a, b, c)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t HPC = get_compile_time_arg_val(1);
    constexpr uint32_t NA = get_compile_time_arg_val(2);
    constexpr uint32_t RT = get_compile_time_arg_val(3);
    constexpr uint32_t NB = get_compile_time_arg_val(4);
    constexpr auto qk_args = TensorAccessorArgs<5>();
    constexpr auto v0_args = TensorAccessorArgs<qk_args.next_compile_time_args_offset()>();
    constexpr auto v1_args = TensorAccessorArgs<v0_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;
    const auto s_qk = TensorAccessor(qk_args, get_common_arg_val<uint32_t>(0), tile);
    const auto s_v0 = TensorAccessor(v0_args, get_common_arg_val<uint32_t>(1), tile);
    const auto s_v1 = TensorAccessor(v1_args, get_common_arg_val<uint32_t>(2), tile);
    const uint32_t hc = get_absolute_logical_x(), r = get_absolute_logical_y();
#ifndef QR_MPC
#define QR_MPC 1
#endif
    for (uint32_t q = 0; q < QR_MPC * 2 * HPC; ++q) {  // (QR_MPC row tiles per core: row tile r QR_MPC + q / (2 HPC))
        const uint32_t i = q % (2 * HPC), rr = r * QR_MPC + q / (2 * HPC);
        const uint32_t h = hc * HPC + i / 2, d = i % 2;
        cb_wait_front(cb_out, 1);
        uint64_t dst;
        if (h < NA) {
            dst = s_qk.get_noc_addr((h * RT + rr) * 2 + d);
        } else if (h < NA + NB) {
            dst = s_v0.get_noc_addr(((h - NA) * RT + rr) * 2 + d);
        } else {
            dst = s_v1.get_noc_addr(((h - NA - NB) * RT + rr) * 2 + d);
        }
        noc_async_write(get_read_ptr(cb_out), dst, tile);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
