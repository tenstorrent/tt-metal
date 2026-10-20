// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 DFlash verify QKV heads split + RoPE, writer (BRISC) (qkv_rope_reader.cpp): head h < ROPE_HEADS (Q, then
// K) into the heads tensor [1, ROPE_HEADS, rows, 64] (page (h RT + r) 2 + d), the V heads into the one-head tensors
// v0 / v1 [1, 1, rows, 64] (page r 2 + d).
// common runtime args: [qk_addr, v0_addr, v1_addr]; compile-time args: [cb_out, HPC, ROPE_HEADS, RT (row tiles),
// accessors (qk, v0, v1)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t HPC = get_compile_time_arg_val(1);
    constexpr uint32_t ROPE_HEADS = get_compile_time_arg_val(2);
    constexpr uint32_t RT = get_compile_time_arg_val(3);
    constexpr auto qk_args = TensorAccessorArgs<4>();
    constexpr auto v0_args = TensorAccessorArgs<qk_args.next_compile_time_args_offset()>();
    constexpr auto v1_args = TensorAccessorArgs<v0_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;
    const auto s_qk = TensorAccessor(qk_args, get_common_arg_val<uint32_t>(0), tile);
    const auto s_v0 = TensorAccessor(v0_args, get_common_arg_val<uint32_t>(1), tile);
    const auto s_v1 = TensorAccessor(v1_args, get_common_arg_val<uint32_t>(2), tile);
    const uint32_t hc = get_absolute_logical_x(), r = get_absolute_logical_y();
    for (uint32_t i = 0; i < 2 * HPC; ++i) {
        const uint32_t h = hc * HPC + i / 2, d = i % 2;
        cb_wait_front(cb_out, 1);
        uint64_t dst;
        if (h < ROPE_HEADS) {
            dst = s_qk.get_noc_addr((h * RT + r) * 2 + d);
        } else if (h == ROPE_HEADS) {
            dst = s_v0.get_noc_addr(r * 2 + d);
        } else {
            dst = s_v1.get_noc_addr(r * 2 + d);
        }
        noc_async_write(get_read_ptr(cb_out), dst, tile);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
