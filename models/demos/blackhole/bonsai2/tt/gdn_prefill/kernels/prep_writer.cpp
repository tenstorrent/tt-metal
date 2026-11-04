// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet chunk preprocessing, writer (BRISC). Per item the compute pushes 101 fp32 tiles in this order:
// L_unit 16, v_beta_sc 16, k_bd_sc 16, intra 16, q_decay 16, k_decay_t 16 (row-major [K, C] tiles), dl_exp 1,
// L_inv 4. They go to item `it`'s tiles of the 8 outputs, as gated_delta_attn_seq reads them.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"

void kernel_main() {
    // common args: 8 output addresses, Nitems, ncores, grid_x
    uint32_t first, n;
    core_span(
        get_common_arg_val<uint32_t>(8),
        get_common_arg_val<uint32_t>(9),
        get_common_arg_val<uint32_t>(10),
        1,
        first,
        n);
    constexpr uint32_t TB = 4096;
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    const auto o0 = TensorAccessor(a0, get_common_arg_val<uint32_t>(0), TB);
    const auto o1 = TensorAccessor(a1, get_common_arg_val<uint32_t>(1), TB);
    const auto o2 = TensorAccessor(a2, get_common_arg_val<uint32_t>(2), TB);
    const auto o3 = TensorAccessor(a3, get_common_arg_val<uint32_t>(3), TB);
    const auto o4 = TensorAccessor(a4, get_common_arg_val<uint32_t>(4), TB);
    const auto o5 = TensorAccessor(a5, get_common_arg_val<uint32_t>(5), TB);
    const auto o6 = TensorAccessor(a6, get_common_arg_val<uint32_t>(6), TB);
    const auto o7 = TensorAccessor(a7, get_common_arg_val<uint32_t>(7), TB);
    constexpr uint32_t cb_out = 16;

    auto put = [&](uint64_t dst) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), dst, TB);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    };
    for (uint32_t it = first; it < first + n; ++it) {
        for (uint32_t k = 0; k < 16; ++k) {
            put(o0.get_noc_addr(it * 16 + k));
        }
        for (uint32_t k = 0; k < 16; ++k) {
            put(o1.get_noc_addr(it * 16 + k));
        }
        for (uint32_t k = 0; k < 16; ++k) {
            put(o2.get_noc_addr(it * 16 + k));
        }
        for (uint32_t k = 0; k < 16; ++k) {
            put(o3.get_noc_addr(it * 16 + k));
        }
        for (uint32_t k = 0; k < 16; ++k) {
            put(o4.get_noc_addr(it * 16 + k));
        }
        for (uint32_t k = 0; k < 16; ++k) {
            put(o5.get_noc_addr(it * 16 + k));
        }
        put(o6.get_noc_addr(it));
        for (uint32_t k = 0; k < 4; ++k) {
            put(o7.get_noc_addr(it * 4 + k));
        }
    }
}
