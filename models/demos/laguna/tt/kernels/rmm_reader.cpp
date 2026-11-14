// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Router logits, K-split (Laguna; router_mm.py), reader: core c = n * KC + kc owns output column n and K chunk kc
// (Kc tiles): the Mt x Kc activation tiles of x [.., Mt * 32, Kt * 32] and the Kc weight tiles of column n of
// w [Kt * 32, Nt * 32].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t KC = get_compile_time_arg_val(4);
    constexpr uint32_t Kc = Kt / KC;
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_x = 0, cb_w = 1;
    const uint32_t c = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t n = c / KC, kc = c % KC;
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), 2048);
    cb_reserve_back(cb_w, Kc);
    for (uint32_t k = 0; k < Kc; ++k) {
        noc_async_read_tile((kc * Kc + k) * Nt + n, w, get_write_ptr(cb_w) + k * 2048);
    }
    cb_reserve_back(cb_x, Mt * Kc);
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t k = 0; k < Kc; ++k) {
            noc_async_read_tile(m * Kt + kc * Kc + k, x, get_write_ptr(cb_x) + (m * Kc + k) * 2048);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, Kc);
    cb_push_back(cb_x, Mt * Kc);
}
