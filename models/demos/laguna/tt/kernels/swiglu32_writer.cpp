// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode routed SwiGLU writer (Laguna): same unit order as the reader; writes unit (a, n) to tile
// (a-th active expert * Nt + n) of the [1, E, 32, I] output. Inactive experts' tiles are not written.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

template <uint32_t num_experts>
static inline uint32_t nth_active(volatile tt_l1_ptr uint16_t* spv, uint32_t a) {
    for (uint32_t e = 0; e < num_experts; ++e) {
        if (spv[e] != 0) {
            if (a == 0) {
                return e;
            }
            --a;
        }
    }
    return 0;
}

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t num_experts = get_compile_time_arg_val(1);
    constexpr uint32_t page = get_compile_time_arg_val(2);
    constexpr uint32_t sp_page = get_compile_time_arg_val(3);
    constexpr uint32_t grid_x = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sp2 = 5, cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr auto sp_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto out = TensorAccessor(out_args, out_addr, page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);

    const uint32_t sp_l1 = get_write_ptr(cb_sp2);
    noc_async_read(sp.get_noc_addr(0), sp_l1, sp_page);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    uint32_t na = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        na += spv[e] != 0 ? 1 : 0;
    }
    const uint32_t units = na * Nt;
    for (uint32_t u = core; u < units; u += num_cores) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(nth_active<num_experts>(spv, u / Nt) * Nt + u % Nt, out, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
