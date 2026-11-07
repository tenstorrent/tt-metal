// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Sum of the active experts' down outputs (Laguna batched decode). Core n owns output column tile n of [1, 1, 32, H]
// and pushes tile (e, n) of the [1, E, 32, H] down output for every active expert e (from the union sparsity row).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t num_experts = get_compile_time_arg_val(1);
    constexpr uint32_t page = get_compile_time_arg_val(2);
    constexpr uint32_t sp_page = get_compile_time_arg_val(3);
    constexpr uint32_t grid_x = get_compile_time_arg_val(4);
    constexpr uint32_t cb_x = 0, cb_meta = 1, cb_sp = 2;
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto sp_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t n = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto x = TensorAccessor(x_args, x_addr, page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);

    const uint32_t sp_l1 = get_write_ptr(cb_sp);
    noc_async_read(sp.get_noc_addr(0), sp_l1, sp_page);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    uint32_t na = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        na += spv[e] != 0 ? 1 : 0;
    }
    cb_reserve_back(cb_meta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = na;
    cb_push_back(cb_meta, 1);
    for (uint32_t e = 0; e < num_experts; ++e) {
        if (spv[e] == 0) {
            continue;
        }
        cb_reserve_back(cb_x, 1);
        noc_async_read_tile(e * Nt + n, x, get_write_ptr(cb_x));
        noc_async_read_barrier();
        cb_push_back(cb_x, 1);
    }
}
