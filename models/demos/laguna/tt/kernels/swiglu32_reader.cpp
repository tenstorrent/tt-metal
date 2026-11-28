// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode routed SwiGLU reader (Laguna). Units are (active local expert a, output column tile n) in
// order a * Nt + n; core c takes units c, c + num_cores, ... Active experts come from the union sparsity row (nonzero
// = some token routed there). Per unit pushes the gate tile, the up tile (columns Nt..2Nt-1 of the packed gate|up
// output) and the expert's routing-weight tile (column 0 = the 32 tokens' weights).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

template <uint32_t num_experts>
static inline uint32_t nth_active_impl(volatile tt_l1_ptr uint16_t* spv, uint32_t a) {
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
    constexpr uint32_t cb_g = 0, cb_u = 1, cb_w = 2, cb_meta = 3, cb_sp = 4;
    constexpr auto gu_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const uint32_t gu_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto gu = TensorAccessor(gu_args, gu_addr, page);
    const auto w = TensorAccessor(w_args, w_addr, page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);

    const uint32_t sp_l1 = get_write_ptr(cb_sp);
    noc_async_read(sp.get_noc_addr(0), sp_l1, sp_page);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    auto nth_active = [](volatile tt_l1_ptr uint16_t* v, uint32_t a) { return nth_active_impl<num_experts>(v, a); };
    uint32_t na = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        na += spv[e] != 0 ? 1 : 0;
    }
    const uint32_t units = na * Nt;
    const uint32_t mine = units > core ? (units - core + num_cores - 1) / num_cores : 0;
    cb_reserve_back(cb_meta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = mine;
    cb_push_back(cb_meta, 1);

    for (uint32_t u = core; u < units; u += num_cores) {
        const uint32_t e = nth_active(spv, u / Nt);
        const uint32_t n = u % Nt;
        cb_reserve_back(cb_g, 1);
        cb_reserve_back(cb_u, 1);
        cb_reserve_back(cb_w, 1);
        noc_async_read_tile(e * 2 * Nt + n, gu, get_write_ptr(cb_g));
        noc_async_read_tile(e * 2 * Nt + Nt + n, gu, get_write_ptr(cb_u));
        noc_async_read_tile(e, w, get_write_ptr(cb_w));
        noc_async_read_barrier();
        cb_push_back(cb_g, 1);
        cb_push_back(cb_u, 1);
        cb_push_back(cb_w, 1);
    }
}
