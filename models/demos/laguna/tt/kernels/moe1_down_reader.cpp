// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode MoE down + expert-sum reader (Laguna). Core nt owns output column tile nt. Per active local
// expert (from the sparsity row; the routing weight is already folded into the gate/up output), pushes row 0 of
// its Kt activation tiles (rows 1-31 zeroed) and its Kt down-weight tiles of column nt.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t num_experts = get_compile_time_arg_val(2);
    constexpr uint32_t x_page = get_compile_time_arg_val(3);
    constexpr uint32_t w_page = get_compile_time_arg_val(4);
    constexpr uint32_t sp_page = get_compile_time_arg_val(5);
    constexpr uint32_t grid_x = get_compile_time_arg_val(6);
    constexpr uint32_t max_active = get_compile_time_arg_val(7);
    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_w = 1;
    constexpr uint32_t cb_meta = 2;
    constexpr uint32_t cb_sp = 3;
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t nt = get_absolute_logical_y() * grid_x + get_absolute_logical_x();

    const auto x = TensorAccessor(x_args, x_addr, x_page);
    const auto w = TensorAccessor(w_args, w_addr, w_page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);

    const uint32_t sp_l1 = get_write_ptr(cb_sp);
    noc_async_read(sp.get_noc_addr(0), sp_l1, sp_page);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    uint32_t experts[max_active];
    uint32_t n = 0;
    for (uint32_t e = 0; e < num_experts && n < max_active; ++e) {
        if (spv[e] != 0) {
            experts[n++] = e;
        }
    }
    cb_reserve_back(cb_meta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = n;
    cb_push_back(cb_meta, 1);

    // rows 1-31 of the activation tiles are zeroed ONCE for both CB slots (each expert overwrites only row 0), so
    // the output tile's padding rows stay zero
    if (n > 0) {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t x_base = get_write_ptr(cb_x);
        for (uint32_t off = 0; off < 2 * Kt * x_page; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, x_base + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }
    for (uint32_t u = 0; u < n; ++u) {
        cb_reserve_back(cb_w, Kt);
        uint32_t wdst = get_write_ptr(cb_w);
        const uint32_t base = experts[u] * Kt * Nt + nt;
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            noc_async_read_tile(base + kt * Nt, w, wdst);
            wdst += w_page;
        }
        cb_reserve_back(cb_x, Kt);
        const uint32_t x_l1 = get_write_ptr(cb_x);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            const uint64_t src = x.get_noc_addr(experts[u] * Kt + kt);
            noc_async_read(src, x_l1 + kt * x_page, 32);
            noc_async_read(src + 512, x_l1 + kt * x_page + 512, 32);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, Kt);
        cb_push_back(cb_w, Kt);
    }
}
