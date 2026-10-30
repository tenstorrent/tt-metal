// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// vp_cross_entropy_bw reader: the grad tile once; per item (row r, slice s): the row's log-normalizer tile, then
// the slice's logits tiles `block` at a time (full blocks; a shorter tail still pushes a full block).

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_logz = 1;
constexpr uint32_t cb_grad = 2;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t Ht = get_compile_time_arg_val(1);
constexpr uint32_t slices = get_compile_time_arg_val(2);
constexpr uint32_t St = get_compile_time_arg_val(3);
constexpr uint32_t block = get_compile_time_arg_val(4);
constexpr auto in_args = TensorAccessorArgs<5>();
constexpr auto logz_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
constexpr auto grad_args = TensorAccessorArgs<logz_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t logz_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    const uint32_t in_tile = get_tile_size(cb_in);
    const uint32_t logz_tile = get_tile_size(cb_logz);
    const uint32_t grad_tile = get_tile_size(cb_grad);
    const auto in_acc = TensorAccessor(in_args, in_addr, in_tile);
    const auto logz_acc = TensorAccessor(logz_args, logz_addr, logz_tile);
    const auto grad_acc = TensorAccessor(grad_args, grad_addr, grad_tile);

    cb_reserve_back(cb_grad, 1);
    noc_async_read_page(0, grad_acc, get_write_ptr(cb_grad));
    noc_async_read_barrier();
    cb_push_back(cb_grad, 1);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / slices;
        const uint32_t s = work - r * slices;
        const uint32_t k0 = s * St;
        const uint32_t nk = (k0 + St <= Wt) ? St : (Wt - k0);

        cb_reserve_back(cb_logz, 1);
        noc_async_read_page(r, logz_acc, get_write_ptr(cb_logz));
        noc_async_read_barrier();
        cb_push_back(cb_logz, 1);

        for (uint32_t k = 0; k < nk; k += block) {
            const uint32_t n = (k + block <= nk) ? block : (nk - k);
            cb_reserve_back(cb_in, block);
            uint32_t l1 = get_write_ptr(cb_in);
            for (uint32_t j = 0; j < n; ++j) {
                noc_async_read_page(r * Wt + k0 + k + j, in_acc, l1);
                l1 += in_tile;
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, block);
        }
    }
}
