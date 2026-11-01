// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// Per item (row r, slice s): the row's max tile (SUM_EXP) and the slice's logits tiles in blocks.
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_max = 1;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t slices = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);
constexpr auto in_args = TensorAccessorArgs<4>();
constexpr auto max_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t max_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    const uint32_t in_tile = get_tile_size(cb_in);
    const auto in_acc = TensorAccessor(in_args, in_addr, in_tile);
#ifdef SUM_EXP
    const uint32_t max_tile = get_tile_size(cb_max);
    const auto max_acc = TensorAccessor(max_args, max_addr, max_tile);
#endif

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / slices;
        const uint32_t s = work - r * slices;
        const uint32_t k0 = s * St;
        const uint32_t nk = (k0 + St <= Wt) ? St : (Wt - k0);
#ifdef SUM_EXP
        cb_reserve_back(cb_max, 1);
        noc_async_read_page(r, max_acc, get_write_ptr(cb_max));
        noc_async_read_barrier();
        cb_push_back(cb_max, 1);
#endif
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
