// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode MoE down writer (Laguna): writes this core's output tile nt, or a zero tile when no local expert
// is active (the compute kernel then produces nothing).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t out_page = get_compile_time_arg_val(0);
    constexpr uint32_t grid_x = get_compile_time_arg_val(1);
    constexpr uint32_t num_experts = get_compile_time_arg_val(2);
    constexpr uint32_t sp_page = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sp2 = 4;
    constexpr uint32_t cb_zero = 5;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t has_shared = get_compile_time_arg_val(4);  // the shared expert always contributes
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr auto sp_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t nt = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const auto out = TensorAccessor(out_args, out_addr, out_page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);

    const uint32_t sp_l1 = get_write_ptr(cb_sp2);
    noc_async_read(sp.get_noc_addr(0), sp_l1, sp_page);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    bool any = has_shared != 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        if (spv[e] != 0) {
            any = true;
            break;
        }
    }
    if (any) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(nt, out, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    } else {
        const uint32_t z = get_write_ptr(cb_zero);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < out_page; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, z + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        noc_async_write_tile(nt, out, z);
        noc_async_write_barrier();
    }
}
