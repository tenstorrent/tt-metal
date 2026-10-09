// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode MoE gate/up writer (Laguna): re-derives this core's active experts from the sparsity row (same
// rule as the reader) and writes each unit's tile to tile (expert * Nt + nt) of the [1, E, 32, N] output.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t num_experts = get_compile_time_arg_val(1);
    constexpr uint32_t out_page = get_compile_time_arg_val(2);
    constexpr uint32_t sp_page = get_compile_time_arg_val(3);
    constexpr uint32_t grid_x = get_compile_time_arg_val(4);
    constexpr uint32_t slot_groups = get_compile_time_arg_val(5);
    constexpr uint32_t slots = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sp2 = 4;
    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr auto sp_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t core_index = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t nt = core_index / slot_groups;
    const uint32_t first_slot = (core_index % slot_groups) * slots;

    const auto out = TensorAccessor(out_args, out_addr, out_page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);

    const uint32_t sp_l1 = get_write_ptr(cb_sp2);
    noc_async_read(sp.get_noc_addr(0), sp_l1, sp_page);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    uint32_t experts[slots];
    uint32_t n = 0;
    uint32_t seen = 0;
    for (uint32_t e = 0; e < num_experts && n < slots; ++e) {
        if (spv[e] != 0) {
            if (seen >= first_slot) {
                experts[n++] = e;
            }
            ++seen;
        }
    }
    for (uint32_t u = 0; u < n; ++u) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(experts[u] * Nt + nt, out, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
