// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Writes this core's CPC output tiles of [1, 1, 32, H] (zero tiles when no local expert is active).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t page = get_compile_time_arg_val(0);
    constexpr uint32_t grid_x = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t sp_page = get_compile_time_arg_val(3);
    constexpr uint32_t CPC = get_compile_time_arg_val(4);
    constexpr uint32_t EG = get_compile_time_arg_val(5);
    constexpr uint32_t Nh = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sp2 = 4, cb_zero = 5, cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr auto sp_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t c0 = (core / EG) * CPC, eg = core % EG;
    const uint32_t row0 = eg * Nh;  // this group's partial-sum slab of the [1, EG, 32, H] output
    const auto out = TensorAccessor(out_args, out_addr, page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);
    const uint32_t sp_l1 = get_write_ptr(cb_sp2);
    noc_async_read(sp.get_noc_addr(0), sp_l1, sp_page);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    bool any = false;
    uint32_t seen = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (spv[e] != 0) {
            any = any || (seen % EG == eg);
            ++seen;
        }
    }
    if (any) {
        cb_wait_front(cb_out, CPC);
        for (uint32_t j = 0; j < CPC; ++j) {
            noc_async_write_tile(row0 + c0 + j, out, get_read_ptr(cb_out) + j * page);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, CPC);
    } else {
        const uint32_t z = get_write_ptr(cb_zero);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < page; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, z + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < CPC; ++j) {
            noc_async_write_tile(row0 + c0 + j, out, z);
        }
        noc_async_write_barrier();
    }
}
