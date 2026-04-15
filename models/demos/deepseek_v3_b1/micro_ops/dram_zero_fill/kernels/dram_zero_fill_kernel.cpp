// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DRAM Zero Fill kernel: writes zero tiles to all pages of a DRAM tensor.
//
// NCRISC: Each core zeroes one L1 tile, then writes it to its assigned
//         range of DRAM pages via noc_async_write_page + TensorAccessor.
// BRISC:  No-op.
// TRISC:  No-op.

#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t output_cb = get_named_compile_time_arg_val("output_cb");
    constexpr uint32_t total_pages = get_named_compile_time_arg_val("total_pages");
    constexpr uint32_t pages_per_core = get_named_compile_time_arg_val("pages_per_core");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t grid_start_x = get_named_compile_time_arg_val("grid_start_x");
    constexpr uint32_t grid_start_y = get_named_compile_time_arg_val("grid_start_y");
    constexpr uint32_t grid_end_x = get_named_compile_time_arg_val("grid_end_x");
    constexpr uint32_t grid_end_y = get_named_compile_time_arg_val("grid_end_y");

    uint32_t core_id = unified_kernels::linear_id_in_grid<true>(grid_start_x, grid_start_y, grid_end_x, grid_end_y);

    uint32_t buffer_addr = get_common_arg_val<uint32_t>(0);
    constexpr auto ta_args = TensorAccessorArgs<0>();
    auto accessor = TensorAccessor(ta_args, buffer_addr, page_size);

    cb_reserve_back(output_cb, 1);
    uint32_t l1_addr = get_write_ptr(output_cb);

    // Zero the L1 scratch tile
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < page_size / sizeof(uint32_t); i++) {
        ptr[i] = 0;
    }

    uint32_t start_page = core_id * pages_per_core;
    uint32_t end_page = start_page + pages_per_core;
    if (end_page > total_pages) {
        end_page = total_pages;
    }

    for (uint32_t page_id = start_page; page_id < end_page; page_id++) {
        noc_async_write_page(page_id, accessor, l1_addr);
    }
    noc_async_write_barrier();
#endif
}
