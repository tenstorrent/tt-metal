// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// Writer kernel: drains output CB row-by-row to DRAM.
void kernel_main() {
    uint32_t arg_idx = 0U;
    const uint32_t output_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto cb_output = tt::CBIndex::c_14;
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    const uint32_t tile_bytes = get_tile_size(cb_output);
    constexpr auto output_args = TensorAccessorArgs<2>();
    const auto output_addr_generator = TensorAccessor(output_args, output_address, tile_bytes);

    for (uint32_t r = start_row; r < (start_row + num_rows_to_process); ++r) {
        write_full_row_tiles(cb_output, output_addr_generator, Wt, block_size, tile_bytes, r * Wt);
    }

#ifdef POLYNORM_DEBUG
    constexpr auto cb_debug = tt::CBIndex::c_18;
    if (num_rows_to_process > 0U && start_row == 0U) {
        cb_wait_front(cb_debug, 6U);
        DPRINT << "POLYNORM_DEBUG: sum_x2" << ENDL();
        print_tile(cb_debug, 0U, false);
        DPRINT << "POLYNORM_DEBUG: sum_x4" << ENDL();
        print_tile(cb_debug, 1U, false);
        DPRINT << "POLYNORM_DEBUG: sum_x6" << ENDL();
        print_tile(cb_debug, 2U, false);
        DPRINT << "POLYNORM_DEBUG: inv_rms_x" << ENDL();
        print_tile(cb_debug, 3U, false);
        DPRINT << "POLYNORM_DEBUG: inv_rms_x2" << ENDL();
        print_tile(cb_debug, 4U, false);
        DPRINT << "POLYNORM_DEBUG: inv_rms_x3" << ENDL();
        print_tile(cb_debug, 5U, false);
        cb_pop_front(cb_debug, 6U);
    }
#endif
}
