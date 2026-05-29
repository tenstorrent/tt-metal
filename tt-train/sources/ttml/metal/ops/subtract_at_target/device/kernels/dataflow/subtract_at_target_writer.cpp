// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_output_idx = tt::CBIndex::c_2;

    const uint32_t tile_bytes = get_tile_size(cb_output_idx);
    const auto output_addr_gen = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t r = start_row; r < start_row + num_rows; ++r) {
        const uint32_t row_start = r * Wt;
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            write_tiles_by_row(cb_output_idx, output_addr_gen, row_start + wt, 1U, tile_bytes, 1U);
        }
    }
}
