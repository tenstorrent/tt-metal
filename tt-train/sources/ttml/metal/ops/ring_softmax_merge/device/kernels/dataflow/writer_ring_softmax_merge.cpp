// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per row tile: the new lse tile back over the running one, the new output
// row back over the running one. In place is safe: the reader on this core
// read the row before the compute produced its replacement, and every row
// tile belongs to one core.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t lse_acc_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_acc_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t rows = get_arg_val<uint32_t>(arg++);
    const uint32_t first_row = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr auto lse_acc_args = TensorAccessorArgs<1>();
    constexpr auto out_acc_args = TensorAccessorArgs<lse_acc_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_lse_new = tt::CBIndex::c_6;
    constexpr uint32_t cb_out_new = tt::CBIndex::c_7;
    const uint32_t fp32_bytes = get_tile_size(cb_lse_new);

    const auto lse_acc = TensorAccessor(lse_acc_args, lse_acc_addr, fp32_bytes);
    const auto out_acc = TensorAccessor(out_acc_args, out_acc_addr, fp32_bytes);

    for (uint32_t r = first_row; r < first_row + rows; ++r) {
        write_tiles_by_row(cb_lse_new, lse_acc, r, 1U, fp32_bytes, 1U);
        write_tiles_by_row(cb_out_new, out_acc, r * Wt, Wt, fp32_bytes, Wt);
    }
}
