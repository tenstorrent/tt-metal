// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writer for the fused residual-add + RMSNorm op: per tile-row, the sum (CB 16, pushed
// by the compute as soon as the add is done) and the normalised row (CB 17).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t sum_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr auto s_args = TensorAccessorArgs<1>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_sum = 16, cb_out = 17;
    const auto ss = TensorAccessor(s_args, sum_addr);
    const auto so = TensorAccessor(o_args, out_addr);
    const uint32_t s_tile = get_tile_size(cb_sum);
    const uint32_t o_tile = get_tile_size(cb_out);

    Noc noc;
    CircularBuffer cs(cb_sum), co(cb_out);
    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t base = (row_start + r) * Wt;
        cs.wait_front(Wt);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc.async_write(cs, ss, s_tile, {.offset_bytes = j * s_tile}, {.page_id = base + j});
        }
        co.wait_front(Wt);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc.async_write(co, so, o_tile, {.offset_bytes = j * o_tile}, {.page_id = base + j});
        }
        noc.async_write_barrier();
        cs.pop_front(Wt);
        co.pop_front(Wt);
    }
}
