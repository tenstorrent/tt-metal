// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the fused RoPE + QK RMSNorm: one Wt-tile row per unit, to q_out or k_out (unit order as the reader).
// Compile-time args: Wt, St, RH, accessor args of q_out, k_out. Runtime: q_out, k_out addresses, num_units, start.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);
    const uint32_t start_unit = get_arg_val<uint32_t>(3);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t RH = get_compile_time_arg_val(2);
    constexpr auto q_args = TensorAccessorArgs<3>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();

    constexpr uint32_t out_cb_id = tt::CBIndex::c_16;
    Noc noc;
    CircularBuffer cb_out(out_cb_id);
    const uint32_t bytes = get_tile_size(out_cb_id);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto k = TensorAccessor(k_args, k_addr);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t st = u / (2 * RH);
        const uint32_t bh = u % (2 * RH);
        const uint32_t page = ((bh % RH) * St + st) * Wt;
        cb_out.wait_front(Wt);
        for (uint32_t j = 0; j < Wt; ++j) {
            if (bh < RH) {
                noc.async_write(cb_out, q, bytes, {.offset_bytes = j * bytes}, {.page_id = page + j});
            } else {
                noc.async_write(cb_out, k, bytes, {.offset_bytes = j * bytes}, {.page_id = page + j});
            }
        }
        noc.async_writes_flushed();
        cb_out.pop_front(Wt);
    }
    noc.async_write_barrier();
}
