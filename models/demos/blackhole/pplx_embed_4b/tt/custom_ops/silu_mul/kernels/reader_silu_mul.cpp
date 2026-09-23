// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// silu(a)*b reader: CH tiles of a and of b per unit (same tile ids), one barrier per unit.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_units = get_arg_val<uint32_t>(2);
    const uint32_t unit_start = get_arg_val<uint32_t>(3);
    constexpr uint32_t CH = get_compile_time_arg_val(0);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(1);
    constexpr auto a_args = TensorAccessorArgs<2>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_a = 0, cb_b = 1;
    const auto sa = TensorAccessor(a_args, a_addr);
    const auto sb = TensorAccessor(b_args, b_addr);
    const uint32_t ta = get_tile_size(cb_a), tb = get_tile_size(cb_b);
    Noc noc;
    CircularBuffer ca(cb_a), cbb(cb_b);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t base = (unit_start + u) * CH;
        ca.reserve_back(CH);
        cbb.reserve_back(CH);
        for (uint32_t i = 0; i < CH; ++i) {
            const uint32_t t = base + i < n_tiles ? base + i : n_tiles - 1;  // ragged tail re-reads the last tile
            noc.async_read(sa, ca, ta, {.page_id = t}, {.offset_bytes = i * ta});
            noc.async_read(sb, cbb, tb, {.page_id = t}, {.offset_bytes = i * tb});
        }
        noc.async_read_barrier();
        ca.push_back(CH);
        cbb.push_back(CH);
    }
}
