// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_units = get_arg_val<uint32_t>(1);
    const uint32_t unit_start = get_arg_val<uint32_t>(2);
    constexpr uint32_t CH = get_compile_time_arg_val(0);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    constexpr uint32_t cb_out = 16;
    const auto so = TensorAccessor(o_args, out_addr);
    const uint32_t to = get_tile_size(cb_out);
    Noc noc;
    CircularBuffer co(cb_out);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t base = (unit_start + u) * CH;
        co.wait_front(CH);
        for (uint32_t i = 0; i < CH; ++i) {
            if (base + i < n_tiles) {
                noc.async_write(co, so, to, {.offset_bytes = i * to}, {.page_id = base + i});
            }
        }
        noc.async_write_barrier();
        co.pop_front(CH);
    }
}
