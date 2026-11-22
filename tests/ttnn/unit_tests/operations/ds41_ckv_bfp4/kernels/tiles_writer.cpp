// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Probe: the first pass's tiles to DRAM (raw tile bytes, page = one tile), the later passes' tiles dropped.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_pass = get_compile_time_arg_val(1);
    constexpr uint32_t passes = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    for (uint32_t t = 0; t < tiles_per_pass * passes; ++t) {
        cb_wait_front(cb_out, 1);
        if (t < tiles_per_pass) {
            noc_async_write(get_read_ptr(cb_out), out.get_noc_addr(t), tile_bytes);
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, 1);
    }
}
