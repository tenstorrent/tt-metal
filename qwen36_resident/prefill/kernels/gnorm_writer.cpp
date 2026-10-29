// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN output gate, writer (BRISC): the 4 output tiles of group (t, h) to token-major tiles
// (t, 4h + 0..3) of y ([C, heads * 128]).
// Compile-time args: 0 heads. Runtime args: 0 first group, 1 groups, 2 y address.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t heads = get_compile_time_arg_val(0);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t groups = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 7;
    constexpr uint32_t tile = get_tile_size(cb_out);
    const InterleavedAddrGenFast<true> y{
        .bank_base_address = get_arg_val<uint32_t>(2), .page_size = tile, .data_format = DataFormat::Float16_b};
    for (uint32_t g = first; g < first + groups; g++) {
        const uint32_t t = g / heads, h = g % heads;
        for (uint32_t d = 0; d < 4; d++) {
            cb_wait_front(cb_out, 1);
            noc_async_write_tile(t * heads * 4 + h * 4 + d, y, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
