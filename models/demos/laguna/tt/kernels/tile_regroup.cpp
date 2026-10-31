// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Tile copy for a 32-row tensor regrouped between [1, 1, 32, n * hd] (heads side by side) and [1, n, 32, hd] (one
// head per batch row) (Laguna DFlash draft; tile_regroup.py). With one 32-row tile row both layouts store a head's
// tiles in the same order, so output piece g is the source's tiles [src_off_g, src_off_g + n_g) unchanged: core c of
// piece g copies tiles [c * per, min((c + 1) * per, n_g)).
// Common runtime args: src address, then per piece: dst address, src tile offset, tile count, first core, tiles per core.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);
    constexpr uint32_t grid_x = get_compile_time_arg_val(1);
    constexpr uint32_t page = get_compile_time_arg_val(2);
    constexpr auto s_args = TensorAccessorArgs<3>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(s_args, get_common_arg_val<uint32_t>(0), page);
    const uint32_t c = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t base = 1 + g * 5;
        const uint32_t n = get_common_arg_val<uint32_t>(base + 2);
        const uint32_t first = get_common_arg_val<uint32_t>(base + 3);
        const uint32_t per = get_common_arg_val<uint32_t>(base + 4);
        if (c < first || c >= first + (n + per - 1) / per) {
            continue;
        }
        const auto dst = TensorAccessor(d_args, get_common_arg_val<uint32_t>(base), page);
        const uint32_t off = get_common_arg_val<uint32_t>(base + 1);
        const uint32_t t0 = (c - first) * per;
        const uint32_t t1 = t0 + per < n ? t0 + per : n;
        const uint32_t buf = get_write_ptr(0);
        for (uint32_t t = t0; t < t1; ++t) {
            noc_async_read(src.get_noc_addr(off + t), buf + (t - t0) * page, page);
        }
        noc_async_read_barrier();
        for (uint32_t t = t0; t < t1; ++t) {
            noc_async_write(buf + (t - t0) * page, dst.get_noc_addr(t), page);
        }
        noc_async_write_barrier();
        return;
    }
}
