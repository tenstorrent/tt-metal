// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Decode attention epilogue (1..32 rows; Laguna), writer: on the cores of token row b, head h's gated row (row h of
// the result tiles) becomes columns h * 128 .. h * 128 + 127 of row s = row_out + b of the [T, H * 128] WO input (any
// layout the accessor describes, e.g. WO's width shards): per tile two 32-byte pieces, face 0 / 2 row -> row s % 32
// of face ((s % 32) / 16) * 2, face 1 / 3 row -> the same row of the next face. Rows past the call's rows are left
// as they are: the matmul keeps rows apart and the all-reduce reads rows < B only.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t TJ = get_compile_time_arg_val(1);  // head_dim tiles per core
    constexpr uint32_t grid_x = get_compile_time_arg_val(2);
    constexpr uint32_t row_out = get_compile_time_arg_val(5);  // output row of token row 0
    constexpr uint32_t Wo = H * 4;                              // output width in tiles
    constexpr auto o_args = TensorAccessorArgs<6>();
    constexpr uint32_t T = 2048, cb_out = 16;
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), T);
    cb_wait_front(cb_out, TJ);
    const uint32_t base = get_read_ptr(cb_out);
    const uint32_t L = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t b = L / (4 / TJ);
    const uint32_t j0 = (L % (4 / TJ)) * TJ;
    const uint32_t s = row_out + b, r = s % 32, t0 = (s / 32) * Wo;
    const uint32_t doff = ((r / 16) * 2) * 512 + (r % 16) * 32;
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t roff = (h < 16 ? 0 : 1024) + (h % 16) * 32;
        for (uint32_t j = 0; j < TJ; ++j) {
            const uint64_t dst = o.get_noc_addr(t0 + h * 4 + j0 + j) + doff;
            noc_async_write(base + j * T + roff, dst, 32);
            noc_async_write(base + j * T + roff + 512, dst + 512, 32);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, TJ);
}
