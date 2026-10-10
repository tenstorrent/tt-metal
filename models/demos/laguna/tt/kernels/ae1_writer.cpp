// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Decode attention epilogue (batch 1, or up to 8 rows; Laguna), writer: on core b, head h's gated row (row h of the 4
// result tiles) becomes columns h * 128 .. h * 128 + 127 of row b of the [B, H * 128] WO input (any layout the
// accessor describes, e.g. WO's width shards): per tile two 32-byte pieces, face 0 / 2 row -> row b of face
// (b / 16) * 2, face 1 / 3 row -> row b of the next face. Rows B..31 of the output are left as they are: the
// matmul keeps rows apart and the all-reduce reads rows < B only.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t TJ = get_compile_time_arg_val(1);  // head_dim tiles per core (core y = tile group)
    constexpr auto o_args = TensorAccessorArgs<2>();
    constexpr uint32_t T = 2048, cb_out = 16;
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), T);
    cb_wait_front(cb_out, TJ);
    const uint32_t base = get_read_ptr(cb_out);
    const uint32_t b = get_absolute_logical_x();
    const uint32_t j0 = get_absolute_logical_y() * TJ;
    const uint32_t doff = ((b / 16) * 2) * 512 + (b % 16) * 32;
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t roff = (h < 16 ? 0 : 1024) + (h % 16) * 32;
        for (uint32_t j = 0; j < TJ; ++j) {
            const uint64_t dst = o.get_noc_addr(h * 4 + j0 + j) + doff;
            noc_async_write(base + j * T + roff, dst, 32);
            noc_async_write(base + j * T + roff + 512, dst + 512, 32);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, TJ);
}
