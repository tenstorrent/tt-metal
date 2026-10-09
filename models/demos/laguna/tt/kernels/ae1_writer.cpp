// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode attention epilogue (Laguna), writer: head h's gated row (row h of the 4 result tiles) becomes columns
// h * 128 .. h * 128 + 127 of row 0 of the [1, H * 128] WO input (any layout the accessor describes, e.g. WO's
// width shards): per tile two 32-byte pieces, face 0 / 2 row -> face 0 row 0, face 1 / 3 row -> face 1 row 0.
// Rows 1..31 of the output are left as they are: the matmul keeps rows apart and the all-reduce reads row 0 only.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr uint32_t T = 2048, cb_out = 16;
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), T);
    cb_wait_front(cb_out, 4);
    const uint32_t base = get_read_ptr(cb_out);
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t roff = (h < 16 ? 0 : 1024) + (h % 16) * 32;
        for (uint32_t j = 0; j < 4; ++j) {
            const uint64_t dst = o.get_noc_addr(h * 4 + j);
            noc_async_write(base + j * T + roff, dst, 32);
            noc_async_write(base + j * T + roff + 512, dst + 512, 32);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 4);
}
