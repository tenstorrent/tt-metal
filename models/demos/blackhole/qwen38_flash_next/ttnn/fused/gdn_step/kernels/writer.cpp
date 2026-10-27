// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per (lane, head) item: the 16 new state tiles to the lane's state and the head's 4 gated tiles to the output (whole
// tiles when rows == 1, else only the lane's row of each tile).
// Compile-time args: 0 rows; then TensorAccessorArgs of state and out.  Runtime args: 0 state, 1 out addresses, then
// items, then (lane, head) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_outs = 25, cb_outg = 26;
constexpr uint32_t value_heads = 12, head_tiles = 4, head_state_tiles = 16;
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096, face_bytes = 512, half_row_bytes = 32;
}  // namespace

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto state_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    uint32_t arg = 0;
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t head = get_arg_val<uint32_t>(arg++);

        cb_wait_front(cb_outs, head_state_tiles);
        {
            const uint32_t l1 = get_read_ptr(cb_outs);
            for (uint32_t t = 0; t < head_state_tiles; ++t) {
                noc_async_write_page(
                    (lane * value_heads + head) * head_state_tiles + t, state, l1 + t * fp32_tile_bytes);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_outs, head_state_tiles);

        cb_wait_front(cb_outg, head_tiles);
        {
            const uint32_t l1 = get_read_ptr(cb_outg);
            for (uint32_t c = 0; c < head_tiles; ++c) {
                const uint32_t page = head * head_tiles + c;
                const uint32_t src = l1 + c * bf16_tile_bytes;
                if constexpr (rows == 1) {
                    noc_async_write_page(page, out, src);
                } else {
                    const uint32_t row = ((lane >> 4) * 2) * face_bytes + (lane & 15) * half_row_bytes;
                    noc_async_write(src + row, out.get_noc_addr(page, row), half_row_bytes);
                    noc_async_write(src + row + face_bytes, out.get_noc_addr(page, row + face_bytes), half_row_bytes);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_outg, head_tiles);
    }
}
