// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per (lane, head) item: the 16 new state tiles to the lane's state, the head's 4 gated tiles to the output (whole
// tiles when rows == 1, else only the lane's row of each tile), and with DEBUG_TAPS the item's 18 tap tiles.
// Compile-time args: 0 rows; then TensorAccessorArgs of state, out [, debug].  Runtime args: 0 state, 1 out
// [, 2 debug] addresses, then items, then (lane, head) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_OUTS = 25, CB_OUTG = 26, CB_DBG = 28;
constexpr uint32_t HEADS = 12, HT = 4, STATE_TILES = 16, DBG_TILES = 28;
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096, FACE_BYTES = 512, HALF_ROW_BYTES = 32;
}  // namespace

void kernel_main() {
    constexpr uint32_t ROWS = get_compile_time_arg_val(0);
    constexpr auto state_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
#ifdef DEBUG_TAPS
    constexpr auto dbg_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
#endif
    uint32_t arg = 0;
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
#ifdef DEBUG_TAPS
    const uint32_t dbg_addr = get_arg_val<uint32_t>(arg++);
    const auto dbg = TensorAccessor(dbg_args, dbg_addr);
#endif
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t item = 0; item < items; ++item) {
        FUSED_ZONE("fz_gs_w_item");
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t head = get_arg_val<uint32_t>(arg++);

        cb_wait_front(CB_OUTS, STATE_TILES);
        {
            const uint32_t l1 = get_read_ptr(CB_OUTS);
            for (uint32_t t = 0; t < STATE_TILES; ++t) {
                noc_async_write_page((lane * HEADS + head) * STATE_TILES + t, state, l1 + t * FP32_TILE);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUTS, STATE_TILES);

        cb_wait_front(CB_OUTG, HT);
        {
            const uint32_t l1 = get_read_ptr(CB_OUTG);
            for (uint32_t c = 0; c < HT; ++c) {
                const uint32_t page = head * HT + c;
                const uint32_t src = l1 + c * BF16_TILE;
                if constexpr (ROWS == 1) {
                    noc_async_write_page(page, out, src);
                } else {
                    const uint32_t row = ((lane >> 4) * 2) * FACE_BYTES + (lane & 15) * HALF_ROW_BYTES;
                    noc_async_write(src + row, out.get_noc_addr(page, row), HALF_ROW_BYTES);
                    noc_async_write(src + row + FACE_BYTES, out.get_noc_addr(page, row + FACE_BYTES), HALF_ROW_BYTES);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUTG, HT);

#ifdef DEBUG_TAPS
        cb_wait_front(CB_DBG, DBG_TILES);
        {
            const uint32_t l1 = get_read_ptr(CB_DBG);
            for (uint32_t t = 0; t < DBG_TILES; ++t) {
                noc_async_write_page((lane * HEADS + head) * DBG_TILES + t, dbg, l1 + t * FP32_TILE);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_DBG, DBG_TILES);
#endif
    }
}
