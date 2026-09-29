// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The lanes commit under the fold: lane u's committed count c_u = (a_u + 1) * active_u (the pass's fp32 [1, B, 1, 1]
// device tensor: page u, element [0, 0]) picks prefix slot lane * R + c_u - 1 (lane u's state after c_u rows) into
// lane u's recurrent state; c_u = 0 (an inactive lane, or one seeded and not yet verified) writes NOTHING, so that
// lane's state stays bitwise -- data movement only, no arithmetic.  The counts are read on the device so a traced
// replay follows the rewritten values.  Per item (lane, head, column tile): the four state tiles (kt, col) of the
// head.  Compile-time args: 0 ROWS (R), 1 LANES (B); then TensorAccessorArgs of counts, prefix, recurrent.
// Runtime args: 0 counts, 1 prefix, 2 recurrent addresses, 3 items, then (lane, head, col) triples.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_COUNTS = 0, CB_TILES = 1;
constexpr uint32_t ROWS = get_compile_time_arg_val(0);
constexpr uint32_t LANES = get_compile_time_arg_val(1);
constexpr uint32_t HEADS = 12, HT = 4, STATE_TILES = HT * HT, FP32_TILE = 4096;
static_assert(ROWS >= 1 && LANES >= 1 && LANES * ROWS <= 32, "B lanes x R rows in one tile");
}  // namespace

void kernel_main() {
    constexpr auto counts_args = TensorAccessorArgs<2>();
    constexpr auto prefix_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
    uint32_t arg = 0;
    const uint32_t counts_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t prefix_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto counts = TensorAccessor(counts_args, counts_addr);
    const auto prefix = TensorAccessor(prefix_args, prefix_addr);
    const auto state = TensorAccessor(state_args, state_addr);

    uint32_t committed[LANES];
    {
        FUSED_ZONE("fz_gsc_lp_counts");
        cb_reserve_back(CB_COUNTS, LANES);
        const uint32_t l1 = get_write_ptr(CB_COUNTS);
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            noc_async_read_page(lane, counts, l1 + lane * FP32_TILE);
        }
        noc_async_read_barrier();
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            const float value = *reinterpret_cast<volatile tt_l1_ptr float*>(l1 + lane * FP32_TILE);
            uint32_t c = value <= 0.0f ? 0u : static_cast<uint32_t>(value);
            committed[lane] = c > ROWS ? ROWS : c;
        }
    }
    cb_reserve_back(CB_TILES, HT);
    const uint32_t tiles = get_write_ptr(CB_TILES);
    for (uint32_t item = 0; item < items; ++item) {
        FUSED_ZONE("fz_gsc_lp_pick");
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t col = get_arg_val<uint32_t>(arg++);
        const uint32_t c = committed[lane];
        if (c == 0) {
            continue;  // the lane keeps its state
        }
        const uint32_t slot = lane * ROWS + c - 1;
        for (uint32_t kt = 0; kt < HT; ++kt) {
            noc_async_read_page((slot * HEADS + head) * STATE_TILES + kt * HT + col, prefix, tiles + kt * FP32_TILE);
        }
        noc_async_read_barrier();
        for (uint32_t kt = 0; kt < HT; ++kt) {
            noc_async_write_page((lane * HEADS + head) * STATE_TILES + kt * HT + col, state, tiles + kt * FP32_TILE);
        }
        noc_async_write_barrier();
    }
}
