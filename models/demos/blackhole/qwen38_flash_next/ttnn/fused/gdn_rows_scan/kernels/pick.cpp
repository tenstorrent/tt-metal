// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The verify rows' commit under the fold: the state after `accepted + 1` rows is prefix slot `accepted`, so the
// commit copies that slot into the layer's recurrent state -- data movement only, no arithmetic.  The accept count
// is the pass's fp32 [1, 1, 1, 1] device scalar (element [0, 0] of one tile), read here so a traced replay follows the
// rewritten value.  Per item (head, column tile): the four state tiles (kt, col) of the head.
// Compile-time args: 0 ROWS; then TensorAccessorArgs of accepted, prefix, recurrent.  Runtime args: 0 accepted,
// 1 prefix, 2 recurrent addresses, 3 items, then (head, col) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_ACCEPT = 0, CB_TILES = 1;
constexpr uint32_t ROWS = get_compile_time_arg_val(0);
constexpr uint32_t HEADS = 12, HT = 4, STATE_TILES = HT * HT, FP32_TILE = 4096;
}  // namespace

void kernel_main() {
    constexpr auto accept_args = TensorAccessorArgs<1>();
    constexpr auto prefix_args = TensorAccessorArgs<accept_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
    uint32_t arg = 0;
    const uint32_t accept_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t prefix_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto accept = TensorAccessor(accept_args, accept_addr);
    const auto prefix = TensorAccessor(prefix_args, prefix_addr);
    const auto state = TensorAccessor(state_args, state_addr);

    uint32_t accepted = 0;
    {
        FUSED_ZONE("fz_gsc_p_accept");
        cb_reserve_back(CB_ACCEPT, 1);
        const uint32_t l1 = get_write_ptr(CB_ACCEPT);
        noc_async_read_page(0, accept, l1);
        noc_async_read_barrier();
        const float value = *reinterpret_cast<volatile tt_l1_ptr float*>(l1);
        accepted = value <= 0.0f ? 0u : static_cast<uint32_t>(value);
        if (accepted >= ROWS) {
            accepted = ROWS - 1;
        }
    }
    cb_reserve_back(CB_TILES, HT);
    const uint32_t tiles = get_write_ptr(CB_TILES);
    for (uint32_t item = 0; item < items; ++item) {
        FUSED_ZONE("fz_gsc_p_pick");
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t col = get_arg_val<uint32_t>(arg++);
        for (uint32_t kt = 0; kt < HT; ++kt) {
            noc_async_read_page(
                (accepted * HEADS + head) * STATE_TILES + kt * HT + col, prefix, tiles + kt * FP32_TILE);
        }
        noc_async_read_barrier();
        for (uint32_t kt = 0; kt < HT; ++kt) {
            noc_async_write_page(head * STATE_TILES + kt * HT + col, state, tiles + kt * FP32_TILE);
        }
        noc_async_write_barrier();
    }
}
