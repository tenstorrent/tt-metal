// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The verify rows' commit under the fold: copies prefix slot `accepted` (the state after accepted + 1 rows) into the
// recurrent state; data movement only. The accept count is the fp32 [1, 1, 1, 1] device scalar, read on device so a
// traced replay follows the rewritten value. Per item (head, column tile): the four state tiles (kt, col).
// Compile-time args: 0 rows; then TensorAccessorArgs of accepted, prefix, recurrent.  Runtime args: 0 accepted,
// 1 prefix, 2 recurrent addresses, 3 items, then (head, col) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_accept = 0, cb_tiles = 1;
constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t value_heads = 12, head_tiles = 4, head_state_tiles = head_tiles * head_tiles;
constexpr uint32_t fp32_tile_bytes = 4096;
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
    const Noc noc;

    uint32_t accepted = 0;
    {
        CircularBuffer(cb_accept).reserve_back(1);
        const CoreLocalMem<volatile float> accept_l1(CircularBuffer(cb_accept).get_write_ptr());
        noc.async_read(accept, accept_l1, accept.get_aligned_page_size(), {.page_id = 0}, {});
        noc.async_read_barrier();
        const float value = *accept_l1;
        accepted = value <= 0.0f ? 0u : static_cast<uint32_t>(value);
        if (accepted >= rows) {
            accepted = rows - 1;
        }
    }
    // cb_tiles is scratch: reserved once and never pushed, each item's tiles are read into it and written out of it
    CircularBuffer(cb_tiles).reserve_back(head_tiles);
    const CoreLocalMem<uint32_t> tiles(CircularBuffer(cb_tiles).get_write_ptr());
    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t col = get_arg_val<uint32_t>(arg++);
        for (uint32_t kt = 0; kt < head_tiles; ++kt) {
            noc.async_read(
                prefix,
                tiles,
                prefix.get_aligned_page_size(),
                {.page_id = (accepted * value_heads + head) * head_state_tiles + kt * head_tiles + col},
                {.offset_bytes = kt * fp32_tile_bytes});
        }
        noc.async_read_barrier();
        for (uint32_t kt = 0; kt < head_tiles; ++kt) {
            noc.async_write(
                tiles,
                state,
                state.get_aligned_page_size(),
                {.offset_bytes = kt * fp32_tile_bytes},
                {.page_id = head * head_state_tiles + kt * head_tiles + col});
        }
        noc.async_write_barrier();
    }
}
