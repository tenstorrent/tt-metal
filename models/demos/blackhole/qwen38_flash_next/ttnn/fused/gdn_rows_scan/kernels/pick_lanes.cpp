// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The lanes commit under the fold: lane u's committed count c_u (fp32 [1, B, 1, 1] device tensor, page u) picks
// prefix slot lane * R + c_u - 1 into lane u's recurrent state; c_u = 0 writes nothing, so that lane's state is kept.
// Data movement only; the counts are read on device so a traced replay follows the rewritten values.
// Per item (lane, head, column tile): the four state tiles (kt, col).
// Compile-time args: 0 rows (R), 1 lanes (B); then TensorAccessorArgs of counts, prefix, recurrent.
// Runtime args: 0 counts, 1 prefix, 2 recurrent addresses, 3 items, then (lane, head, col) triples.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_counts = 0, cb_tiles = 1;
constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t lanes = get_compile_time_arg_val(1);
constexpr uint32_t value_heads = 12, head_tiles = 4, head_state_tiles = head_tiles * head_tiles;
constexpr uint32_t fp32_tile_bytes = 4096;
static_assert(rows >= 1 && lanes >= 1 && lanes * rows <= 32, "B lanes x R rows in one tile");
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
    const Noc noc;

    uint32_t committed[lanes];
    {
        CircularBuffer(cb_counts).reserve_back(lanes);
        const CoreLocalMem<volatile float> counts_l1(CircularBuffer(cb_counts).get_write_ptr());
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            noc.async_read(
                counts,
                counts_l1,
                counts.get_aligned_page_size(),
                {.page_id = lane},
                {.offset_bytes = lane * fp32_tile_bytes});
        }
        noc.async_read_barrier();
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const float value = counts_l1[lane * fp32_tile_bytes / sizeof(float)];
            const uint32_t count = value <= 0.0f ? 0u : static_cast<uint32_t>(value);
            committed[lane] = count > rows ? rows : count;
        }
    }
    // cb_tiles is scratch: reserved once and never pushed, each item's tiles are read into it and written out of it
    CircularBuffer(cb_tiles).reserve_back(head_tiles);
    const CoreLocalMem<uint32_t> tiles(CircularBuffer(cb_tiles).get_write_ptr());
    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t col = get_arg_val<uint32_t>(arg++);
        const uint32_t count = committed[lane];
        if (count == 0) {
            continue;  // the lane keeps its state
        }
        const uint32_t slot = lane * rows + count - 1;
        for (uint32_t kt = 0; kt < head_tiles; ++kt) {
            noc.async_read(
                prefix,
                tiles,
                prefix.get_aligned_page_size(),
                {.page_id = (slot * value_heads + head) * head_state_tiles + kt * head_tiles + col},
                {.offset_bytes = kt * fp32_tile_bytes});
        }
        noc.async_read_barrier();
        for (uint32_t kt = 0; kt < head_tiles; ++kt) {
            noc.async_write(
                tiles,
                state,
                state.get_aligned_page_size(),
                {.offset_bytes = kt * fp32_tile_bytes},
                {.page_id = (lane * value_heads + head) * head_state_tiles + kt * head_tiles + col});
        }
        noc.async_write_barrier();
    }
}
