// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the fused QKV head split + RoPE: rotated Q/K rows from the compute kernel and V rows from the
// reader go to [B, H, S, Dh] outputs. A block of units is written, flushed and popped together; one write
// barrier at the end.
//
// Compile-time args: head_tiles, num_heads, seq_tiles, units_per_block,
//                    TensorAccessorArgs(q), TensorAccessorArgs(k), TensorAccessorArgs(v)
// Runtime args: q_addr, k_addr, v_addr, num_units, unit_start

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);
    const uint32_t unit_start = get_arg_val<uint32_t>(4);

    constexpr uint32_t head_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_heads = get_compile_time_arg_val(1);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t units_per_block = get_compile_time_arg_val(3);
    constexpr auto q_args = TensorAccessorArgs<4>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t v_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t qk_cb_id = tt::CBIndex::c_16;

    Noc noc;
    CircularBuffer cb_qk(qk_cb_id);
    CircularBuffer cb_v(v_cb_id);
    const uint32_t tile_bytes = get_tile_size(qk_cb_id);
    const uint32_t v_tile_bytes = get_tile_size(v_cb_id);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto k = TensorAccessor(k_args, k_addr);
    const auto v = TensorAccessor(v_args, v_addr);

    for (uint32_t done = 0; done < num_units; done += units_per_block) {
        const uint32_t n = (num_units - done) < units_per_block ? (num_units - done) : units_per_block;
        cb_v.wait_front(head_tiles * n);
        cb_qk.wait_front(2 * head_tiles * n);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t unit = unit_start + done + i;
            const uint32_t head = unit % num_heads;
            const uint32_t row = unit / num_heads;
            const uint32_t batch = row / seq_tiles;
            const uint32_t s_tile = row - batch * seq_tiles;
            const uint32_t page = ((batch * num_heads + head) * seq_tiles + s_tile) * head_tiles;
            const uint32_t q_off = 2 * i * head_tiles * tile_bytes;
            const uint32_t k_off = q_off + head_tiles * tile_bytes;
            const uint32_t v_off = i * head_tiles * v_tile_bytes;
            for (uint32_t j = 0; j < head_tiles; ++j) {
                noc.async_write(cb_qk, q, tile_bytes, {.offset_bytes = q_off + j * tile_bytes}, {.page_id = page + j});
                noc.async_write(cb_qk, k, tile_bytes, {.offset_bytes = k_off + j * tile_bytes}, {.page_id = page + j});
                noc.async_write(
                    cb_v, v, v_tile_bytes, {.offset_bytes = v_off + j * v_tile_bytes}, {.page_id = page + j});
            }
        }
        noc.async_writes_flushed();
        cb_qk.pop_front(2 * head_tiles * n);
        cb_v.pop_front(head_tiles * n);
    }
    noc.async_write_barrier();
}
