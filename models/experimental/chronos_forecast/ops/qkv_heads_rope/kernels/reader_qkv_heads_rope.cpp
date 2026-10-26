// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the fused QKV head split + RoPE. A work unit is one (batch, seq tile, head): the head's
// Q and K tiles go to the compute kernel, its V tiles straight to the writer. cos/sin are batch-shared
// and read once per core. Reads are batched per block of units with one barrier per block.
//
// Compile-time args: head_tiles, num_heads, seq_tiles, units_per_block, scalar_value (bf16),
//                    TensorAccessorArgs(xqkv), TensorAccessorArgs(cos), TensorAccessorArgs(sin)
// Runtime args: xqkv_addr, cos_addr, sin_addr, num_units, unit_start

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_units = get_arg_val<uint32_t>(3);
    const uint32_t unit_start = get_arg_val<uint32_t>(4);

    constexpr uint32_t head_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_heads = get_compile_time_arg_val(1);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t units_per_block = get_compile_time_arg_val(3);
    constexpr uint16_t scalar_value = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto cos_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    constexpr uint32_t row_tiles = 3 * num_heads * head_tiles;
    constexpr uint32_t cs_tiles = seq_tiles * head_tiles;

    constexpr uint32_t qk_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t v_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t cos_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t sin_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t scalar_cb_id = tt::CBIndex::c_4;

    Noc noc;
    CircularBuffer cb_qk(qk_cb_id);
    CircularBuffer cb_v(v_cb_id);
    CircularBuffer cb_cos(cos_cb_id);
    CircularBuffer cb_sin(sin_cb_id);
    CircularBuffer cb_scalar(scalar_cb_id);

    const uint32_t tile_bytes = get_tile_size(qk_cb_id);
    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const auto src = TensorAccessor(src_args, src_addr);
    const auto cos = TensorAccessor(cos_args, cos_addr);
    const auto sin = TensorAccessor(sin_args, sin_addr);

    cb_scalar.reserve_back(1);
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_scalar.get_write_ptr());
    scalar_buffer[0] = scalar_value;
    cb_scalar.push_back(1);

    cb_cos.reserve_back(cs_tiles);
    cb_sin.reserve_back(cs_tiles);
    for (uint32_t t = 0; t < cs_tiles; ++t) {
        noc.async_read(cos, cb_cos, cos_tile_bytes, {.page_id = t}, {.offset_bytes = t * cos_tile_bytes});
        noc.async_read(sin, cb_sin, sin_tile_bytes, {.page_id = t}, {.offset_bytes = t * sin_tile_bytes});
    }
    noc.async_read_barrier();
    cb_cos.push_back(cs_tiles);
    cb_sin.push_back(cs_tiles);

    for (uint32_t done = 0; done < num_units; done += units_per_block) {
        const uint32_t n = (num_units - done) < units_per_block ? (num_units - done) : units_per_block;
        cb_qk.reserve_back(2 * head_tiles * n);
        cb_v.reserve_back(head_tiles * n);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t unit = unit_start + done + i;
            const uint32_t head = unit % num_heads;
            const uint32_t q_page = (unit / num_heads) * row_tiles + head * head_tiles;
            const uint32_t q_off = 2 * i * head_tiles * tile_bytes;
            const uint32_t k_off = q_off + head_tiles * tile_bytes;
            const uint32_t v_off = i * head_tiles * tile_bytes;
            for (uint32_t j = 0; j < head_tiles; ++j) {
                noc.async_read(
                    src, cb_qk, tile_bytes, {.page_id = q_page + j}, {.offset_bytes = q_off + j * tile_bytes});
                noc.async_read(
                    src,
                    cb_qk,
                    tile_bytes,
                    {.page_id = q_page + num_heads * head_tiles + j},
                    {.offset_bytes = k_off + j * tile_bytes});
                noc.async_read(
                    src,
                    cb_v,
                    tile_bytes,
                    {.page_id = q_page + 2 * num_heads * head_tiles + j},
                    {.offset_bytes = v_off + j * tile_bytes});
            }
        }
        noc.async_read_barrier();
        cb_qk.push_back(2 * head_tiles * n);
        cb_v.push_back(head_tiles * n);
    }
}
