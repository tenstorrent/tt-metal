// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE gate reader, one core: the key and query global norms (bf16 TILE [1,1,4,2560], global_tiles tiles each) into
// CBs 0 / 1, the scale tile (fp32, every element 2560^-0.5, the chain's scalar-filled operand) into CB 5, the value
// row (bf16 TILE [1,1,1,640], local_tiles tiles; row 0 valid) repeated over the four branch rows into CB 9 (the
// chain's ttnn.repeat: rows 0-3 = row 0, the rest 0 -- each tile a NoC copy of the bf16 zero tile, then row 0's two
// face rows (32 bytes each) copied into rows 0-3 by the RISC), and the reduce scaler tile into CB 3 (waited on and
// ignored by the accurate fold).
// Compile-time args: 0 global_tiles (the builder's GLOBAL_TILES, 80), 1 local_tiles (LOCAL_TILES, 20), then
// TensorAccessorArgs(key), (query), (value), (scale tile), (zero tile).
// Runtime args: the five buffer addresses in that order, 5 the first key/query tile (the lane's global_tiles block),
// 6 the first value tile (the lane's local_tiles block); the 1-row form passes 0, 0.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t global_tiles = get_compile_time_arg_val(0);
constexpr uint32_t local_tiles = get_compile_time_arg_val(1);
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096, bf16_face_bytes = 512, bf16_face_row_bytes = 32;
constexpr uint32_t cb_key = 0, cb_query = 1, cb_scaler = 3, cb_scale = 5, cb_value = 9;

template <typename Accessor>
FORCE_INLINE void read_tiles(
    const Accessor& tensor, DataflowBuffer& buffer, uint32_t tiles, uint32_t tile_bytes, uint32_t first) {
    const Noc noc;
    buffer.reserve_back(tiles);
    for (uint32_t t = 0; t < tiles; ++t) {
        noc.async_read(
            tensor, buffer, tile_bytes, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = t * tile_bytes});
    }
}

void kernel_main() {
    constexpr auto key_args = TensorAccessorArgs<2>();
    constexpr auto query_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto zero_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    const auto key = TensorAccessor(key_args, get_arg_val<uint32_t>(0));
    const auto query = TensorAccessor(query_args, get_arg_val<uint32_t>(1));
    const auto value = TensorAccessor(value_args, get_arg_val<uint32_t>(2));
    const auto scale = TensorAccessor(scale_args, get_arg_val<uint32_t>(3));
    const auto zero = TensorAccessor(zero_args, get_arg_val<uint32_t>(4));
    const uint32_t first_norm_tile = get_arg_val<uint32_t>(5);
    const uint32_t first_value_tile = get_arg_val<uint32_t>(6);
    const Noc noc;
    DataflowBuffer key_dfb(cb_key), query_dfb(cb_query), scale_dfb(cb_scale), value_dfb(cb_value);
    {
        read_tiles(key, key_dfb, global_tiles, bf16_tile_bytes, first_norm_tile);
        read_tiles(query, query_dfb, global_tiles, bf16_tile_bytes, first_norm_tile);
        scale_dfb.reserve_back(1);
        noc.async_read(scale, scale_dfb, fp32_tile_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    }
    // value branches: zero tiles first, then row 0 of each value tile (face 0 and face 1 rows, 32 bytes each)
    value_dfb.reserve_back(local_tiles);
    const uint32_t value_base = value_dfb.get_write_ptr();
    {
        for (uint32_t t = 0; t < local_tiles; ++t) {
            noc.async_read(
                zero,
                value_dfb,
                bf16_tile_bytes,
                {.page_id = 0, .offset_bytes = 0},
                {.offset_bytes = t * bf16_tile_bytes});
        }
        noc.async_read_barrier();
        key_dfb.push_back(global_tiles);
        query_dfb.push_back(global_tiles);
        scale_dfb.push_back(1);
    }
    // row 0 of value tile t into row 0 of its zeroed tile (the zero copies have landed), then duplicated into rows
    // 1-3 on the RISC
    {
        for (uint32_t t = 0; t < local_tiles; ++t) {
            noc.async_read(
                value,
                value_dfb,
                bf16_face_row_bytes,
                {.page_id = first_value_tile + t, .offset_bytes = 0},
                {.offset_bytes = t * bf16_tile_bytes});
            noc.async_read(
                value,
                value_dfb,
                bf16_face_row_bytes,
                {.page_id = first_value_tile + t, .offset_bytes = bf16_face_bytes},
                {.offset_bytes = t * bf16_tile_bytes + bf16_face_bytes});
        }
        noc.async_read_barrier();
        CoreLocalMem<volatile uint32_t> value_words(value_base);
        for (uint32_t t = 0; t < local_tiles; ++t) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t row0 = (t * bf16_tile_bytes + face * bf16_face_bytes) / 4;  // 8 words per bf16 face row
                for (uint32_t r = 1; r < 4; ++r) {
                    for (uint32_t w = 0; w < 8; ++w) {
                        value_words[row0 + r * 8 + w] = value_words[row0 + w];
                    }
                }
            }
        }
        value_dfb.push_back(local_tiles);
    }
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
}
