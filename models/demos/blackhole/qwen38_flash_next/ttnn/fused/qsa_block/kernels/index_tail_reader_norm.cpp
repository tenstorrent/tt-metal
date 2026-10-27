// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, norm core reader: the constant tiles (1.0 and 0.25 reduce scalers, eps, ones), the gamma rows, the index
// query tile row (do_query), then per lane the raw-key ring with row (P % 4) replaced by the lane's raw key (-0 -> +0,
// as the SFPU one-hot select does).
// Compile-time args: 0 eps bits, then TensorAccessorArgs: index_q, raw_key, kv_block_start, gamma_q, gamma_k, ring,
// kv_row_hit. Runtime args: 0 index_q, 1 raw_key, 2 kv_block_start, 3 gamma_q, 4 gamma_k, 5 ring, 6 kv_row_hit
// addresses, 7 rows, 8 first tile of the index query in index_q, 9 first tile of the raw key in raw_key, 10 lane_first,
// 11 lane_count, 12 do_query.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "index_tail_cbs.h"
#include "positions.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t index_query_addr = get_arg_val<uint32_t>(0);
    const uint32_t raw_key_addr = get_arg_val<uint32_t>(1);
    const uint32_t block_start_addr = get_arg_val<uint32_t>(2);
    const uint32_t gamma_query_addr = get_arg_val<uint32_t>(3);
    const uint32_t gamma_key_addr = get_arg_val<uint32_t>(4);
    const uint32_t ring_addr = get_arg_val<uint32_t>(5);
    const uint32_t row_hit_addr = get_arg_val<uint32_t>(6);
    const uint32_t rows = get_arg_val<uint32_t>(7);
    const uint32_t query_first_tile = get_arg_val<uint32_t>(8);
    const uint32_t raw_key_first_tile = get_arg_val<uint32_t>(9);
    const uint32_t lane_first = get_arg_val<uint32_t>(10);
    const uint32_t lane_count = get_arg_val<uint32_t>(11);
    const uint32_t do_query = get_arg_val<uint32_t>(12);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(0);
    constexpr auto index_query_args = TensorAccessorArgs<1>();
    constexpr auto raw_key_args = TensorAccessorArgs<index_query_args.next_compile_time_args_offset()>();
    constexpr auto block_start_args = TensorAccessorArgs<raw_key_args.next_compile_time_args_offset()>();
    constexpr auto gamma_query_args = TensorAccessorArgs<block_start_args.next_compile_time_args_offset()>();
    constexpr auto gamma_key_args = TensorAccessorArgs<gamma_query_args.next_compile_time_args_offset()>();
    constexpr auto ring_args = TensorAccessorArgs<gamma_key_args.next_compile_time_args_offset()>();
    constexpr auto row_hit_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    const auto index_query = TensorAccessor(index_query_args, index_query_addr);
    const auto raw_key = TensorAccessor(raw_key_args, raw_key_addr);
    const auto block_start = TensorAccessor(block_start_args, block_start_addr);
    const auto gamma_query = TensorAccessor(gamma_query_args, gamma_query_addr);
    const auto gamma_key = TensorAccessor(gamma_key_args, gamma_key_addr);
    const auto ring = TensorAccessor(ring_args, ring_addr);
    const auto row_hit = TensorAccessor(row_hit_args, row_hit_addr);

    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler_ring, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
        0.25f);
    generate_bcast_col_scalar(CircularBuffer(cb_eps), eps_bits);
    CircularBuffer(cb_ones).reserve_back(1);
    tile_rows::fill_words(get_write_ptr(cb_ones), tile_bytes / 4, 0x3F803F80u);
    CircularBuffer(cb_ones).push_back(1);

    CircularBuffer(cb_gamma_q).reserve_back(head_tiles);
    CircularBuffer(cb_gamma_k).reserve_back(head_tiles);
    CircularBuffer(cb_x).reserve_back(head_tiles);
    CircularBuffer(cb_raw).reserve_back(head_tiles);
    CircularBuffer(cb_pos_r).reserve_back(1);
    CircularBuffer(cb_poscr).reserve_back(1);
    const uint32_t gamma_query_l1 = get_write_ptr(cb_gamma_q), gamma_key_l1 = get_write_ptr(cb_gamma_k);
    const uint32_t x_l1 = get_write_ptr(cb_x), raw_key_l1 = get_write_ptr(cb_raw),
                   positions_l1 = get_write_ptr(cb_pos_r);
    {
        for (uint32_t c = 0; c < head_tiles; ++c) {
            noc_async_read_page(c, gamma_key, gamma_key_l1 + c * tile_bytes);
            noc_async_read_page(raw_key_first_tile + c, raw_key, raw_key_l1 + c * tile_bytes);
            if (do_query) {
                noc_async_read_page(c, gamma_query, gamma_query_l1 + c * tile_bytes);
                noc_async_read_page(query_first_tile + c, index_query, x_l1 + c * tile_bytes);
            }
        }
        noc_async_read_barrier();
        tile_rows::read_positions(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(positions_l1),
            block_start,
            row_hit,
            rows,
            get_write_ptr(cb_poscr));
        CircularBuffer(cb_gamma_k).push_back(head_tiles);
        if (do_query) {
            CircularBuffer(cb_gamma_q).push_back(head_tiles);
            CircularBuffer(cb_x).push_back(head_tiles);
        }
    }
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(positions_l1);

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        const uint32_t slot = positions[lane] & 3;
        CircularBuffer(cb_ring).reserve_back(head_tiles);
        const uint32_t ring_l1 = get_write_ptr(cb_ring);
        for (uint32_t c = 0; c < head_tiles; ++c) {
            noc_async_read_page(lane * head_tiles + c, ring, ring_l1 + c * tile_bytes);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t c = 0; c < head_tiles; ++c) {
            copy_tile_row(raw_key_l1 + c * tile_bytes, lane, ring_l1 + c * tile_bytes, slot, true);
        }
        CircularBuffer(cb_ring).push_back(head_tiles);
    }
}
