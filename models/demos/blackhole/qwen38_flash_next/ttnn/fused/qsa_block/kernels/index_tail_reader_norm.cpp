// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, norm core reader (one core pair per lane; the first pair also takes the index query tile): the constant
// tiles (1.0 and 0.25 reduce scalers, eps, ones), the gamma rows, the index query tile row (do_query), then for this
// core's lanes (lane_first .. lane_first + lane_count - 1) the lane's raw-key ring with row (P % 4) replaced by the
// lane's raw key (-0 -> +0 as the chain's SFPU one-hot select does).
// Compile-time args: 0 eps bits, then TensorAccessorArgs: index_q, raw_key, kv_block_start, gamma_q, gamma_k, ring,
// kv_row_hit. Runtime args: 0 index_q, 1 raw_key, 2 kv_block_start, 3 gamma_q, 4 gamma_k, 5 ring, 6 kv_row_hit
// addresses, 7 rows, 8 first tile of the index query in index_q, 9 first tile of the raw key in raw_key (0 for the
// separate projection shards; the windows' first tiles when both are the merged projection shard), 10 lane_first,
// 11 lane_count (this core pair's lanes), 12 do_query (the first pair also takes the index query tile: its rows are all
// the lanes' rows).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "index_tail_cbs.h"
#include "positions.h"
#include "../../kernels/zones.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t raw_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t gq_addr = get_arg_val<uint32_t>(3);
    const uint32_t gk_addr = get_arg_val<uint32_t>(4);
    const uint32_t ring_addr = get_arg_val<uint32_t>(5);
    const uint32_t hit_addr = get_arg_val<uint32_t>(6);
    const uint32_t rows = get_arg_val<uint32_t>(7);
    const uint32_t q_first = get_arg_val<uint32_t>(8);
    const uint32_t raw_first = get_arg_val<uint32_t>(9);
    const uint32_t lane_first = get_arg_val<uint32_t>(10);
    const uint32_t lane_count = get_arg_val<uint32_t>(11);
    const uint32_t do_query = get_arg_val<uint32_t>(12);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(0);
    constexpr auto q_args = TensorAccessorArgs<1>();
    constexpr auto raw_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<raw_args.next_compile_time_args_offset()>();
    constexpr auto gq_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto gk_args = TensorAccessorArgs<gq_args.next_compile_time_args_offset()>();
    constexpr auto ring_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();
    constexpr auto hit_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(q_args, q_addr);
    const auto raw = TensorAccessor(raw_args, raw_addr);
    const auto pos = TensorAccessor(pos_args, pos_addr);
    const auto gq = TensorAccessor(gq_args, gq_addr);
    const auto gk = TensorAccessor(gk_args, gk_addr);
    const auto ring = TensorAccessor(ring_args, ring_addr);
    const auto hit = TensorAccessor(hit_args, hit_addr);

    dataflow_kernel_lib::prepare_reduce_scaler<CB_SCALER, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    dataflow_kernel_lib::prepare_reduce_scaler<CB_SCALER_RING, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
        0.25f);
    generate_bcast_col_scalar(CircularBuffer(CB_EPS), eps_bits);
    cb_reserve_back(CB_ONES, 1);
    tile_rows::fill_words(get_write_ptr(CB_ONES), TILE_BYTES / 4, 0x3F803F80u);
    cb_push_back(CB_ONES, 1);

    cb_reserve_back(CB_GAMMA_Q, HEAD_TILES);
    cb_reserve_back(CB_GAMMA_K, HEAD_TILES);
    cb_reserve_back(CB_X, HEAD_TILES);
    cb_reserve_back(CB_RAW, HEAD_TILES);
    cb_reserve_back(CB_POS_R, 1);
    cb_reserve_back(CB_POSCR, 1);
    const uint32_t gq_l1 = get_write_ptr(CB_GAMMA_Q), gk_l1 = get_write_ptr(CB_GAMMA_K);
    const uint32_t x_l1 = get_write_ptr(CB_X), raw_l1 = get_write_ptr(CB_RAW), pos_l1 = get_write_ptr(CB_POS_R);
    {
        FUSED_ZONE("fz_qs_it_rn_setup");
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            noc_async_read_page(c, gk, gk_l1 + c * TILE_BYTES);
            noc_async_read_page(raw_first + c, raw, raw_l1 + c * TILE_BYTES);
            if (do_query) {
                noc_async_read_page(c, gq, gq_l1 + c * TILE_BYTES);
                noc_async_read_page(q_first + c, q, x_l1 + c * TILE_BYTES);
            }
        }
        noc_async_read_barrier();
        tile_rows::read_positions(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1), pos, hit, rows, get_write_ptr(CB_POSCR));
        cb_push_back(CB_GAMMA_K, HEAD_TILES);
        if (do_query) {
            cb_push_back(CB_GAMMA_Q, HEAD_TILES);
            cb_push_back(CB_X, HEAD_TILES);
        }
    }
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);

    for (uint32_t i = 0; i < lane_count; ++i) {
        FUSED_ZONE("fz_qs_it_rn_lane");
        const uint32_t lane = lane_first + i;
        const uint32_t slot = positions[lane] & 3;
        cb_reserve_back(CB_RING, HEAD_TILES);
        const uint32_t ring_l1 = get_write_ptr(CB_RING);
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            noc_async_read_page(lane * HEAD_TILES + c, ring, ring_l1 + c * TILE_BYTES);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            copy_tile_row(raw_l1 + c * TILE_BYTES, lane, ring_l1 + c * TILE_BYTES, slot, true);
        }
        cb_push_back(CB_RING, HEAD_TILES);
    }
}
