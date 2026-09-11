// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Operands of one block pair, from DRAM into L1.
//
// A stand-in for the relay's packet arrival and column residency: here every
// operand simply comes from DRAM, because this step is about the arithmetic.
// The causal mask tile is generated on the core rather than read, as
// sdpa_bw's writer does.
//
// Several pairs can be fed in sequence, each naming its row block and column
// block, so the compute kernel can accumulate across them the way a streak
// accumulates dQ and a column residency accumulates dK and dV.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t value_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lse_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t u_scalar_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t num_pairs = get_arg_val<uint32_t>(arg++);
    const uint32_t pair_args = arg;  // then (row block, column block) per pair

    constexpr uint32_t qWt = get_compile_time_arg_val(0);
    constexpr uint32_t vWt = get_compile_time_arg_val(1);
    constexpr auto query_args = TensorAccessorArgs<2>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto grad_output_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto lse_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto u_args = TensorAccessorArgs<lse_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_lse = tt::CBIndex::c_4;
    constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
#ifdef DIAGONAL_BLOCK
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
    generate_causal_mask_tile(cb_attn_mask);
#endif

    const uint32_t tile_bytes = get_tile_size(cb_query);
    const uint32_t interm_bytes = get_tile_size(cb_lse);

    const auto query = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto grad_output = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto lse = TensorAccessor(lse_args, lse_addr, interm_bytes);
    const auto u_scalar = TensorAccessor(u_args, u_scalar_addr, interm_bytes);

    for (uint32_t pair = 0; pair < num_pairs; ++pair) {
        const uint32_t row_block = get_arg_val<uint32_t>(pair_args + 2u * pair);
        const uint32_t col_block = get_arg_val<uint32_t>(pair_args + 2u * pair + 1u);

        // Row-side operands follow the row block, column-side the column block.
        read_tiles_by_row(cb_query, query, row_block * qWt, qWt, tile_bytes, qWt);
        read_tiles_by_row(cb_key, key, col_block * qWt, qWt, tile_bytes, qWt);
        read_tiles_by_row(cb_value, value, col_block * vWt, vWt, tile_bytes, vWt);
        read_tiles_by_row(cb_grad_output, grad_output, row_block * vWt, vWt, tile_bytes, vWt);
        read_one_tile(cb_lse, lse, row_block);
        read_one_tile(cb_u_scalar, u_scalar, row_block);
    }
}
