// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t num_inner = get_compile_time_arg_val(2);

// CBs with input data / scalar parameters
constexpr auto cb_input_pass_1 = tt::CBIndex::c_0;
constexpr auto cb_input_pass_2 = tt::CBIndex::c_1;
constexpr auto cb_scaler = tt::CBIndex::c_2;
constexpr auto cb_eps = tt::CBIndex::c_3;
constexpr auto cb_w0 = tt::CBIndex::c_4;
constexpr auto cb_w1 = tt::CBIndex::c_5;
constexpr auto cb_w2 = tt::CBIndex::c_6;
constexpr auto cb_bias = tt::CBIndex::c_7;
// CBs with intermediate computations
constexpr auto cb_sum_pows = tt::CBIndex::c_8;  // Queue of row sums: [sum(x^2), sum(x^4), sum(x^6)]
constexpr auto cb_inv_rms = tt::CBIndex::c_9;   // Queue of inv_rms: [inv_rms(x), inv_rms(x^2), inv_rms(x^3)]
// CB with output data
constexpr auto cb_output = tt::CBIndex::c_10;

// Consume one sum tile from cb_sum and produce one inverse-RMS tile in cb_inv_rms:
// inv_rms = 1 / sqrt(sum * (1/C) + eps).
//
// Important ordering note for this kernel:
// - cb_sum_pows front order is [sum(x^2), sum(x^4), sum(x^6)].
// - kernel_main calls this helper three times in that order, so cb_inv_rms is produced as
//   [inv_rms(x), inv_rms(x^2), inv_rms(x^3)] and later indexed by emit_output_for_row().
void reduce_sum_to_inv_rms(const uint32_t cb_sum, const uint32_t cb_inv_rms) {
    cb_wait_front(cb_sum, onetile);
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);

    tile_regs_acquire();
    constexpr uint32_t reg_acc = 0U;
    constexpr uint32_t reg_eps = 1U;
    reconfig_data_format(cb_sum, cb_scaler);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, cb_inv_rms);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, 0, 0, reg_acc);
    reduce_uninit();

    reconfig_data_format_srca(cb_eps);
    copy_tile_init(cb_eps);
    copy_tile(cb_eps, 0, reg_eps);

    reconfig_data_format(cb_sum, cb_eps);
    add_binary_tile_init();
    add_binary_tile(reg_acc, reg_eps, reg_acc);

    sqrt_tile_init();
    sqrt_tile(reg_acc);
    recip_tile_init();
    recip_tile(reg_acc);

    tile_regs_commit();
    pack_and_push(reg_acc, cb_inv_rms);
    cb_pop_front(cb_sum, onetile);
}

// Pass-1 (per row):
// - Stream input tiles from cb_input_pass_1.
// - Compute x^2, x^4, x^6 per tile in DEST regs [0,1,2].
// - Accumulate these three quantities across the whole row into a single 3-tile CB (cb_sum_pows):
//     tile 0 -> sum(x^2), tile 1 -> sum(x^4), tile 2 -> sum(x^6).
//
// We reserve/push all 3 sum tiles once per row and use L1-accumulation to fold every row tile
// into the same 3 output slots.
void accumulate_sum_x2_x4_x6_for_row() {
    constexpr uint32_t reg_x2 = 0U;
    constexpr uint32_t reg_x4 = 1U;
    constexpr uint32_t reg_x6 = 2U;
    constexpr uint32_t reg_x = 3U;

    bool first_tile = true;
    cb_reserve_back(cb_sum_pows, /*num_tiles=*/3U);

    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_1, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            tile_regs_acquire();
            copy_tile_init(cb_input_pass_1);
            copy_tile(cb_input_pass_1, block_idx, reg_x);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_x, reg_x2);    // x^2
            mul_binary_tile(reg_x2, reg_x2, reg_x4);  // x^4
            mul_binary_tile(reg_x4, reg_x2, reg_x6);  // x^6

            tile_regs_commit();
            pack_l1_acc_block(cb_sum_pows, /*first_block=*/first_tile, /*num_tiles=*/3U, /*dst_start_index=*/0U);
            first_tile = false;
        }
        cb_pop_front(cb_input_pass_1, block_size);
    }

    cb_push_back(cb_sum_pows, /*num_tiles=*/3U);
}

// Pass-2 (per row):
// - Consume the 3 inv_rms tiles produced from Pass-1 reductions:
//     inv_rms tile 2: for x^3 term, tile 1: for x^2 term, tile 0: for x term.
// - For each input tile, compute and accumulate:
//     w0 * (x^3 * inv_rms_x3) + w1 * (x^2 * inv_rms_x2) + w2 * (x * inv_rms_x) + bias
// - Emit one output tile to cb_output at the matching block index.
void emit_output_for_row() {
    constexpr uint32_t reg_acc = 0U;
    constexpr uint32_t reg_x = 1U;
    constexpr uint32_t reg_tmp = 2U;
    constexpr uint32_t reg_bcast_or_scalar = 3U;

    cb_wait_front(cb_inv_rms, /*num_tiles=*/3U);

    for (uint32_t col = 0; col < num_inner; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_inner - col);
        cb_wait_front(cb_input_pass_2, block_size);
        cb_reserve_back(cb_output, block_size);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            tile_regs_acquire();

            // term3 (cubic branch): w0 * norm(x^3), where norm(x^3) = x^3 * inv_rms_x3.
            unary_bcast_init<BroadcastType::COL>(cb_inv_rms, cb_inv_rms);
            unary_bcast<BroadcastType::COL>(cb_inv_rms, /*tile_idx=*/2U, reg_bcast_or_scalar);
            copy_tile_init(cb_input_pass_2);
            copy_tile(cb_input_pass_2, block_idx, reg_x);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_x, reg_tmp);    // x^2
            mul_binary_tile(reg_tmp, reg_x, reg_tmp);  // x^3
            reconfig_data_format(cb_input_pass_2, cb_inv_rms);
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // norm(x^3)
            copy_tile_init(cb_w0);
            copy_tile(cb_w0, 0, reg_bcast_or_scalar);
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_acc);  // accumulator init

            // term2 (quadratic branch): w1 * norm(x^2), where norm(x^2) = x^2 * inv_rms_x2.
            unary_bcast_init<BroadcastType::COL>(cb_inv_rms, cb_inv_rms);
            unary_bcast<BroadcastType::COL>(cb_inv_rms, /*tile_idx=*/1U, reg_bcast_or_scalar);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_x, reg_tmp);  // x^2
            reconfig_data_format(cb_input_pass_2, cb_inv_rms);
            mul_binary_tile_init();
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // norm(x^2)
            copy_tile_init(cb_w1);
            copy_tile(cb_w1, 0, reg_bcast_or_scalar);
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // weighted term2
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);

            // term1 (linear branch): w2 * norm(x), where norm(x) = x * inv_rms_x.
            unary_bcast_init<BroadcastType::COL>(cb_inv_rms, cb_inv_rms);
            unary_bcast<BroadcastType::COL>(cb_inv_rms, /*tile_idx=*/0U, reg_bcast_or_scalar);
            reconfig_data_format(cb_input_pass_2, cb_inv_rms);
            mul_binary_tile_init();
            mul_binary_tile(reg_x, reg_bcast_or_scalar, reg_tmp);  // norm(x)
            copy_tile_init(cb_w2);
            copy_tile(cb_w2, 0, reg_bcast_or_scalar);
            mul_binary_tile(reg_tmp, reg_bcast_or_scalar, reg_tmp);  // weighted term1
            add_binary_tile_init();
            add_binary_tile(reg_acc, reg_tmp, reg_acc);

            // Final affine shift and output write for this tile.
            copy_tile_init(cb_bias);
            copy_tile(cb_bias, 0, reg_bcast_or_scalar);
            add_binary_tile(reg_acc, reg_bcast_or_scalar, reg_acc);

            tile_regs_commit();
            // Keep reserve(block_size)/push(block_size) invariant, but only fill current_block_size slots.
            pack_l1_acc_block(cb_output, /* first_block */ true, /* num_tiles */ 1U, /* dst_start_index */ block_idx);
        }

        cb_push_back(cb_output, block_size);
        cb_pop_front(cb_input_pass_2, block_size);
    }

    cb_pop_front(cb_inv_rms, /*num_tiles=*/3U);
}

// Main row pipeline on this core:
//   1) accumulate row sums for x^2/x^4/x^6
//   2) convert each sum to inv_rms (three calls preserve sum order)
//   3) consume inv_rms triplet and emit final PolyNorm output tiles
void kernel_main() {
    cb_wait_front(cb_scaler, onetile);
    cb_wait_front(cb_eps, onetile);
    cb_wait_front(cb_w0, onetile);
    cb_wait_front(cb_w1, onetile);
    cb_wait_front(cb_w2, onetile);
    cb_wait_front(cb_bias, onetile);

    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        (void)row;
        accumulate_sum_x2_x4_x6_for_row();

        reduce_sum_to_inv_rms(cb_sum_pows, cb_inv_rms);
        reduce_sum_to_inv_rms(cb_sum_pows, cb_inv_rms);
        reduce_sum_to_inv_rms(cb_sum_pows, cb_inv_rms);

        emit_output_for_row();
    }

    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_eps, onetile);
    cb_pop_front(cb_w0, onetile);
    cb_pop_front(cb_w1, onetile);
    cb_pop_front(cb_w2, onetile);
    cb_pop_front(cb_bias, onetile);
}
