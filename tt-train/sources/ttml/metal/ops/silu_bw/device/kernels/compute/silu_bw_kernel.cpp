// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/ops/common/compute_utils.hpp"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_1;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_2;
// CBs with intermediate computations
constexpr uint32_t cb_sigmoid_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_one_minus_sigmoid_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_times_input_plus_one_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_times_sigmoid_idx = tt::CBIndex::c_6;

// ============================================================================
// Functions below compute final results for a pipeline stage and write them to
// circular buffers (CBs). These functions produce outputs that are consumed by
// later stages.
// ============================================================================

inline void compute_sigmoid() {
    // Compute: sigmoid(x) = 1 / (1 + exp(-x))
    // The result is stored in cb_sigmoid_idx.

    tile_regs_acquire();
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        copy_tile_init(cb_input_idx);
        copy_tile(cb_input_idx, /* tile_index */ block_idx, /* register_idx */ block_idx);

        sigmoid_tile_init();
        sigmoid_tile(/* register_idx */ block_idx);
    }
    tile_regs_commit();

    pack_and_push_block(cb_sigmoid_idx, block_size);
}

inline void compute_one_minus_sigmoid() {
    // Compute: 1 - sigmoid(x)
    // The result is stored in cb_one_minus_sigmoid_idx.
    cb_wait_front(cb_sigmoid_idx, block_size);

    const uint32_t one = 0x3F800000;  // FP32 encoding of 1.0

    tile_regs_acquire();
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        copy_tile_init(cb_sigmoid_idx);
        copy_tile(cb_sigmoid_idx, /* tile_index */ block_idx, /* register_idx */ block_idx);

        binop_with_scalar_tile_init();
        rsub_unary_tile(/* register_idx */ block_idx, one);  // 1.0F is the constant to subtract from.
    }
    tile_regs_commit();

    pack_and_push_block(cb_one_minus_sigmoid_idx, block_size);
}

inline void compute_times_input_plus_one() {
    // Compute: (1 - sigmoid(x)) * input + 1
    // The result is stored in cb_times_input_plus_one_idx.
    cb_wait_front(cb_one_minus_sigmoid_idx, block_size);

    const uint32_t one = 0x3F800000;  // FP32 encoding of 1.0

    tile_regs_acquire();
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        mul_tiles_init(cb_input_idx, cb_one_minus_sigmoid_idx);
        mul_tiles(
            cb_input_idx,
            cb_one_minus_sigmoid_idx,
            /* tile_index */ block_idx,
            /* tile_index */ block_idx,
            /* register_idx */ block_idx);

        binop_with_scalar_tile_init();
        add_unary_tile(/* register_idx */ block_idx, one);  // Add 1.0F to the result.
    }
    tile_regs_commit();

    pack_and_push_block(cb_times_input_plus_one_idx, block_size);
}

inline void compute_times_sigmoid() {
    // Compute: ((1 - sigmoid(x)) * input + 1) * sigmoid(x)
    // The result is stored in cb_times_sigmoid_idx.
    cb_wait_front(cb_times_input_plus_one_idx, block_size);

    tile_regs_acquire();
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        mul_tiles_init(cb_sigmoid_idx, cb_times_input_plus_one_idx);
        mul_tiles(
            cb_sigmoid_idx,
            cb_times_input_plus_one_idx,
            /* tile_index */ block_idx,
            /* tile_index */ block_idx,
            /* register_idx */ block_idx);
    }
    tile_regs_commit();

    pack_and_push_block(cb_times_sigmoid_idx, block_size);
}

inline void compute_times_grad() {
    // Compute: ((1 - sigmoid(x)) * input + 1) * sigmoid(x) * dL_dout
    // The result is stored in cb_dL_da_idx.
    cb_wait_front(cb_times_sigmoid_idx, block_size);

    tile_regs_acquire();
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        mul_tiles_init(cb_times_sigmoid_idx, cb_dL_out_idx);
        mul_tiles(
            cb_times_sigmoid_idx,
            cb_dL_out_idx,
            /* tile_index */ block_idx,
            /* tile_index */ block_idx,
            /* register_idx */ block_idx);
    }
    tile_regs_commit();

    pack_and_push_block(cb_dL_da_idx, block_size);
}

inline void MAIN {
    init_sfpu(cb_input_idx, cb_dL_da_idx);
    binary_op_init_common(cb_input_idx, cb_dL_out_idx, cb_dL_da_idx);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_wait_front(cb_input_idx, block_size);
            cb_wait_front(cb_dL_out_idx, block_size);

            compute_sigmoid();
            compute_one_minus_sigmoid();
            compute_times_input_plus_one();
            compute_times_sigmoid();
            compute_times_grad();

            cb_pop_front(cb_sigmoid_idx, block_size);
            cb_pop_front(cb_one_minus_sigmoid_idx, block_size);
            cb_pop_front(cb_times_input_plus_one_idx, block_size);
            cb_pop_front(cb_times_sigmoid_idx, block_size);

            cb_pop_front(cb_input_idx, block_size);
            cb_pop_front(cb_dL_out_idx, block_size);
        }
    }
}

}  // namespace NAMESPACE
