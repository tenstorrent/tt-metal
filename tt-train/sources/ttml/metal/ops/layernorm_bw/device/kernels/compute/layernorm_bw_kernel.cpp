// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/ops/common/compute_utils.hpp"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t mask_w = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);

// CBs with input data
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_0;      // 1/N scaler
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;      // mask for width dimension
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_2;       // gamma (scale parameter)
constexpr uint32_t cb_x_hat_idx = tt::CBIndex::c_3;       // x_hat (normalized input) from forward pass
constexpr uint32_t cb_rstd_idx = tt::CBIndex::c_4;        // rstd from forward pass
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;      // upstream gradient
constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_6;  // reduction vector

// CBs with output data
constexpr uint32_t cb_dx_idx = tt::CBIndex::c_10;                // dx (input gradient)
constexpr uint32_t cb_dgamma_components = tt::CBIndex::c_11;     // dgamma components
constexpr uint32_t cb_dbeta_components = tt::CBIndex::c_12;      // dbeta components
constexpr uint32_t cb_debug_scaled_sum_idx = tt::CBIndex::c_20;  // DEBUG: scaled dy*gamma sum for inspection

// CBs with intermediate computations
constexpr uint32_t cb_x_normalized_idx = tt::CBIndex::c_13;         // x_normalized (loaded from x_hat)
constexpr uint32_t cb_dy_gamma_idx = tt::CBIndex::c_14;             // dy * gamma
constexpr uint32_t cb_dy_gamma_sum_idx = tt::CBIndex::c_15;         // sum(dy * gamma)
constexpr uint32_t cb_dy_gamma_xnorm_sum_idx = tt::CBIndex::c_16;   // sum(dy * gamma * x_normalized)
constexpr uint32_t cb_scaled_dy_gamma_sum_idx = tt::CBIndex::c_17;  // (1/N) * sum(dy * gamma) - pre-scaled
constexpr uint32_t cb_scaled_dy_gamma_xnorm_sum_idx =
    tt::CBIndex::c_18;  // (1/N) * sum(dy * gamma * x_normalized) - pre-scaled
constexpr uint32_t cb_zero = tt::CBIndex::c_19;  // (1/N) * sum(dy * gamma * x_normalized) - pre-scaled

constexpr uint32_t onetile = 1;

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

inline void zero_dst() {
    copy_tile_init(cb_zero);
    for (int i = 0; i < 4; i++) {
        copy_tile(cb_zero, 0, i);
    }
}

inline void zero_dst_reg(uint32_t i) {
    copy_tile_init(cb_zero);
    copy_tile(cb_zero, 0, i);
}

inline void compute_dy_gamma(
    uint32_t dy_tile_idx, uint32_t gamma_tile_idx, uint32_t dy_gamma_register, bool print_debug = false) {
    // Multiply dy * gamma where gamma is broadcasted across rows
    // dy: [32x32] tile, gamma: [1x32] broadcasted to each row
    zero_dst_reg(dy_gamma_register);
    // if(print_debug){
    //     cb_reserve_back(cb_debug_scaled_sum_idx, onetile);
    //     tile_regs_commit();
    //     pack_and_push(dy_gamma_register, cb_debug_scaled_sum_idx);
    //     tile_regs_acquire();
    // }
    reconfig_data_format(cb_dL_out_idx, cb_gamma_idx);
    mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
    mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, dy_tile_idx, gamma_tile_idx, dy_gamma_register);
}

#ifdef EVERYTHING_FITS_IN_L1
inline void compute_dy_gamma_sum(const uint32_t row) {
    // Computes sum(dy * gamma) across width dimension when everything fits in L1
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t temp_register = 2U;

    tile_regs_acquire();

    for (uint32_t col = 0; col < Wt; ++col) {
        mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
        mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, col, col, sum_register);

        // Mask the tile if needed
        if constexpr (do_mask_w) {
            if (col + 1 == Wt) {
                // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                const uint32_t mask_register = sum_register + 1U;

                copy_tile_init(cb_mask_w_idx);
                copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                mask_tile_init();
                mask_tile(sum_register, mask_register);
            }
        }
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_dy_gamma_sum_idx);

    // Reduce sum across inner dimension using matmul
    cb_wait_front(cb_dy_gamma_sum_idx, onetile);

    const uint32_t reduced_sum_register = 0U;
    tile_regs_acquire();

    reconfig_data_format(cb_dy_gamma_sum_idx, cb_mat_mul_reduce);
    mm_init(cb_dy_gamma_sum_idx, cb_mat_mul_reduce, cb_dy_gamma_sum_idx, 0);
    matmul_tiles(
        cb_dy_gamma_sum_idx,
        cb_mat_mul_reduce,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        reduced_sum_register,
        /* transpose */ 0);

    tile_regs_commit();
    pack_and_push(reduced_sum_register, cb_dy_gamma_sum_idx);
    cb_pop_front(cb_dy_gamma_sum_idx, onetile);

    // Scale the sum by 1/N and store in cb_scaled_dy_gamma_sum_idx
    cb_wait_front(cb_dy_gamma_sum_idx, onetile);
    cb_wait_front(cb_scaler_idx, onetile);
    tile_regs_acquire();

    const uint32_t scaled_sum_register = 0U;
    const uint32_t scaler_register = 1U;

    // Explicitly reset unpacker - cb_dy_gamma_sum_idx was used in matmul
    reconfig_data_format(cb_dy_gamma_sum_idx, cb_dy_gamma_sum_idx);
    copy_tile_init(cb_dy_gamma_sum_idx);
    copy_tile(cb_dy_gamma_sum_idx, 0, scaled_sum_register);

    reconfig_data_format(cb_scaler_idx, cb_scaler_idx);
    copy_tile_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, 0, scaler_register);

    mul_binary_tile_init();
    mul_binary_tile(scaled_sum_register, scaler_register, scaled_sum_register);

    tile_regs_commit();
    pack_and_push(scaled_sum_register, cb_scaled_dy_gamma_sum_idx);
    cb_wait_front(cb_scaled_dy_gamma_sum_idx, onetile);
    unary_bcast_init<BroadcastType::COL>(cb_scaled_dy_gamma_sum_idx, cb_scaled_dy_gamma_sum_idx);
    unary_bcast<BroadcastType::COL>(cb_scaled_dy_gamma_sum_idx, /* tile idx */ 0, /* reg tile idx */ 0);
    tile_regs_commit();
    pack_and_push(0, cb_scaled_dy_gamma_sum_idx);
    cb_pop_front(cb_scaled_dy_gamma_sum_idx, onetile);
}

inline void compute_dy_gamma_xnorm_sum(const uint32_t row) {
    // Computes 1/N * sum(dy * gamma * x_normalized) across width dimension when everything fits in L1
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t x_norm_register = 2U;
    const uint32_t temp_register = 3U;

    tile_regs_acquire();

    // tile_regs_commit();
    // tile_regs_wait();
    // tile_regs_release();
    // tile_regs_acquire();
    // cb_reserve_back(cb_debug_scaled_sum_idx, onetile);
    // const uint32_t test_reg = 3;
    // copy_tile_init(cb_gamma_idx);
    // copy_tile(cb_gamma_idx, 0, test_reg);
    // tile_regs_commit();
    // pack_and_push(0, cb_debug_scaled_sum_idx);
    // tile_regs_acquire();

    for (uint32_t col = 0; col < Wt; ++col) {
        auto target_register = (col == 0) ? sum_register : working_register;

        // Compute x_normalized for this tile
        copy_tile_init(cb_x_hat_idx);
        copy_tile(cb_x_hat_idx, col, x_norm_register);

        // Compute dy * gamma for this tile
        compute_dy_gamma(col, col, target_register);

        // Multiply: (dy * gamma) * x_normalized
        mul_binary_tile_init();
        mul_binary_tile(target_register, x_norm_register, target_register);

        // Mask the tile if needed
        if constexpr (do_mask_w) {
            if (col + 1 == Wt) {
                // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                const uint32_t mask_register = target_register + 1U;

                copy_tile_init(cb_mask_w_idx);
                copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                mask_tile_init();
                mask_tile(target_register, mask_register);
            }
        }

        // Accumulate to sum
        if (col > 0) {
            add_binary_tile_init();
            add_binary_tile(sum_register, working_register, sum_register);
        }
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_dy_gamma_xnorm_sum_idx);

    // Reduce sum across inner dimension using matmul
    cb_wait_front(cb_dy_gamma_xnorm_sum_idx, onetile);
    const uint32_t reduced_sum_register = 0U;
    tile_regs_acquire();

    reconfig_data_format(cb_dy_gamma_xnorm_sum_idx, cb_mat_mul_reduce);
    mm_init(cb_dy_gamma_xnorm_sum_idx, cb_mat_mul_reduce, cb_dy_gamma_xnorm_sum_idx, 0);
    matmul_tiles(
        cb_dy_gamma_xnorm_sum_idx,
        cb_mat_mul_reduce,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ reduced_sum_register,
        /* transpose */ 0);

    cb_reserve_back(cb_dy_gamma_xnorm_sum_idx, onetile);
    tile_regs_commit();
    pack_and_push(reduced_sum_register, cb_dy_gamma_xnorm_sum_idx);
    cb_pop_front(cb_dy_gamma_xnorm_sum_idx, onetile);

    // Scale the sum by 1/N and store in cb_scaled_dy_gamma_xnorm_sum_idx
    cb_wait_front(cb_dy_gamma_xnorm_sum_idx, onetile);
    cb_wait_front(cb_scaler_idx, onetile);
    tile_regs_acquire();

    const uint32_t scaled_sum_register = 0U;
    const uint32_t scaler_register = 1U;
    copy_tile_init(cb_dy_gamma_xnorm_sum_idx);
    copy_tile(cb_dy_gamma_xnorm_sum_idx, 0, scaled_sum_register);
    copy_tile_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, 0, scaler_register);
    mul_binary_tile_init();
    mul_binary_tile(scaled_sum_register, scaler_register, scaled_sum_register);

    tile_regs_commit();
    pack_and_push(scaled_sum_register, cb_scaled_dy_gamma_xnorm_sum_idx);
    cb_wait_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    unary_bcast_init<BroadcastType::COL>(cb_scaled_dy_gamma_xnorm_sum_idx, cb_scaled_dy_gamma_xnorm_sum_idx);
    unary_bcast<BroadcastType::COL>(cb_scaled_dy_gamma_xnorm_sum_idx, /* tile idx */ 0, /* reg tile idx */ 0);
    tile_regs_commit();
    pack_and_push(0, cb_scaled_dy_gamma_xnorm_sum_idx);
    cb_pop_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
}
#else
// Non-L1 versions would be similar but with proper CB management

// cb_scaled_dy_gamma_sum_idx [[1/N * sum_i(dy[0, :] * gamma[:]), 1/N * sum_i(dy[0, :] * gamma[:]), ...],
//                             [1/N * sum_i(dy[1, :] * gamma[:]), 1/N * sum_i(dy[1, :] * gamma[:]), ...],
//                             ...]
// cb_dL_out_idx, cb_gamma_idx blocks are read
inline void compute_dy_gamma_sum(const uint32_t row) {
    // Block-based processing when not everything fits in L1
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t temp_register = 2U;

    tile_regs_acquire();

    for (uint32_t col = 0; col < Wt; col += block_size) {
        cb_wait_front(cb_dL_out_idx, block_size);
        cb_wait_front(cb_gamma_idx, block_size);

        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
            mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, block_idx, block_idx, sum_register);

            // Mask the tile if needed
            if constexpr (do_mask_w) {
                if (col + block_idx + 1 == Wt) {
                    // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                    const uint32_t mask_register = sum_register + 1U;

                    copy_tile_init(cb_mask_w_idx);
                    copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                    mask_tile_init();
                    mask_tile(sum_register, mask_register);
                }
            }
        }

        cb_pop_front(cb_dL_out_idx, block_size);
        cb_pop_front(cb_gamma_idx, block_size);
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_dy_gamma_sum_idx);

    // Reduce using matmul
    cb_wait_front(cb_dy_gamma_sum_idx, onetile);

    const uint32_t reduced_sum_register = 0U;

    tile_regs_acquire();
    reconfig_data_format(cb_dy_gamma_sum_idx, cb_mat_mul_reduce);
    mm_init(cb_dy_gamma_sum_idx, cb_mat_mul_reduce, cb_dy_gamma_sum_idx, 0);
    matmul_tiles(
        cb_dy_gamma_sum_idx,
        cb_mat_mul_reduce,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ reduced_sum_register,
        /* transpose */ 0);

    tile_regs_commit();
    pack_and_push(reduced_sum_register, cb_dy_gamma_sum_idx);
    cb_pop_front(cb_dy_gamma_sum_idx, onetile);

    // Scale the sum by 1/N and store in cb_scaled_dy_gamma_sum_idx
    cb_wait_front(cb_dy_gamma_sum_idx, onetile);
    cb_wait_front(cb_scaler_idx, onetile);
    tile_regs_acquire();

    const uint32_t scaled_sum_register = 0U;
    const uint32_t scaler_register = 1U;
    copy_tile_init(cb_dy_gamma_sum_idx);
    copy_tile(cb_dy_gamma_sum_idx, 0, scaled_sum_register);
    copy_tile_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, 0, scaler_register);
    mul_binary_tile_init();
    mul_binary_tile(scaled_sum_register, scaler_register, scaled_sum_register);

    tile_regs_commit();
    pack_and_push(scaled_sum_register, cb_scaled_dy_gamma_sum_idx);
    cb_wait_front(cb_scaled_dy_gamma_sum_idx, onetile);
    unary_bcast_init<BroadcastType::COL>(cb_scaled_dy_gamma_sum_idx, cb_scaled_dy_gamma_sum_idx);
    unary_bcast<BroadcastType::COL>(cb_scaled_dy_gamma_sum_idx, /* tile idx */ 0, /* reg tile idx */ 0);
    tile_regs_commit();
    pack_and_push(0, cb_scaled_dy_gamma_sum_idx);
    cb_pop_front(cb_scaled_dy_gamma_sum_idx, onetile);
    cb_pop_front(cb_dy_gamma_sum_idx, onetile);
}

inline void compute_dy_gamma_xnorm_sum(const uint32_t row) {
    // Similar implementation for non-L1 case
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t x_norm_register = 2U;
    const uint32_t temp_register = 3U;

    tile_regs_acquire();

    for (uint32_t col = 0; col < Wt; col += block_size) {
        cb_wait_front(cb_x_hat_idx, block_size);
        cb_wait_front(cb_dL_out_idx, block_size);
        cb_wait_front(cb_gamma_idx, block_size);

        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            uint32_t global_col = col + block_idx;
            auto target_register = (global_col == 0) ? sum_register : working_register;

            // Load x_normalized (x_hat)
            copy_tile_init(cb_x_hat_idx);
            copy_tile(cb_x_hat_idx, block_idx, x_norm_register);

            // Compute dy * gamma
            compute_dy_gamma(block_idx, block_idx, target_register);

            // Multiply: (dy * gamma) * x_normalized
            mul_binary_tile_init();
            mul_binary_tile(target_register, x_norm_register, target_register);

            // Mask the tile if needed
            if constexpr (do_mask_w) {
                if (global_col + 1 == Wt) {
                    // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                    const uint32_t mask_register = target_register + 1U;

                    copy_tile_init(cb_mask_w_idx);
                    copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                    mask_tile_init();
                    mask_tile(target_register, mask_register);
                }
            }

            // Accumulate to sum
            if (global_col > 0) {
                add_binary_tile_init();
                add_binary_tile(sum_register, working_register, sum_register);
            }
        }

        cb_pop_front(cb_x_hat_idx, block_size);
        cb_pop_front(cb_dL_out_idx, block_size);
        cb_pop_front(cb_gamma_idx, block_size);
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_dy_gamma_xnorm_sum_idx);

    // Reduce using matmul
    const uint32_t reduced_sum_register = 0U;
    cb_wait_front(cb_dy_gamma_xnorm_sum_idx, onetile);

    tile_regs_acquire();
    reconfig_data_format(cb_dy_gamma_xnorm_sum_idx, cb_mat_mul_reduce);
    mm_init(cb_dy_gamma_xnorm_sum_idx, cb_mat_mul_reduce, cb_dy_gamma_xnorm_sum_idx, 0);
    matmul_tiles(
        cb_dy_gamma_xnorm_sum_idx,
        cb_mat_mul_reduce,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ reduced_sum_register,
        /* transpose */ 0);

    tile_regs_commit();
    pack_and_push(reduced_sum_register, cb_dy_gamma_xnorm_sum_idx);
    cb_pop_front(cb_dy_gamma_xnorm_sum_idx, onetile);
    tile_regs_acquire();

    // Scale the sum by 1/N and store in cb_scaled_dy_gamma_xnorm_sum_idx
    const uint32_t scaled_sum_register = 0U;
    const uint32_t scaler_register = 1U;
    cb_wait_front(cb_dy_gamma_xnorm_sum_idx, onetile);
    cb_wait_front(cb_scaler_idx, onetile);
    copy_tile_init(cb_dy_gamma_xnorm_sum_idx);
    copy_tile(cb_dy_gamma_xnorm_sum_idx, 0, scaled_sum_register);
    copy_tile_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, 0, scaler_register);
    mul_binary_tile_init();
    mul_binary_tile(scaled_sum_register, scaler_register, scaled_sum_register);

    tile_regs_commit();
    pack_and_push(scaled_sum_register, cb_scaled_dy_gamma_xnorm_sum_idx);
    cb_wait_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    unary_bcast_init<BroadcastType::COL>(cb_scaled_dy_gamma_xnorm_sum_idx, cb_scaled_dy_gamma_xnorm_sum_idx);
    unary_bcast<BroadcastType::COL>(cb_scaled_dy_gamma_xnorm_sum_idx, /* tile idx */ 0, /* reg tile idx */ 0);
    cb_pop_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    tile_regs_commit();
    pack_and_push(0, cb_scaled_dy_gamma_xnorm_sum_idx);
    cb_pop_front(cb_dy_gamma_xnorm_sum_idx, onetile);
}
#endif  // EVERYTHING_FITS_IN_L1

// uses 3 registers starting from dx_register (reduced from 4!)
inline void compute_dx(const uint32_t input_tile_idx, const uint32_t dx_register) {
    // Computes dx = (1/rstd) * (dy*gamma - (1/N) * sum(dy*gamma) - x_normalized * (1/N) * sum(dy*gamma * x_normalized))

    const uint32_t x_norm_register = dx_register + 1;
    const uint32_t temp_register = dx_register + 2;

    // Compute dy * gamma
    compute_dy_gamma(input_tile_idx, input_tile_idx, dx_register);

    // if (input_tile_idx == 0) {
    //     cb_reserve_back(cb_debug_scaled_sum_idx, onetile);
    //     const uint32_t test_reg = 3;
    //     copy_tile_init(cb_scaled_dy_gamma_sum_idx);
    //     copy_tile(cb_scaled_dy_gamma_sum_idx, input_tile_idx, test_reg);
    //     tile_regs_commit();
    //     pack_and_push(0, cb_debug_scaled_sum_idx);
    // }

    // Load pre-scaled sum: (1/N) * sum(dy*gamma)
    copy_tile_init(cb_scaled_dy_gamma_sum_idx);
    copy_tile(cb_scaled_dy_gamma_sum_idx, 0, temp_register);

    // Subtract: dy*gamma - (1/N) * sum(dy*gamma)
    sub_binary_tile_init();
    sub_binary_tile(dx_register, temp_register, dx_register);

    // Load x_normalized
    copy_tile_init(cb_x_hat_idx);
    copy_tile(cb_x_hat_idx, input_tile_idx, x_norm_register);

    // Load pre-scaled sum: (1/N) * sum(dy*gamma*x_norm)
    copy_tile_init(cb_scaled_dy_gamma_xnorm_sum_idx);
    copy_tile(cb_scaled_dy_gamma_xnorm_sum_idx, 0, temp_register);

    // Multiply by x_normalized: x_normalized * (1/N) * sum(dy*gamma * x_normalized)
    mul_binary_tile_init();
    mul_binary_tile(x_norm_register, temp_register, temp_register);

    // Subtract: result - x_normalized * (1/N) * sum(...)
    sub_binary_tile_init();
    sub_binary_tile(dx_register, temp_register, dx_register);
    // Multiply by rstd: dx = rstd * (...)
    copy_tile_init(cb_rstd_idx);
    copy_tile(cb_rstd_idx, 0, temp_register);
    mul_binary_tile_init();
    mul_binary_tile(dx_register, temp_register, dx_register);
}

// uses 3 registers starting from dgamma_register
inline void compute_dgamma_components(const uint32_t input_tile_idx, const uint32_t dgamma_register) {
    // Computes dgamma_components = dy * x_normalized

    const uint32_t x_norm_register = dgamma_register + 1;
    const uint32_t dy_register = dgamma_register + 2;

    copy_tile_init(cb_x_hat_idx);
    copy_tile(cb_x_hat_idx, input_tile_idx, x_norm_register);

    // Load dy
    copy_tile_init(cb_dL_out_idx);
    copy_tile(cb_dL_out_idx, input_tile_idx, dy_register);

    // Multiply: dy * x_normalized
    sub_binary_tile_init();
    sub_binary_tile(dgamma_register, dgamma_register, dgamma_register);
    mul_binary_tile_init();
    mul_binary_tile(dy_register, x_norm_register, dgamma_register);
}

inline void compute_dbeta_components(const uint32_t dy_tile_idx, const uint32_t dbeta_register) {
    // Computes dbeta_components = dy (simple copy)
    copy_tile_init(cb_dL_out_idx);
    copy_tile(cb_dL_out_idx, dy_tile_idx, dbeta_register);
}

inline void MAIN {
    if constexpr (do_mask_w) {
        cb_wait_front(cb_mask_w_idx, onetile);
    }
    cb_wait_front(cb_scaler_idx, onetile);
    cb_wait_front(cb_mat_mul_reduce, onetile);
    cb_wait_front(cb_zero, onetile);

    init_sfpu(cb_x_hat_idx, cb_dx_idx);
    binary_op_init_common(cb_x_hat_idx, cb_gamma_idx, cb_dx_idx);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Wait for rstd (per row)
        // [[rstd_{1, 1}, rstd_{1, 2}, ...], [rstd_{1, 1}, rstd_{1, 2}, ...], ...]
        cb_wait_front(cb_rstd_idx, onetile);
        tile_regs_acquire();
        unary_bcast_init<BroadcastType::COL>(cb_rstd_idx, cb_rstd_idx);
        unary_bcast<BroadcastType::COL>(cb_rstd_idx, /* tile idx */ 0, /* reg tile idx */ 0);
        tile_regs_commit();
        pack_and_push(0, cb_rstd_idx);
        cb_wait_front(cb_rstd_idx, onetile);
        tile_regs_acquire();

#ifdef EVERYTHING_FITS_IN_L1
        // Wait for x_hat for entire row when everything fits in L1
        cb_wait_front(cb_x_hat_idx, Wt);
        cb_wait_front(cb_dL_out_idx, Wt);
        // If everything fits in L1, wait for gamma only once as its shared across all rows
        if (row == 0) {
            cb_wait_front(cb_gamma_idx, Wt);
        }
#endif
        // tile_regs_commit();
        // tile_regs_wait();
        // tile_regs_release();
        // tile_regs_acquire();
        // cb_reserve_back(cb_debug_scaled_sum_idx, onetile);
        // tile_regs_acquire();

        // const uint32_t test_reg = 0;
        // // reconfig_data_format(cb_scaled_dy_gamma_sum_idx, cb_scaled_dy_gamma_sum_idx);
        // copy_tile_init(cb_scaled_dy_gamma_xnorm_sum_idx);
        // copy_tile(cb_scaled_dy_gamma_xnorm_sum_idx, 0, test_reg);
        // add_tiles will overwrite the register, no need to zero it first
        // tile_regs_commit();
        // pack_and_push(0, cb_debug_scaled_sum_idx);
        // tile_regs_acquire();

        compute_dy_gamma_sum(row);
        cb_wait_front(cb_scaled_dy_gamma_sum_idx, onetile);

        compute_dy_gamma_xnorm_sum(row);
        cb_wait_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);

        // Process each block of tiles in the row
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_reserve_back(cb_dx_idx, block_size);
            cb_reserve_back(cb_dgamma_components, block_size);
            cb_reserve_back(cb_dbeta_components, block_size);

            // Compute dx
            {
#ifndef EVERYTHING_FITS_IN_L1
                cb_wait_front(cb_x_hat_idx, block_size);
                cb_wait_front(cb_dL_out_idx, block_size);
                cb_wait_front(cb_gamma_idx, block_size);
#endif
                uint32_t dx_register;
                tile_regs_acquire();

                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t input_tile_idx = col + block_idx;
#else
                    const uint32_t input_tile_idx = block_idx;
#endif
                    dx_register = block_idx;
                    compute_dx(input_tile_idx, dx_register);
                    // if (row == num_rows_per_core-1 and block_idx == block_size-1) {
                    // cb_reserve_back(cb_debug_scaled_sum_idx, onetile);
                    // tile_regs_acquire();

                    // const uint32_t test_reg = 0;
                    // // reconfig_data_format(cb_scaled_dy_gamma_sum_idx, cb_scaled_dy_gamma_sum_idx);
                    // copy_tile_init(cb_scaled_dy_gamma_xnorm_sum_idx);
                    // copy_tile(cb_scaled_dy_gamma_xnorm_sum_idx, 0, test_reg);
                    // // add_tiles will overwrite the register, no need to zero it first
                    // tile_regs_commit();
                    // pack_and_push(dx_register, cb_debug_scaled_sum_idx);
                    // }
                }
                tile_regs_commit();
                pack_and_push_block(cb_dx_idx, block_size);
                // cb_wait_front(cb_dx_idx, block_size);
                // if(row == num_rows_per_core-1 and col == 0) {
                //     cb_reserve_back(cb_debug_scaled_sum_idx, onetile);
                //     tile_regs_acquire();

                //     const uint32_t test_reg = 0;
                //     // reconfig_data_format(cb_scaled_dy_gamma_sum_idx, cb_scaled_dy_gamma_sum_idx);
                //     copy_tile_init(cb_scaled_dy_gamma_sum_idx);
                //     copy_tile(cb_scaled_dy_gamma_sum_idx, 0, test_reg);
                //     // add_tiles will overwrite the register, no need to zero it first
                //     tile_regs_commit();
                //     pack_and_push(test_reg, cb_debug_scaled_sum_idx);
                // }
#ifndef EVERYTHING_FITS_IN_L1
                cb_pop_front(cb_x_hat_idx, block_size);
                cb_pop_front(cb_dL_out_idx, block_size);
                cb_pop_front(cb_gamma_idx, block_size);
#endif
            }

            // Compute dgamma_components
            {
#ifndef EVERYTHING_FITS_IN_L1
                cb_wait_front(cb_x_hat_idx, block_size);
                cb_wait_front(cb_dL_out_idx, block_size);
#endif
                uint32_t dgamma_register;
                tile_regs_acquire();

                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t input_tile_idx = col + block_idx;
#else
                    const uint32_t input_tile_idx = block_idx;
#endif
                    dgamma_register = block_idx;
                    compute_dgamma_components(input_tile_idx, dgamma_register);
                }
                tile_regs_commit();
                pack_and_push_block(cb_dgamma_components, block_size);
#ifndef EVERYTHING_FITS_IN_L1
                cb_pop_front(cb_x_hat_idx, block_size);
                cb_pop_front(cb_dL_out_idx, block_size);
#endif
            }

            // Compute dbeta_components
            {
#ifndef EVERYTHING_FITS_IN_L1
                cb_wait_front(cb_dL_out_idx, block_size);
#endif
                uint32_t dbeta_register;
                tile_regs_acquire();
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t dy_tile_idx = col + block_idx;
#else
                    const uint32_t dy_tile_idx = block_idx;
#endif
                    dbeta_register = block_idx;
                    compute_dbeta_components(dy_tile_idx, dbeta_register);
                }
                tile_regs_commit();
                pack_and_push_block(cb_dbeta_components, block_size);
#ifndef EVERYTHING_FITS_IN_L1
                cb_pop_front(cb_dL_out_idx, block_size);
#endif
            }
        }

#ifdef EVERYTHING_FITS_IN_L1
        // Pop per-row data
        cb_pop_front(cb_dL_out_idx, Wt);
        cb_pop_front(cb_x_hat_idx, Wt);

        // Pop gamma only on the last row (since it's shared across all rows)
        if (row == num_rows_per_core - 1) {
            cb_pop_front(cb_gamma_idx, Wt);
        }
#endif

        // Pop the row-level tensors
        cb_pop_front(cb_rstd_idx, 2 * onetile);
        cb_pop_front(cb_scaled_dy_gamma_sum_idx, onetile);
        cb_pop_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    }
    if constexpr (do_mask_w) {
        cb_pop_front(cb_mask_w_idx, onetile);
    }
}

}  // namespace NAMESPACE
