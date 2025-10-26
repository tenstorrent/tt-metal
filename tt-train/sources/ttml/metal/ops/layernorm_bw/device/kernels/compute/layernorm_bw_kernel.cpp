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
constexpr uint32_t cb_x_hat_idx = tt::CBIndex::c_3;       // x_hat (computed as (input - mean) * rstd)
constexpr uint32_t cb_rstd_idx = tt::CBIndex::c_4;        // rstd from forward pass
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;      // upstream gradient
constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_6;  // reduction vector
constexpr uint32_t cb_input_idx = tt::CBIndex::c_7;       // input tensor
constexpr uint32_t cb_mean_idx = tt::CBIndex::c_8;        // mean from forward pass
constexpr uint32_t cb_mean_bcast_idx = tt::CBIndex::c_9;  // broadcasted mean (to avoid conflict with reader)

// CBs with output data
constexpr uint32_t cb_dx_idx = tt::CBIndex::c_10;                // dx (input gradient)
constexpr uint32_t cb_dgamma_components = tt::CBIndex::c_11;     // dgamma components
constexpr uint32_t cb_dbeta_components = tt::CBIndex::c_12;      // dbeta components
constexpr uint32_t cb_rstd_bcast_idx = tt::CBIndex::c_13;        // broadcasted rstd (to avoid conflict with reader)

// CBs with intermediate computations
constexpr uint32_t cb_dy_gamma_sum_idx = tt::CBIndex::c_15;         // sum(dy * gamma)
constexpr uint32_t cb_dy_gamma_xnorm_sum_idx = tt::CBIndex::c_16;   // sum(dy * gamma * x_normalized)
constexpr uint32_t cb_scaled_dy_gamma_sum_idx = tt::CBIndex::c_17;  // (1/N) * sum(dy * gamma) - pre-scaled
constexpr uint32_t cb_scaled_dy_gamma_xnorm_sum_idx =
    tt::CBIndex::c_18;  // (1/N) * sum(dy * gamma * x_normalized) - pre-scaled

constexpr uint32_t onetile = 1;

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

inline void zero_dst_reg(uint32_t i) {
    float zero = 0.0f;
    fill_tile_int(i, zero);
    fill_tile(i, zero);
}

// Compute x_hat = (input - mean) * rstd
// input is in cb_input_idx, mean is in cb_mean_idx (broadcasted), rstd is in cb_rstd_idx (broadcasted)
// result is stored in cb_x_hat_idx
inline void compute_x_hat_preprocessing(uint32_t num_tiles) {
    // mean and rstd are already broadcasted across the row
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        tile_regs_acquire();
        uint32_t x_hat_reg = 0;
        uint32_t mean_reg = 1;
        uint32_t rstd_reg = 2;

        // Load input tile
        copy_tile_init(cb_input_idx);
        copy_tile(cb_input_idx, tile_idx, x_hat_reg);

        // Load broadcasted mean
        copy_tile_init(cb_mean_bcast_idx);
        copy_tile(cb_mean_bcast_idx, 0, mean_reg);

        // Subtract mean: (input - mean)
        sub_binary_tile_init();
        sub_binary_tile(x_hat_reg, mean_reg, x_hat_reg);

        // Load broadcasted rstd
        copy_tile_init(cb_rstd_bcast_idx);
        copy_tile(cb_rstd_bcast_idx, 0, rstd_reg);

        // Multiply by rstd: (input - mean) * rstd
        mul_binary_tile_init();
        mul_binary_tile(x_hat_reg, rstd_reg, x_hat_reg);

        // Store result in cb_x_hat
        cb_reserve_back(cb_x_hat_idx, 1);
        tile_regs_commit();
        pack_and_push(x_hat_reg, cb_x_hat_idx);
    }
}

#ifdef EVERYTHING_FITS_IN_L1
// cb_scaled_dy_gamma_sum_idx (the result is broadcasted across rows)
//                             [[1/N * sum_i(dy[0, :] * gamma[:]), 1/N * sum_i(dy[0, :] * gamma[:]), ...], // shape: [1,
//                             Wt] [1/N * sum_i(dy[1, :] * gamma[:]), 1/N * sum_i(dy[1, :] * gamma[:]), ...], // shape:
//                             [1, Wt]
//                             ...]
// cb_dL_out_idx, cb_gamma_idx blocks are read
// acquire in the beginning, release in the end
inline void compute_dy_gamma_sum(const uint32_t row) {
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t temp_register = 2U;

    tile_regs_acquire();

    zero_dst_reg(sum_register);

    for (uint32_t col = 0; col < Wt; ++col) {
        // Mask the tile if needed
        if constexpr (do_mask_w) {
            if (col + 1 == Wt) {
                mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
                mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, col, col, working_register);

                // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                const uint32_t mask_register = working_register + 1U;

                copy_tile_init(cb_mask_w_idx);
                copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                mask_tile_init();
                mask_tile(working_register, mask_register);

                add_binary_tile_init();
                add_binary_tile(sum_register, working_register, sum_register);
            }
        } else {
            mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
            mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, col, col, sum_register);
        }
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_dy_gamma_sum_idx);
    tile_regs_acquire();

    // tile_regs_acquire();

    // Reduce sum across inner dimension using matmul
    cb_wait_front(cb_dy_gamma_sum_idx, onetile);

    const uint32_t reduced_sum_register = 0U;

    // reconfig_data_format(cb_dy_gamma_sum_idx, cb_mat_mul_reduce);
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
    tile_regs_acquire();

    // Scale the sum by 1/N and store in cb_scaled_dy_gamma_sum_idx
    cb_wait_front(cb_dy_gamma_sum_idx, onetile);
    cb_wait_front(cb_scaler_idx, onetile);

    const uint32_t scaled_sum_register = 0U;
    const uint32_t scaler_register = 1U;

    // reconfig_data_format(cb_dy_gamma_sum_idx, cb_dy_gamma_sum_idx);
    copy_tile_init(cb_dy_gamma_sum_idx);
    copy_tile(cb_dy_gamma_sum_idx, 0, scaled_sum_register);
    // reconfig_data_format(cb_scaler_idx, cb_scaler_idx);
    copy_tile_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, 0, scaler_register);
    mul_binary_tile_init();
    mul_binary_tile(scaled_sum_register, scaler_register, scaled_sum_register);

    tile_regs_commit();
    cb_pop_front(cb_dy_gamma_sum_idx, onetile);
    pack_and_push(scaled_sum_register, cb_scaled_dy_gamma_sum_idx);
    tile_regs_acquire();
    cb_wait_front(cb_scaled_dy_gamma_sum_idx, onetile);
    unary_bcast_init<BroadcastType::COL>(cb_scaled_dy_gamma_sum_idx, cb_scaled_dy_gamma_sum_idx);
    unary_bcast<BroadcastType::COL>(cb_scaled_dy_gamma_sum_idx, /* tile idx */ 0, /* reg tile idx */ 0);
    tile_regs_commit();
    cb_pop_front(cb_scaled_dy_gamma_sum_idx, onetile);
    pack_and_push(0, cb_scaled_dy_gamma_sum_idx);
}

// cb_scaled_dy_gamma_xnorm_sum_idx (the result is broadcasted across rows)
//                             [[1/N * sum_i(dy[0, :] * gamma[:] * x_normalized[0, :]), 1/N * sum_i(dy[0, :] * gamma[:]
//                             * x_normalized[0, :]), ...], // shape: [1, Wt] [1/N * sum_i(dy[1, :] * gamma[:] *
//                             x_normalized[1, :]), 1/N * sum_i(dy[1, :] * gamma[:] * x_normalized[1, :]), ...], //
//                             shape: [1, Wt]
//                             ...]
// cb_dL_out_idx, cb_gamma_idx, cb_x_hat_idx blocks are read
// acquire in the beginning, release in the end
inline void compute_dy_gamma_xnorm_sum(const uint32_t row) {
    // Computes 1/N * sum(dy * gamma * x_normalized) across width dimension when everything fits in L1
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t x_norm_register = 2U;
    const uint32_t temp_register = 3U;

    tile_regs_acquire();

    for (uint32_t col = 0; col < Wt; ++col) {
        auto target_register = (col == 0) ? sum_register : working_register;

        // Compute x_normalized for this tile
        copy_tile_init(cb_x_hat_idx);
        copy_tile(cb_x_hat_idx, col, x_norm_register);

        // Compute dy * gamma for this tile
        zero_dst_reg(target_register);
        mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
        mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, col, col, target_register);

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
    tile_regs_acquire();

    // Reduce sum across inner dimension using matmul
    cb_wait_front(cb_dy_gamma_xnorm_sum_idx, onetile);
    const uint32_t reduced_sum_register = 0U;

    // reconfig_data_format(cb_dy_gamma_xnorm_sum_idx, cb_mat_mul_reduce);
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
    tile_regs_acquire();

    // Scale the sum by 1/N and store in cb_scaled_dy_gamma_xnorm_sum_idx
    cb_wait_front(cb_dy_gamma_xnorm_sum_idx, onetile);
    cb_wait_front(cb_scaler_idx, onetile);

    const uint32_t scaled_sum_register = 0U;
    const uint32_t scaler_register = 1U;
    copy_tile_init(cb_dy_gamma_xnorm_sum_idx);
    copy_tile(cb_dy_gamma_xnorm_sum_idx, 0, scaled_sum_register);
    copy_tile_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, 0, scaler_register);
    mul_binary_tile_init();
    mul_binary_tile(scaled_sum_register, scaler_register, scaled_sum_register);

    tile_regs_commit();
    cb_pop_front(cb_dy_gamma_xnorm_sum_idx, onetile);
    pack_and_push(scaled_sum_register, cb_scaled_dy_gamma_xnorm_sum_idx);
    tile_regs_acquire();
    cb_wait_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    unary_bcast_init<BroadcastType::COL>(cb_scaled_dy_gamma_xnorm_sum_idx, cb_scaled_dy_gamma_xnorm_sum_idx);
    unary_bcast<BroadcastType::COL>(cb_scaled_dy_gamma_xnorm_sum_idx, /* tile idx */ 0, /* reg tile idx */ 0);
    tile_regs_commit();
    cb_pop_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    pack_and_push(0, cb_scaled_dy_gamma_xnorm_sum_idx);
}
#else
// cb_scaled_dy_gamma_sum_idx (the result is broadcasted across rows)
//                             [[1/N * sum_i(dy[0, :] * gamma[:]), 1/N * sum_i(dy[0, :] * gamma[:]), ...], // shape: [1,
//                             Wt] [1/N * sum_i(dy[1, :] * gamma[:]), 1/N * sum_i(dy[1, :] * gamma[:]), ...], // shape:
//                             [1, Wt]
//                             ...]
// cb_dL_out_idx, cb_gamma_idx blocks are read
// acquire in the beginning, release in the end
inline void compute_dy_gamma_sum(const uint32_t row) {
    // Block-based processing when not everything fits in L1
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t temp_register = 2U;

    tile_regs_acquire();

    zero_dst_reg(sum_register);

    for (uint32_t col = 0; col < Wt; col += block_size) {
        cb_wait_front(cb_dL_out_idx, block_size);
        cb_wait_front(cb_gamma_idx, block_size);

        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            uint32_t global_col = col + block_idx;
            if (global_col + 1 > Wt) {
                break;
            }
            // Mask the tile if needed
            if constexpr (do_mask_w) {
                if (global_col + 1 == Wt) {
                    mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
                    mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, block_idx, block_idx, working_register);

                    // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                    const uint32_t mask_register = working_register + 1U;

                    copy_tile_init(cb_mask_w_idx);
                    copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                    mask_tile_init();
                    mask_tile(working_register, mask_register);

                    add_binary_tile_init();
                    add_binary_tile(sum_register, working_register, sum_register);
                }
            } else {
                mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
                mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, block_idx, block_idx, sum_register);
            }
        }

        cb_pop_front(cb_dL_out_idx, block_size);
        cb_pop_front(cb_gamma_idx, block_size);
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_dy_gamma_sum_idx);
    tile_regs_acquire();

    // Reduce using matmul
    cb_wait_front(cb_dy_gamma_sum_idx, onetile);
    const uint32_t reduced_sum_register = 0U;

    // reconfig_data_format(cb_dy_gamma_sum_idx, cb_mat_mul_reduce);
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
    tile_regs_acquire();

    // Scale the sum by 1/N and store in cb_scaled_dy_gamma_sum_idx
    cb_wait_front(cb_dy_gamma_sum_idx, onetile);
    cb_wait_front(cb_scaler_idx, onetile);

    const uint32_t scaled_sum_register = 0U;
    const uint32_t scaler_register = 1U;
    copy_tile_init(cb_dy_gamma_sum_idx);
    copy_tile(cb_dy_gamma_sum_idx, 0, scaled_sum_register);
    copy_tile_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, 0, scaler_register);
    mul_binary_tile_init();
    mul_binary_tile(scaled_sum_register, scaler_register, scaled_sum_register);

    tile_regs_commit();
    cb_pop_front(cb_dy_gamma_sum_idx, onetile);
    pack_and_push(scaled_sum_register, cb_scaled_dy_gamma_sum_idx);
    tile_regs_acquire();
    cb_wait_front(cb_scaled_dy_gamma_sum_idx, onetile);
    unary_bcast_init<BroadcastType::COL>(cb_scaled_dy_gamma_sum_idx, cb_scaled_dy_gamma_sum_idx);
    unary_bcast<BroadcastType::COL>(cb_scaled_dy_gamma_sum_idx, /* tile idx */ 0, /* reg tile idx */ 0);
    tile_regs_commit();
    cb_pop_front(cb_scaled_dy_gamma_sum_idx, onetile);
    pack_and_push(0, cb_scaled_dy_gamma_sum_idx);
}

// cb_scaled_dy_gamma_xnorm_sum_idx (the result is broadcasted across rows)
//                             [[1/N * sum_i(dy[0, :] * gamma[:] * x_normalized[0, :]), 1/N * sum_i(dy[0, :] * gamma[:]
//                             * x_normalized[0, :]), ...], // shape: [1, Wt] [1/N * sum_i(dy[1, :] * gamma[:] *
//                             x_normalized[1, :]), 1/N * sum_i(dy[1, :] * gamma[:] * x_normalized[1, :]), ...], //
//                             shape: [1, Wt]
//                             ...]
// cb_dL_out_idx, cb_gamma_idx, cb_x_hat_idx blocks are read
// acquire in the beginning, release in the end
inline void compute_dy_gamma_xnorm_sum(const uint32_t row) {
    // Similar implementation for non-L1 case
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t x_norm_register = 2U;
    const uint32_t temp_register = 3U;

    tile_regs_acquire();

    for (uint32_t col = 0; col < Wt; col += block_size) {
        // Compute x_hat from input for this block
        cb_wait_front(cb_input_idx, block_size);
        compute_x_hat_preprocessing(block_size);
        tile_regs_acquire();

        cb_wait_front(cb_x_hat_idx, block_size);
        cb_wait_front(cb_dL_out_idx, block_size);
        cb_wait_front(cb_gamma_idx, block_size);

        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            uint32_t global_col = col + block_idx;
            if (global_col + 1 > Wt) {
                break;
            }
            auto target_register = (global_col == 0) ? sum_register : working_register;

            // Load x_normalized (x_hat)
            copy_tile_init(cb_x_hat_idx);
            copy_tile(cb_x_hat_idx, block_idx, x_norm_register);

            // Compute dy * gamma
            zero_dst_reg(target_register);
            mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
            mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, block_idx, block_idx, target_register);

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
        cb_pop_front(cb_input_idx, block_size);
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_dy_gamma_xnorm_sum_idx);
    tile_regs_acquire();

    // Reduce using matmul
    const uint32_t reduced_sum_register = 0U;
    cb_wait_front(cb_dy_gamma_xnorm_sum_idx, onetile);

    // reconfig_data_format(cb_dy_gamma_xnorm_sum_idx, cb_mat_mul_reduce);
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
    cb_pop_front(cb_dy_gamma_xnorm_sum_idx, onetile);
    pack_and_push(scaled_sum_register, cb_scaled_dy_gamma_xnorm_sum_idx);
    tile_regs_acquire();
    cb_wait_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    unary_bcast_init<BroadcastType::COL>(cb_scaled_dy_gamma_xnorm_sum_idx, cb_scaled_dy_gamma_xnorm_sum_idx);
    unary_bcast<BroadcastType::COL>(cb_scaled_dy_gamma_xnorm_sum_idx, /* tile idx */ 0, /* reg tile idx */ 0);
    tile_regs_commit();
    cb_pop_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    pack_and_push(0, cb_scaled_dy_gamma_xnorm_sum_idx);
}
#endif  // EVERYTHING_FITS_IN_L1

// uses 3 registers starting from dx_register
// Computes dx = (1/rstd) * (dy*gamma - (1/N) * sum(dy*gamma) - x_normalized * (1/N) * sum(dy*gamma * x_normalized))
// result is in dx_register
// acquire in the inner loop, push after block is processed
inline void compute_dx(const uint32_t input_tile_idx, const uint32_t dx_register, const uint32_t global_col) {
    if (global_col + 1 > Wt) {
        zero_dst_reg(dx_register);
        return;
    }
    const uint32_t x_norm_register = dx_register + 1;
    const uint32_t temp_register = dx_register + 2;

    // Compute dy * gamma
    zero_dst_reg(dx_register);
    mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
    mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, input_tile_idx, input_tile_idx, dx_register);

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
    copy_tile_init(cb_rstd_bcast_idx);
    copy_tile(cb_rstd_bcast_idx, 0, temp_register);
    mul_binary_tile_init();
    mul_binary_tile(dx_register, temp_register, dx_register);

    if constexpr (do_mask_w) {
        if (global_col + 1 == Wt) {
            const uint32_t mask_register = dx_register + 1U;
            copy_tile_init(cb_mask_w_idx);
            copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);
            mask_tile_init();
            mask_tile(dx_register, mask_register);
        }
    }
}

// uses 3 registers starting from dgamma_register
// result is in dgamma_register
// acquire in the inner loop, push after block is processed
inline void compute_dgamma_components(
    const uint32_t input_tile_idx, const uint32_t dgamma_register, const uint32_t global_col) {
    // Computes dgamma_components = dy * x_normalized

    if (global_col + 1 > Wt) {
        zero_dst_reg(dgamma_register);
        return;
    }

    const uint32_t x_norm_register = dgamma_register + 1;

    // Load dy
    copy_tile_init(cb_dL_out_idx);
    copy_tile(cb_dL_out_idx, input_tile_idx, dgamma_register);

    // Load x_normalized
    copy_tile_init(cb_x_hat_idx);
    copy_tile(cb_x_hat_idx, input_tile_idx, x_norm_register);

    // Multiply: dy * x_normalized
    mul_binary_tile_init();
    mul_binary_tile(dgamma_register, x_norm_register, dgamma_register);

    // Mask dgamma_register if needed
    if constexpr (do_mask_w) {
        if (global_col + 1 == Wt) {
            // Limitation: mask_tile only works when the mask register is immediately next to the data register.
            const uint32_t mask_register = dgamma_register + 1U;

            copy_tile_init(cb_mask_w_idx);
            copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

            mask_tile_init();
            mask_tile(dgamma_register, mask_register);
        }
    }
}

// Computes dbeta_components = dy (simple copy)
// result is in dbeta_register
// acquire in the inner loop, push after block is processed
inline void compute_dbeta_components(
    const uint32_t dy_tile_idx, const uint32_t dbeta_register, const uint32_t global_col) {
    if (global_col + 1 > Wt) {
        zero_dst_reg(dbeta_register);
        return;
    }

    copy_tile_init(cb_dL_out_idx);
    copy_tile(cb_dL_out_idx, dy_tile_idx, dbeta_register);

    // Mask dbeta_register if needed
    if constexpr (do_mask_w) {
        if (global_col + 1 == Wt) {
            // Limitation: mask_tile only works when the mask register is immediately next to the data register.
            const uint32_t mask_register = dbeta_register + 1U;
            copy_tile_init(cb_mask_w_idx);
            copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

            mask_tile_init();
            mask_tile(dbeta_register, mask_register);
        }
    }
}

inline void MAIN {
    if constexpr (do_mask_w) {
        cb_wait_front(cb_mask_w_idx, onetile);
    }
    cb_wait_front(cb_scaler_idx, onetile);
    cb_wait_front(cb_mat_mul_reduce, onetile);

    init_sfpu(cb_x_hat_idx, cb_dx_idx);
    binary_op_init_common(cb_x_hat_idx, cb_gamma_idx, cb_dx_idx);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Wait for rstd and mean (per row)
        // Both are broadcasted across rows: [[value, value, ...], [value, value, ...], ...]
        tile_regs_acquire();
        cb_wait_front(cb_rstd_idx, onetile);
        unary_bcast_init<BroadcastType::COL>(cb_rstd_idx, cb_rstd_idx);
        unary_bcast<BroadcastType::COL>(cb_rstd_idx, /* tile idx */ 0, /* reg tile idx */ 0);
        tile_regs_commit();
        cb_pop_front(cb_rstd_idx, onetile);
        pack_and_push(0, cb_rstd_bcast_idx);
        cb_wait_front(cb_rstd_bcast_idx, onetile);

        tile_regs_acquire();
        cb_wait_front(cb_mean_idx, onetile);
        unary_bcast_init<BroadcastType::COL>(cb_mean_idx, cb_mean_idx);
        unary_bcast<BroadcastType::COL>(cb_mean_idx, /* tile idx */ 0, /* reg tile idx */ 0);
        tile_regs_commit();
        cb_pop_front(cb_mean_idx, onetile);
        pack_and_push(0, cb_mean_bcast_idx);
        cb_wait_front(cb_mean_bcast_idx, onetile);

#ifdef EVERYTHING_FITS_IN_L1
        // Wait for input for entire row when everything fits in L1
        cb_wait_front(cb_input_idx, Wt);
        // Compute x_hat = (input - mean) * rstd for the entire row
        compute_x_hat_preprocessing(Wt);
        cb_wait_front(cb_x_hat_idx, Wt);

        cb_wait_front(cb_dL_out_idx, Wt);
        // If everything fits in L1, wait for gamma only once as its shared across all rows
        if (row == 0) {
            cb_wait_front(cb_gamma_idx, Wt);
        }
#endif
        compute_dy_gamma_sum(row);
        cb_wait_front(cb_scaled_dy_gamma_sum_idx, onetile);

        compute_dy_gamma_xnorm_sum(row);
        cb_wait_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);

        // Process each block of tiles in the row
        for (uint32_t col = 0; col < Wt; col += block_size) {
            // Calculate actual number of tiles in this block (handles last block when Wt % block_size != 0)
            const uint32_t current_block_size = (col + block_size > Wt) ? (Wt - col) : block_size;

            // Compute dx
            {
#ifndef EVERYTHING_FITS_IN_L1
                // Compute x_hat for this block
                cb_wait_front(cb_input_idx, current_block_size);
                compute_x_hat_preprocessing(current_block_size);
                cb_wait_front(cb_x_hat_idx, current_block_size);

                cb_wait_front(cb_dL_out_idx, current_block_size);
                cb_wait_front(cb_gamma_idx, current_block_size);
#endif
                uint32_t dx_register;
                tile_regs_acquire();

                for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t input_tile_idx = col + block_idx;
#else
                    const uint32_t input_tile_idx = block_idx;
#endif
                    dx_register = block_idx;
                    compute_dx(input_tile_idx, dx_register, col + block_idx);
                }
                tile_regs_commit();
                pack_and_push_block(cb_dx_idx, current_block_size);
#ifndef EVERYTHING_FITS_IN_L1
                cb_pop_front(cb_gamma_idx, current_block_size);
                cb_pop_front(cb_dL_out_idx, current_block_size);
                cb_pop_front(cb_x_hat_idx, current_block_size);
                cb_pop_front(cb_input_idx, current_block_size);
#endif
            }

            // Compute dgamma_components
            {
#ifndef EVERYTHING_FITS_IN_L1
                // Compute x_hat for this block
                cb_wait_front(cb_input_idx, current_block_size);
                compute_x_hat_preprocessing(current_block_size);
                cb_wait_front(cb_x_hat_idx, current_block_size);

                cb_wait_front(cb_dL_out_idx, current_block_size);
#endif
                uint32_t dgamma_register;
                tile_regs_acquire();

                for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t input_tile_idx = col + block_idx;
#else
                    const uint32_t input_tile_idx = block_idx;
#endif
                    dgamma_register = block_idx;
                    compute_dgamma_components(input_tile_idx, dgamma_register, col + block_idx);
                }
                tile_regs_commit();
                pack_and_push_block(cb_dgamma_components, current_block_size);
#ifndef EVERYTHING_FITS_IN_L1
                cb_pop_front(cb_x_hat_idx, current_block_size);
                cb_pop_front(cb_dL_out_idx, current_block_size);
                cb_pop_front(cb_input_idx, current_block_size);
#endif
            }

            // Compute dbeta_components
            {
#ifndef EVERYTHING_FITS_IN_L1
                cb_wait_front(cb_dL_out_idx, current_block_size);
#endif
                uint32_t dbeta_register;
                tile_regs_acquire();

                for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t dy_tile_idx = col + block_idx;
#else
                    const uint32_t dy_tile_idx = block_idx;
#endif
                    dbeta_register = block_idx;
                    compute_dbeta_components(dy_tile_idx, dbeta_register, col + block_idx);
                }
                tile_regs_commit();
                pack_and_push_block(cb_dbeta_components, current_block_size);
#ifndef EVERYTHING_FITS_IN_L1
                cb_pop_front(cb_dL_out_idx, current_block_size);
#endif
            }
        }

        // Pop the row-level tensors
        cb_pop_front(cb_rstd_bcast_idx, onetile);
        cb_pop_front(cb_mean_bcast_idx, onetile);
        cb_pop_front(cb_scaled_dy_gamma_sum_idx, onetile);
        cb_pop_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
#ifdef EVERYTHING_FITS_IN_L1
        // Pop per-row data
        cb_pop_front(cb_dL_out_idx, Wt);
        cb_pop_front(cb_x_hat_idx, Wt);
        cb_pop_front(cb_input_idx, Wt);

        // Pop gamma only on the last row (since it's shared across all rows)
        if (row == num_rows_per_core - 1) {
            cb_pop_front(cb_gamma_idx, Wt);
        }
#endif
    }
    if constexpr (do_mask_w) {
        cb_pop_front(cb_mask_w_idx, onetile);
    }
}

}  // namespace NAMESPACE
