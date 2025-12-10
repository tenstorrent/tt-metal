// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t mask_w = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);
constexpr uint32_t closest_to_Wt_multiple_of_block_size = ((Wt + block_size - 1) / block_size) * block_size;

// CBs with input data
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_0;      // 1/N scaler
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;      // mask for width dimension
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_2;       // gamma (scale parameter)
constexpr uint32_t cb_x_hat_idx = tt::CBIndex::c_3;       // x_hat (computed as (input - mean) * rstd)
constexpr uint32_t cb_rstd_idx = tt::CBIndex::c_4;        // rstd from forward pass
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;      // upstream gradient
constexpr uint32_t cb_input_idx = tt::CBIndex::c_6;       // input tensor
constexpr uint32_t cb_mean_idx = tt::CBIndex::c_7;        // mean from forward pass
constexpr uint32_t cb_mean_bcast_idx = tt::CBIndex::c_8;  // broadcasted mean (to avoid conflict with reader)

// CBs with output data
constexpr uint32_t cb_dx_idx = tt::CBIndex::c_9;              // dx (input gradient)
constexpr uint32_t cb_dgamma_components = tt::CBIndex::c_10;  // dgamma components
constexpr uint32_t cb_dbeta_components = tt::CBIndex::c_11;   // dbeta components
constexpr uint32_t cb_rstd_bcast_idx = tt::CBIndex::c_12;     // broadcasted rstd (to avoid conflict with reader)

// CBs with intermediate computations
constexpr uint32_t cb_scaled_dy_gamma_sum_idx = tt::CBIndex::c_13;  // (1/N) * sum(dy * gamma) - pre-scaled
constexpr uint32_t cb_scaled_dy_gamma_xnorm_sum_idx =
    tt::CBIndex::c_14;  // (1/N) * sum(dy * gamma * x_normalized) - pre-scaled

constexpr uint32_t onetile = 1;

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

// Compute x_hat = (input - mean) * rstd
// input is in cb_input_idx, mean is in cb_mean_idx (broadcasted), rstd is in cb_rstd_idx (broadcasted)
// result is stored in cb_x_hat_idx
inline void compute_x_hat_preprocessing(const uint32_t num_tiles) {
    // mean and rstd are already broadcasted across the row
    reconfig_data_format(cb_input_idx, cb_input_idx);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx += block_size) {
        const uint32_t current_block_size = std::min(block_size, num_tiles - tile_idx);
        tile_regs_acquire();

        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            const uint32_t x_hat_reg = block_idx;
            const uint32_t temp_reg = x_hat_reg + 1;

            // Subtract mean: (input - mean)
            sub_tiles_init(cb_input_idx, cb_mean_bcast_idx);
            sub_tiles(cb_input_idx, cb_mean_bcast_idx, tile_idx + block_idx, 0, x_hat_reg);

            // Load broadcasted rstd
            copy_tile_init(cb_rstd_bcast_idx);
            copy_tile(cb_rstd_bcast_idx, 0, temp_reg);

            // Multiply by rstd: (input - mean) * rstd
            mul_binary_tile_init();
            mul_binary_tile(x_hat_reg, temp_reg, x_hat_reg);
        }

        tile_regs_commit();
        pack_and_push_block(cb_x_hat_idx, block_size);
    }
}

#ifdef EVERYTHING_FITS_IN_L1

// cb_scaled_dy_gamma_sum_idx (the result is broadcasted across rows)
//                             [[1/N * sum_i(dy[0, :] * gamma[:]), 1/N * sum_i(dy[0, :] * gamma[:]), ...], // shape: [1,
//                             Wt] [1/N * sum_i(dy[1, :] * gamma[:]), 1/N * sum_i(dy[1, :] * gamma[:]), ...], // shape:
//                             [1, Wt]
//                             ...]
// cb_dL_out_idx, cb_gamma_idx, cb_x_hat_idx blocks are read
// acquire in the beginning, release in the end
inline void compute_dy_gamma_sum(const uint32_t row) {
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t temp_register = 2U;

    tile_regs_acquire();

    reconfig_data_format(cb_dL_out_idx, cb_gamma_idx);

    // Accumulate dy * gamma into sum_register
    for (uint32_t col = 0; col < Wt; ++col) {
        // Mask the tile if needed
        if constexpr (do_mask_w) {
            if (col + 1 == Wt) {
                const uint32_t target_register = (col == 0) ? sum_register : working_register;
                mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
                mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, col, col, target_register);

                // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                const uint32_t mask_register = target_register + 1U;

                copy_tile_init(cb_mask_w_idx);
                copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                mask_tile_init();
                mask_tile(target_register, mask_register);

                if (col > 0) {
                    add_binary_tile_init();
                    add_binary_tile(sum_register, target_register, sum_register);
                }
            } else {
                mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
                mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, col, col, sum_register);
            }
        } else {
            mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
            mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, col, col, sum_register);
        }
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_scaled_dy_gamma_sum_idx);

    // Reduce sum across inner dimension and scale by 1/N using matmul
    const uint32_t reduced_sum_register = 0U;
    tile_regs_acquire();
    cb_wait_front(cb_scaled_dy_gamma_sum_idx, onetile);

    reconfig_data_format(cb_scaled_dy_gamma_sum_idx, cb_scaler_idx);
    mm_init(cb_scaled_dy_gamma_sum_idx, cb_scaler_idx, cb_scaled_dy_gamma_sum_idx, 0);
    matmul_tiles(
        cb_scaled_dy_gamma_sum_idx,
        cb_scaler_idx,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ reduced_sum_register);

    tile_regs_commit();
    cb_pop_front(cb_scaled_dy_gamma_sum_idx, onetile);
    pack_and_push(reduced_sum_register, cb_scaled_dy_gamma_sum_idx);
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
    const uint32_t temp_register = 2U;

    tile_regs_acquire();

    reconfig_data_format(cb_dL_out_idx, cb_dL_out_idx);

    // Accumulate dy * gamma * x_normalized into sum_register
    for (uint32_t col = 0; col < Wt; ++col) {
        const uint32_t target_register = (col == 0) ? sum_register : working_register;

        // compute dy * gamma for dy_gamma_xnorm
        zero_dst_reg(target_register);
        mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
        mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, col, col, target_register);

        // Load x_normalized for this tile
        copy_tile_init(cb_x_hat_idx);
        copy_tile(cb_x_hat_idx, col, temp_register);

        // Multiply: (dy * gamma) * x_normalized
        mul_binary_tile_init();
        mul_binary_tile(target_register, temp_register, target_register);

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
    pack_and_push(sum_register, cb_scaled_dy_gamma_xnorm_sum_idx);

    // Reduce sum across inner dimension and scale by 1/N using matmul
    const uint32_t reduced_sum_register = 0U;
    tile_regs_acquire();
    cb_wait_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);

    reconfig_data_format(cb_scaled_dy_gamma_xnorm_sum_idx, cb_scaler_idx);
    mm_init(cb_scaled_dy_gamma_xnorm_sum_idx, cb_scaler_idx, cb_scaled_dy_gamma_xnorm_sum_idx, 0);
    matmul_tiles(
        cb_scaled_dy_gamma_xnorm_sum_idx,
        cb_scaler_idx,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ reduced_sum_register);

    tile_regs_commit();
    cb_pop_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    pack_and_push(reduced_sum_register, cb_scaled_dy_gamma_xnorm_sum_idx);
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

    tile_regs_acquire();

    reconfig_data_format(cb_dL_out_idx, cb_gamma_idx);

    // Accumulate dy * gamma into sum_register
    for (uint32_t col = 0; col < Wt; col += block_size) {
        cb_wait_front(cb_dL_out_idx, block_size);
        cb_wait_front(cb_gamma_idx, block_size);

        const uint32_t current_block_size = std::min(block_size, Wt - col);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            uint32_t global_col = col + block_idx;
            // Mask the tile if needed
            if constexpr (do_mask_w) {
                if (global_col + 1 == Wt) {
                    const uint32_t target_register = (global_col == 0) ? sum_register : working_register;
                    mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
                    mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, block_idx, block_idx, target_register);

                    // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                    const uint32_t mask_register = target_register + 1U;

                    copy_tile_init(cb_mask_w_idx);
                    copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                    mask_tile_init();
                    mask_tile(target_register, mask_register);

                    if (global_col > 0) {
                        add_binary_tile_init();
                        add_binary_tile(sum_register, target_register, sum_register);
                    }
                } else {
                    mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
                    mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, block_idx, block_idx, sum_register);
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
    pack_and_push(sum_register, cb_scaled_dy_gamma_sum_idx);

    // Reduce using matmul and scale by 1/N
    const uint32_t reduced_sum_register = 0U;
    tile_regs_acquire();
    cb_wait_front(cb_scaled_dy_gamma_sum_idx, onetile);

    reconfig_data_format(cb_scaled_dy_gamma_sum_idx, cb_scaler_idx);
    mm_init(cb_scaled_dy_gamma_sum_idx, cb_scaler_idx, cb_scaled_dy_gamma_sum_idx, 0);
    matmul_tiles(
        cb_scaled_dy_gamma_sum_idx,
        cb_scaler_idx,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ reduced_sum_register);

    tile_regs_commit();
    cb_pop_front(cb_scaled_dy_gamma_sum_idx, onetile);
    pack_and_push(reduced_sum_register, cb_scaled_dy_gamma_sum_idx);
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
    tile_regs_acquire();

    reconfig_data_format(cb_dL_out_idx, cb_dL_out_idx);

    // Accumulate dy * gamma * x_normalized into sum_register
    for (uint32_t col = 0; col < Wt; col += block_size) {
        // Compute x_hat from input for this block
        cb_wait_front(cb_input_idx, block_size);
        compute_x_hat_preprocessing(block_size);

        cb_wait_front(cb_x_hat_idx, block_size);
        cb_wait_front(cb_dL_out_idx, block_size);
        cb_wait_front(cb_gamma_idx, block_size);

        const uint32_t current_block_size = std::min(block_size, Wt - col);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            uint32_t global_col = col + block_idx;
            const uint32_t target_register = (global_col == 0) ? sum_register : working_register;

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
    pack_and_push(sum_register, cb_scaled_dy_gamma_xnorm_sum_idx);

    // Reduce using matmul and scale by 1/N
    const uint32_t reduced_sum_register = 0U;
    tile_regs_acquire();
    cb_wait_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);

    reconfig_data_format(cb_scaled_dy_gamma_xnorm_sum_idx, cb_scaler_idx);
    mm_init(cb_scaled_dy_gamma_xnorm_sum_idx, cb_scaler_idx, cb_scaled_dy_gamma_xnorm_sum_idx, 0);
    matmul_tiles(
        cb_scaled_dy_gamma_xnorm_sum_idx,
        cb_scaler_idx,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ reduced_sum_register);

    tile_regs_commit();
    cb_pop_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
    pack_and_push(reduced_sum_register, cb_scaled_dy_gamma_xnorm_sum_idx);
}
#endif  // EVERYTHING_FITS_IN_L1

// uses 3 registers starting from dx_register
// Computes dx = (1/rstd) * (dy*gamma - (1/N) * sum(dy*gamma) - x_normalized * (1/N) * sum(dy*gamma * x_normalized))
// result is in dx_register
// acquire in the inner loop, push after block is processed
inline void compute_dx(const uint32_t input_tile_idx, const uint32_t dx_register, const uint32_t global_col) {
    const uint32_t temp_register = dx_register + 1U;

    // Compute dy * gamma
    zero_dst_reg(dx_register);
    reconfig_data_format(cb_dL_out_idx, cb_gamma_idx);
    mul_bcast_rows_init_short(cb_dL_out_idx, cb_gamma_idx);
    mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, input_tile_idx, input_tile_idx, dx_register);

    // Load pre-scaled sum: (1/N) * sum(dy*gamma)
    reconfig_data_format(cb_scaled_dy_gamma_sum_idx, cb_scaled_dy_gamma_sum_idx);
    copy_tile_init(cb_scaled_dy_gamma_sum_idx);
    copy_tile(cb_scaled_dy_gamma_sum_idx, 0, temp_register);

    // Subtract: dy*gamma - (1/N) * sum(dy*gamma)
    sub_binary_tile_init();
    sub_binary_tile(dx_register, temp_register, dx_register);

    // Multiply by x_normalized: x_normalized * (1/N) * sum(dy*gamma * x_normalized)
    reconfig_data_format(cb_x_hat_idx, cb_scaled_dy_gamma_xnorm_sum_idx);
    mul_tiles_init(cb_x_hat_idx, cb_scaled_dy_gamma_xnorm_sum_idx);
    mul_tiles(cb_x_hat_idx, cb_scaled_dy_gamma_xnorm_sum_idx, input_tile_idx, 0, temp_register);

    // Subtract: result - x_normalized * (1/N) * sum(...)
    sub_binary_tile_init();
    sub_binary_tile(dx_register, temp_register, dx_register);

    // Multiply by rstd: dx = rstd * (...)
    reconfig_data_format(cb_rstd_bcast_idx, cb_rstd_bcast_idx);
    copy_tile_init(cb_rstd_bcast_idx);
    copy_tile(cb_rstd_bcast_idx, 0, temp_register);
    mul_binary_tile_init();
    mul_binary_tile(dx_register, temp_register, dx_register);
}

// uses 3 registers starting from dgamma_register
// result is in dgamma_register
// acquire in the inner loop, push after block is processed
inline void compute_dgamma_components(
    const uint32_t input_tile_idx, const uint32_t dgamma_register, const uint32_t global_col) {
    // Computes dgamma_components = dy * x_normalized
    // Load x_normalized
    mul_tiles_init(cb_dL_out_idx, cb_x_hat_idx);
    mul_tiles(cb_dL_out_idx, cb_x_hat_idx, input_tile_idx, input_tile_idx, dgamma_register);
}

// Computes dbeta_components = dy (simple copy)
// result is in dbeta_register
// acquire in the inner loop, push after block is processed
inline void compute_dbeta_components(
    const uint32_t dy_tile_idx, const uint32_t dbeta_register, const uint32_t global_col) {
    copy_tile_init(cb_dL_out_idx);
    copy_tile(cb_dL_out_idx, dy_tile_idx, dbeta_register);
}

inline void MAIN {
    if constexpr (do_mask_w) {
        cb_wait_front(cb_mask_w_idx, onetile);
    }
    cb_wait_front(cb_scaler_idx, onetile);

    init_sfpu(cb_x_hat_idx, cb_dx_idx);
    binary_op_init_common(cb_x_hat_idx, cb_gamma_idx, cb_dx_idx);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Wait for rstd and mean (per row)
        // Both are broadcasted across rows: [[value, value, ...], [value, value, ...], ...]
        tile_regs_acquire();
        cb_wait_front(cb_rstd_idx, onetile);
        reconfig_data_format(cb_rstd_idx, cb_rstd_idx);
        unary_bcast_init<BroadcastType::COL>(cb_rstd_idx, cb_rstd_idx);
        unary_bcast<BroadcastType::COL>(cb_rstd_idx, /* tile idx */ 0, /* reg tile idx */ 0);
        tile_regs_commit();
        cb_pop_front(cb_rstd_idx, onetile);
        pack_and_push(0, cb_rstd_bcast_idx);
        cb_wait_front(cb_rstd_bcast_idx, onetile);

        tile_regs_acquire();
        cb_wait_front(cb_mean_idx, onetile);
        reconfig_data_format(cb_mean_idx, cb_mean_idx);
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
        cb_wait_front(cb_x_hat_idx, closest_to_Wt_multiple_of_block_size);

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
            const uint32_t current_block_size = std::min(block_size, Wt - col);

            // Compute dx
#ifndef EVERYTHING_FITS_IN_L1
            // Compute x_hat for this block
            cb_wait_front(cb_input_idx, block_size);
            compute_x_hat_preprocessing(current_block_size);
            cb_wait_front(cb_x_hat_idx, block_size);

            cb_wait_front(cb_dL_out_idx, block_size);
            cb_wait_front(cb_gamma_idx, block_size);
#endif
            {
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
                pack_and_push_block(cb_dx_idx, block_size);
            }

            reconfig_data_format(cb_dL_out_idx, cb_dL_out_idx);
            // Compute dgamma_components
            {
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
                pack_and_push_block(cb_dgamma_components, block_size);
            }

            // Compute dbeta_components
            {
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
                pack_and_push_block(cb_dbeta_components, block_size);
            }
#ifndef EVERYTHING_FITS_IN_L1
            cb_pop_front(cb_gamma_idx, block_size);
            cb_pop_front(cb_dL_out_idx, block_size);
            cb_pop_front(cb_x_hat_idx, block_size);
            cb_pop_front(cb_input_idx, block_size);
#endif
        }

        // Pop the row-level tensors
        cb_pop_front(cb_rstd_bcast_idx, onetile);
        cb_pop_front(cb_mean_bcast_idx, onetile);
        cb_pop_front(cb_scaled_dy_gamma_sum_idx, onetile);
        cb_pop_front(cb_scaled_dy_gamma_xnorm_sum_idx, onetile);
#ifdef EVERYTHING_FITS_IN_L1
        // Pop per-row data
        cb_pop_front(cb_dL_out_idx, Wt);
        cb_pop_front(cb_x_hat_idx, closest_to_Wt_multiple_of_block_size);
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
