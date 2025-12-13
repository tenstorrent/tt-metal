// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t mask_w = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);
constexpr uint32_t closest_to_Wt_multiple_of_block_size = ((Wt + block_size - 1) / block_size) * block_size;

// CBs with input data
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_0;  // 1/N scaler
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;  // mask for width dimension
constexpr uint32_t cb_eps_idx = tt::CBIndex::c_2;     // epsilon for numerical stability
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;   // gamma (scale parameter)
constexpr uint32_t cb_beta_idx = tt::CBIndex::c_4;    // beta (shift parameter)
constexpr uint32_t cb_input_idx = tt::CBIndex::c_5;   // input tensor

// CBs with output data
constexpr uint32_t cb_output_idx = tt::CBIndex::c_6;  // normalized output
constexpr uint32_t cb_mean_idx = tt::CBIndex::c_7;    // mean (for backward pass)
constexpr uint32_t cb_rstd_idx = tt::CBIndex::c_8;    // rstd (for backward pass)

// CBs with intermediate computations
constexpr uint32_t cb_sum_idx = tt::CBIndex::c_9;                   // sum of inputs
constexpr uint32_t cb_mean_bcast_idx = tt::CBIndex::c_10;           // broadcasted mean
constexpr uint32_t cb_variance_sum_idx = tt::CBIndex::c_11;         // sum((x - mean)^2)
constexpr uint32_t cb_rstd_bcast_idx = tt::CBIndex::c_12;           // broadcasted rstd
constexpr uint32_t cb_x_hat_idx = tt::CBIndex::c_13;                // normalized x_hat

constexpr uint32_t cb_output_intermediate_idx = tt::CBIndex::c_14;  // intermediate for x_hat * gamma

constexpr uint32_t onetile = 1;

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

#ifdef RETURN_MEAN_RSTD
constexpr bool return_mean_rstd = true;
#else
constexpr bool return_mean_rstd = false;
#endif

// Step 1: Compute sum of all input tiles in the row, then scale by 1/N to get mean
// When EVERYTHING_FITS_IN_L1: all Wt tiles are available, process them directly
#ifdef EVERYTHING_FITS_IN_L1
inline void compute_sum() {
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;

    tile_regs_acquire();

    // Accumulate all input tiles: sum = x0 + x1 + x2 + ... + x_{Wt-1}
    for (uint32_t col = 0; col < Wt; ++col) {
        auto target_register = (col == 0) ? sum_register : working_register;

        copy_tile_init(cb_input_idx);
        copy_tile(cb_input_idx, col, target_register);

        // Mask the tile if needed
        if constexpr (do_mask_w) {
            if (col + 1 == Wt) {
                const uint32_t mask_register = target_register + 1U;
                copy_tile_init(cb_mask_w_idx);
                copy_tile(cb_mask_w_idx, 0, mask_register);
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
    pack_and_push(sum_register, cb_sum_idx);

    // Reduce the resulting tile and scale by 1/N to get mean: mean = (1/N) * sum
    const uint32_t mean_register = 0U;
    tile_regs_acquire();
    cb_wait_front(cb_sum_idx, onetile);
    mm_init_short(cb_sum_idx, cb_scaler_idx, 0);
    matmul_tiles(
        cb_sum_idx,
        cb_scaler_idx,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ mean_register);

    tile_regs_commit();
    cb_pop_front(cb_sum_idx, onetile);
    pack_and_push(mean_register, cb_mean_bcast_idx);
}
#else
inline void compute_sum() {
    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;

    tile_regs_acquire();

    for (uint32_t col = 0; col < Wt; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, Wt - col);
        cb_wait_front(cb_input_idx, block_size);

        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            uint32_t global_col = col + block_idx;
            auto target_register = (global_col == 0) ? sum_register : working_register;

            copy_tile_init(cb_input_idx);
            copy_tile(cb_input_idx, block_idx, target_register);

            // Mask the tile if needed
            if constexpr (do_mask_w) {
                if (global_col + 1 == Wt) {
                    const uint32_t mask_register = target_register + 1U;
                    copy_tile_init(cb_mask_w_idx);
                    copy_tile(cb_mask_w_idx, 0, mask_register);
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

        cb_pop_front(cb_input_idx, block_size);
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_sum_idx);

    const uint32_t mean_register = 0U;
    tile_regs_acquire();
    cb_wait_front(cb_sum_idx, onetile);
    mm_init_short(cb_sum_idx, cb_scaler_idx, 0);
    matmul_tiles(
        cb_sum_idx,
        cb_scaler_idx,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ mean_register);

    tile_regs_commit();
    cb_pop_front(cb_sum_idx, onetile);
    pack_and_push(mean_register, cb_mean_bcast_idx);
}
#endif

// Step 2a: Compute variance = (1/N) * sum((x - mean)^2)
// Accumulate squared deviations across all tiles
#ifdef EVERYTHING_FITS_IN_L1
inline void compute_variance_reduced_to_one_tile() {
    cb_wait_front(cb_mean_bcast_idx, onetile);

    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t mean_register = 2U;

    tile_regs_acquire();

    // Load broadcasted mean once
    copy_tile_init(cb_mean_bcast_idx);
    copy_tile(cb_mean_bcast_idx, 0, mean_register);

    for (uint32_t col = 0; col < Wt; ++col) {
        auto target_register = (col == 0) ? sum_register : working_register;

        // Load input
        copy_tile_init(cb_input_idx);
        copy_tile(cb_input_idx, col, target_register);

        // Subtract mean: (x - mean)
        sub_binary_tile_init();
        sub_binary_tile(target_register, mean_register, target_register);

        // Square: (x - mean)^2
        mul_binary_tile_init();
        mul_binary_tile(target_register, target_register, target_register);

        // Mask the tile if needed
        if constexpr (do_mask_w) {
            if (col + 1 == Wt) {
                const uint32_t mask_register = target_register + 1U;
                copy_tile_init(cb_mask_w_idx);
                copy_tile(cb_mask_w_idx, 0, mask_register);
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
    pack_and_push(sum_register, cb_variance_sum_idx);
}
#else
// Compute variance = sum((x - mean)^2) across tiles
inline void compute_variance_reduced_to_one_tile() {
    cb_wait_front(cb_mean_bcast_idx, onetile);

    const uint32_t sum_register = 0U;
    const uint32_t working_register = 1U;
    const uint32_t mean_register = 2U;

    tile_regs_acquire();

    // Load broadcasted mean once
    copy_tile_init(cb_mean_bcast_idx);
    copy_tile(cb_mean_bcast_idx, 0, mean_register);

    for (uint32_t col = 0; col < Wt; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, Wt - col);
        cb_wait_front(cb_input_idx, block_size);

        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            uint32_t global_col = col + block_idx;
            auto target_register = (global_col == 0) ? sum_register : working_register;

            // Load input
            copy_tile_init(cb_input_idx);
            copy_tile(cb_input_idx, block_idx, target_register);

            // Subtract mean: (x - mean)
            sub_binary_tile_init();
            sub_binary_tile(target_register, mean_register, target_register);

            // Square: (x - mean)^2
            mul_binary_tile_init();
            mul_binary_tile(target_register, target_register, target_register);

            // Mask the tile if needed
            if constexpr (do_mask_w) {
                if (global_col + 1 == Wt) {
                    const uint32_t mask_register = target_register + 1U;
                    copy_tile_init(cb_mask_w_idx);
                    copy_tile(cb_mask_w_idx, 0, mask_register);
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

        cb_pop_front(cb_input_idx, block_size);
    }

    tile_regs_commit();
    pack_and_push(sum_register, cb_variance_sum_idx);
}
#endif

// Step 2b: Compute reciprocal standard deviation: rstd = 1 / sqrt(variance + eps)
// First computes variance = (1/N) * sum((x - mean)^2), then adds epsilon and takes rsqrt
inline void compute_rstd() {
    compute_variance_reduced_to_one_tile();

    cb_wait_front(cb_variance_sum_idx, onetile);
    cb_wait_front(cb_eps_idx, onetile);

    const uint32_t rstd_register = 0U;
    const uint32_t eps_register = 1U;

    tile_regs_acquire();

    // Reduce variance sum and scale by 1/N
    mm_init_short(cb_variance_sum_idx, cb_scaler_idx, 0);
    matmul_tiles(
        cb_variance_sum_idx,
        cb_scaler_idx,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        /* idst */ rstd_register);

    // Load epsilon
    copy_tile_init(cb_eps_idx);
    copy_tile(cb_eps_idx, 0, eps_register);

    // Add epsilon: variance + eps
    add_binary_tile_init();
    add_binary_tile(rstd_register, eps_register, rstd_register);

    // Compute rstd = 1 / sqrt(variance + eps) using rsqrt
    rsqrt_tile_init();
    rsqrt_tile(rstd_register);

    tile_regs_commit();
    cb_pop_front(cb_variance_sum_idx, onetile);
    pack_and_push(rstd_register, cb_rstd_bcast_idx);
}

// Step 3: Normalize input: x_hat = (input - mean) * rstd
// Only used in EVERYTHING_FITS_IN_L1 mode; in block mode this is fused with output computation
#ifdef EVERYTHING_FITS_IN_L1
inline void compute_x_hat() {
    cb_wait_front(cb_mean_bcast_idx, onetile);
    cb_wait_front(cb_rstd_bcast_idx, onetile);

    for (uint32_t col = 0; col < Wt; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, Wt - col);

        tile_regs_acquire();

        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            uint32_t x_hat_reg = block_idx;
            uint32_t temp_reg = x_hat_reg + 1;
            uint32_t input_tile_idx = col + block_idx;

            // Subtract mean: (input - mean)
            sub_tiles_init(cb_input_idx, cb_mean_bcast_idx);
            sub_tiles(cb_input_idx, cb_mean_bcast_idx, input_tile_idx, 0, x_hat_reg);

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
#endif

// Compute output = x_hat * gamma + beta
#ifdef EVERYTHING_FITS_IN_L1
inline void compute_output() {
    compute_x_hat();

    cb_wait_front(cb_x_hat_idx, closest_to_Wt_multiple_of_block_size);

    for (uint32_t col = 0; col < Wt; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, Wt - col);

        tile_regs_acquire();

        // First multiply x_hat by gamma -> store in intermediate CB
        mul_bcast_rows_init_short(cb_x_hat_idx, cb_gamma_idx);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            uint32_t input_tile_idx = col + block_idx;
            mul_tiles_bcast_rows(cb_x_hat_idx, cb_gamma_idx, input_tile_idx, input_tile_idx, block_idx);
        }
        tile_regs_commit();
        pack_and_push_block(cb_output_intermediate_idx, block_size);
        cb_wait_front(cb_output_intermediate_idx, block_size);

        // Then add beta from intermediate CB -> store in output CB
        tile_regs_acquire();
        add_bcast_rows_init_short(cb_output_intermediate_idx, cb_beta_idx);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            uint32_t input_tile_idx = col + block_idx;
            add_tiles_bcast_rows(cb_output_intermediate_idx, cb_beta_idx, block_idx, input_tile_idx, block_idx);
        }

        tile_regs_commit();
        cb_pop_front(cb_output_intermediate_idx, block_size);
        pack_and_push_block(cb_output_idx, block_size);
    }
}
#else
// For non-L1 case: compute x_hat and output together block-by-block to avoid CB overflow
inline void compute_output() {
    cb_wait_front(cb_mean_bcast_idx, onetile);
    cb_wait_front(cb_rstd_bcast_idx, onetile);

    for (uint32_t col = 0; col < Wt; col += block_size) {
        const uint32_t current_block_size = std::min(block_size, Wt - col);
        cb_wait_front(cb_input_idx, block_size);
        cb_wait_front(cb_gamma_idx, block_size);
        cb_wait_front(cb_beta_idx, block_size);

        tile_regs_acquire();

        // First compute x_hat = (input - mean) * rstd for this block
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            uint32_t x_hat_reg = block_idx;
            uint32_t temp_reg = x_hat_reg + 1;

            // Subtract mean: (input - mean)
            sub_tiles_init(cb_input_idx, cb_mean_bcast_idx);
            sub_tiles(cb_input_idx, cb_mean_bcast_idx, block_idx, 0, x_hat_reg);

            // Load broadcasted rstd
            copy_tile_init(cb_rstd_bcast_idx);
            copy_tile(cb_rstd_bcast_idx, 0, temp_reg);

            // Multiply by rstd: (input - mean) * rstd = x_hat
            mul_binary_tile_init();
            mul_binary_tile(x_hat_reg, temp_reg, x_hat_reg);
        }

        tile_regs_commit();
        pack_and_push_block(cb_x_hat_idx, block_size);
        cb_wait_front(cb_x_hat_idx, block_size);

        // Now compute output = x_hat * gamma + beta
        // Multiply x_hat by gamma -> store in intermediate CB
        tile_regs_acquire();
        mul_bcast_rows_init_short(cb_x_hat_idx, cb_gamma_idx);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            mul_tiles_bcast_rows(cb_x_hat_idx, cb_gamma_idx, block_idx, block_idx, block_idx);
        }

        tile_regs_commit();
        cb_pop_front(cb_x_hat_idx, block_size);
        pack_and_push_block(cb_output_intermediate_idx, block_size);
        cb_wait_front(cb_output_intermediate_idx, block_size);

        // Then add beta from intermediate CB -> store in output CB
        tile_regs_acquire();
        add_bcast_rows_init_short(cb_output_intermediate_idx, cb_beta_idx);
        for (uint32_t block_idx = 0; block_idx < current_block_size; ++block_idx) {
            add_tiles_bcast_rows(cb_output_intermediate_idx, cb_beta_idx, block_idx, block_idx, block_idx);
        }

        tile_regs_commit();
        cb_pop_front(cb_output_intermediate_idx, block_size);
        pack_and_push_block(cb_output_idx, block_size);
        cb_pop_front(cb_input_idx, block_size);
        cb_pop_front(cb_gamma_idx, block_size);
        cb_pop_front(cb_beta_idx, block_size);
    }
}
#endif

// Copy mean for output (for backward pass)
inline void copy_mean_to_output() {
    if constexpr (return_mean_rstd) {
        cb_wait_front(cb_mean_bcast_idx, onetile);

        const uint32_t mean_register = 0U;

        tile_regs_acquire();
        copy_tile_init(cb_mean_bcast_idx);
        copy_tile(cb_mean_bcast_idx, 0, mean_register);

        tile_regs_commit();
        pack_and_push(mean_register, cb_mean_idx);
    }
}

// Copy rstd for output (for backward pass)
inline void copy_rstd_to_output() {
    if constexpr (return_mean_rstd) {
        cb_wait_front(cb_rstd_bcast_idx, onetile);

        const uint32_t rstd_register = 0U;

        tile_regs_acquire();
        copy_tile_init(cb_rstd_bcast_idx);
        copy_tile(cb_rstd_bcast_idx, 0, rstd_register);

        tile_regs_commit();
        pack_and_push(rstd_register, cb_rstd_idx);
    }
}

inline void MAIN {
    // Wait for constant inputs
    cb_wait_front(cb_scaler_idx, onetile);
    cb_wait_front(cb_eps_idx, onetile);

    if constexpr (do_mask_w) {
        cb_wait_front(cb_mask_w_idx, onetile);
    }

#ifdef EVERYTHING_FITS_IN_L1
    cb_wait_front(cb_gamma_idx, Wt);
    cb_wait_front(cb_beta_idx, Wt);
#endif

    init_sfpu(cb_input_idx, cb_output_idx);
    binary_op_init_common(cb_input_idx, cb_gamma_idx, cb_output_idx);
    mm_init(cb_sum_idx, cb_scaler_idx, cb_mean_bcast_idx);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
#ifdef EVERYTHING_FITS_IN_L1
        cb_wait_front(cb_input_idx, Wt);
#endif

        // Step 1: Compute mean = (1/N) * sum(x)
        compute_sum();

        // Step 2: Compute rstd = 1 / sqrt((1/N) * sum((x - mean)^2) + eps)
        compute_rstd();

        // Step 4: Compute output = x_hat * gamma + beta (this includes x_hat computation)
        compute_output();

        // Step 5: Save mean and rstd for backward pass (if needed)
        copy_mean_to_output();
        copy_rstd_to_output();

        // Cleanup
        cb_pop_front(cb_mean_bcast_idx, onetile);
        cb_pop_front(cb_rstd_bcast_idx, onetile);

#ifdef EVERYTHING_FITS_IN_L1
        cb_pop_front(cb_input_idx, Wt);
        cb_pop_front(cb_x_hat_idx, closest_to_Wt_multiple_of_block_size);
#endif
    }

#ifdef EVERYTHING_FITS_IN_L1
    cb_pop_front(cb_gamma_idx, Wt);
    cb_pop_front(cb_beta_idx, Wt);
#endif

    cb_pop_front(cb_scaler_idx, onetile);
    cb_pop_front(cb_eps_idx, onetile);

    if constexpr (do_mask_w) {
        cb_pop_front(cb_mask_w_idx, onetile);
    }
}

}  // namespace NAMESPACE
