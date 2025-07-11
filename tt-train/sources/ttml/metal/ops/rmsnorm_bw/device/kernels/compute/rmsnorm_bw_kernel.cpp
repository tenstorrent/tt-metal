// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
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
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_6;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_7;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_8;
// CBs with intermediate computations
constexpr uint32_t cb_recip_rms_a_bcasted_idx = tt::CBIndex::c_9;
constexpr uint32_t cb_scale_idx = tt::CBIndex::c_10;
constexpr uint32_t cb_scale_bcasted_idx = tt::CBIndex::c_11;

constexpr uint32_t onetile = 1;

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

// ============================================================================
// The following functions are register-only helpers. They perform intermediate
// computations using registers but do not write results directly to CBs.
// ============================================================================

inline void compute_gained_dL_dout(uint32_t col, uint32_t working_register, uint32_t tile_register) {
    // The output of this function is gained_dL_dout, which is stored in working_register.

    // Compute scaled_gain = gamma * 1/rms_a
    // NOTE: cb_recip_rms_a_bcasted is ready to be used, so we do not need to wait for it.
    mul_tiles_init(cb_gamma_idx, cb_recip_rms_a_bcasted_idx);
    mul_tiles(cb_gamma_idx, cb_recip_rms_a_bcasted_idx, col, 0, working_register);

    // Compute gained_dL_dout = scaled_gain * dL_out
    copy_tile_init(cb_dL_out_idx);
    copy_tile(cb_dL_out_idx, col, tile_register);
    mul_binary_tile_init();
    mul_binary_tile(working_register, tile_register);
}

inline void compute_dL_da(
    const uint32_t input_tile_idx,
    const uint32_t dL_da_register,
    const uint32_t rhs_register,
    const uint32_t tile_register) {
    // Computes gradient with respect to input activation (dL_da) for RMSNorm backward pass.
    // Formula: dL_da = gained_dL_dout - (scale * a) / (c * rms_a^2)
    // where gained_dL_dout = (gamma / rms_a) * dL_dout

    // Compute scaled_outer = scale * a (broadcasted multiplication)
    reconfig_data_format(cb_input_idx, cb_scale_bcasted_idx);
    mul_tiles_init(cb_input_idx, cb_scale_bcasted_idx);
    mul_tiles(cb_input_idx, cb_scale_bcasted_idx, /* tile_idx */ input_tile_idx, /* tile_idx */ 0, rhs_register);

    // Compute rhs = scaled_outer / (c * rms_a^2)
    // Uses pre-computed reciprocal of rms_a and scaler (1/c) for efficiency
    reconfig_data_format(cb_recip_rms_a_bcasted_idx, cb_recip_rms_a_bcasted_idx);
    copy_tile_init(cb_recip_rms_a_bcasted_idx);
    copy_tile(cb_recip_rms_a_bcasted_idx, /* tile_idx */ 0, tile_register);
    mul_binary_tile_init();
    mul_binary_tile(rhs_register, tile_register);
    mul_binary_tile(rhs_register, tile_register);
    copy_tile_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, /* tile_idx */ 0, tile_register);
    mul_binary_tile_init();
    mul_binary_tile(rhs_register, tile_register);

    // Compute final result: dL_da = gained_dL_dout - rhs
    compute_gained_dL_dout(input_tile_idx, dL_da_register, tile_register);
    sub_binary_tile_init();
    sub_binary_tile(dL_da_register, rhs_register);
}

inline void compute_dL_dgamma_components(
    const uint32_t input_tile_idx, const uint32_t dL_dg_components_register, const uint32_t tile_register) {
    // Computes gradient components with respect to gamma (dL_dgamma_components) for RMSNorm backward pass.
    // Formula: dL_dgamma_components = dL_dout * (a / rms_a)
    // These components will be reduced across batches and sequence dimensions outside the kernel.

    // Compute normalized_a = a / rms_a (using pre-computed reciprocal)
    reconfig_data_format(cb_input_idx, cb_recip_rms_a_bcasted_idx);
    mul_tiles_init(cb_input_idx, cb_recip_rms_a_bcasted_idx);
    mul_tiles(
        cb_input_idx,
        cb_recip_rms_a_bcasted_idx,
        /* tile_idx */ input_tile_idx,
        /* tile_idx */ 0,
        dL_dg_components_register);

    // Compute dL_dgamma_components = normalized_a * dL_dout
    copy_tile_init(cb_dL_out_idx);
    copy_tile(cb_dL_out_idx, /* tile_idx */ input_tile_idx, tile_register);
    mul_binary_tile_init();
    mul_binary_tile(dL_dg_components_register, tile_register);
}

// ============================================================================
// Functions below compute final results for a pipeline stage and write them to
// circular buffers (CBs). These functions produce outputs that are consumed by
// later stages.
// ============================================================================

inline void compute_recip_rms_a_bcasted() {
    // Computes reciprocal of RMS activation (1/rms_a) and broadcasts it across columns.
    const uint32_t reg_rms_a = 0;
    tile_regs_acquire();
    unary_bcast_init<BroadcastType::COL>(cb_rms_a_idx, cb_recip_rms_a_bcasted_idx);
    unary_bcast<BroadcastType::COL>(cb_rms_a_idx, /* tile idx */ 0, /* reg tile idx */ reg_rms_a);
    recip_tile_init();
    recip_tile(reg_rms_a);
    tile_regs_commit();
    pack_and_push(reg_rms_a, cb_recip_rms_a_bcasted_idx);
}

#ifdef EVERYTHING_FITS_IN_L1
inline void compute_scale(const uint32_t row) {
    // Computes scale factor for RMSNorm backward pass: scale = sum(a * gained_dL_dout, dim=C)
    // where gained_dL_dout = (gamma / rms_a) * dL_dout
    // The result is reduced along the inner dimension and used in dL_da computation.

    const uint32_t scale_register = 0U;
    const uint32_t intermediate_register = 1U;
    const uint32_t tile_register = 2U;
    tile_regs_acquire();
    // Gamma is constant among all rows so if everything fits in L1, we can read it only once.
    if (row == 0) {
        cb_wait_front(cb_gamma_idx, Wt);
    }
    cb_wait_front(cb_dL_out_idx, Wt);
    cb_wait_front(cb_input_idx, Wt);
    for (uint32_t col = 0; col < Wt; col++) {
        // If col == 0, we use scale_register as the working register to aviod unnecessary copying of data.
        auto working_register = col == 0 ? scale_register : intermediate_register;

        compute_gained_dL_dout(col, working_register, tile_register);

        // Compute and accumulate scale components: a * gained_dL_dout
        copy_tile_init(cb_input_idx);
        copy_tile(cb_input_idx, col, tile_register);
        mul_binary_tile_init();
        mul_binary_tile(working_register, tile_register);

        // Mask the tile if needed
        if constexpr (do_mask_w) {
            if (col + 1 == Wt) {
                // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                const uint32_t mask_register = working_register + 1U;

                copy_tile_init(cb_mask_w_idx);
                copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                mask_tile_init();
                mask_tile(working_register, mask_register);
            }
        }

        // Add scale_components to scale
        if (col > 0) {
            add_binary_tile_init();
            add_binary_tile(scale_register, working_register);
        }
    }
    tile_regs_commit();

    pack_and_push(scale_register, cb_scale_idx);

    cb_wait_front(cb_scale_idx, onetile);
    const uint32_t reducted_scale_register = 0U;
    tile_regs_acquire();

    // Reduce scale along the inner dimension (C).

    // NOTE: Currently, there is a bug in reduce_tile that causes precision issues. To avoid this, we use a
    // workaround of matmul with appropriate scale. Once the bug is fixed, we can switch back to reduce_tile.
    reconfig_data_format(cb_scale_idx, cb_mat_mul_reduce);
    mm_init(cb_scale_idx, cb_mat_mul_reduce, cb_scale_idx, 0);
    matmul_tiles(
        cb_scale_idx,
        cb_mat_mul_reduce,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        reducted_scale_register,
        /* transpose */ 0);
    tile_regs_commit();

    // Pop the non-reduced scale tile from the CB.
    cb_pop_front(cb_scale_idx, onetile);
    pack_and_push(reducted_scale_register, cb_scale_idx);
}
#else
inline void compute_scale(const uint32_t row) {
    // Computes scale factor for RMSNorm backward pass: scale = sum(a * gained_dL_dout, dim=C)
    // where gained_dL_dout = (gamma / rms_a) * dL_dout
    // The result is reduced along the inner dimension and used in dL_da computation.

    const uint32_t scale_register = 0U;
    const uint32_t intermediate_register = 1U;
    const uint32_t tile_register = 2U;
    tile_regs_acquire();
    for (uint32_t col = 0; col < Wt;) {
        cb_wait_front(cb_gamma_idx, block_size);
        cb_wait_front(cb_dL_out_idx, block_size);
        cb_wait_front(cb_input_idx, block_size);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++col) {
            // If col == 0, we use scale_register as the working register to aviod unnecessary copying of data.
            auto working_register = col == 0 ? scale_register : intermediate_register;

            compute_gained_dL_dout(block_idx, working_register, tile_register);

            // Compute and accumulate scale components: a * gained_dL_dout
            copy_tile_init(cb_input_idx);
            copy_tile(cb_input_idx, block_idx, tile_register);
            mul_binary_tile_init();
            mul_binary_tile(working_register, tile_register);

            // Mask the tile if needed
            if constexpr (do_mask_w) {
                if (col + 1 == Wt) {
                    // Limitation: mask_tile only works when the mask register is immediately next to the data register.
                    const uint32_t mask_register =
                        working_register + 1U;  // mask register should be next to data register

                    copy_tile_init(cb_mask_w_idx);
                    copy_tile(cb_mask_w_idx, /* tile_idx */ 0, /* register idx */ mask_register);

                    mask_tile_init();
                    mask_tile(working_register, mask_register);
                }
            }

            // Add scale_components to scale
            if (col > 0) {
                add_binary_tile_init();
                add_binary_tile(scale_register, working_register);
            }
        }
        // Pop tiles that are not needed anymore.
        cb_pop_front(cb_gamma_idx, block_size);
        cb_pop_front(cb_dL_out_idx, block_size);
        cb_pop_front(cb_input_idx, block_size);
    }
    tile_regs_commit();

    pack_and_push(scale_register, cb_scale_idx);

    cb_wait_front(cb_scale_idx, onetile);
    const uint32_t reducted_scale_register = 0U;
    tile_regs_acquire();

    // Reduce scale along the inner dimension (C).

    // NOTE: Currently, there is a bug in reduce_tile that causes precision issues. To avoid this, we use a
    // workaround of matmul with appropriate scale. Once the bug is fixed, we can switch back to reduce_tile.
    reconfig_data_format(cb_scale_idx, cb_mat_mul_reduce);
    mm_init(cb_scale_idx, cb_mat_mul_reduce, cb_scale_idx, 0);
    matmul_tiles(
        cb_scale_idx,
        cb_mat_mul_reduce,
        /* tile_idx */ 0,
        /* tile_idx */ 0,
        reducted_scale_register,
        /* transpose */ 0);
    tile_regs_commit();

    // Pop the non-reduced scale tile from the CB.
    cb_pop_front(cb_scale_idx, onetile);
    pack_and_push(reducted_scale_register, cb_scale_idx);
}
#endif  // EVERYTHING_FITS_IN_L1

inline void bcast_scale() {
    // Broadcasts the reduced scale factor across columns.
    cb_reserve_back(cb_scale_bcasted_idx, onetile);
    const uint32_t reg_scale_bcasted = 0;
    tile_regs_acquire();
    reconfig_data_format(cb_scale_idx, cb_scale_bcasted_idx);

    unary_bcast_init<BroadcastType::COL>(cb_scale_idx, cb_scale_bcasted_idx);
    unary_bcast<BroadcastType::COL>(cb_scale_idx, /* tile idx */ 0, /* reg tile idx */ reg_scale_bcasted);
    tile_regs_commit();

    pack_and_push(reg_scale_bcasted, cb_scale_bcasted_idx);
}

inline void MAIN {
    if constexpr (do_mask_w) {
        cb_wait_front(cb_mask_w_idx, onetile);
    }
    cb_wait_front(cb_scaler_idx, onetile);
    cb_wait_front(cb_mat_mul_reduce, onetile);

    init_sfpu(cb_input_idx, cb_dL_da_idx);
    binary_op_init_common(cb_input_idx, cb_gamma_idx, cb_dL_da_idx);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_rms_a_idx, onetile);
        // This value is constant for the whole row, so we can compute it once per row.
        compute_recip_rms_a_bcasted();
        cb_wait_front(cb_recip_rms_a_bcasted_idx, onetile);
        // To compute scale we must iterate over all inner dimension.
        compute_scale(row);
        // Wait for the reducted cb_scale to be ready.
        cb_wait_front(cb_scale_idx, onetile);
        bcast_scale();
        cb_wait_front(cb_scale_bcasted_idx, onetile);

        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_reserve_back(cb_dL_da_idx, block_size);
            cb_reserve_back(cb_dL_dgamma_components, block_size);
#ifndef EVERYTHING_FITS_IN_L1
            cb_wait_front(cb_gamma_idx, block_size);
            cb_wait_front(cb_dL_out_idx, block_size);
            cb_wait_front(cb_input_idx, block_size);
#endif
            // Compute dL_da.
            {
                uint32_t dL_da_register;  // Depending on the block_idx, it will be 0 or 1.
                const uint32_t rhs_register = 2U;
                const uint32_t tile_register = 3U;
                tile_regs_acquire();
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t input_tile_idx = col + block_idx;
#else
                    const uint32_t input_tile_idx = block_idx;
#endif
                    dL_da_register = block_idx;
                    compute_dL_da(
                        input_tile_idx,
                        dL_da_register,
                        rhs_register,
                        tile_register);  // Compute dL_da for the current tile.
                }
                tile_regs_commit();

                pack_and_push_block(cb_dL_da_idx, block_size);
            }

            // Compute dL_dgamma_components.
            {
                uint32_t dL_dg_components_register;  // Depending on the block_idx, it will be 0 or 1.
                const uint32_t tile_register = 2U;
                tile_regs_acquire();
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t input_tile_idx = col + block_idx;
#else
                    const uint32_t input_tile_idx = block_idx;
#endif
                    dL_dg_components_register = block_idx;
                    compute_dL_dgamma_components(
                        input_tile_idx,
                        dL_dg_components_register,
                        tile_register);  // Compute dL_dg_components for the current tile.
                }
                tile_regs_commit();

                pack_and_push_block(cb_dL_dgamma_components, block_size);
            }
#ifndef EVERYTHING_FITS_IN_L1
            cb_pop_front(cb_gamma_idx, block_size);
            cb_pop_front(cb_dL_out_idx, block_size);
            cb_pop_front(cb_input_idx, block_size);
#endif
        }
#ifdef EVERYTHING_FITS_IN_L1
        cb_pop_front(cb_dL_out_idx, Wt);
        cb_pop_front(cb_input_idx, Wt);
#endif
        cb_pop_front(cb_rms_a_idx, onetile);
        cb_pop_front(cb_recip_rms_a_bcasted_idx, onetile);
        cb_pop_front(cb_scale_idx, onetile);
        cb_pop_front(cb_scale_bcasted_idx, onetile);
    }
#ifdef EVERYTHING_FITS_IN_L1
    cb_pop_front(cb_gamma_idx, Wt);
#endif
    cb_pop_front(cb_scaler_idx, onetile);
    cb_pop_front(cb_mat_mul_reduce, onetile);
    if constexpr (do_mask_w) {
        cb_pop_front(cb_mask_w_idx, onetile);
    }
}

}  // namespace NAMESPACE
