// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// TODO: Do it in everyfile! (company name has been changed)

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t mask_w = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);

// TODO: Consider moving these constants to compile-time arguments to avoid index-mismatch issues while developing.
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;  // 1/c - used for scaling
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_one_idx = tt::CBIndex::c_6;  // Used to reduce scale to a single value
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
    // 4. Compute:
    // auto scaled_outer = ttnn::multiply(
    //     scale,
    //     a,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [B,1,S,1] x [B,1,S,C] -> [B,1,S,C] (bcast)
    mul_tiles_init(cb_input_idx, cb_scale_bcasted_idx);
    mul_tiles(cb_input_idx, cb_scale_bcasted_idx, /* tile_idx */ input_tile_idx, /* tile_idx */ 0, rhs_register);

    // 5. Compute:
    // auto ms_a = ttnn::square(rms_a);  // [B,1,S,1] -> [B,1,S,1]
    // auto c_by_ms_a = ttnn::multiply(
    //     ms_a, c, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);  //
    //     [B,1,S,1] x [1]
    //     ->
    // [B,1,S,1] (bcast)
    // auto rhs = ttnn::divide(
    //     scaled_outer,
    //     c_by_ms_a,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [B,1,S,C] x [B,1,S,1] -> [B,1,S,C] (bcast)
    // NOTE: Both recip_rms_a and scaler are already broadcasted, so we can use them directly.
    // Scaler is 1/C, so we multiply not divide by it.
    copy_tile_init(cb_recip_rms_a_bcasted_idx);
    copy_tile(cb_recip_rms_a_bcasted_idx, /* tile_idx */ 0, tile_register);
    mul_binary_tile_init();
    mul_binary_tile(rhs_register, tile_register);
    mul_binary_tile(rhs_register, tile_register);
    copy_tile_init(cb_scaler_idx);
    copy_tile(cb_scaler_idx, /* tile_idx */ 0, tile_register);
    mul_binary_tile_init();
    mul_binary_tile(rhs_register, tile_register);

    // 6. Compute:
    // auto dL_da = ttnn::subtract(
    //     gained_dL_dout,
    //     rhs,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]
    compute_gained_dL_dout(input_tile_idx, dL_da_register, tile_register);
    sub_binary_tile_init();
    sub_binary_tile(dL_da_register, rhs_register);
}

inline void compute_dL_dgamma_components(
    const uint32_t input_tile_idx, const uint32_t dL_dg_components_register, const uint32_t tile_register) {
    // 7. Compute:
    // auto dL_dg_components = ttnn::multiply(
    //     dL_dout,
    //     ttnn::divide(a, rms_a, std::nullopt, std::nullopt, std::nullopt, none, none, none,
    //     false), std::nullopt, std::nullopt, std::nullopt, none, none, none, false);  // [B,1,S,C]
    //     x [B,1,S,1] -> [B,1,S,C] (bcast); checked by add_grad
    mul_tiles_init(cb_input_idx, cb_recip_rms_a_bcasted_idx);
    mul_tiles(
        cb_input_idx,
        cb_recip_rms_a_bcasted_idx,
        /* tile_idx */ input_tile_idx,
        /* tile_idx */ 0,
        dL_dg_components_register);
    copy_tile_init(cb_dL_out_idx);
    copy_tile(cb_dL_out_idx, /* tile_idx */ input_tile_idx, tile_register);
    mul_binary_tile_init();
    mul_binary_tile(dL_dg_components_register, tile_register);
    // auto dL_dg = ttnn::sum(
    //     dL_dg_components,
    //     /* dim_arg */ ttnn::SmallVector<int>{0, 1, 2},
    //     /* keep_dim */ true,
    //     /* output_mem_config */ std::nullopt,
    //     /*compute_kernel_config */ core::ComputeKernelConfig::precise());  // [B,1,S,C] ->
    //     [1,1,1,C]
    // NOTE: To compute dL_dg, we need to process all batches. Therefore, we will compute here only
    // dL_dg_components, and then store them in CB. The reduction will be done outside the kernel.
}

// ============================================================================
// Functions below compute final results for a pipeline stage and write them to
// circular buffers (CBs). These functions produce outputs that are consumed by
// later stages.
// ============================================================================

inline void compute_recip_rms_a_bcasted() {
    const uint32_t reg_rms_a = 0;
    tile_regs_acquire();
    unary_bcast_init<BroadcastType::COL>(cb_rms_a_idx, cb_recip_rms_a_bcasted_idx);
    unary_bcast<BroadcastType::COL>(cb_rms_a_idx, /* tile idx */ 0, /* reg tile idx */ reg_rms_a);
    recip_tile_init();
    recip_tile(reg_rms_a);
    tile_regs_commit();
    pack_and_push(reg_rms_a, cb_recip_rms_a_bcasted_idx);
}

// TODO: There was an idea to move compute_scale to separate files and include here only one based on the
// EVERYTHING_FITS_IN_L1 compile-time argument. However, I don't find it nice because both implementations reuse
// compute_gained_dL_dout, so either we would have a circular dependency or we would have to duplicate the code moving
// this function to those new files. So I vote to keep it here. Maybe there is an alternative of merging both
// implementations into one, but I don't see it right now, mostly due to cb_wait_front and cb_pop_front calls. Using a
// lot of #ifdefs is not nice, and I guess it can only make the code more complex and less readable.
#ifdef EVERYTHING_FITS_IN_L1
inline void compute_scale() {
    // Compute:
    // auto scaled_gain = ttnn::divide(
    //     g,
    //     rms_a,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [1,1,1,C] x [B,1,S,1] -> [B,1,S,C] (bcast)
    // auto gained_dL_dout = ttnn::multiply(
    //     scaled_gain,
    //     dL_dout,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]
    // auto scale = ttml::ttnn_fixed::sum_over_dim(
    //     ttnn::multiply(a, gained_dL_dout, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
    //     3);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C] -> [B,1,S,1]
    //
    const uint32_t scale_register = 0U;         // Register to use for scale computations.
    const uint32_t intermediate_register = 1U;  // Register to use for intermediate computations.
    const uint32_t tile_register = 2U;          // Register to use for tile.
    tile_regs_acquire();
    cb_wait_front(cb_gamma_idx, Wt);
    cb_wait_front(cb_dL_out_idx, Wt);
    cb_wait_front(cb_input_idx, Wt);
    for (uint32_t col = 0; col < Wt; col++) {
        // If col == 0, we use scale_register as the working register to aviod unnecessary copying of data.
        auto working_register = col == 0 ? scale_register : intermediate_register;

        compute_gained_dL_dout(col, working_register, tile_register);

        // Compute scale_components = a * gained_dL_dout
        // NOTE: Reduction requires packing scale_components to a CB. To minimize synchronization time, we can
        // reduce the components along the inner dimension (C) after all blocks are processed artificially
        // multiplying by 1.
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
    cb_wait_front(cb_one_idx, onetile);
    const uint32_t reducted_scale_register = 0U;  // Register to use for reduced scale computations.
    tile_regs_acquire();

    // Reduce scale along the inner dimension (C).
    reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scale_idx, cb_one_idx, cb_scale_idx);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scale_idx, cb_one_idx, 0, 0, reducted_scale_register);
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_scale_idx);
    tile_regs_commit();

    // Pop the non-reduced scale tile from the CB.
    cb_pop_front(cb_scale_idx, onetile);
    pack_and_push(reducted_scale_register, cb_scale_idx);
}
#else
inline void compute_scale() {
    // Compute:
    // auto scaled_gain = ttnn::divide(
    //     g,
    //     rms_a,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [1,1,1,C] x [B,1,S,1] -> [B,1,S,C] (bcast)
    // auto gained_dL_dout = ttnn::multiply(
    //     scaled_gain,
    //     dL_dout,
    //     std::nullopt,
    //     std::nullopt,
    //     std::nullopt,
    //     none,
    //     none,
    //     none,
    //     false);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]
    // auto scale = ttml::ttnn_fixed::sum_over_dim(
    //     ttnn::multiply(a, gained_dL_dout, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
    //     3);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C] -> [B,1,S,1]
    //
    const uint32_t scale_register = 0U;         // Register to use for scale computations.
    const uint32_t intermediate_register = 1U;  // Register to use for intermediate computations.
    const uint32_t tile_register = 2U;          // Register to use for tile.
    tile_regs_acquire();
    for (uint32_t col = 0; col < Wt;) {
        cb_wait_front(cb_gamma_idx, block_size);
        cb_wait_front(cb_dL_out_idx, block_size);
        cb_wait_front(cb_input_idx, block_size);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx, ++col) {
            // If col == 0, we use scale_register as the working register to aviod unnecessary copying of data.
            auto working_register = col == 0 ? scale_register : intermediate_register;

            compute_gained_dL_dout(col, working_register, tile_register);

            // Compute scale_components = a * gained_dL_dout
            // NOTE: Reduction requires packing scale_components to a CB. To minimize synchronization time, we can
            // reduce the components along the inner dimension (C) after all blocks are processed artificially
            // multiplying by 1.
            copy_tile_init(cb_input_idx);
            copy_tile(cb_input_idx, col, tile_register);
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
        // Pop the CBs that are not needed anymore.
        cb_pop_front(cb_gamma_idx, block_size);
        cb_pop_front(cb_dL_out_idx, block_size);
        cb_pop_front(cb_input_idx, block_size);
    }
    tile_regs_commit();

    pack_and_push(scale_register, cb_scale_idx);

    cb_wait_front(cb_scale_idx, onetile);
    const uint32_t reducted_scale_register = 0U;  // Register to use for reduced scale computations.
    tile_regs_acquire();

    // Reduce scale along the inner dimension (C).
    reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scale_idx, cb_one_idx, cb_scale_idx);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_scale_idx, cb_one_idx, 0, 0, reducted_scale_register);
    reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_scale_idx);
    tile_regs_commit();

    // Pop the non-reduced scale tile from the CB.
    cb_pop_front(cb_scale_idx, onetile);
    pack_and_push(reducted_scale_register, cb_scale_idx);
}
#endif  // EVERYTHING_FITS_IN_L1

// NOTE: Maybe this should be merged with compute_scale, because it is just a broadcast of the scale
// along the inner dimension (C). We can't skip packing step, becasue bcast operates on CBs but I guess it would be
// cleaner then.
inline void bcast_scale() {
    cb_reserve_back(cb_scale_bcasted_idx, onetile);
    const uint32_t reg_scale_bcasted = 0;
    tile_regs_acquire();

    // Broadcast scale along inner dimension (C).
    unary_bcast_init<BroadcastType::COL>(cb_scale_idx, cb_scale_bcasted_idx);
    unary_bcast<BroadcastType::COL>(cb_scale_idx, /* tile idx */ 0, /* reg tile idx */ reg_scale_bcasted);
    tile_regs_commit();

    pack_and_push(reg_scale_bcasted, cb_scale_bcasted_idx);
}

// TODO: Ask about bfloat16 vs float32 for regs, cbs and computations on LLK channel
// TASK: see how to prepare PR - have a look at Slava's past PRs
inline void MAIN {
    if constexpr (do_mask_w) {
        cb_wait_front(cb_mask_w_idx, onetile);
    }
    cb_wait_front(cb_scaler_idx, onetile);
    cb_wait_front(cb_one_idx, onetile);

    init_sfpu(cb_input_idx, cb_dL_da_idx);
    binary_op_init_common(cb_input_idx, cb_gamma_idx, cb_dL_da_idx);

    // TODO: Reduction should be done on float32, not bfloat16. Currently reduce_tile suffers from precision issues. One
    // option is to use matmul with onehot vector instead of reduce_tile, or wait for the fix of reduce_tile to be
    // merged. Then one need to remember to call reconfig_data_format and pack_reconfig_data_format in appropriate
    // places.
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_rms_a_idx, onetile);
        // This value is constant for the whole row, so we can compute it once per row.
        compute_recip_rms_a_bcasted();
        cb_wait_front(cb_recip_rms_a_bcasted_idx, onetile);
        // To compute scale we must iterate over all inner dimension.
        compute_scale();
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
                uint32_t dL_da_register;  // Register to use for dL_da computations. Depending on the block_idx, it will
                                          // be 0 or 1.
                const uint32_t rhs_register = 2U;   // Register to use for intermediate computations of rhs.
                const uint32_t tile_register = 3U;  // Register to use for tile.
                tile_regs_acquire();
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t input_tile_idx = col + block_idx;
#else
                    const uint32_t input_tile_idx = block_idx;
#endif
                    dL_da_register = block_idx;  // 0 or 1 depending on the block_idx.
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
                uint32_t dL_dg_components_register;  // Register to use for dL_dg_components computations. Depending on
                                                     // the block_idx, it will be 0 or 1.
                const uint32_t tile_register = 2U;   // Register to use for tile.
                tile_regs_acquire();
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
#ifdef EVERYTHING_FITS_IN_L1
                    const uint32_t input_tile_idx = col + block_idx;
#else
                    const uint32_t input_tile_idx = block_idx;
#endif
                    dL_dg_components_register = block_idx;  // 0 or 1 depending on the block_idx.
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
        cb_pop_front(cb_rms_a_idx, onetile);
        cb_pop_front(cb_recip_rms_a_bcasted_idx, onetile);
        cb_pop_front(cb_scale_idx, onetile);
        cb_pop_front(cb_scale_bcasted_idx, onetile);
    }

#ifdef EVERYTHING_FITS_IN_L1
    cb_pop_front(cb_gamma_idx, Wt);
    cb_pop_front(cb_dL_out_idx, Wt);
    cb_pop_front(cb_input_idx, Wt);
#endif
    cb_pop_front(cb_scaler_idx, onetile);
    cb_pop_front(cb_gamma_idx, onetile);
    cb_pop_front(cb_one_idx, onetile);
    if constexpr (do_mask_w) {
        cb_pop_front(cb_mask_w_idx, onetile);
    }
}

}  // namespace NAMESPACE
