#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reduce.h"
#include "tt_metal/hw/inc/debug/dprint_tensix.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_binary.h"
#include "llk_math_reduce_api.h"
#include "ckernel_sfpu.h"
#endif

// clang-format off
/**
 * Initializes the fused eltwise binary reduce operation.
 *
 * This function initializes all necessary components (UNPACK, MATH, PACK) for the fused operation.
 * Must be called before fused_eltwise_binary_reduce().
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | cb_inp0        | Input circular buffer 0                                       | uint32_t | 0 to 31     | True     |
 * | cb_inp1        | Input circular buffer 1                                       | uint32_t | 0 to 31     | True     |
 * | acc_to_dest    | Whether to accumulate to destination                          | bool     | true/false  | False    |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type = ELTWISE_OP_TYPE,
    PoolType reduce_type = REDUCE_OP,
    ReduceDim reduce_dim = REDUCE_DIM,
    bool full_init = true>
ALWI void fused_eltwise_binary_reduce_init(uint32_t cb_inp0, uint32_t cb_inp1, bool acc_to_dest = false) {
    // UNPACK initialization - configure and init for both input CBs
    UNPACK((llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(cb_inp0, cb_inp1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_inp0, cb_inp1)));

    // MATH initialization - configure sync and hardware, then init eltwise binary
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure_disaggregated(cb_inp0, cb_inp1)));
    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, NONE, MATH_FIDELITY>(0 /*transpose*/, acc_to_dest)));

    // // Optional full initialization for UNPACK (conditional compilation), not sure about full init
    // if constexpr (full_init) {
    //     UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_inp0, cb_inp1, 0 /*transpose*/, acc_to_dest)));
    // }
    PACK((llk_pack_init()));
    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(16)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}

// clang-format off
/**
 * Performs a complete fused eltwise binary reduce operation on multiple tiles.
 *
 * This function performs the following operations in sequence:
 * 1. Eltwise binary operation on all tiles (storing results in destination register)
 * 2. Reuses destination data by moving it to source registers
 * 3. Populates the first tile with ones for the reduce operation
 * 4. Performs reduce operation across all tiles
 *
 * CRITICAL: When packing the result, you MUST use tile index 0 (pack_tile(0, cb_out)).
 * The reduce operation uses the entire destination register (indices 0-7) and overwrites
 * the first tile with the result, so the entire fused operation resolves around this.
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | cb_inp0        | Input circular buffer 0                                       | uint32_t | 0 to 31     | True     |
 * | cb_inp1        | Input circular buffer 1                                       | uint32_t | 0 to 31     | True     |
 * | itile0         | Input tile 0 index                                           | uint32_t | 0+          | True     |
 * | itile1         | Input tile 1 index                                           | uint32_t | 0+          | True     |
 * | tile_cnt       | Number of tiles to process                                    | uint32_t | 1 to 8      | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type = ELTWISE_OP_TYPE,
    PoolType reduce_type = REDUCE_OP,
    ReduceDim reduce_dim = REDUCE_DIM,
    bool fp32_transpose = false>
ALWI void fused_eltwise_binary_reduce(
    uint32_t cb_inp0, uint32_t cb_inp1, uint32_t itile0, uint32_t itile1, uint32_t tile_cnt) {
    // Step 1: Perform eltwise binary operations on all tiles
    for (uint32_t i = 0; i < tile_cnt; ++i) {
        UNPACK((llk_unpack_AB(cb_inp0, cb_inp1, i, i)));
        MATH((llk_math_eltwise_binary<
              eltwise_binary_type,
              NONE,
              DST_ACCUM_MODE,
              MATH_FIDELITY,
              EltwiseBinaryReuseDestType::NONE>(i)));
    }

    // Step 2: Reset counters before reduce operation
    MATH((_fused_eltwise_binary_uninit_()));

    // Step 3: Switch banks - UNPACK thread switches srcA/srcB banks for reduce
    UNPACK((llk_unpack_AB_fused()));

    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>()));

    // Step 3a: Prepare data for reduce operation
    MATH(eltwise_binary_reuse_dest_as_src_tile<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0));  // Move tile 0 to srcA

    MATH(ckernel::sfpu::_populate_first_tile_with_ones_());                               // Fill tile 0 with ones
    MATH(eltwise_binary_reuse_dest_as_src_tile<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0));  // Move tile 0 to srcB


    // Could be replaced with zeroacc for DEST, this was replaced with SFPUs calculate_fill due to debugging issues
    MATH(ckernel::sfpu::_populate_first_tile_with_zeroes_());


    // Step 4: Initialize reduce operation
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, reduce_dim>()));

    // Step 5: Perform reduce operation (result stored in tile 0)
    // Note: This compute processes up to 8 tiles in a loop
    uint32_t reduce_dst_idx = 0;

    for (uint32_t i = 0; i < tile_cnt; ++i) {
        if (i != 0) {
            MATH(eltwise_binary_reuse_dest_as_src_tile<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i));
        }
        MATH((llk_math_reduce_column<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY, false, fp32_transpose>(
            reduce_dst_idx)));
    }

    // Step 6: Clear data valid flags after reduce loop
    MATH((llk_math_reduce_clear_dvalid_after_for_loop()));
}

// clang-format off
/**
 * Uninitializes the fused eltwise binary reduce operation.
 *
 * This function cleans up the reduce operation and should be called after
 * all fused operations are complete.
 */
// clang-format on
ALWI void fused_eltwise_binary_reduce_uninit() { PACK((llk_pack_reduce_mask_clear())); }
