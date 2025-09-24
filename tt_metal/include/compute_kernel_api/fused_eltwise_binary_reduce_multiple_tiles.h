#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reduce.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_binary.h"
#include "llk_math_reduce_api.h"
#include "ckernel_sfpu.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Simplified interface that handles the complete fused operation
 * Following the exact algorithm step by step as specified
 *
 * | Argument       | Description                                                   | Type     | Valid Range | Required |
 * |----------------|---------------------------------------------------------------|----------|-------------|----------|
 * | cb_inp0        | Input circular buffer 0                                       | uint32_t | 0 to 31     | True     |
 * | cb_inp1        | Input circular buffer 1                                       | uint32_t | 0 to 31     | True     |
 * | cb_scaler      | Scaler circular buffer                                        | uint32_t | 0 to 31     | True     |
 * | cb_out         | Output circular buffer                                        | uint32_t | 0 to 31     | True     |
 * | itile0         | Input tile 0 index                                           | uint32_t | 0+          | True     |
 * | itile1         | Input tile 1 index                                           | uint32_t | 0+          | True     |
 * | iscaler        | Scaler tile index                                             | uint32_t | 0+          | True     |
 * | idst           | Destination tile index                                        | uint32_t | 0+          | True     |
 */
// clang-format on
// =============================================================================
// PHASE 1: ELTWISE BINARY INITIALIZATION (NO PACKER)
// =============================================================================
template <EltwiseBinaryType eltwise_binary_type = ELTWISE_OP_TYPE, bool full_init = true>
ALWI void fused_eltwise_binary_init(uint32_t cb_inp0, uint32_t cb_inp1, bool acc_to_dest = false) {
    // UNPACK initialization - configure and init for both input CBs
    UNPACK((llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(cb_inp0, cb_inp1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_inp0, cb_inp1)));

    // MATH initialization - configure sync and hardware, then init eltwise binary
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure_disaggregated(cb_inp0, cb_inp1)));
    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, NONE, MATH_FIDELITY>(0 /*transpose*/, acc_to_dest)));

    // // Optional full initialization for UNPACK (conditional compilation)
    // if constexpr (full_init) {
    //     UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_inp0, cb_inp1, 0 /*transpose*/, acc_to_dest)));
    // }
    PACK((llk_pack_init()));
    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(16)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}

// =============================================================================
// PHASE 2: ELTWISE BINARY OPERATION
// =============================================================================
template <EltwiseBinaryType eltwise_binary_type = ELTWISE_OP_TYPE, uint32_t idst>
ALWI void fused_eltwise_binary_compute(uint32_t cb_inp0, uint32_t cb_inp1, uint32_t itile0, uint32_t itile1) {
    // Use the low-level LLK calls directly since we're in a fused operation
    UNPACK((llk_unpack_AB(cb_inp0, cb_inp1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          eltwise_binary_type,
          NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(idst)));
}

// Runtime version for multiple tiles (when idst is not known at compile time)
template <EltwiseBinaryType eltwise_binary_type = ELTWISE_OP_TYPE>
ALWI void fused_eltwise_binary_compute(
    uint32_t cb_inp0, uint32_t cb_inp1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    // Use the low-level LLK calls directly since we're in a fused operation
    UNPACK((llk_unpack_AB(cb_inp0, cb_inp1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          eltwise_binary_type,
          NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(idst)));
}

// =============================================================================
// PHASE 3: DESTINATION REUSE
// =============================================================================
ALWI void fused_eltwise_binary_reuse_dest() {
    // 4. eltwise_binary_reuse_dest_as_src, moving the result to srcA
    MATH(eltwise_binary_reuse_dest_as_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>());

    // 5. populate dest with ones
    MATH(ckernel::sfpu::_populate_first_tile_with_ones_());

    // 6. eltwise_binary_reuse_dest_as_src, moving the result to srcB
    MATH(eltwise_binary_reuse_dest_as_src<EltwiseBinaryReuseDestType::DEST_TO_SRCB>());
}

// =============================================================================
// PHASE 3B: DESTINATION REUSE FOR MULTIPLE TILES
// =============================================================================
// should be called only once
ALWI void fused_reduce_populate_ones() {
    MATH(ckernel::sfpu::_populate_first_tile_with_ones_());
    // Use explicit tile index 0 to ensure we move tile 0 to srcB
    MATH(eltwise_binary_reuse_dest_as_src<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0));
}

ALWI void fused_eltwise_binary_reuse_dest_multiple_tiles(uint32_t idst) {
    MATH(eltwise_binary_reuse_dest_as_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(idst));
}

// =============================================================================
// PHASE 4: REDUCE INITIALIZATION (NO UNPACKER)
// =============================================================================
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
ALWI void fused_reduce_init() {
    // MATH initialization - init reduce operation (no unpacker calls)
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>()));

    // PACK initialization - configure the reduce mask to ensure only the final
    // reduced result is packed and intermediate/partial results are masked out
    PACK((llk_pack_reduce_mask_config<false /*untilize*/, reduce_dim>()));
}

// =============================================================================
// PHASE 5: REDUCE OPERATION
// =============================================================================
template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool fp32_transpose = false>
ALWI void fused_reduce_compute(
    uint32_t idst, uint32_t operandA, uint32_t operandB, uint32_t tile_index_a, uint32_t tile_index_b) {
    // 8. reduce operation (using srcA from dest reuse, cb_scaler used as dummy first param)
    // **FIXED: Use llk_math_reduce_fused which doesn't clear data valid flags**
    MATH((llk_math_reduce_fused<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY, false, fp32_transpose>(idst)));
    UNPACK((llk_unpack_AB_but_fused_so_no_mop(operandA, operandB, tile_index_a, tile_index_b)));
}

// =============================================================================
// PHASE 5B: CLEAR DVALID AFTER FOR LOOP
// =============================================================================
ALWI void fused_reduce_clear_dvalid_after_for_loop() { MATH((llk_math_reduce_clear_dvalid_after_for_loop())); }

// =============================================================================
// PHASE 6A: ELTWISE BINARY CLEANUP
// =============================================================================
ALWI void fused_eltwise_binary_uninit() { MATH((_fused_eltwise_binary_uninit_())); }

// =============================================================================
// PHASE 6B: REDUCE CLEANUP
// =============================================================================
ALWI void fused_reduce_uninit() {
    // 9. reduce_uninit
    reduce_uninit();
}
}  // namespace ckernel
