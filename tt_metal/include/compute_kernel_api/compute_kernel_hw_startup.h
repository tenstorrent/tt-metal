// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the required hardware initialization for all subsequent operations in the compute kernel. This function should be
 * called exactly once at the very beginning of the kernel, before any operation-specific initialization functions (such as
 * reduce_init, tilize_init, etc.). The circular buffer (CB) IDs provided to this function must match those used in the next
 * operation-specific initialization function. If the operands for the next operation require a different data format than
 * what was configured here, you must call one of the reconfig_data_format functions before proceeding with the next
 * initialization. Similarly, if the next operation requires different properties (such as tile or face dimensions), you must
 * ensure that the same CB IDs are used as in this function.
 *
 * NOTE: This function performs MMIO writes, which are slow and almost exclusively require the idle state of the execution
 * units that should be configured (PACK, MATH, UNPACK, CFG, etc.). This is why it is unsafe to call this function in the
 * middle of a kernel execution. This function should be called only once at the beginning of the kernel, before any other
 * calls to Compute API are made (either init or other). Calling this function after other API calls may lead cause race
 * conditions and undefined behavior which can be hard to debug.
 *
 * Return value: None
 *
 * | Param Type | Name  | Description                                                     | Type     | Valid Range | Required |
 * |------------|-------|-----------------------------------------------------------------|----------|-------------|----------|
 * | Function   | icb0  | The identifier of the circular buffer (CB) containing operand A | uint32_t | 0 to 31     | True     |
 * | Function   | icb1  | The identifier of the circular buffer (CB) containing operand B | uint32_t | 0 to 31     | True     |
 * | Function   | ocb   | The identifier of the output circular buffer (CB)               | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    PACK((llk_pack_init<false /*untilize*/, false /*zero_output*/, false /*tilize*/>(ocb)));
    PACK((llk_pack_hw_configure_disaggregated<
          DST_ACCUM_MODE,
          false /*untilize*/,
          ReluType::NO_RELU,
          0 /*relu_treshold*/,
          false /*tilize*/>(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false /*untilize*/>(ocb)));
}

// clang-format off
/**
 * Convenience overload for hardware initialization when only one input circular buffer is used.
 * Both input operands (srcA and srcB) will be programmed using the same circular buffer identifier (`icb0`).
 * Internally, this calls the three-parameter version with `icb0` passed for both input operands.
 *
 * | Param Type | Name  | Description                                                        | Type     | Valid Range | Required |
 * |------------|-------|--------------------------------------------------------------------|----------|-------------|----------|
 * | Function   | icb0  | The identifier of the circular buffer (CB) used for both input ops | uint32_t | 0 to 31     | True     |
 * | Function   | ocb   | The identifier of the output circular buffer (CB)                  | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void compute_kernel_hw_startup(uint32_t icb0, uint32_t ocb) { compute_kernel_hw_startup(icb0, icb0, ocb); }

}  // namespace ckernel
