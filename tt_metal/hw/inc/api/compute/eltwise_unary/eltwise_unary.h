// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "api/compute/tile_move_copy.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

// =====================================================================================================================
// Deprecated API
//
// The eltwise-unary short init is identical to the copy short init, so there is no separate unary
// init - copy_init serves both. The functions below are the old all-in-one inits, kept as thin
// forwarders to the current programming model:
//   compute_kernel_hw_startup(icb, ocb);  // once, as the first Compute API call in MAIN
//   copy_init(icb);                       // before the datacopy that feeds the SFPU op
// =====================================================================================================================

// clang-format off
/**
 * Legacy combined hardware + pipeline init for eltwise-unary / SFPU ops. Forwards to
 * compute_kernel_hw_startup(icb, ocb) followed by copy_init(icb), which together reproduce the old
 * all-in-one behavior. Like compute_kernel_hw_startup it performs one-time MMIO configuration and so
 * must still be the first Compute API call in the kernel; it is not safe mid-kernel or in a loop. To
 * re-init for a new operand mid-kernel, call a reconfig_data_format function then copy_init(), not this.
 *
 * Return value: None
 *
 * | Argument | Description                                       | Type     | Valid Range | Required |
 * |----------|---------------------------------------------------|----------|-------------|----------|
 * | icb      | The identifier of the input circular buffer (CB)  | uint32_t | 0 to 31     | True     |
 * | ocb      | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
// clang-format on
[[deprecated(
    "Use compute_kernel_hw_startup(icb, ocb) once at kernel start, then copy_init(icb). This will be removed after "
    "15-09-2026.")]] ALWI void
unary_op_init_common(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    compute_kernel_hw_startup(icb, ocb);
    copy_init(icb, 0, 0, call_line);
}

// clang-format off
/**
 * Legacy SFPU init. Superseded by the same model as unary_op_init_common:
 * compute_kernel_hw_startup(icb, ocb) once at kernel start, then copy_init(icb) before the op.
 *
 * Return value: None
 *
 * | Argument | Description                                       | Type     | Valid Range | Required |
 * |----------|---------------------------------------------------|----------|-------------|----------|
 * | icb      | The identifier of the input circular buffer (CB)  | uint32_t | 0 to 31     | True     |
 * | ocb      | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
// clang-format on
[[deprecated(
    "Use compute_kernel_hw_startup(icb, ocb) once at kernel start, then copy_init(icb). This will be removed after "
    "15-09-2026.")]] ALWI void
init_sfpu(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    compute_kernel_hw_startup(icb, ocb);
    copy_init(icb, 0, 0, call_line);
}

}  // namespace ckernel
