// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "experimental/llk_math_compressed_custom_mm.h"
#include "llk_math_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MATH CUSTOM_MM
 *
 * Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
 * in0 tile shape: [{1, 2, 4, 8}, 32]
 * in1 tile shape: [32, 32]
 * rt_dim: 1
 * ct_dim: any integer from 1 to 64, as long as the output tiles fit in DEST (see CustomMmDestLayout)
 * kt_dim: even number from 2 to 256 (inclusive)
 * fidelity: LoFi only
 * throttle: not supported
 *
 * Uses llk_math_custom_mm.h as the low-level implementation.
 *************************************************************************/

template <bool transpose = false, ckernel::CustomMmDestLayout layout = ckernel::CustomMmDestLayout::Default>
inline void llk_math_compressed_custom_mm_init(const std::uint32_t operand0, const std::uint32_t operand1) {
    SAN_HOOK(unsupported());
    // The math thread needs nothing from the operands: unlike plain custom_mm, it has no face-row-dependent setup.
    _llk_math_compressed_custom_mm_init_<transpose, layout>();
}

// Shared by the CB-id execute below and the LLKOperand ones (api/compute/experimental/2_0/compressed_custom_mm.h).
template <ckernel::CustomMmDestLayout layout = ckernel::CustomMmDestLayout::Default>
inline void llk_math_compressed_custom_mm_impl(
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    LLK_ASSERT(
        (custom_mm_fits_dest<DST_SYNC_MODE, layout>(dst_index, ct_dim)),
        "compressed_custom_mm: dst_index + ct_dim output tiles overrun the DEST sync region");

    _llk_math_compressed_custom_mm_<layout>(base_address_meta, dst_index, kt_dim, ct_dim);
}

template <ckernel::CustomMmDestLayout layout = ckernel::CustomMmDestLayout::Default>
inline void llk_math_compressed_custom_mm(
    const std::uint32_t operand0,
    const std::uint32_t operand1,
    const std::uint32_t base_address_meta,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    llk_math_compressed_custom_mm_impl<layout>(base_address_meta, dst_index, kt_dim, ct_dim);
}
