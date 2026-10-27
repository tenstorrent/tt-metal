// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "experimental/llk_math_custom_mm_api.h"  // legacy CB-id API + shared llk_math_custom_mm_impl
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MATH CUSTOM_MM -- LLKOperand (id-free, compile-time NTTP) overload (execute only)
 *
 * The execute needs in0's face row count, which the CB-id overload looks up from the operand id. This one
 * takes it from the in0 (SrcB) descriptor instead. The init takes it as a plain argument already, so the
 * id-free path calls _llk_math_custom_mm_init_ directly.
 *************************************************************************/

template <
    ckernel::CustomMmDestLayout layout = ckernel::CustomMmDestLayout::Default,
    bool finalize = ckernel::custom_mm_split_acc(layout),
    ckernel::experimental::LLKMemDescriptor IN0_DESC>
inline void llk_math_custom_mm(
    const std::uint32_t dst_index, const std::uint32_t kt_dim, const std::uint32_t ct_dim = 1) {
    SAN_HOOK(unsupported());
    llk_math_custom_mm_impl<layout, finalize>(IN0_DESC.shape.face_r_dim, dst_index, kt_dim, ct_dim);
}
