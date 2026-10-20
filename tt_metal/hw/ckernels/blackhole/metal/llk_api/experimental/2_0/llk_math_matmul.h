// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_matmul_api.h"  // legacy CB-id API + unified llk_math_matmul_init_impl; execute reused as-is
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK MATMUL MATH -- LLKOperand (id-free, compile-time NTTP) overload (init only)
 *
 * Only the init needs an id-free overload: it passes the two descriptor shapes (IN0 -> SrcB, IN1 -> SrcA)
 * to the LLK, which derives the tile dims and partial_face from them. Matmul math is FORMAT-FREE. The
 * matmul math EXECUTE (llk_math_matmul<fidelity, throttle>(idst, ct, rt)) takes no operand id, so the
 * id-free path reuses the legacy execute directly -- no overload here.
 *************************************************************************/

template <
    ckernel::experimental::LLKMemDescriptor IN0_DESC,
    ckernel::experimental::LLKMemDescriptor IN1_DESC,
    MathFidelity math_fidelity,
    int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t transpose = 0, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    SAN_HOOK(unsupported());
    llk_math_matmul_init_impl<math_fidelity, THROTTLE_LEVEL>(IN0_DESC.shape, IN1_DESC.shape, transpose, ct_dim, rt_dim);
}
