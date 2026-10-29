// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "experimental/llk_unpack_block_max8_fpu.h"
#include "llk_unpack_common_api.h"
#include "sanitizer/api.h"

namespace ckernel {

/**
 * @brief Publish the SrcA (negative infinity) and SrcB (1.0) sources for one FPU block-max call.
 * @note Call once per @ref llk_math_block_max8_fpu, after any unpack the math thread consumes first.
 */
inline void llk_unpack_block_max8_fpu() {
    SAN_HOOK(unsupported());
    _llk_unpack_block_max8_fpu_();
}

}  // namespace ckernel
