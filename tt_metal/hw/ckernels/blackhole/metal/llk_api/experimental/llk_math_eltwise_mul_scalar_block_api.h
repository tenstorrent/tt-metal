// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_math_eltwise_mul_scalar_block.h"
#include "llk_math_common_api.h"

#include "sanitizer/api.h"

namespace ckernel {

inline void llk_math_eltwise_mul_scalar_block_init() {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_mul_scalar_block_init_();
}

inline void llk_math_eltwise_mul_scalar_block(const std::uint32_t dst_index, const std::uint32_t block_size) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_mul_scalar_block_(dst_index, block_size);
}

}  // namespace ckernel
