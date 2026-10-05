// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_bf16_clamped_affine.h"

namespace ckernel::sfpu::bf16 {
template <typename Config, int Iterations = 8>
inline void calculate_clamped_affine() {
    {
        clamped_affine_pairs<Config, Iterations>();
    }
}
}  // namespace ckernel::sfpu::bf16
