// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu::ttpoly::erfc {
using ::sfpi::DataLayout;
using ::sfpi::vFloat;
template <typename Reciprocal>
__attribute__((always_inline)) inline vFloat correction_finite_reciprocal(
    vFloat x, [[maybe_unused]] Reciprocal reciprocal) {
    vFloat y = sfpi::approx_recip(x);
    vFloat residual;
    residual = x * y - 2.0f;
    return y * -residual - 0.0f;
}
}  // namespace ckernel::sfpu::ttpoly::erfc
