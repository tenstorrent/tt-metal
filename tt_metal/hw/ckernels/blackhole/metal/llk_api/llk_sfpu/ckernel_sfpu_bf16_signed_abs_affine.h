// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
#include "sfpu/ckernel_sfpu_bf16_signed_abs_affine.h"

// Included after polynomial_horner.h and the target's existing LLK headers.
// Config carries the selected coefficient/terminal words. Callers own traversal.
namespace sfpi {
template <typename Config, uint32_t J>
inline void signed_abs_affine_pin() {
    if constexpr (signed_abs_affine_addend_reg<Config, J>() == 8u - J) {
        ::ckernel::sfpu::bf16_sfpi::sfploadi(8u - J, SFPLOADI_MOD0_UPPER, Config::kCoefficientBits[6u - J] >> 16);
        ::ckernel::sfpu::bf16_sfpi::sfploadi(8u - J, SFPLOADI_MOD0_LOWER, Config::kCoefficientBits[6u - J] & 0xffffu);
    }
}

template <typename Config>
inline void signed_abs_affine_pin_coefficients() {
    signed_abs_affine_pin<Config, 1>();
    signed_abs_affine_pin<Config, 2>();
    signed_abs_affine_pin<Config, 3>();
}

inline void signed_abs_affine_drain() {
    // Preserve the canonical five issued slots before counter/commit changes.
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
    ::ckernel::sfpu::bf16_sfpi::sfpnop();
}
}  // namespace sfpi
