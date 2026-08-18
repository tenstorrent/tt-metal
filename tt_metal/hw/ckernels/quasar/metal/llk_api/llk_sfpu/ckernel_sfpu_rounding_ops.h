// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel::sfpu {

// For |value| < 2**23, adding and subtracting 2**23 rounds the magnitude to
// nearest-even. Correct a rounded-up value by one to obtain truncation. Values
// at or above 2**23 are already integral in fp32, so return them unchanged.
// Keeping the operation in the float domain avoids the range limit of Quasar's
// vSMag16 conversion.
sfpi_inline sfpi::vFloat _trunc_body_(sfpi::vFloat value) {
    sfpi::vFloat magnitude = sfpi::setsgn(value, 0);
    sfpi::vFloat result = value;
    v_if(magnitude < 0x1p23f) {
        sfpi::vFloat integer = magnitude + 0x1p23f;
        integer = integer - 0x1p23f;
        v_if(integer > magnitude) { integer = integer - 1.0f; }
        v_endif;
        result = sfpi::copysgn(integer, value);
    }
    v_endif;
    return result;
}

sfpi_inline sfpi::vFloat _floor_body_(sfpi::vFloat value) {
    sfpi::vFloat result = _trunc_body_(value);
    v_if(value < result) { result = result - 1.0f; }
    v_endif;
    return result;
}

sfpi_inline sfpi::vFloat _ceil_body_(sfpi::vFloat value) {
    sfpi::vFloat result = _trunc_body_(value);
    v_if(value > result) { result = result + 1.0f; }
    v_endif;
    return result;
}

}  // namespace ckernel::sfpu
