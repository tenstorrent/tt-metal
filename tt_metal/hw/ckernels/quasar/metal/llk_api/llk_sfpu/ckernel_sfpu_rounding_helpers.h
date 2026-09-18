// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel::sfpu {

// Clear fractional mantissa bits, preserving signed zero and already integral values.
sfpi_inline sfpi::vFloat _trunc_body_(sfpi::vFloat value) {
    sfpi::vInt exponent = sfpi::exexp(value);
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(value);
    v_if(exponent < 0) { bits = bits & 0x80000000u; }
    v_elseif(exponent < 23) {
        sfpi::vUInt mask = sfpi::shft(sfpi::vUInt(0xffffffffu), 23 - exponent, sfpi::ShiftMode::Logical);
        bits = bits & mask;
    }
    v_endif;
    return sfpi::as<sfpi::vFloat>(bits);
}

sfpi_inline sfpi::vFloat _floor_body_(sfpi::vFloat value) {
    sfpi::vFloat result = _trunc_body_(value);
    v_if(result > value) { result = result - 1.0f; }
    v_endif;
    return result;
}

}  // namespace ckernel::sfpu
