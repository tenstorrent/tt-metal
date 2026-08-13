// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_polyval.h"
#include "sfpi.h"

namespace ckernel::sfpu {

constexpr float CW_INV_LN2 = 1.4426950408889634f;
constexpr float CW_NEG_LN2_HI = -0.6931152343750000f;
constexpr float CW_NEG_LN2_LO = -3.19461832987e-05f;

sfpi_inline sfpi::vFloat expm1_cw_clamped(sfpi::vFloat x) {
    // Blackhole uses sfpi::max here. Quasar's vector min/max orders negative lanes by their sign-magnitude encoding,
    // so use a signed comparison to preserve Blackhole's lower clamp.
    v_if(x < -87.0f) { x = -87.0f; }
    v_endif;

    // Cody-Waite range reduction: x = k*ln(2) + r.
    const sfpi::vFloat c231 = Converter::as_float(0x4B400000U);
    sfpi::vFloat tmp = x * CW_INV_LN2 + c231;
    sfpi::vFloat k_f = tmp - c231;
    sfpi::vFloat r = k_f * CW_NEG_LN2_HI + x;
    r = r + k_f * CW_NEG_LN2_LO;

    // expm1(r) = r * h(r), matching Blackhole's Horner polynomial.
#ifdef INP_FLOAT32
    sfpi::vFloat h = PolynomialEvaluator::eval(
        r, 1.0f, 5.0000000000e-01f, 1.6666504741e-01f, 4.1666239500e-02f, 8.3691505715e-03f, 1.3948583510e-03f);
#else
    sfpi::vFloat h =
        PolynomialEvaluator::eval(r, 1.0f, 4.9999371171e-01f, 1.6666433215e-01f, 4.1875664145e-02f, 8.3751315251e-03f);
#endif
    h = r * h;

    // Blackhole reconstructs 2^k with setexp from its sign-magnitude integer path. Quasar's vInt ALU is
    // two's-complement, so construct the same IEEE exponent explicitly after the shared round-to-nearest operation.
    sfpi::vInt k_int = sfpi::as<sfpi::vInt>(tmp) - sfpi::as<sfpi::vInt>(c231);
    sfpi::vFloat two_k = sfpi::as<sfpi::vFloat>((k_int + 127) << 23);
    return (two_k - 1.0f) + two_k * h;
}

}  // namespace ckernel::sfpu
