// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
#include "sfpi.h"
namespace ttpoly_generated {
struct AcosBf16Config {
    static constexpr uint32_t kDegree = 5u;
    static constexpr bool kSafeInput = true;
    static constexpr uint32_t kCoefficientBits[] = {
        0x3fc90fd2u, 0xbe5ba9b0u, 0x3db4017au, 0xbd386186u, 0x3c9f23c6u, 0xbb8f523eu};
    inline sfpi::vFloat operator[](uint32_t index) const {
        return sfpi::as<sfpi::vFloat>(sfpi::vUInt(kCoefficientBits[index]));
    }
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_sqrt_factored.h"
#ifndef TT_POLY_LLK_SQRT_FACTORED_MIRRORED_V1
#error "typed reflected-root runtime required"
#endif
