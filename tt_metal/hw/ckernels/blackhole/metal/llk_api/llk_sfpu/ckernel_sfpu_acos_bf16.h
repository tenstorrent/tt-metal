// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "sfpi.h"
namespace ckernel::sfpu {
struct AcosBf16Config {
    static constexpr uint32_t kDegree = 5u;
    static constexpr bool kSafeInput = false;
    // Fit: minimax degree-5 polynomial on [0, 1], 1 segment, max pure (continuous) ULP 1.02.
    static constexpr uint32_t kCoefficientBits[] = {
        0x3fc90fd2u, 0xbe5ba9b0u, 0x3db4017au, 0xbd386186u, 0x3c9f23c6u, 0xbb8f523eu};
    inline sfpi::vFloat operator[](uint32_t index) const {
        return sfpi::as<sfpi::vFloat>(sfpi::vUInt(kCoefficientBits[index]));
    }
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_sqrt_factored.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_acos_bf16() {
    ckernel::sfpu::bf16::calculate_sqrt_factored<ckernel::sfpu::AcosBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
