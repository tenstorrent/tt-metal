// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct Log2Bf16Config {
    static constexpr uint32_t kDegree = 6u;
    // Fit: minimax degree-6 polynomial on [1, 2], 1 segment, max pure (continuous) ULP 0.499.
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u, 0x3fb8a9d6u, 0xbf386deeu, 0x3ef03878u, 0xbe9b298cu, 0x3e158d8cu, 0xbd0d1240u};
    static constexpr uint32_t kScaleBits = 0x3f800000u;
    static constexpr uint32_t kBodySlots = 29u;
    static constexpr uint32_t kTailSlots = 0u;
    static constexpr uint32_t kInputMinNormalBits = 0x00800000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_log2.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_log2_bf16() {
    ckernel::sfpu::bf16::calculate_log2<ckernel::sfpu::Log2Bf16Config, ITERATIONS>();
}
inline void init_log2_bf16() { ckernel::sfpu::bf16::init_log2<ckernel::sfpu::Log2Bf16Config>(); }

}  // namespace ckernel::sfpu
