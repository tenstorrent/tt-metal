// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct AbsBf16Config {
    static constexpr uint32_t kEqualMask = 0x80ffu;
    static constexpr uint32_t kEqualValue = 0x80ffu;
    static constexpr uint32_t kNonzeroMask = 0x7f00u;
    static constexpr uint32_t kOutputBf16 = 0xff80u;
    static constexpr uint32_t kTruePatterns = 127u;
    static constexpr uint32_t kSelectedBodySlots = 25u;
    static constexpr uint32_t kSelectedPeakLregs = 8u;
    static constexpr uint32_t kLoadFormat = 0u;
    static constexpr uint32_t kAbsMode = 1u;
    static constexpr uint32_t kStoreFormat = 0u;
    static constexpr uint32_t kBodySlots = 6u;
    static constexpr uint32_t kPeakLregs = 2u;
    static constexpr uint32_t kProvedPatterns = 65536u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_abs_value.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_abs_bf16() {
    ckernel::sfpu::bf16::calculate_abs_value<ckernel::sfpu::AbsBf16Config, ITERATIONS>();
}
inline void init_abs_bf16() { ckernel::sfpu::bf16::init_abs_value<ckernel::sfpu::AbsBf16Config>(); }

}  // namespace ckernel::sfpu
