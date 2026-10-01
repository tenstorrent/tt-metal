// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct HardshrinkBf16Config {
    static constexpr uint32_t kKind = 0x00000001u;
    static constexpr uint32_t kThresholdBits = 0x3f000000u;
    static constexpr uint32_t kComparatorBf16 = 0x00003f00u;
    static constexpr uint32_t kBodySlots = 0x0000000du;
    static constexpr uint32_t kRowsPerReplay = 0x00000002u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_simple_forward.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_hardshrink_bf16() {
    ckernel::sfpu::bf16::calculate_simple_forward<ckernel::sfpu::HardshrinkBf16Config, ITERATIONS>();
}
inline void init_hardshrink_bf16() { ckernel::sfpu::bf16::init_simple_forward<ckernel::sfpu::HardshrinkBf16Config>(); }

}  // namespace ckernel::sfpu
