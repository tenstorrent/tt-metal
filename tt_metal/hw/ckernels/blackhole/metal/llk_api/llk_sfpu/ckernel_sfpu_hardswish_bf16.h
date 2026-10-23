// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct HardswishBf16Config {
    static constexpr uint32_t kKind = 0x00000003u;
    static constexpr uint32_t kSlopeBits = 0x3e2aaaabu;
    static constexpr uint32_t kInterceptBits = 0x3f000000u;
    static constexpr uint32_t kRawEqual = 0x000080ffu;
    static constexpr uint32_t kBodySlots = 0x00000019u;
    static constexpr uint32_t kRowsPerReplay = 0x00000001u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_hardswish_bf16_simple_forward.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 32>
inline void calculate_hardswish_bf16() {
    ckernel::sfpu::bf16::hardswish::calculate_simple_forward<ckernel::sfpu::HardswishBf16Config, ITERATIONS>();
}
inline void init_hardswish_bf16() {
    ckernel::sfpu::bf16::hardswish::init_simple_forward<ckernel::sfpu::HardswishBf16Config>();
}

}  // namespace ckernel::sfpu
