// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include <cstdint>
namespace ttpoly_generated {
struct HardswishBf16Config {
    static constexpr uint32_t kKind = 0x00000003u;
    static constexpr uint32_t kThresholdBits = 0x00000000u;
    static constexpr uint32_t kComparatorBf16 = 0x00000000u;
    static constexpr uint32_t kSlopeBits = 0x3e2aaaabu;
    static constexpr uint32_t kInterceptBits = 0x3f000000u;
    static constexpr uint32_t kRawEqual = 0x000080ffu;
    static constexpr uint32_t kBodySlots = 0x00000016u;
    static constexpr uint32_t kRowsPerReplay = 0x00000001u;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_simple_forward.h"
#ifndef TT_POLY_LLK_SIMPLE_FORWARD_SELECTED_V1
#error "selected simple forward runtime required"
#endif
#define TT_POLY_HARDSWISH_BF16_AVAILABLE 1
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_hardswish_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_simple_forward<ttpoly_generated::HardswishBf16Config, ITERATIONS>();
}
inline void init_hardswish_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::init_simple_forward<ttpoly_generated::HardswishBf16Config>();
}
#endif

}  // namespace ckernel::sfpu
