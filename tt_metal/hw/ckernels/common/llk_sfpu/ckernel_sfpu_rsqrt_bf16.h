// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct RsqrtBf16Config {
    static constexpr uint32_t kKind = 1u;
    static constexpr uint32_t kMagic = 0x5f1110a0u;
    static constexpr uint32_t kC0Bits = 0x00000000u;
    static constexpr uint32_t kC1Bits = 0x401214c9u;
    static constexpr uint32_t kC2Bits = 0x40103626u;
    static constexpr uint32_t kBodySlots = 30u;
    static constexpr uint32_t kNegativeExponentZeroOutput = 0x7f80u;
};
}  // namespace ttpoly_generated
#include "ckernel_sfpu_tt_poly_newton_root.h"
#ifndef TT_POLY_LLK_NEWTON_ROOT_SELECTED_V1
#error "typed selected Newton runtime required"
#endif
