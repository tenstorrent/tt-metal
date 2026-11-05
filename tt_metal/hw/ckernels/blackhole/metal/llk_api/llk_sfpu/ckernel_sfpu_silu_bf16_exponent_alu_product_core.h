// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_silu_bf16_replay_window.h"
// Selected product bodies. Caller owns initialization and SFPU start/done.
namespace ckernel::sfpu::ttpoly::silu {
using namespace ::sfpi;
using ::sfpi::DataLayout;
inline void product_symmetric_load_macro_init() {
#if !defined(DISABLE_SFPLOADMACRO)
    // The LOADMACRO's RNE rounds L0 and its store writes the row its load addressed, three slots later.
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN, 0, 0, ckernel::p_sfpu::LREG0, 13, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    constexpr uint32_t kSymmetricProductLmSequence = 0x13850000u;
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (kSymmetricProductLmSequence >> 16));
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, (kSymmetricProductLmSequence & 0xFFFFu));
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
    TTI_SFPNOP;
#endif
}
}  // namespace ckernel::sfpu::ttpoly::silu
