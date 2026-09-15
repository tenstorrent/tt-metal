// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Private BF16 experiment. Include after the frozen compute_common.hpp, then
// rename exp_packthread_tile_init only while including compute_streaming.hpp.
// No extra per-vector operations: change the native integer exp grid from
// 256 steps/exponent to 64. Every positive normal result is exactly E8M6,
// so BF16 denominator reduction and LoFi PV consume the same P bits.
// This does not promise exact recurrence or a correctly rounded exponential.
#if defined(SDPA_LOFI_EXP_GRID7) && defined(DISABLE_SFPLOADMACRO)
#error "Grid7 requires native unclamped approximate-exp LOADMACRO"
#endif

#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
template <uint32_t scale>
inline void init_lofi_grid7_constants() {
    constexpr float a = (256.0f * 1.4426950408889634f) * __builtin_bit_cast(float, scale) * 0.25f;
    constexpr float b = 32500.818359375f * 0.25f;
    TTI_SFPLOADI(0, 0xA, lo16(a));
    TTI_SFPLOADI(0, 0x8, hi16(a));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, lo16(b));
    TTI_SFPLOADI(0, 0x8, hi16(b));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 0xA, 17);
    TTI_SFPLOADI(0, 0x8, 0);
    TTI_SFPCONFIG(0, 14, 0);
}
}  // namespace ckernel::sfpu
#endif

namespace ckernel {
template <
    bool approx = false,
    uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void lofi_grid7_exp_packthread_tile_init() {
    exp_packthread_tile_init<approx, scale, input_clamping, is_fp32_dest_acc_en>();
#ifdef SDPA_LOFI_EXP_GRID7
    static_assert(!is_fp32_dest_acc_en, "Grid7 experiment is BF16 DST only");
    if constexpr (approx && input_clamping == InputClamping::None) {
        // The native init has just restored the full macro/replay program.
        // Only its three constants change; accurate rescale exponentials and
        // clamped normalization initialization retain their original settings.
        PACK((sfpu::init_lofi_grid7_constants<scale>()));
    }
#endif
}
}  // namespace ckernel
