// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Quasar: x_raw is x as loaded from Dest (a float), not its raw 16-bit Dest bits; see
// ckernel_sfpu_erf_bw_bf16_mirrored_terminals.h.
inline void apply_target_raw_pos_nan_policy(vFloat x_raw, vFloat& result) {
    positive_nan_class_terminal<TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT>(
        x_raw, result, TT_TARGET_BH_BF16_RAW_POS_NAN_CONSTANT);
}

inline void apply_target_raw_neg_nan_policy(vFloat x_raw, vFloat& result) {
    if constexpr (TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT <= 4) {
        negative_nan_class_terminal<TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT, true>(x_raw, result);
    } else {
        v_if(is_neg(x_raw) && exexp(x_raw, ExponentMode::Biased) >= 255) {
            v_if(exman(x_raw) != 0) {
                result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT>(
                    result, TT_TARGET_BH_BF16_RAW_NEG_NAN_CONSTANT);
            }
            v_endif;
        }
        v_endif;
    }
}
