// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

inline void apply_target_raw_pos_nan_policy(vUInt raw_u16, vFloat& result) {
    positive_nan_class_terminal<TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT>(
        raw_u16, result, TT_TARGET_BH_BF16_RAW_POS_NAN_CONSTANT);
}

inline void apply_target_raw_neg_nan_policy(vUInt raw_u16, vFloat& result) {
    if constexpr (TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT <= 4) {
        negative_nan_class_terminal<TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT, true>(raw_u16, result);
    } else {
        vUInt exponent_and_sign_delta = (raw_u16 ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
        v_if(exponent_and_sign_delta == 0u) {
            vUInt mantissa = raw_u16 & vUInt(0x7f00u);
            v_if(mantissa != 0u) {
                result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT>(
                    result, TT_TARGET_BH_BF16_RAW_NEG_NAN_CONSTANT);
            }
            v_endif;
        }
        v_endif;
    }
}
