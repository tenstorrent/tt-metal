// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

inline void apply_target_raw_pos_nan_policy(vUInt raw_u16, vFloat& result) {
    positive_nan_class_terminal<TT_TARGET_BF16_RAW_POS_NAN_RESULT>(
        raw_u16, result, TT_TARGET_BF16_RAW_POS_NAN_CONSTANT);
}

inline void apply_target_raw_neg_nan_policy(vUInt raw_u16, vFloat& result) {
    dense_negative_nan_constant<__builtin_bit_cast(uint32_t, TT_TARGET_BF16_RAW_NEG_NAN_CONSTANT)>(raw_u16, result);
}
