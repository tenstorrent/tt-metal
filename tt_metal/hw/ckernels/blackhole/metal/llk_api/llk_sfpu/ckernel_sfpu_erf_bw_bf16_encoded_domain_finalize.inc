// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

inline void finalize_encoded_raw_domain_actions(vUInt raw_u16, vFloat& result) {
    apply_target_raw_pos_nan_policy(raw_u16, result);
    apply_target_raw_neg_nan_policy(raw_u16, result);
}

inline void finalize_encoded_raw_domain_actions(vFloat x_raw, vUInt raw_u16, vFloat& result) {
    (void)x_raw;
    finalize_encoded_raw_domain_actions(raw_u16, result);
}

inline void finalize_raw_domain_actions(vFloat x_raw, vUInt raw_u16, vFloat& result) {
    finalize_raw_domain_actions(x_raw, result);
    finalize_encoded_raw_domain_actions(x_raw, raw_u16, result);
}
