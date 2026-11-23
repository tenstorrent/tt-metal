// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Quasar: the encoded (NaN class) actions read x as loaded from Dest, not its raw 16-bit Dest bits.
inline void finalize_encoded_raw_domain_actions(vFloat x_raw, vFloat& result) {
    apply_target_raw_pos_nan_policy(x_raw, result);
    apply_target_raw_neg_nan_policy(x_raw, result);
}
