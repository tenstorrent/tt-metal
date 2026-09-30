// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
inline vFloat target_domain_action_coordinate(vFloat x_raw) {
#if defined(TT_TARGET_BH_BF16_ACTION_COORDINATE)
    // Float16_b exponent-zero inputs expand into ordinary FP32 values in DST.
    // The bound target declares DAZ before mathematical dispatch, so normalize
    // exactly that open interval here, once, before actions and finalization.
    // The two ordered comparisons also canonicalize either signed zero to +0.
    return raw_daz_action_coordinate(x_raw);
#else
    return x_raw;
#endif
}
