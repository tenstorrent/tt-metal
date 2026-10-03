// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The selected derivative arrives rounded, or unrounded under FUSE_GRAD_SINGLE_ROUNDING. Raw input still occupies
// this destination row; classification must happen before its final store.
inline vFloat finalize_zone_gradient(vFloat raw, vUInt raw_u16, vFloat factor, vFloat grad) {
    vFloat result = factor * grad;
    v_if(raw < TT_ZONE_GRADIENT_LOWER) { result = 0.0f; }
    v_endif;
    v_if(raw > TT_ZONE_GRADIENT_UPPER) { result = grad; }
    v_endif;
    // Same encoded BF16 exponent/mantissa predicate as nan_union_class_terminal.
    // Numeric ingress cannot distinguish negative NaN from negative infinity.
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x00ffu);
    v_if(exponent_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) { result = grad; }
        v_endif;
    }
    v_endif;
    return result;
}
