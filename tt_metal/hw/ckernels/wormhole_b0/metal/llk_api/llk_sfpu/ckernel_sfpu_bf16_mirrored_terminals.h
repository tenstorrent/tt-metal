// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Include inside namespace sfpi. Typed callers retain the domain-action proof.
template <uint32_t BoundBits, bool SignedEndpoints>
inline void mirrored_class_terminals(vFloat x_raw, vFloat& result) {
    vFloat absolute_raw = setsgn(x_raw, 0);
    if constexpr (SignedEndpoints) {
        v_if(absolute_raw > __builtin_bit_cast(float, BoundBits)) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(absolute_raw >= __builtin_bit_cast(float, BoundBits)) {
            vFloat endpoint_inf = std::numeric_limits<float>::infinity();
            result = copysgn(endpoint_inf, x_raw);
        }
        v_endif;
    } else {
        v_if(absolute_raw > __builtin_bit_cast(float, BoundBits)) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
}
