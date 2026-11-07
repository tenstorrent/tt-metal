// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define TT_ASYMPTOTIC_IF(x) if constexpr (true)
#define TT_ASYMPTOTIC_END_IF

inline vFloat apply_asymptotic_scale(vFloat value) {
    value = value * ASYMPTOTIC_SCALE;
    return value;
}
