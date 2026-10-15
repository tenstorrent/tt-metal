// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The CSV codegen declares whether the asymptotic correction is active on the
// left tail, right tail, or whole domain.  Keep that decision compile-time so a
// right-tail factor does not materialize the old 1e38 upper-bound comparison
// (and, more importantly, does not run on the non-tail segments).
#define TT_ASYMPTOTIC_IF(x) if constexpr (true)
#define TT_ASYMPTOTIC_END_IF

inline vFloat apply_asymptotic_scale(vFloat value) {
    value = value * ASYMPTOTIC_SCALE;
    return value;
}
