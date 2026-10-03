// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Single-pass evaluator integration point.  Every evaluator calls this while
// its raw-coordinate vector is still live, after reconstruction/postcompose
// and before precision conversion, gradient multiplication, or store.  The
// no-action arm compiles away, preserving legacy generated source behavior.
inline void finalize_raw_domain_actions(vFloat x_raw, vFloat& result) {
    [[maybe_unused]] vFloat x_effective = target_domain_action_coordinate(x_raw);
    (void)x_raw;
    (void)result;
    // Ordered finite/domain return_class actions own their lanes first.  The
    // effective special policy then has final class authority; target rounding
    // and FTZ happen in the evaluator's existing conversion/store epilogue.
    apply_target_special_policy(x_effective, result);
}
