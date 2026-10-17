// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The activation policy is mathematical: it applies after the compiler-bound
// target ingress transform above. Codegen supplies class codes from that
// target record, never from an activation name.
inline void apply_target_special_policy(vFloat x_effective, vFloat& result) {
    (void)x_effective;
    (void)result;
}
