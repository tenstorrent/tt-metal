// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

inline void finalize_raw_domain_actions(vFloat x_raw, vFloat& result) {
    [[maybe_unused]] vFloat x_effective = target_domain_action_coordinate(x_raw);
    apply_raw_pre_domain_actions(x_effective, result);
    apply_target_special_policy(x_effective, result);
}
