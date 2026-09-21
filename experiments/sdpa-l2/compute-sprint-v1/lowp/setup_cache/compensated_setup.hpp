// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// E/G fixed Q256/K512/D128 only. Keep original paired update arithmetic.
// The PACK correction exp uses SFPI instructions, not native exp macro/replay
// initialization. Restore its modified address modifiers, retain both programs.
#if defined(TRISC_PACK) || defined(TRISC_MATH)
namespace ckernel::sfpu {
static bool sprint_compensation_ready = false;
inline void invalidate_sprint_compensation() { sprint_compensation_ready = false; }
inline void init_sprint_cached_compensation() {
    if (!sprint_compensation_ready) {
        init_sdpa_compensated_state_macros();
        init_sdpa_compensated_sum_replay();
        sprint_compensation_ready = true;
    } else {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
    }
}
inline void sprint_compensation_sum_ready() {}
} // namespace ckernel::sfpu
#endif

