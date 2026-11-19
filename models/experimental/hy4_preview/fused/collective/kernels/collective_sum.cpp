// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of the standalone all_reduce_sum (fused/collective/collective.py): per stage, the sum of the
// landing's slots in ring order into cb_out. The data-movement kernel hands a stage's sum over as the next stage's
// payload, and the next stage writes its own sum over it.
//
// Compile-time args: stages (up to 4), tiles, cb_out, then per stage: members, landing CB of parity 0, of parity 1.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"

#include "collective_compute.hpp"

constexpr uint32_t stages = get_compile_time_arg_val(0);
constexpr uint32_t tiles = get_compile_time_arg_val(1);
constexpr uint32_t cb_out = get_compile_time_arg_val(2);
constexpr uint32_t MAX_STAGES = 4;
static_assert(stages >= 1 && stages <= MAX_STAGES, "one to four stages");
// members, landing CB of parity 0, landing CB of parity 1
constexpr uint32_t stage_args[MAX_STAGES][3] = {
    {get_compile_time_arg_val(3), get_compile_time_arg_val(4), get_compile_time_arg_val(5)},
    {get_compile_time_arg_val(6), get_compile_time_arg_val(7), get_compile_time_arg_val(8)},
    {get_compile_time_arg_val(9), get_compile_time_arg_val(10), get_compile_time_arg_val(11)},
    {get_compile_time_arg_val(12), get_compile_time_arg_val(13), get_compile_time_arg_val(14)},
};

void kernel_main() {
    compute_kernel_hw_startup(stage_args[0][1], cb_out);

    for (uint32_t s = 0; s < stages; ++s) {
        const uint32_t members = stage_args[s][0];
        const uint32_t landing = hy4_collective::wait_landing(stage_args[s][1], stage_args[s][2], members * tiles);
        hy4_collective::sum_slots(landing, members, tiles, cb_out);
        cb_pop_front(landing, members * tiles);
    }
}
