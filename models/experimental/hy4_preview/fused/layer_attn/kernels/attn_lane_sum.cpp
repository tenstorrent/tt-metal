// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the query hub core in program P1 of the attention half (fused/layer_attn/program.py): the
// sum over the chips of the lane sums of q_a's squares. Every chip's statistic tile (a Float32 [8, 32] tile, 1 KB:
// the first quarter of a Float32 [32, 32] page whose rest stays zero) has landed in slot c of the lane landing; the
// slots are added in chip order with Float32 adds in the SFPU (fused/collective/kernels/collective_compute.hpp
// sum_slots, the collective's own reduce), bit-equal on every chip.
//
// Compile-time args: the landing's circular buffer (unpacked to DEST as Float32), the sum's, the slots.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"

#include "collective_compute.hpp"

constexpr uint32_t cb_land = get_compile_time_arg_val(0);
constexpr uint32_t cb_sum = get_compile_time_arg_val(1);
constexpr uint32_t slots = get_compile_time_arg_val(2);

void kernel_main() {
    compute_kernel_hw_startup(cb_land, cb_sum);
    cb_wait_front(cb_land, slots);
    hy4_collective::sum_slots(cb_land, slots, 1, cb_sum);
    cb_pop_front(cb_land, slots);
}
