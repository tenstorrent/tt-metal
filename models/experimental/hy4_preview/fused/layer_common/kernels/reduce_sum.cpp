// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// A reduce core of a fused layer program (fused/layer_common/transport.py): compute. Sums the slots of the stage-1
// landing (n1 chips of a mesh row) and, on the root chip, of the stage-2 landing (n2 row sums), each in slot order
// with Float32 adds in the SFPU and one rounding to BFloat16 per stage (fused/collective/kernels/
// collective_compute.hpp sum_slots, as the collective's own reduce).
//
// Compile-time args: cb_s1, cb_sum, cb_s2, cb_f, tiles, n1, n2, is_root

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"

#include "collective_compute.hpp"

constexpr uint32_t cb_s1 = get_compile_time_arg_val(0);
constexpr uint32_t cb_sum = get_compile_time_arg_val(1);
constexpr uint32_t cb_s2 = get_compile_time_arg_val(2);
constexpr uint32_t cb_f = get_compile_time_arg_val(3);
constexpr uint32_t tiles = get_compile_time_arg_val(4);
constexpr uint32_t n1 = get_compile_time_arg_val(5);
constexpr uint32_t n2 = get_compile_time_arg_val(6);
constexpr bool is_root = get_compile_time_arg_val(7) == 1;

void kernel_main() {
    compute_kernel_hw_startup(cb_s1, cb_sum);
    cb_wait_front(cb_s1, n1 * tiles);
    hy4_collective::sum_slots(cb_s1, n1, tiles, cb_sum);
    cb_pop_front(cb_s1, n1 * tiles);
    if constexpr (is_root) {
        cb_wait_front(cb_s2, n2 * tiles);
        hy4_collective::sum_slots(cb_s2, n2, tiles, cb_f);
        cb_pop_front(cb_s2, n2 * tiles);
    }
}
