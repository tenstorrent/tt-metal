// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce_custom.h"

// Compiles the num_faces contract of the reduce_block_max_row compute API; never executed on a device.
// The num_faces convenience overloads build a TensorShape from a flat face count, and the LLK only
// implements two full 16-row column faces per face-row: 4 faces (32x32) or 2 faces (16x32). Any other
// count has to be rejected at compile time rather than pool SrcA rows the unpacker never wrote.
// Compile-time arg 1 selects ONE overload per build so each static_assert is checked on its own.
void kernel_main() {
    constexpr std::uint32_t num_faces = get_compile_time_arg_val(0);
    constexpr std::uint32_t overload = get_compile_time_arg_val(1);
    constexpr std::uint32_t block_ct_dim = 2;
    constexpr bool respect_trigger = false;
    constexpr std::uint32_t row_start_index = 0;
    constexpr std::uint32_t idst = 0;
    if constexpr (overload == 0) {
        reduce_block_max_row_init<block_ct_dim, respect_trigger, num_faces>(tt::CBIndex::c_16);
    } else if constexpr (overload == 1) {
        reduce_block_max_row<block_ct_dim, respect_trigger, num_faces>(
            tt::CBIndex::c_0, tt::CBIndex::c_1, row_start_index, idst);
    }
#ifdef ARCH_BLACKHOLE
    else if constexpr (overload == 2) {
        reduce_block_max_row_reinit_short<block_ct_dim, respect_trigger, num_faces>(tt::CBIndex::c_16);
    } else if constexpr (overload == 3) {
        reduce_block_max_row_reinit_minimal<block_ct_dim, respect_trigger, num_faces>(tt::CBIndex::c_16);
    }
#endif
    reduce_block_max_row_uninit(tt::CBIndex::c_0);
}
