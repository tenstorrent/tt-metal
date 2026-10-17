// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce_custom.h"

// Compiles the num_faces contract of the reduce_block_max_row compute API; never executed on a device.
// The num_faces convenience overloads build a TensorShape from a flat face count, and the LLK only
// implements two full 16-row column faces per face-row: 4 faces (32x32) or 2 faces (16x32). Any other
// count has to be rejected at compile time rather than pool SrcA rows the unpacker never wrote.
void kernel_main() {
    constexpr std::uint32_t num_faces = get_compile_time_arg_val(0);
    reduce_block_max_row_init<2, false, num_faces>(tt::CBIndex::c_16);
    reduce_block_max_row<2, false, num_faces>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0);
#ifdef ARCH_BLACKHOLE
    reduce_block_max_row_reinit_short<2, false, num_faces>(tt::CBIndex::c_16);
    reduce_block_max_row_reinit_minimal<2, false, num_faces>(tt::CBIndex::c_16);
#endif
    reduce_block_max_row_uninit(tt::CBIndex::c_0);
}
