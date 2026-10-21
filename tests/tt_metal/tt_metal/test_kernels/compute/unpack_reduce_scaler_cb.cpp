// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile coverage for the single-operand reduce wrappers llk_unpack_reduce_init / llk_unpack_reduce
// (hw/ckernels/<arch>/metal/llk_api/llk_unpack_reduce_api.h). Nothing else in tree instantiates them, so
// this kernel exercises the scaler CB address math, a nonzero scaler tile index, the SrcB format selection
// and the bounds asserts for REDUCE_COL SUM and REDUCE_ROW MAX. It is only compiled, never run.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"

namespace {

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_out = 16;
constexpr uint32_t scaler_tile_index = 1;

template <PoolType pool_type, ReduceDim reduce_dim>
void reduce_with_scaler_cb(uint32_t num_tiles) {
    UNPACK((llk_unpack_reduce_init<pool_type, reduce_dim>(cb_scaler, 0 /*within_face_16x16_transpose*/)));
    MATH((llk_math_reduce_init<pool_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>(cb_in, cb_scaler)));
    PACK((llk_pack_reduce_mask_config<reduce_dim, PackMode::Default>(cb_out)));

    cb_wait_front(cb_scaler, scaler_tile_index + 1);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        UNPACK((llk_unpack_reduce<pool_type, reduce_dim>(cb_in, cb_scaler, 0, scaler_tile_index)));
        MATH((llk_math_reduce<pool_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>(cb_in, cb_scaler, 0)));
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_in, 1);
    }
    cb_pop_front(cb_scaler, scaler_tile_index + 1);

    MATH((llk_math_reduce_uninit()));
    PACK((llk_pack_reduce_mask_clear()));
}

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_compile_time_arg_val(0);
    // Once per kernel; the scaler CB is the SrcB operand of the startup, as the wrapper's contract requires.
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    reduce_with_scaler_cb<PoolType::SUM, ReduceDim::REDUCE_COL>(num_tiles);
    reduce_with_scaler_cb<PoolType::MAX, ReduceDim::REDUCE_ROW>(num_tiles);
}
