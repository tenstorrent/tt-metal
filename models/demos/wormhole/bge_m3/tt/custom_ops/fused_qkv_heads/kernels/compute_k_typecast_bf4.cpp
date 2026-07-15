// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 head-split K-typecast compute kernel (BF8 -> BF4).
//
// Folds the standalone `ttnn.typecast(k, bfloat4_b)` into the head-split op so
// the BF8 K tiles produced by the reader are converted to BF4 in-place on the
// TRISC before the writer stores them to DRAM. This removes one program launch
// and the BF8 K DRAM write + read round trip per layer.
//
// Q and V stay on the direct reader->writer data-movement path (BF8, untouched).
// Only K flows reader(BF8) -> cb_k_in -> this compute -> cb_k_out(BF4) -> writer.
//
// Compile-time args:
//   0: group_kv_tiles  (K tiles per work unit = heads_per_group * head_dim_tiles)
//   1: cb_k_in         (BF8 input CB index)
//   2: cb_k_out        (BF4 output CB index)
//
// Runtime args:
//   0: num_work_units  (work units assigned to this core)

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t group_kv_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_k_out = get_compile_time_arg_val(2);

    const uint32_t num_work_units = get_arg_val<uint32_t>(0);

    // Bfp8_b = 6, Bfp4_b = 7 (tt_backend_api_types.hpp).
    constexpr uint32_t IN_FMT = 6u;
    constexpr uint32_t OUT_FMT = 7u;

    CircularBuffer cb_in(cb_k_in);
    CircularBuffer cb_out(cb_k_out);

    init_sfpu(cb_k_in, cb_k_out);
    const uint32_t total_k_tiles = num_work_units * group_kv_tiles;
    for (uint32_t t = 0; t < total_k_tiles; ++t) {
        cb_out.reserve_back(1);
        cb_in.wait_front(1);

        tile_regs_acquire();
        copy_tile(cb_k_in, 0, 0);
        typecast_tile_init<IN_FMT, OUT_FMT>();
        typecast_tile<IN_FMT, OUT_FMT>(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_k_out);
        tile_regs_release();

        cb_in.pop_front(1);
        cb_out.push_back(1);
    }
}
