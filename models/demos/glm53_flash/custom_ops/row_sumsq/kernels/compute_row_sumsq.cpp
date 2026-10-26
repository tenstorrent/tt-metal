// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the fused row sum of squares (tt-blaze G53F-0209): ttnn.sum(ttnn.multiply(x, x), dim=-1) without the
// x*x tensor. The two ops it replaces, as they run on an fp32 input with fp32 dest:
//   * ttnn.multiply (binary_ng, fp32 x fp32 -> the SFPU path): mul_binary_tile on the two operands in dest;
//   * ttnn.sum (reduce W, default config fp32_dest_acc_en -> "accurate fp32", the SFPU reduce of
//     compute_kernel_lib::reduce): the row's first tile copied into dest, every next tile copied into a work slot and
//     folded with add_binary_tile in tile order, then the intra-tile sfpu_reduce<SUM, Float32, REDUCE_ROW>.
// Here each tile is copied into dest (fp32 unpack-to-dest: exact, as the round trip through DRAM is), squared in
// place with the same mul_binary_tile, and folded with the same add_binary_tile in the same order, so every partial
// sum and the result are the composite's bit for bit.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reduce.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t NCORES = get_compile_time_arg_val(2);
constexpr uint32_t GRID_X = get_compile_time_arg_val(3);
constexpr uint32_t cb_in_id = get_compile_time_arg_val(4);
constexpr uint32_t cb_out_id = get_compile_time_arg_val(5);

void kernel_main() {
    const uint32_t core_idx = get_absolute_logical_y() * GRID_X + get_absolute_logical_x();
    const uint32_t base = Ht / NCORES, extra = Ht % NCORES;
    const uint32_t n = base + (core_idx < extra ? 1u : 0u);
    if (n == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_in_id, cb_out_id);
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_ROW, PackMode::Default>(cb_out_id)));
    CircularBuffer cb_in(cb_in_id), cb_out(cb_out_id);
    constexpr uint32_t ACC = 0, WORK = 1;
    for (uint32_t r = 0; r < n; ++r) {
        tile_regs_acquire();
        copy_init(cb_in_id);
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            const uint32_t d = wt == 0 ? ACC : WORK;
            cb_in.wait_front(1);
            copy_tile(cb_in_id, 0, d);
            cb_in.pop_front(1);
            mul_binary_tile_init();
            mul_binary_tile(d, d, d);  // ttnn.multiply(x, x)'s SFPU product
            if (wt > 0) {
                add_binary_tile_init();
                add_binary_tile(ACC, WORK, ACC);  // ttnn.sum's fold of tile wt
            }
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(ACC, 1, 1);
        cb_out.reserve_back(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(ACC, cb_out_id);
        tile_regs_release();
        cb_out.push_back(1);
    }
    PACK((llk_pack_reduce_mask_clear()));
}
