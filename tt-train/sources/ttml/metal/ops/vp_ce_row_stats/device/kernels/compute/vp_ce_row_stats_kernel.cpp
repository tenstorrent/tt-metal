// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/common/sdpa_compute_utils_common.hpp"

// Per item (row r, slice s of St tiles):
//   ROW_MAX: the row max over the slice's tiles (FPU reduce, one DEST register held over the blocks), packed to a
//            scratch tile, then column 0 broadcast over the tile's columns (so the caller's max over the partial
//            columns is exact) into the output tile.
//   SUM_EXP: the item in one DEST pass: exp(x - max) on the SFPU at fp32 two tiles at a time, added elementwise
//            onto an accumulator register, then the SFPU row reduce at full fp32 (column 0, the other columns
//            masked to zero).
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_max = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_ones = 3;
constexpr uint32_t cb_tmp = 5;
constexpr uint32_t cb_out = 6;

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t slices = get_compile_time_arg_val(2);
constexpr uint32_t St = get_compile_time_arg_val(3);
constexpr uint32_t block = get_compile_time_arg_val(4);

void kernel_main() {
    const uint32_t work_start = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_ones, 1);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t s = work - (work / slices) * slices;
        const uint32_t k0 = s * St;
        const uint32_t nk = (k0 + St <= Wt) ? St : (Wt - k0);
        cb_reserve_back(cb_out, 1);
#ifdef ROW_MAX
        reconfig_data_format(cb_in, cb_scaler);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_tmp);
        tile_regs_acquire();
        for (uint32_t k = 0; k < nk; k += block) {
            const uint32_t n = (k + block <= nk) ? block : (nk - k);
            cb_wait_front(cb_in, block);
            for (uint32_t j = 0; j < n; ++j) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, j, 0, /* dst */ 0);
            }
            cb_pop_front(cb_in, block);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_tmp, 1);
        pack_reconfig_data_format(cb_tmp);
        pack_tile(0, cb_tmp);
        tile_regs_release();
        cb_push_back(cb_tmp, 1);
        // Column 0 broadcast over the columns.
        cb_wait_front(cb_tmp, 1);
        tile_regs_acquire();
        reconfig_data_format_srcb(cb_tmp);
        UNPACK((llk_unpack_A_init<BroadcastType::COL, false, EltwiseBinaryReuseDestType::NONE, false>(
            false, false, cb_tmp)));
        MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::B2D, DST_ACCUM_MODE, BroadcastType::COL>(
            cb_tmp)));
        unary_bcast<BroadcastType::COL>(cb_tmp, /* tile_idx */ 0, /* dst */ 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_pop_front(cb_tmp, 1);
#else
        // The whole item in one DEST pass: per block of two tiles exp(x - max) at fp32 on the SFPU (DST[0..1]),
        // added elementwise onto the accumulator DST[2] (exact fp32); then the accumulator's row sums by the SFPU
        // reduce at full fp32, masked to column 0 (DST[3] holds the ones-column tile). No tile leaves DEST before
        // the output, no source-register rounding of the exp values.
        static_assert(block == 2U, "SUM_EXP: two exp tiles, the accumulator and the mask fill the fp32 DEST");
        cb_wait_front(cb_max, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_in, cb_max);
        for (uint32_t k = 0, b = 0; k < nk; k += block, ++b) {
            const uint32_t n = (k + block <= nk) ? block : (nk - k);
            cb_wait_front(cb_in, block);
            sub_bcast_cols_init(cb_in, cb_max);
            for (uint32_t j = 0; j < n; ++j) {
                sub_tiles_bcast_cols(cb_in, cb_max, j, 0, j);
            }
            exp_tile_init</* fast_and_approx */ false>();  // the accurate fp32 exp (the SDPA's is bf16-grade)
            for (uint32_t j = 0; j < n; ++j) {
                exp_tile</* fast_and_approx */ false>(j);
            }
            add_binary_tile_init();
            if (b == 0U) {
                if (n == 2U) {
                    add_binary_tile(0, 1, 2);
                } else {
                    copy_dest_values_init();
                    copy_dest_values(2, 0);
                }
            } else {
                for (uint32_t j = 0; j < n; ++j) {
                    add_binary_tile(2, j, 2);
                }
            }
            cb_pop_front(cb_in, block);
        }
        cb_pop_front(cb_max, 1);
        // Row sums into column 0 at full fp32, the other columns zeroed through the ones-column mask.
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2);
        reconfig_data_format_srca(cb_in, cb_ones);
        copy_init(cb_ones);
        copy_tile(cb_ones, 0, 3);
        mask_tile_init();
        mask_tile(2, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(2, cb_out);
        tile_regs_release();
#endif
        cb_push_back(cb_out, 1);
    }
}
