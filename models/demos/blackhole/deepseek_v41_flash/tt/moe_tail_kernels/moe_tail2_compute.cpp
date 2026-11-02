// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Tilize variant: act tiles = hardware tilize of the row-major segments (32 rows per slot k, TPC tiles per block), then
// out_tile[j] = sum_k act_tile[k, j] * score_col[k]  (MAC with column broadcast, same maths as the plain kernel).
// TAIL_FOLD (DSV41_TAIL_FOLD=1): the tile is rounded to bf16 into CB_MID, then out = bf16(mid + fold) (the
// shared-expert partial), as in moe_tail_compute.cpp.
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

using namespace ckernel;

constexpr uint32_t TPC = get_compile_time_arg_val(0);
constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t cb_a = get_compile_time_arg_val(2);
constexpr uint32_t cb_s = get_compile_time_arg_val(3);
constexpr uint32_t cb_o = get_compile_time_arg_val(4);
constexpr uint32_t cb_rm = get_compile_time_arg_val(5);

void kernel_main() {
    CircularBuffer ca(cb_a), cs(cb_s), co(cb_o);
    compute_kernel_hw_startup(
        cb_rm, cb_a);  // tilize first; the bcast MAC below re-inits (as sparse_sdpa does after its Q tilize)
    compute_kernel_lib::tilize<TPC, cb_rm, cb_a>(/*num_blocks=*/K, /*total_input_pages=*/32 * K);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_a, cb_s);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(cb_a, cb_s, 1)));
    reconfig_data_format(cb_a, cb_s);
    cs.wait_front(K);
    ca.wait_front(K * TPC);
#ifdef TAIL_FOLD
    CircularBuffer cm(CB_MID), cf(CB_FOLD);
    constexpr uint32_t cb_dst = CB_MID;
#else
    constexpr uint32_t cb_dst = cb_o;
#endif
    for (uint32_t j = 0; j < TPC; ++j) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            mul_tiles_bcast_cols(cb_a, cb_s, k * TPC + j, k, 0);
        }
        tile_regs_commit();
#ifdef TAIL_FOLD
        cm.reserve_back(1);
#else
        co.reserve_back(1);
#endif
        pack_reconfig_data_format(cb_dst);
        tile_regs_wait();
        pack_tile(0, cb_dst);
        tile_regs_release();
#ifdef TAIL_FOLD
        cm.push_back(1);
#else
        co.push_back(1);
#endif
    }
    ca.pop_front(K * TPC);
    cs.pop_front(K);
#ifdef TAIL_FOLD
    cm.wait_front(TPC);
    cf.wait_front(TPC);
    reconfig_data_format(CB_MID, CB_FOLD);
    add_init(CB_MID, CB_FOLD);
    for (uint32_t j = 0; j < TPC; ++j) {
        tile_regs_acquire();
        add_tiles(CB_MID, CB_FOLD, j, j, 0);
        tile_regs_commit();
        co.reserve_back(1);
        pack_reconfig_data_format(cb_o);
        tile_regs_wait();
        pack_tile(0, cb_o);
        tile_regs_release();
        co.push_back(1);
    }
    cm.pop_front(TPC);
    cf.pop_front(TPC);
#endif
}
