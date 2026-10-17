// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/quantization.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Math replay-buffer co-residency probe: one kernel runs an FPU reduce (REDUCE_ROW SUM, whose MOP replays
// a body recorded in the FPU half of the math replay buffer) and an SFPU dequant (whose body is recorded in
// the SFPU half). Both inits run before either op executes, so each op replays the body it recorded only if
// the two recordings do not overlap. The init order is a compile-time arg so both "who records last" cases
// are covered:
//   get_compile_time_arg_val(0) == 0 : reduce_init, then dequant_tile_init  (dequant records last)
//   get_compile_time_arg_val(0) == 1 : dequant_tile_init, then reduce_init  (reduce records last)
// CBs: c_0 reduce data (bf16), c_1 reduce scaler (bf16, row 0 of each face = 1.0), c_2 dequant input
// (Int32), c_3 dequant scale (fp32), c_16 output (fp32): tile 0 = row sums in column 0, tile 1 = dequant.
// Runtime arg 0: fp32 bit pattern of the negated zero point (the dequant body adds it, which is the
// quantization op's convention).

namespace {
constexpr std::uint32_t kInitOrder = get_compile_time_arg_val(0);

constexpr auto cb_data = tt::CBIndex::c_0;
constexpr auto cb_scaler = tt::CBIndex::c_1;
constexpr auto cb_q = tt::CBIndex::c_2;
constexpr auto cb_scale = tt::CBIndex::c_3;
constexpr auto cb_out = tt::CBIndex::c_16;

constexpr std::uint32_t kDstReduce = 0;
constexpr std::uint32_t kDstQ = 0;
constexpr std::uint32_t kDstScale = 1;
}  // namespace

void kernel_main() {
    const std::uint32_t neg_zero_point = get_arg_val<std::uint32_t>(0);

    CircularBuffer data(cb_data);
    CircularBuffer scaler(cb_scaler);
    CircularBuffer q(cb_q);
    CircularBuffer scale(cb_scale);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup(cb_data, cb_scaler, cb_out);
    // REDUCE_ROW SUM runs with swapped operands (scaler -> SrcA, data -> SrcB); see reduce_init.
    reconfig_data_format(cb_scaler, cb_data);

    if constexpr (kInitOrder == 0) {
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_data, cb_scaler, cb_out);
        dequant_tile_init(neg_zero_point);
    } else {
        dequant_tile_init(neg_zero_point);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_data, cb_scaler, cb_out);
    }

    data.wait_front(1);
    scaler.wait_front(1);
    q.wait_front(1);
    scale.wait_front(1);
    out.reserve_back(2);

    // Tile 0: the reduce. Its MOP replays the FPU-half body.
    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_data, cb_scaler, 0, 0, kDstReduce);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(kDstReduce, cb_out);
    tile_regs_release();
    reduce_uninit();

    // Tile 1: the dequant. Int32 input and fp32 scale go to Dest first (as the quantization op's binary
    // SFPU kernels do), then the SFPU body replays the SFPU-half recording.
    reconfig_data_format_srca(cb_scaler, cb_q);
    copy_init(cb_q);
    tile_regs_acquire();
    copy_tile(cb_q, 0, kDstQ);
    reconfig_data_format_srca(cb_q, cb_scale);
    copy_init(cb_scale);
    copy_tile(cb_scale, 0, kDstScale);
    dequant_tile(kDstQ, kDstScale, kDstQ);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(kDstQ, cb_out);
    tile_regs_release();

    data.pop_front(1);
    scaler.pop_front(1);
    q.pop_front(1);
    scale.pop_front(1);
    out.push_back(2);
}
