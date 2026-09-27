// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The online-softmax merge of a step's partial attention into the running
// accumulators, per row tile, exact Float32 on the vector unit throughout.
//
//   a = lse (running), b = lse_s (the step's); m = max(a, b)
//   lse <- m + ln(1 + exp(-|a - b|))
//   w_a = exp(a - m) / (1 + exp(-|a - b|)),  w_b = exp(b - m) / (1 + exp(-|a - b|))
//   O   <- w_a O + w_b O_s
//
// Precision. Every Float32 that passes through the matmul unit's source
// registers keeps 19 of its 32 bits, which rounded the lse by 2e-3 and the
// output by 5e-4 at every merge in a first version and cost the backward a
// factor of six on dQ. So every Float32 here is unpacked straight into the
// DST registers (UnpackToDestFp32), the column broadcast of the statistics
// included -- the API's 32-bit broadcast path does that -- and the merge
// runs on the vector unit in Float32. The statistic tiles carry the row's value in column 0
// and whatever the producing kernels left in the other columns; those
// columns are computed on but never read. A running lse of -inf (nothing
// merged yet) gives w_a = 0, w_b = 1.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t Wt = get_compile_time_arg_val(0);

constexpr uint32_t cb_lse_acc = tt::CBIndex::c_0;   // Float32, unpack to dest
constexpr uint32_t cb_step_lse = tt::CBIndex::c_1;  // Float32, unpack to dest
constexpr uint32_t cb_out_acc = tt::CBIndex::c_2;   // Float32, unpack to dest
constexpr uint32_t cb_step_out = tt::CBIndex::c_3;  // bfloat16 (exact through the source registers)
constexpr uint32_t cb_wa = tt::CBIndex::c_4;        // Float32, unpack to dest
constexpr uint32_t cb_wb = tt::CBIndex::c_5;        // Float32, unpack to dest
constexpr uint32_t cb_lse_new = tt::CBIndex::c_6;
constexpr uint32_t cb_out_new = tt::CBIndex::c_7;

constexpr uint32_t kOneBits = 0x3F800000u;

// Copy a tile from a CB into a DST register, exactly for the CBs in
// unpack-to-dest mode.
inline void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    binary_op_init_common(cb_out_acc, cb_wa, cb_out_new);

    for (uint32_t r = 0; r < rows; ++r) {
        // ---- 2. the new lse and the two weights, on broadcast tiles
        cb_wait_front(cb_lse_acc, 1);
        cb_wait_front(cb_step_lse, 1);
        cb_reserve_back(cb_wa, 1);
        cb_reserve_back(cb_wb, 1);
        cb_reserve_back(cb_lse_new, 1);
        tile_regs_acquire();
        // The 32-bit column broadcast unpacks straight to dest (the API's own
        // path for Float32, since the source register is 19 bits wide), so a
        // and b arrive whole and broadcast across the columns.
        unary_bcast_init<BroadcastType::COL>(cb_lse_acc);
        unary_bcast<BroadcastType::COL>(cb_lse_acc, 0, 0);  // a
        unary_bcast_init<BroadcastType::COL>(cb_step_lse);
        unary_bcast<BroadcastType::COL>(cb_step_lse, 0, 1);  // b
        binary_max_tile_init();
        binary_max_tile(0, 1, 2);  // m
        sub_binary_tile_init();
        sub_binary_tile(0, 2, 0);  // a - m
        sub_binary_tile(1, 2, 1);  // b - m
        add_binary_tile_init();
        add_binary_tile(0, 1, 3);  // -|a - b|
        exp_tile_init<false>();
        exp_tile<false>(0);  // exp(a - m)
        exp_tile<false>(1);  // exp(b - m)
        exp_tile<false>(3);  // e = exp(-|a - b|)
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(3, 4);
        log1p_tile_init<false>();
        log1p_tile<false>(3);  // ln(1 + e)
        add_binary_tile_init();
        add_binary_tile(3, 2, 3);  // the new lse
        binop_with_scalar_tile_init();
        add_unary_tile(4, kOneBits);  // 1 + e
        recip_tile_init();
        recip_tile(4);
        mul_binary_tile_init();
        mul_binary_tile(0, 4, 0);  // w_a
        mul_binary_tile(1, 4, 1);  // w_b
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_wa);
        pack_tile(0, cb_wa);
        pack_tile(1, cb_wb);
        pack_tile(3, cb_lse_new);
        tile_regs_release();
        cb_push_back(cb_wa, 1);
        cb_push_back(cb_wb, 1);
        cb_push_back(cb_lse_new, 1);
        cb_pop_front(cb_lse_acc, 1);
        cb_pop_front(cb_step_lse, 1);

        // ---- 3. the output row: O <- w_a O + w_b O_s, every operand exact
        cb_wait_front(cb_wa, 1);
        cb_wait_front(cb_wb, 1);
        cb_wait_front(cb_out_acc, Wt);
        cb_wait_front(cb_step_out, Wt);
        cb_reserve_back(cb_out_new, Wt);
        for (uint32_t c = 0; c < Wt; ++c) {
            tile_regs_acquire();
            load(cb_out_acc, c, 0);
            load(cb_wa, 0, 1);
            load(cb_step_out, c, 2);
            load(cb_wb, 0, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(2, 3, 2);
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out_new);
            pack_tile(0, cb_out_new);
            tile_regs_release();
        }
        cb_push_back(cb_out_new, Wt);
        cb_pop_front(cb_out_acc, Wt);
        cb_pop_front(cb_step_out, Wt);
        cb_pop_front(cb_wa, 1);
        cb_pop_front(cb_wb, 1);
    }
}
