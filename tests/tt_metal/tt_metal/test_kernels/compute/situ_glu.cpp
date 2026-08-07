// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drives the fused situ_glu binary SFPU op over a tile pair at a time so a host
// test can compare against the torch reference. gate arrives in c_0, up in c_1;
// the result is packed to c_16.
//   compile_time_args = [num_tiles, fp32_dest_acc_en, cap_up]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/situ_glu_sfpu.h"

namespace {
// Kimi betas, up half left untransformed (situ_glu beta2 = None).
struct SituGluNoCap {
    static constexpr float beta_gate = 4.0f;
    static constexpr float beta_up = 25.0f;
    static constexpr bool cap_up = false;
};
}  // namespace

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr bool fp32_dest = get_compile_time_arg_val(1) != 0;
    constexpr bool cap_up = get_compile_time_arg_val(2) != 0;

    constexpr uint32_t cb_gate = tt::CBIndex::c_0;
    constexpr uint32_t cb_up = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    CircularBuffer gate_cb(cb_gate);
    CircularBuffer up_cb(cb_up);
    CircularBuffer out_cb(cb_out);

    init_sfpu(cb_gate, cb_out);
    // Single SFPU op; nothing else reprograms the tanh constants between tiles.
    MATH((ckernel::llk_math_eltwise_binary_sfpu_situ_glu_init()));

    for (uint32_t t = 0; t < num_tiles; ++t) {
        gate_cb.wait_front(1);
        up_cb.wait_front(1);
        out_cb.reserve_back(1);

        tile_regs_acquire();
        copy_tile(cb_gate, 0, 0);  // gate -> dst[0]
        copy_tile(cb_up, 0, 1);    // up   -> dst[1]
        MATH((ckernel::llk_math_eltwise_binary_sfpu_situ_glu<
              fp32_dest,
              std::conditional_t<cap_up, ckernel::sfpu::SituGluConfigKimi, SituGluNoCap>>(0, 1, 0)));
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        out_cb.push_back(1);
        gate_cb.pop_front(1);
        up_cb.pop_front(1);
    }
}
