// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DFlash2 fused grouped dynamic conv -- compute: per column tile, fp32 DST (fp32_dest_acc_en, HiFi4):
//   dst1 = 0; dst1 += CF[tap 1] @ E[tap 1]; dst1 += J[k1] @ B            (coef_1, exact)
//   dst0 = 0; dst0 += SH @ X                                             (x_prev, exact)
//   dst0 = dst0 * dst1                                                   (fp32 SFPU)
//   dst1 = 0; dst1 += CF[tap 0] @ E[tap 0]; dst1 += J[k0] @ B            (coef_0, exact)
//   dst2 = X                                                             (bf16 -> fp32, exact)
//   dst2 = dst2 * dst1; dst0 = dst0 + dst2                               (fp32 SFPU) -> pack bf16
// The DST tiles that accumulate matmuls are zeroed first by a datacopy of the zero tile (an overwrite), so the
// program never relies on the DST clear-on-release; the SrcOrder::Reverse startup is the matmul's (in0 -> SrcB,
// in1 -> SrcA; the datacopies read SrcA); every CB is bf16, so no data-format reconfiguration between the ops.
// Runtime args: n (column tiles of this core).
// DF2_CONV_RES (df2_conv_common.hpp): before the body, X = bf16(X32) by DST <- X32 (fp32 unpack-to-dest, exact) ->
// pack bf16 into CB_X (the packer's rounding) -> the body reads CB_X as X; after the body dst0 += RES (fp32 DST add
// of the bf16 residual) -> pack bf16. Every CB but X32 is bf16; the unpacker is re-pointed by copy_init / matmul_init.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#include "df2_conv_common.hpp"

using namespace df2conv;

namespace {

FORCE_INLINE void zero_dst(uint32_t d) {
    copy_init(CB_ZERO);
    copy_tile(CB_ZERO, 0, d);
}

// dst[d] += CF[tap] @ E[tap] + J[tap] @ B   (the coefficient tile of tap t; the J CB holds k0 at 0, k1 at 1)
FORCE_INLINE void coef_into(uint32_t d, uint32_t tap) {
    matmul_init(CB_CF, CB_E);
    matmul_tiles(CB_CF, CB_E, tap, tap, d);
    matmul_init(CB_J, CB_B);
    matmul_tiles(CB_J, CB_B, tap, 0, d);
}

}  // namespace

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_CF, CB_E, CB_OUT);
    CircularBuffer cb_x(CB_X), cb_cf(CB_CF), cb_e(CB_E), cb_j(CB_J), cb_b(CB_B), cb_sh(CB_SH), cb_zero(CB_ZERO),
        cb_out(CB_OUT);
#if defined(DF2_CONV_RES)
    CircularBuffer cb_x32(CB_X32), cb_res(CB_RES);
#endif
    cb_j.wait_front(2);
    cb_sh.wait_front(1);
    cb_zero.wait_front(1);
    for (uint32_t i = 0; i < n; ++i) {
#if defined(DF2_CONV_RES)
        // x = bf16(X32): the round trip through CB_X (reserve -> pack -> push; the body waits on it below)
        cb_x32.wait_front(1);
        cb_x.reserve_back(1);
        tile_regs_acquire();
        copy_init(CB_X32);
        copy_tile(CB_X32, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_X);
        tile_regs_release();
        cb_x.push_back(1);
        cb_x32.pop_front(1);
#endif
        cb_x.wait_front(1);
        cb_cf.wait_front(2);
        cb_e.wait_front(2);
        cb_b.wait_front(1);
        cb_out.reserve_back(1);
        tile_regs_acquire();
        zero_dst(0);
        zero_dst(1);
        coef_into(1, 1);
        matmul_init(CB_SH, CB_X);
        matmul_tiles(CB_SH, CB_X, 0, 0, 0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        zero_dst(1);
        coef_into(1, 0);
        copy_init(CB_X);
        copy_tile(CB_X, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(2, 1, 2);
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
#if defined(DF2_CONV_RES)
        cb_res.wait_front(1);
        copy_init(CB_RES);
        copy_tile(CB_RES, 0, 3);
        add_binary_tile_init();
        add_binary_tile(0, 3, 0);  // out = conv + residual (fp32), packed bf16
        cb_res.pop_front(1);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_out.push_back(1);
        cb_x.pop_front(1);
        cb_cf.pop_front(2);
        cb_e.pop_front(2);
        cb_b.pop_front(1);
    }
    cb_j.pop_front(2);
    cb_sh.pop_front(1);
    cb_zero.pop_front(1);
}
