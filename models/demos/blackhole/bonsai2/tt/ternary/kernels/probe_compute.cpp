// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Probe: out = act (bf16 tile) @ W, W a ternary tile stored as 256 B of Bfp2_b data with no exponent section.
// Unpacker A (srcA = in1 = W) is switched to Bfp2_b with the forced shared exponent `SHARED_EXP`.
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_act = tt::CBIndex::c_0;
    constexpr uint32_t cb_w = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t shared_exp = get_compile_time_arg_val(0);
    constexpr uint32_t force = get_compile_time_arg_val(1);
    constexpr uint32_t num_w = get_compile_time_arg_val(2);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_out);
    matmul_init(cb_act, cb_w);
    UNPACK(({
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            (uint32_t)DataFormat::Bfp2_b, (uint32_t)DataFormat::Float16_b, 256 >> 4, FACE_R_DIM, 4);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(force);
        cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(shared_exp);
    }));
    cb_wait_front(cb_act, 1);
    for (uint32_t i = 0; i < num_w; ++i) {
        cb_wait_front(cb_w, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        matmul_tiles(cb_act, cb_w, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_w, 1);
    }
    cb_pop_front(cb_act, 1);
}
