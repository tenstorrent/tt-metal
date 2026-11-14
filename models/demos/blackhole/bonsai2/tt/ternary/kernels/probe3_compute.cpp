// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Probe: custom_mm of a [1,32] bf16 in0 tile with one ternary in1 tile (Bfp2_b, forced exponent, 256 B), packed out.
#include <cstdint>
#include "api/compute/reconfig_data_format.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_x = tt::CBIndex::c_0, cb_w = tt::CBIndex::c_1, cb_out = tt::CBIndex::c_16;
    custom_mm_block_init<false, CustomMmDestLayout::Default>(cb_x, cb_w, cb_out);
    UNPACK(({
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            (uint32_t)DataFormat::Bfp2_b, (uint32_t)DataFormat::Float16_b, 256 >> 4, FACE_R_DIM, 4);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(1);
        cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(127);
    }));
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_w, 1);
    tile_regs_acquire();
    custom_mm_block<CustomMmDestLayout::Default, false>(cb_x, cb_w, 0, 0, 0, 1);
    custom_mm_block_uninit<CustomMmDestLayout::Default>(cb_out);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_init(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
