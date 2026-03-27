// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu_params.h"
#include "ckernel_sfpu_mac.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS = 8>
inline void llk_math_eltwise_ternary_sfpu_mac(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_mac<APPROXIMATE, is_fp32_dest_acc_en, data_format, ITERATIONS>,
        dst_index0,
        dst_index1,
        dst_index2,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_mac_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::mac>();
    // eltwise_ternary_sfpu_configure_addrmod only sets ADDR_MOD_6 (dest.incr=2)
    // for SfpuType::where.  mac's replay sequence uses ADDR_MOD_2 on SFPSTORE
    // (which maps to physical slot 6 after set_addr_mod_base() adds 4), so we
    // must configure it explicitly here.
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }.set(ADDR_MOD_6);
}

}  // namespace ckernel
