// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    uint32_t sub_Mt = get_compile_time_arg_val(0);
    uint32_t Kt = get_compile_time_arg_val(1);
    uint32_t sub_Nt = get_compile_time_arg_val(2);

    constexpr int onetile = 1;

    mm_init();

    for (uint32_t mt = 0; mt < sub_Mt; ++mt) {
        for (uint32_t nt = 0; nt < sub_Nt; ++nt) {
            acquire_dst();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, mt * Kt + kt, nt * Kt + kt, 0, false);
            }
            cb_reserve_back(tt::CB::c_out0, onetile);
            pack_tile(0, tt::CB::c_out0);
            cb_push_back(tt::CB::c_out0, onetile);
            release_dst();
        }
    }
}
}  // namespace NAMESPACE
