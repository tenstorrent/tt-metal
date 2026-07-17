// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    uint32_t sub_Mt = get_compile_time_arg_val(0);
    uint32_t Kt = get_compile_time_arg_val(1);
    uint32_t sub_Nt = get_compile_time_arg_val(2);

    constexpr int onetile = 1;

    compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);

    for (uint32_t mt = 0; mt < sub_Mt; ++mt) {
        for (uint32_t nt = 0; nt < sub_Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, mt * Kt + kt, nt * Kt + kt, 0);
            }
            cb_reserve_back(tt::CBIndex::c_16, onetile);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, tt::CBIndex::c_16);
            cb_push_back(tt::CBIndex::c_16, onetile);
            tile_regs_release();
        }
    }
}
