// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// For test_srca_forced_exp.cpp: c_0 bf16 in0 times c_1's BFP2_b mantissas at exponent 127 (srca_forced_exp_bfp2),
// then, after srca_forced_exp_off and a SrcA reconfig, times c_2's bf16 tile; both products to c_16 (Float32).

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/srca_forced_exp.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"

void kernel_main() {
    using namespace ckernel;
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0, cb_mant = tt::CBIndex::c_1, cb_bf16 = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_mant, cb_out);
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_mant, 1);
    cb_wait_front(cb_bf16, 1);
    cb_reserve_back(cb_out, 2);

    matmul_init(cb_in0, cb_mant);
    srca_forced_exp_bfp2(127);
    tile_regs_acquire();
    matmul_tiles(cb_in0, cb_mant, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    srca_forced_exp_off();
    reconfig_data_format_srca(cb_mant, cb_bf16);
    matmul_init(cb_in0, cb_bf16);
    tile_regs_acquire();
    matmul_tiles(cb_in0, cb_bf16, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 2);
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_mant, 1);
    cb_pop_front(cb_bf16, 1);
}
