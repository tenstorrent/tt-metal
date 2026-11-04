// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 combine "rm" -- compute (moe_combine_rm_reader.cpp): per token, the K gathered row parts x their weights
// (SCALAR broadcast) accumulated in fp32 in L1 (packer L1 accumulation into CB_RM, row t at pages t PT ..), then the
// chunk's 32 rows tilized into CB_OUT (32 PT bf16 tiles). Compile-time: K, PT. Runtime: unit count.
// fp32_dest_acc_en: a row part (PT <= 4 tiles) fits the half-sync DEST.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t PT = get_compile_time_arg_val(1);
    constexpr uint32_t CB_IN = 0;
    constexpr uint32_t CB_W = 1;
    constexpr uint32_t CB_OUT = 16;
    constexpr uint32_t CB_RM = 17;
    static_assert(PT <= 4, "a row part must fit the fp32 half-sync DEST");
    const uint32_t nu = get_arg_val<uint32_t>(0);
    if (nu == 0) {
        return;
    }

    compute_kernel_hw_startup(CB_IN, CB_W, CB_RM);
    for (uint32_t u = 0; u < nu; ++u) {
        reconfig_data_format(CB_IN, CB_W);
        pack_reconfig_data_format(CB_RM);
        mul_bcast_scalar_init(CB_IN, CB_W);
        cb_reserve_back(CB_RM, 32 * PT);
        for (uint32_t t = 0; t < 32; ++t) {
            for (uint32_t j = 0; j < K; ++j) {
                cb_wait_front(CB_IN, PT);
                cb_wait_front(CB_W, 1);
                pack_reconfig_l1_acc(j == 0 ? 0 : 1);
                tile_regs_acquire();
                for (uint32_t x = 0; x < PT; ++x) {
                    mul_tiles_bcast<BroadcastType::SCALAR>(CB_IN, CB_W, x, 0, x);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t x = 0; x < PT; ++x) {
                    pack_tile<true>(x, CB_RM, t * PT + x);
                }
                tile_regs_release();
                cb_pop_front(CB_IN, PT);
                cb_pop_front(CB_W, 1);
            }
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(CB_RM, 32 * PT);
        compute_kernel_lib::tilize<32 * PT, CB_RM, CB_OUT>(1);
    }
}
