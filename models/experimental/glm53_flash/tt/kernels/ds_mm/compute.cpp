// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DS matmul — compute: per K block, NR / SUBW output subblocks of matmul_block (KB inner tiles), each packed onto the
// same fp32 partial tiles (packer L1 accumulation from the second block on; the partial CB is exactly NR tiles, so
// popping a block's partials rewinds to the same addresses); after the last block the partials are unpacked exactly
// (fp32 unpack-to-dest) and packed to the output dtype. HiFi2 / fp32 DST as today's decode DS linears.
// Compile-time args: NB, KB, NR, SUBW.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#include "ds_mm_common.hpp"

using namespace dsmm;

void kernel_main() {
    constexpr uint32_t NB = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NR = get_compile_time_arg_val(2);
    constexpr uint32_t SUBW = get_compile_time_arg_val(3);
    static_assert(NR % SUBW == 0, "subblocks");

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, CB_W, CB_PART);
    matmul_block_init(CB_IN0, CB_W, false, SUBW, 1, KB);
    for (uint32_t blk = 0; blk < NB; ++blk) {
        cb_wait_front(CB_IN0, KB);
        cb_wait_front(CB_W, KB * NR);
        for (uint32_t sb = 0; sb < NR / SUBW; ++sb) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < KB; ++k) {
                matmul_block(CB_IN0, CB_W, k, k * NR + sb * SUBW, 0, false, SUBW, 1, KB);
            }
            tile_regs_commit();
            cb_reserve_back(CB_PART, SUBW);
            tile_regs_wait();
            PACK((llk_pack_reconfig_l1_acc(blk > 0 ? 1 : 0)));
            for (uint32_t i = 0; i < SUBW; ++i) {
                pack_tile(i, CB_PART);
            }
            tile_regs_release();
            cb_push_back(CB_PART, SUBW);
        }
        cb_pop_front(CB_IN0, KB);
        cb_pop_front(CB_W, KB * NR);
        if (blk + 1 < NB) {
            cb_wait_front(CB_PART, NR);  // rewind: the next block accumulates onto the same tiles
            cb_pop_front(CB_PART, NR);
        }
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    cb_wait_front(CB_PART, NR);
    cb_reserve_back(CB_OUT, NR);
    reconfig_data_format_srca(CB_PART);
    pack_reconfig_data_format(CB_OUT);
    copy_tile_to_dst_init_short(CB_PART);
    for (uint32_t j = 0; j < NR; ++j) {
        tile_regs_acquire();
        copy_tile(CB_PART, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
    }
    cb_push_back(CB_OUT, NR);
    cb_pop_front(CB_PART, NR);
}
