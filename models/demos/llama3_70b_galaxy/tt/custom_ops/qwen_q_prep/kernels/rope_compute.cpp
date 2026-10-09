// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// R-core compute: per tile row, ttnn.experimental.rotary_embedding_llama (prefill, interleaved) then
// ttnn.typecast(bf16 -> bf8), the two ops _qwen_local_q_prep runs after q_norm.
//
// Runs with the rotary_embedding_llama default compute config (rotary_embedding_llama_device_operation.cpp:
// init_device_compute_kernel_config(..., HiFi4, approx, fp32_dest_acc_en = false, ...)): HiFi4, 16-bit DEST.
// The RoPE part mirrors rotary_embedding_llama.cpp (RELOAD_IMPL == 1, which the stock factory picks
// for this shape: 2 * S / 8 / 32 * 8 heads / 70 cores rows per core > 8) line by line: same hw startup
// calls, reconfigs, inits (including mul_init's default acc_to_dest), operand order and pack order.
//
// Typecast stage: ttnn.typecast bf16 -> bf8 is copy + pack (typecast_tile<Float16_b, Bfp8_b> has no
// SFPU work) with fp32_dest_acc_en = false and bfp8_pack_precise = true (typecast.cpp), i.e. the
// packer reads Float16_b DEST and rounds to bf8. The stock RoPE packs its final add to a bf16 tile
// from Float16_b DEST, which is exact, and the typecast copies that tile back to Float16_b DEST
// unchanged. So packing the final add straight to bf8 here, with this kernel built with
// bfp8_pack_precise = true, packs the same DEST bits through the same packer path.
//
// Each cos/sin row is kept for the consecutive rows of one position block (head-fastest order, see
// rope_reader.cpp) instead of being re-read per head as RELOAD_IMPL == 1 does; the tile data is the same.
//
// Runtime args: 0 num_rows, 1 h0 (head of the first row), 2 n_heads

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "qprep_common.hpp"

using namespace qprep;

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    uint32_t h = get_arg_val<uint32_t>(1);
    const uint32_t n_heads = get_arg_val<uint32_t>(2);

    // rotary_embedding_llama.cpp:85, :89
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_NORMED, CB_TRANS, CB_QOUT);
    compute_kernel_hw_startup(CB_COS_I, CB_SIN_I, CB_QOUT);

    cb_wait_front(CB_TRANS, 1);

    for (uint32_t r = 0; r < num_rows; ++r) {
        // :109-118
        cb_wait_front(CB_NORMED, WT);
        cb_wait_front(CB_SIN, WT);
        cb_wait_front(CB_COS, WT);
        cb_reserve_back(CB_ROT, WT);
        cb_reserve_back(CB_SIN_I, WT);
        cb_reserve_back(CB_COS_I, WT);
        cb_reserve_back(CB_QOUT, WT);

        // rotated = x @ trans_mat (:122-130)
        reconfig_data_format(CB_COS_I, CB_TRANS, CB_SIN_I, CB_NORMED);
        pack_reconfig_data_format(CB_QOUT, CB_ROT);
        matmul_init(CB_NORMED, CB_TRANS);
        ACQ();
        for (uint32_t j = 0; j < WT; ++j) {
            matmul_tiles(CB_NORMED, CB_TRANS, j, 0, j);
            pack_tile(j, CB_ROT, j);
        }
        REL();
        cb_push_back(CB_ROT, WT);

        // sin_interm = rotated * sin (:132-142; caller-owned init, one block of 4, pops at end)
        reconfig_data_format(CB_TRANS, CB_ROT, CB_NORMED, CB_SIN);
        pack_reconfig_data_format(CB_ROT, CB_SIN_I);
        mul_init(CB_ROT, CB_SIN);
        cb_wait_front(CB_ROT, WT);
        tile_regs_acquire();
        for (uint32_t j = 0; j < WT; ++j) {
            mul_tiles(CB_ROT, CB_SIN, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < WT; ++j) {
            pack_tile(j, CB_SIN_I);
        }
        tile_regs_release();
        cb_push_back(CB_SIN_I, WT);
        cb_pop_front(CB_ROT, WT);

        // cos_interm = x * cos (:144-158; reuses the mul_init above)
        reconfig_data_format(CB_ROT, CB_NORMED, CB_SIN, CB_COS);
        pack_reconfig_data_format(CB_SIN_I, CB_COS_I);
        tile_regs_acquire();
        for (uint32_t j = 0; j < WT; ++j) {
            mul_tiles(CB_NORMED, CB_COS, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < WT; ++j) {
            pack_tile(j, CB_COS_I);
        }
        tile_regs_release();
        cb_push_back(CB_COS_I, WT);
        cb_pop_front(CB_NORMED, WT);
        if (++h == n_heads || r + 1 == num_rows) {
            cb_pop_front(CB_SIN, WT);
            cb_pop_front(CB_COS, WT);
            h = (h == n_heads) ? 0 : h;
        }

        // out = cos_interm + sin_interm (:160-164, ckl::add: add_init(acc_to_dest = false)),
        // packed straight to bf8 (the typecast).
        reconfig_data_format(CB_NORMED, CB_COS_I, CB_COS, CB_SIN_I);
        pack_reconfig_data_format(CB_COS_I, CB_QOUT);
        add_init(CB_COS_I, CB_SIN_I, false);
        cb_wait_front(CB_COS_I, WT);
        cb_wait_front(CB_SIN_I, WT);
        tile_regs_acquire();
        for (uint32_t j = 0; j < WT; ++j) {
            add_tiles(CB_COS_I, CB_SIN_I, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < WT; ++j) {
            pack_tile(j, CB_QOUT);
        }
        tile_regs_release();
        cb_push_back(CB_QOUT, WT);
        cb_pop_front(CB_COS_I, WT);
        cb_pop_front(CB_SIN_I, WT);
    }

    cb_pop_front(CB_TRANS, 1);
}
