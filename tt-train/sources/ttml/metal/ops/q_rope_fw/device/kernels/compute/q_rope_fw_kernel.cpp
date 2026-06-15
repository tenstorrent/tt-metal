// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

// q_nope bypasses compute (reader -> writer). Per block:
//   out_pe = (q_pe * cos) + (rotate(q_pe) * sin)
namespace ckernel {

// reconfig + mul init (acc_to_dest=true). The reconfig is required when switching unpacker
// inputs under FP32_DEST_ACC_EN (see reconfig_data_format); it is a no-op when formats match.
ALWI void mul_tiles_init_with_dt(uint32_t icb0, uint32_t icb1) {
    reconfig_data_format(icb0, icb1);
    mul_tiles_init(icb0, icb1);
}

}  // namespace ckernel

constexpr uint32_t q_pe_cb = get_compile_time_arg_val(0);
constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);
constexpr uint32_t rotated_in_cb = get_compile_time_arg_val(4);
constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(5);
constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(6);
constexpr uint32_t rope_out_cb = get_compile_time_arg_val(7);
constexpr uint32_t Tr = get_compile_time_arg_val(8);
constexpr uint32_t n_heads = get_compile_time_arg_val(9);
constexpr uint32_t kDstBatchHeads = get_compile_time_arg_val(10);
constexpr uint32_t kTailHeads = n_heads % kDstBatchHeads;
constexpr uint32_t kFullBatchHeads = n_heads - kTailHeads;
constexpr uint32_t kFullBatchTiles = kDstBatchHeads * Tr;
constexpr uint32_t kTailBatchTiles = kTailHeads * Tr;

ALWI void process_rope_group(const uint32_t num_tiles) {
    cb_wait_front(q_pe_cb, num_tiles);
    cb_reserve_back(rotated_in_cb, num_tiles);

    // 1) rotate: DST[t] = matmul(q_pe[t], trans_mat) -> rotated_in.
    mm_init_short(q_pe_cb, trans_mat_cb);
    tile_regs_acquire();
    for (uint32_t t = 0U; t < num_tiles; ++t) {
        matmul_tiles(q_pe_cb, trans_mat_cb, t, 0U, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(rotated_in_cb);
    for (uint32_t t = 0U; t < num_tiles; ++t) {
        pack_tile(t, rotated_in_cb, t);
    }
    tile_regs_release();
    cb_push_back(rotated_in_cb, num_tiles);
    cb_wait_front(rotated_in_cb, num_tiles);

    // 2) DST[t] = rotated_in[t] * sin[t % Tr] -> sin_interm.
    cb_reserve_back(sin_interm_cb, num_tiles);
    ckernel::mul_tiles_init_with_dt(rotated_in_cb, sin_cb);
    tile_regs_acquire();
    for (uint32_t t = 0U; t < num_tiles; ++t) {
        mul_tiles(rotated_in_cb, sin_cb, t, t % Tr, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(sin_interm_cb);
    for (uint32_t t = 0U; t < num_tiles; ++t) {
        pack_tile(t, sin_interm_cb, t);
    }
    tile_regs_release();
    cb_push_back(sin_interm_cb, num_tiles);
    cb_pop_front(rotated_in_cb, num_tiles);
    cb_wait_front(sin_interm_cb, num_tiles);

    // fp32_dest_acc_en (DST_ACCUM_MODE): use add_tiles from CBs, not dest-reuse ELWADD.
    // Reproducer for LLK ELWADD dest-reuse gap under fp32 dest: comment out this entire
    // `if constexpr (DST_ACCUM_MODE)` block so fp32 kernels compile the `else` path below
    // (binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>). All QRopeFp32DestAccShapes tests
    // then fail with q_pe value mismatch; the non-fp32 path still passes. ELWMUL dest-reuse
    // has fp32-specific handling in the LLK; ELWADD does not (see move_d2b_fixed_face).
    if constexpr (DST_ACCUM_MODE) {
        // 3) DST[t] = q_pe[t] * cos[t % Tr] -> cos_interm.
        cb_reserve_back(cos_interm_cb, num_tiles);
        ckernel::mul_tiles_init_with_dt(q_pe_cb, cos_cb);
        tile_regs_acquire();
        for (uint32_t t = 0U; t < num_tiles; ++t) {
            mul_tiles(q_pe_cb, cos_cb, t, t % Tr, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cos_interm_cb);
        for (uint32_t t = 0U; t < num_tiles; ++t) {
            pack_tile(t, cos_interm_cb, t);
        }
        tile_regs_release();
        cb_push_back(cos_interm_cb, num_tiles);
        cb_pop_front(q_pe_cb, num_tiles);
        cb_wait_front(cos_interm_cb, num_tiles);

        // 4) DST[t] = cos_interm[t] + sin_interm[t] -> rope_out.
        add_tiles_init(cos_interm_cb, sin_interm_cb);
        tile_regs_acquire();
        for (uint32_t t = 0U; t < num_tiles; ++t) {
            add_tiles(cos_interm_cb, sin_interm_cb, t, t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(rope_out_cb);
        for (uint32_t tile = 0U; tile < num_tiles; tile += Tr) {
            cb_reserve_back(rope_out_cb, Tr);
            for (uint32_t j = 0U; j < Tr; ++j) {
                pack_tile(tile + j, rope_out_cb, j);
            }
            cb_push_back(rope_out_cb, Tr);
        }
        tile_regs_release();
        cb_pop_front(cos_interm_cb, num_tiles);
        cb_pop_front(sin_interm_cb, num_tiles);
    } else {
        // Fused combine via dest-reuse ELWADD (see reproducer comment above if fp32).
        // 3) DST[t] = q_pe[t] * cos[t % Tr]; DST[t] += sin_interm[t]  (one pack).
        tile_regs_acquire();
        ckernel::mul_tiles_init_with_dt(q_pe_cb, cos_cb);
        for (uint32_t t = 0U; t < num_tiles; ++t) {
            mul_tiles(q_pe_cb, cos_cb, t, t % Tr, t);
        }
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
            sin_interm_cb);
        for (uint32_t t = 0U; t < num_tiles; ++t) {
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                sin_interm_cb, t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(rope_out_cb);
        for (uint32_t tile = 0U; tile < num_tiles; tile += Tr) {
            cb_reserve_back(rope_out_cb, Tr);
            for (uint32_t j = 0U; j < Tr; ++j) {
                pack_tile(tile + j, rope_out_cb, j);
            }
            cb_push_back(rope_out_cb, Tr);
        }
        tile_regs_release();
        cb_pop_front(sin_interm_cb, num_tiles);
        cb_pop_front(q_pe_cb, num_tiles);
    }
}

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    mm_init(q_pe_cb, trans_mat_cb, rotated_in_cb);
    binary_op_init_common(rotated_in_cb, cos_cb, rope_out_cb);

    cb_wait_front(trans_mat_cb, 1U);

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        cb_wait_front(cos_cb, Tr);
        cb_wait_front(sin_cb, Tr);

        for (uint32_t head_base = 0U; head_base < kFullBatchHeads; head_base += kDstBatchHeads) {
            process_rope_group(kFullBatchTiles);
        }
        if constexpr (kTailHeads != 0U) {
            process_rope_group(kTailBatchTiles);
        }

        cb_pop_front(cos_cb, Tr);
        cb_pop_front(sin_cb, Tr);
    }

    cb_pop_front(trans_mat_cb, 1U);
}
