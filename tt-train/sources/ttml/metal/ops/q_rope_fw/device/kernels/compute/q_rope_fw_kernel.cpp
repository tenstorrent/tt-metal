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

// Compute kernel for the Q RoPE forward op. Applies rotary position embeddings to the rope part of
// each Q head while passing the nope part through unchanged.
//
// Q is split per head into [q_nope (Tn tiles) | q_pe (Tr tiles)] (Th = Tn + Tr). For every
// (batch, sequence-tile) block this kernel processes all n_heads heads:
//   - q_nope: copied straight to the output, no rotation.
//   - q_pe:   rotated via RoPE, computed as  out = (q_pe * cos) + (rotate(q_pe) * sin), where
//             rotate(q_pe) is a matmul of q_pe with the constant trans_mat (the "swap/negate halves"
//             matrix) and cos/sin are the per-sequence-tile caches (Tr tiles each, shared by all heads).
//
// When Tn exceeds the fp32/bf16 nope DST limit, compile with Q_ROPE_CHUNKED_NOPE and copy nope in
// Q_ROPE_NOPE_CHUNK_TILES batches (4 when fp32 dest acc is on, 8 when off). Tr is always <= 8
// (qk_rope_dim <= 256), so q_pe is always processed in one DST batch.
//
// cos/sin are read once per block by the reader and reused across heads; trans_mat is loaded once for
// the whole kernel. Intermediate products use scratch CBs (rotated_in / sin_interm / cos_interm).
void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Tn = get_compile_time_arg_val(8);
    constexpr uint32_t Tr = get_compile_time_arg_val(9);
    constexpr uint32_t Th = get_compile_time_arg_val(10);
    constexpr uint32_t n_heads = get_compile_time_arg_val(11);

    // Matmul packs to rotated_in_interm_cb; out_cb is filled via copy_tile (nope) and add_tiles (rope).
    mm_init(in_cb, trans_mat_cb, rotated_in_interm_cb);
    binary_op_init_common(rotated_in_interm_cb, cos_cb, out_cb);

    cb_wait_front(trans_mat_cb, 1U);

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        cb_wait_front(cos_cb, Tr);
        cb_wait_front(sin_cb, Tr);

        for (uint32_t head = 0U; head < n_heads; ++head) {
            cb_wait_front(in_cb, Th);
            cb_reserve_back(rotated_in_interm_cb, Tr);
            cb_reserve_back(sin_interm_cb, Tr);
            cb_reserve_back(cos_interm_cb, Tr);
            cb_reserve_back(out_cb, Th);

#ifdef Q_ROPE_CHUNKED_NOPE
            constexpr uint32_t kNopeChunkTiles = Q_ROPE_NOPE_CHUNK_TILES;
            for (uint32_t tile = 0U; tile < Tn; tile += kNopeChunkTiles) {
                const uint32_t chunk = (tile + kNopeChunkTiles <= Tn) ? kNopeChunkTiles : (Tn - tile);
                copy_tile_to_dst_init_short(in_cb);
                tile_regs_acquire();
                for (uint32_t j = 0U; j < chunk; ++j) {
                    copy_tile(in_cb, tile + j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(out_cb);
                for (uint32_t j = 0U; j < chunk; ++j) {
                    pack_tile(j, out_cb, tile + j);
                }
                tile_regs_release();
            }
#else
            copy_tile_to_dst_init_short(in_cb);
            tile_regs_acquire();
            for (uint32_t j = 0U; j < Tn; ++j) {
                copy_tile(in_cb, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(out_cb);
            for (uint32_t j = 0U; j < Tn; ++j) {
                pack_tile(j, out_cb, j);
            }
            tile_regs_release();
#endif

            // q_pe: rotate via matmul with trans_mat -> rotated_in_interm_cb (Tr <= 8, single DST batch).
            mm_init_short(in_cb, trans_mat_cb);
            tile_regs_acquire();
            for (uint32_t j = 0U; j < Tr; ++j) {
                const uint32_t in_tile = Tn + j;
                matmul_tiles(in_cb, trans_mat_cb, in_tile, 0U, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(rotated_in_interm_cb);
            for (uint32_t j = 0U; j < Tr; ++j) {
                pack_tile(j, rotated_in_interm_cb, j);
            }
            tile_regs_release();
            cb_push_back(rotated_in_interm_cb, Tr);
            cb_wait_front(rotated_in_interm_cb, Tr);

            // rotated_q_pe * sin -> sin_interm_cb.
            mul_tiles_init(rotated_in_interm_cb, sin_cb);
            tile_regs_acquire();
            for (uint32_t j = 0U; j < Tr; ++j) {
                mul_tiles(rotated_in_interm_cb, sin_cb, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(sin_interm_cb);
            for (uint32_t j = 0U; j < Tr; ++j) {
                pack_tile(j, sin_interm_cb, j);
            }
            tile_regs_release();
            cb_push_back(sin_interm_cb, Tr);
            cb_pop_front(rotated_in_interm_cb, Tr);

            // q_pe * cos -> cos_interm_cb.
            tile_regs_acquire();
            mul_tiles_init(in_cb, cos_cb);
            for (uint32_t j = 0U; j < Tr; ++j) {
                const uint32_t in_tile = Tn + j;
                mul_tiles(in_cb, cos_cb, in_tile, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cos_interm_cb);
            for (uint32_t j = 0U; j < Tr; ++j) {
                pack_tile(j, cos_interm_cb, j);
            }
            tile_regs_release();
            cb_push_back(cos_interm_cb, Tr);
            cb_pop_front(in_cb, Th);

            cb_wait_front(sin_interm_cb, Tr);
            cb_wait_front(cos_interm_cb, Tr);
            // cos_interm + sin_interm -> out_cb (final RoPE result for q_pe).
            add_tiles_init(cos_interm_cb, sin_interm_cb);
            tile_regs_acquire();
            for (uint32_t j = 0U; j < Tr; ++j) {
                add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(out_cb);
            for (uint32_t j = 0U; j < Tr; ++j) {
                const uint32_t out_tile = Tn + j;
                pack_tile(j, out_cb, out_tile);
            }
            tile_regs_release();
            cb_push_back(out_cb, Th);
            cb_pop_front(sin_interm_cb, Tr);
            cb_pop_front(cos_interm_cb, Tr);
        }

        cb_pop_front(cos_cb, Tr);
        cb_pop_front(sin_cb, Tr);
    }

    cb_pop_front(trans_mat_cb, 1U);
}
