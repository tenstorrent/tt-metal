// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tt-metalium/constants.hpp>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Compute kernel: full-FPU scaled-accumulate-untilize.
//
// Per chunk (one block):
//   A. tilize cb_existing_rm (row-major, TILE_HEIGHT row-pages) → cb_existing_tile
//   B. mul cb_src * cb_w COL-broadcast in DST, add cb_existing_tile with dest reuse → cb_combined
//   C. untilize cb_combined → cb_out0 (row-major)

void kernel_main() {
    constexpr uint32_t cb_id_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_existing_rm = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_existing_tile = get_compile_time_arg_val(5);
    constexpr uint32_t cb_id_combined = get_compile_time_arg_val(6);
    constexpr uint32_t cb_id_ctrl = get_compile_time_arg_val(7);
    constexpr uint32_t tile_row_blocks_per_chunk = 1U;

    binary_op_init_common(cb_id_src, cb_id_w, cb_id_combined);

    // NCRISC reader publishes active_steps * num_chunks after reading offsets.
    // Each iteration below processes one tiles_per_chunk chunk.
    // read_tile_value uses the unpack→math/pack mailbox so all three threads
    // see the same value before the loop. Replaces the CT worst-case bound.
    cb_wait_front(cb_id_ctrl, 1U);
    uint32_t total_work_chunks = read_tile_value(cb_id_ctrl, 0U, 0U);
    cb_pop_front(cb_id_ctrl, 1U);

    for (uint32_t work_chunk = 0; work_chunk < total_work_chunks; ++work_chunk) {
        // ---- Phase A: tilize existing_rm → existing_tile ----
        compute_kernel_lib::tilize<
            tiles_per_chunk,
            cb_id_existing_rm,
            cb_id_existing_tile,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(
            tile_row_blocks_per_chunk, tt::constants::TILE_HEIGHT);

        // ---- Phase B: mul cb_src * broadcast(cb_w col 0), add existing_tile in DST → cb_combined ----
        cb_wait_front(cb_id_src, tiles_per_chunk);
        cb_wait_front(cb_id_w, 1U);
        cb_wait_front(cb_id_existing_tile, tiles_per_chunk);
        cb_reserve_back(cb_id_combined, tiles_per_chunk);

        for (uint32_t i = 0; i < tiles_per_chunk; ++i) {
            tile_regs_acquire();
            mul_bcast_cols_init_short(cb_id_src, cb_id_w);
            mul_tiles_bcast_cols(cb_id_src, cb_id_w, i, 0U, 0U);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_id_existing_tile);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_id_existing_tile, i, 0U);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0U, cb_id_combined, i);
            tile_regs_release();
        }

        cb_push_back(cb_id_combined, tiles_per_chunk);
        cb_pop_front(cb_id_src, tiles_per_chunk);
        cb_pop_front(cb_id_w, 1U);
        cb_pop_front(cb_id_existing_tile, tiles_per_chunk);

        // ---- Phase C: untilize cb_combined → cb_out ----
        compute_kernel_lib::untilize<
            tiles_per_chunk,
            cb_id_combined,
            cb_id_out,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(
            tile_row_blocks_per_chunk);
    }
}
