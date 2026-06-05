// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"

// Largest pack_untilize block width (<= DEST tile capacity) dividing full_ct_dim.
constexpr uint32_t untilize_pack_block_ct(uint32_t full_ct_dim) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (full_ct_dim % bct == 0) {
            return bct;
        }
    }
    return 1;
}

ALWI void MUL_TILES(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles, uint32_t in1_idx) {
    // Multiply input by cos
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, in1_idx + 1);
    cb_reserve_back(out_cb, num_tiles);

#ifdef DECODE_MODE
    tile_regs_acquire();
    mul_bcast_rows_init_short(in0_cb, in1_cb);
    mul_tiles_bcast_rows(in0_cb, in1_cb, 0, in1_idx, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
// We don't pop in1 in decode which is sin/cos since we don't stream
#else
    tile_regs_acquire();
    mul_tiles_init(in0_cb, in1_cb);
    mul_tiles(in0_cb, in1_cb, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(in1_cb, num_tiles);
#endif
}

template <uint32_t num_tiles>
ALWI void UNTILIZE_TILES(uint32_t in0_cb, uint32_t out_cb) {
    constexpr uint32_t block_ct = untilize_pack_block_ct(num_tiles);
    constexpr uint32_t num_blocks = num_tiles / block_ct;
    pack_untilize_init<block_ct, num_tiles>(in0_cb, out_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        pack_untilize_block<block_ct, num_tiles>(in0_cb, 1, out_cb, b);
        cb_pop_front(in0_cb, block_ct);
    }
    cb_push_back(out_cb, num_tiles);
    pack_untilize_uninit(out_cb);
}

ALWI void TILIZE_ROWS(uint32_t in0_cb, uint32_t sync_cb, uint32_t out_cb, uint32_t num_tiles) {
    tilize_init(in0_cb, num_tiles, out_cb);
    cb_wait_front(sync_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    tilize_block(in0_cb, num_tiles, out_cb);
    cb_push_back(out_cb, num_tiles);

    // Pop shared cbs after tilize
    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(sync_cb, num_tiles);
    tilize_uninit(in0_cb, out_cb);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(4);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t half_Wt = get_compile_time_arg_val(11);

    binary_op_init_common(in_cb, cos_cb, out_cb);

    cb_wait_front(scalar_cb, onetile);

    uint32_t updated_cos_cb = cos_cb;
    uint32_t updated_sin_cb = sin_cb;

#ifdef DECODE_MODE
    constexpr uint32_t untilized_cos_cb = get_compile_time_arg_val(12);
    constexpr uint32_t untilized_cos_sync_cb = get_compile_time_arg_val(13);
    constexpr uint32_t untilized_sin_cb = get_compile_time_arg_val(14);
    constexpr uint32_t untilized_sin_sync_cb = get_compile_time_arg_val(15);
    constexpr uint32_t retilized_cos_cb = get_compile_time_arg_val(16);
    constexpr uint32_t retilized_sin_cb = get_compile_time_arg_val(17);
    UNTILIZE_TILES<Wt>(sin_cb, untilized_sin_cb);
    UNTILIZE_TILES<Wt>(cos_cb, untilized_cos_cb);
    TILIZE_ROWS(untilized_sin_cb, untilized_sin_sync_cb, retilized_sin_cb, Wt);
    TILIZE_ROWS(untilized_cos_cb, untilized_cos_sync_cb, retilized_cos_cb, Wt);
    updated_cos_cb = retilized_cos_cb;
    updated_sin_cb = retilized_sin_cb;
#endif
    uint32_t in1_idx = 0;
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = 0; j < Wt; j++) {
#ifdef DECODE_MODE
            in1_idx = j;
#endif
            if (j < half_Wt) {
                // Multiply half of the rotated input by scalar (-1)
                cb_wait_front(rotated_in_cb, onetile);
                cb_reserve_back(rotated_in_interm_cb, onetile);
                tile_regs_acquire();
                mul_tiles_bcast_scalar_init_short(rotated_in_cb, scalar_cb);
                mul_tiles_bcast_scalar(rotated_in_cb, scalar_cb, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, rotated_in_interm_cb);
                tile_regs_release();
                cb_push_back(rotated_in_interm_cb, onetile);
                cb_pop_front(rotated_in_cb, onetile);
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_interm_cb, updated_sin_cb, sin_interm_cb, onetile, in1_idx);
            } else {
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_cb, updated_sin_cb, sin_interm_cb, onetile, in1_idx);
            }

            // Multiply input by cos
            MUL_TILES(in_cb, updated_cos_cb, cos_interm_cb, onetile, in1_idx);

            // Add applied sin/cos tensors
            cb_wait_front(cos_interm_cb, onetile);
            cb_wait_front(sin_interm_cb, onetile);
            cb_reserve_back(out_cb, onetile);

            tile_regs_acquire();
            add_tiles_init(cos_interm_cb, sin_interm_cb);
            add_tiles(cos_interm_cb, sin_interm_cb, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb);
            tile_regs_release();
            cb_push_back(out_cb, onetile);
            cb_pop_front(cos_interm_cb, onetile);
            cb_pop_front(sin_interm_cb, onetile);
        }
    }
}
