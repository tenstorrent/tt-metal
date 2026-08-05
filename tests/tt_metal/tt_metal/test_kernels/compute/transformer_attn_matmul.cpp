// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"

using std::uint32_t;

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

// Untilize `full_ct_dim` tiles from icb to ocb using pack_untilize (replaces the removed
// unpack-based untilize op). Handles the full cb hand-off (wait/reserve/pop/push).
template <uint32_t full_ct_dim>
ALWI void untilize_to_cb(uint32_t icb, uint32_t ocb) {
    constexpr uint32_t block_ct = untilize_pack_block_ct(full_ct_dim);
    constexpr uint32_t num_blocks = full_ct_dim / block_ct;
    pack_untilize_init<block_ct, full_ct_dim>(icb, ocb);
    cb_wait_front(icb, full_ct_dim);
    cb_reserve_back(ocb, full_ct_dim);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        pack_untilize_block<block_ct, full_ct_dim>(icb, 1, ocb, b);
        cb_pop_front(icb, block_ct);
    }
    cb_push_back(ocb, full_ct_dim);
    pack_untilize_uninit(ocb);
}

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t transpose_hw = get_compile_time_arg_val(0);
    uint32_t batch = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_intermed0 = 24;
    constexpr uint32_t cb_intermed1 = 25;
    constexpr uint32_t cb_intermed2 = 26;
    constexpr uint32_t out_cb_id = 16;

    constexpr uint32_t num_rows_in_one_tile = 32;

    compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_0, tt::CBIndex::c_1, out_cb_id);
    matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1, transpose_hw);

    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {    // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
                    tile_regs_acquire();
                    for (uint32_t kt = 0; kt < Kt; kt++) {
                        if (tile_row_id == 0) {
                            cb_wait_front(tt::CBIndex::c_0, kt + 1);
                        }
                        cb_wait_front(tt::CBIndex::c_1, onetile);

                        matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, kt, 0, 0);

                        cb_pop_front(tt::CBIndex::c_1, onetile);
                    }

                    tile_regs_commit();
                    tile_regs_wait();

                    cb_reserve_back(cb_intermed0, onetile);
                    pack_tile(0, cb_intermed0);
                    tile_regs_release();
                    cb_push_back(cb_intermed0, onetile);

                    // untilize tile and write to CBIndex::c_25
                    untilize_to_cb<onetile>(cb_intermed0, cb_intermed1);

                    matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1, transpose_hw);
                }
                cb_pop_front(tt::CBIndex::c_0, Kt);

                // cb_intermed2 comes from reader; untilized row-major tile
                cb_wait_front(cb_intermed2, 1);
                cb_reserve_back(tt::CBIndex::c_16, onetile);

                // tilize CB::intermed2 and write to CBIndex::c_16
                tilize_init(cb_intermed2, 1, out_cb_id);
                tilize_block(cb_intermed2, 1, out_cb_id);
                cb_push_back(out_cb_id, 1);

                cb_pop_front(cb_intermed2, 1);
                tilize_uninit(cb_intermed2, out_cb_id);

                matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1, transpose_hw);
            }
        }
    }
}
