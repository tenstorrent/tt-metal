// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implemented based on bmm.cpp
#include "compute_kernel_api/matmul.h"

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh.h"
#include "debug_print.h"

using std::uint32_t;
ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;

    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);
    uint32_t transpose_a = get_compile_time_arg_val(4);
    uint32_t transpose_b = get_compile_time_arg_val(5);

    if (transpose_a || transpose_b) {
        uint32_t mm_src0 = (transpose_a) ? (tt::CB::c_intermed1) : (tt::CB::c_in0);
        uint32_t mm_src1 = (transpose_b) ? (tt::CB::c_intermed2) : (tt::CB::c_in1);

        mm_init(mm_src0, mm_src1);

        if (transpose_a)
            transpose_wh_init(tt::CB::c_in0);

        if (transpose_b)
            transpose_wh_init(tt::CB::c_in1);

        for (uint32_t nb = 0; nb < batch; nb++)
            for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C)      // output tile of C
                for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
                {
                    bool spill = Kt > 1;
                    bool enable_reload = false;
                    for (uint32_t kt = 0; kt < Kt; kt++) {
                        bool last_out = kt == (Kt - 1);
                        cb_wait_front(tt::CB::c_in0, onetile);
                        cb_wait_front(tt::CB::c_in1, onetile);

                        // transpose tile
                        if (transpose_a) {
                            ACQ();
                            transpose_wh_init_short(tt::CB::c_in0);
                            transpose_wh_tile(tt::CB::c_in0, 0, 0);
                            cb_reserve_back(mm_src0, onetile);
                            pack_tile(0, mm_src0);
                            cb_push_back(mm_src0, onetile);
                            REL();
                        }

                        if (transpose_b) {
                            ACQ();
                            transpose_wh_init_short(tt::CB::c_in1);
                            transpose_wh_tile(tt::CB::c_in1, 0, 0);
                            cb_reserve_back(mm_src1, onetile);
                            pack_tile(0, mm_src1);
                            cb_push_back(mm_src1, onetile);
                            REL();
                        }

                        // matmul tile
                        ACQ();
                        if (enable_reload) {
                            cb_wait_front(tt::CB::c_intermed0, onetile);
                            copy_tile_to_dst_init_short();
                            copy_tile(tt::CB::c_intermed0, 0, 0);
                            cb_pop_front(tt::CB::c_intermed0, onetile);
                        }

                        if (transpose_a)
                            cb_wait_front(mm_src0, onetile);
                        if (transpose_b)
                            cb_wait_front(mm_src1, onetile);

                        mm_init_short();
                        matmul_tiles(mm_src0, mm_src1, 0, 0, 0, false);

                        cb_pop_front(tt::CB::c_in0, onetile);
                        cb_pop_front(tt::CB::c_in1, onetile);

                        if (transpose_a)
                            cb_pop_front(mm_src0, onetile);
                        if (transpose_b)
                            cb_pop_front(mm_src1, onetile);

                        if (last_out) {
                            cb_reserve_back(tt::CB::c_out0, onetile);
                            pack_tile(0, tt::CB::c_out0);
                            cb_push_back(tt::CB::c_out0, onetile);
                        } else {
                            cb_reserve_back(tt::CB::c_intermed0, onetile);
                            pack_tile(0, tt::CB::c_intermed0);
                            cb_push_back(tt::CB::c_intermed0, onetile);
                        }
                        REL();

                        if (spill)
                            enable_reload = true;
                    }
                }
    } else {
        // the simplest possible version of outer product blocked matmul
        // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
        mm_init();
        for (uint32_t nb = 0; nb < batch; nb++)
            for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C)      // output tile of C
                for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
                {
                    ACQ();
                    for (uint32_t kt = 0; kt < Kt; kt++) {
                        cb_wait_front(tt::CB::c_in0, onetile);
                        cb_wait_front(tt::CB::c_in1, onetile);

                        matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0, false);

                        cb_pop_front(tt::CB::c_in0, onetile);
                        cb_pop_front(tt::CB::c_in1, onetile);
                    }

                    cb_reserve_back(tt::CB::c_out0, onetile);
                    pack_tile(0, tt::CB::c_out0);
                    cb_push_back(tt::CB::c_out0, onetile);
                    REL();
                }
    }
}
}  // namespace NAMESPACE
