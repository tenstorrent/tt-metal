// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GGUF decode and dest_srca_mm through the compute API (gguf_decode.h, dest_srca_mm.h), for test_gguf_decode_mm.cpp.
// c_0: one UInt32 page per K tile, c_1: one bf16 in0 tile per K tile, c_16: Float32 output tiles.
//   GGUF_MM = 0: each page decoded and packed (K tiles of [K][N] weights).
//   GGUF_MM = 1: one output tile, the sum over K tiles of in0 * decoded weight tile. Per K tile, DEST tiles 1 / 4 hold
//                the decoded tile, 2 / 5 its page, 3 / 6 scratch, alternating; tile 0 accumulates.
//   GGUF_OVERLAP = 1: the pack thread decodes (gguf_decode_overlap_*).
//   GGUF_SPLIT = 1: the next page is copied while this one decodes (page_post / page_unpack / page_wait); 0: after it
//                   (page_to_dest).
// Runtime arg: K tiles. Defines: GGUF_FMT (one-page K tiles, groups of one), GGUF_IN0_ROWS.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/dest_srca_mm.h"
#include "api/compute/experimental/gguf_decode.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

void kernel_main() {
    using namespace ckernel;
    const uint32_t kt_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_page = tt::CBIndex::c_0;
    constexpr uint32_t cb_in0 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr auto thr = GGUF_OVERLAP ? GgufDecodeThread::Pack : GgufDecodeThread::Math;

    compute_kernel_hw_startup(cb_page, cb_in0, cb_out);
    gguf_decode_init<GGUF_FMT, thr>();
    dest_srca_mm_init<GGUF_IN0_ROWS>(cb_in0, cb_page);
    cb_wait_front(cb_page, kt_count);
    cb_wait_front(cb_in0, kt_count);

#if GGUF_MM
#if GGUF_OVERLAP
    gguf_decode_overlap_init();
#endif
    tile_regs_acquire();
#if GGUF_OVERLAP
    gguf_decode_overlap_acquire();
#else
    gguf_zero_tile<thr>(0);
#endif
    dest_srca_mm_page_to_dest(cb_page, 0, 2);
#if GGUF_OVERLAP
    gguf_decode_overlap_page_ready();
#endif
    uint32_t set = 0;
    for (uint32_t kt = 0; kt < kt_count; kt++) {
        const uint32_t out = set ? 4 : 1;
        const uint32_t next_page = set ? 2 : 5;
        const bool next = kt + 1 < kt_count;
        if (GGUF_SPLIT && next) {
            dest_srca_mm_page_post(next_page);
        }
#if GGUF_OVERLAP
        gguf_decode_overlap_wait_page();
        if (kt == 0) {
            gguf_zero_tile<thr>(0);
        }
        gguf_decode_tile<GGUF_FMT, thr>(out, 0);
        gguf_decode_overlap_tile_done();
#else
        gguf_decode_tile<GGUF_FMT, thr>(out, 0);
#endif
        dest_srca_mm_unpack(cb_in0, kt);
        if (next) {
            if (GGUF_SPLIT) {
                dest_srca_mm_page_unpack(cb_page, kt + 1);
                dest_srca_mm_page_wait(cb_page, next_page);
            } else {
                dest_srca_mm_page_to_dest(cb_page, kt + 1, next_page);
            }
#if GGUF_OVERLAP
            gguf_decode_overlap_page_ready();
#endif
        }
#if GGUF_OVERLAP
        gguf_decode_overlap_wait_tile();
#endif
        dest_srca_mm_math<GGUF_IN0_ROWS>(out, 0);
#if GGUF_OVERLAP
        gguf_decode_overlap_matmul_done();
#endif
        set ^= 1;
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
#else
    for (uint32_t kt = 0; kt < kt_count; kt++) {
        tile_regs_acquire();
        dest_srca_mm_page_to_dest(cb_page, kt, 1);
        gguf_decode_tile<GGUF_FMT, thr>(0, 0);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
#endif
    cb_pop_front(cb_page, kt_count);
    cb_pop_front(cb_in0, kt_count);
    dest_srca_mm_uninit();
}
