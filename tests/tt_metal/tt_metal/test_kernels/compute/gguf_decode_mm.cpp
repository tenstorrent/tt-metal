// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GGUF decode and dest_srca_mm through the compute API for test_gguf_decode_mm.cpp; c_0 pages, c_1 bf16 in0 tiles,
// c_16 Float32 outputs (GGUF_MM 0: each decoded tile, 1: in0 times the decoded weights).

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/dest_srca_mm.h"
#include "api/compute/experimental/gguf_decode.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "experimental/ckernel_gguf_page.h"
#include "experimental/gguf_page.h"

namespace {

constexpr auto L = ckernel::gguf::page_layout<static_cast<ckernel::gguf::Format>(GGUF_FMT)>();
constexpr bool codes = ckernel::gguf::bfp_codes<GGUF_FMT>;
constexpr bool group_step = ckernel::gguf::group_scales<GGUF_FMT>;
constexpr auto thr = GGUF_OVERLAP ? ckernel::GgufDecodeThread::Pack : ckernel::GgufDecodeThread::Math;

// Page tile_index of c_0 to DEST tile page, codes (block-float formats) to the rows of tile out.
ALWI void page_to_dest(uint32_t tile_index, uint32_t page, uint32_t out) {
    if constexpr (codes) {
        ckernel::dest_srca_mm_page_to_dest_codes(
            tt::CBIndex::c_0,
            tile_index,
            page,
            out,
            ckernel::gguf::BFP_CODES,
            ckernel::gguf::BFP_CODE_EXP,
            L.scale_offset,
            ckernel::gguf::bfp_scale_words<GGUF_FMT>(),
            experimental::gguf::paired_words_shift<static_cast<experimental::gguf::Format>(GGUF_FMT)>(tile_index));
    } else {
        ckernel::dest_srca_mm_page_to_dest(tt::CBIndex::c_0, tile_index, page);
    }
}

// The group step at tile out for a group's first tile; GGUF_GROUP_COPY: also copied to the tiles 3 further.
ALWI void group(uint32_t out, bool both_sets) {
    if constexpr (group_step) {
        if (!both_sets) {
            ckernel::gguf_decode_group<GGUF_FMT, thr>(out);
        } else if constexpr (GGUF_GROUP_COPY) {
            ckernel::gguf_decode_group<GGUF_FMT, thr>(out);
            ckernel::gguf_decode_group_copy<GGUF_FMT, 3, thr>(out);
        } else {
            ckernel::gguf_decode_group<GGUF_FMT, thr, 3>(out);
        }
    }
}

}  // namespace

void kernel_main() {
    using namespace ckernel;
    const uint32_t kt_count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_page = tt::CBIndex::c_0;
    constexpr uint32_t cb_in0 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_page, cb_in0, cb_out);
    gguf_decode_init<GGUF_FMT, thr>();
    dest_srca_mm_init<GGUF_IN0_ROWS>(cb_in0, cb_page);
    cb_wait_front(cb_page, kt_count);
    cb_wait_front(cb_in0, kt_count);

#if GGUF_MM
    // Per K tile, DEST tiles 1 / 4 hold the decoded tile, 2 / 5 its page, 3 / 6 scratch, alternating; tile 0 sums.
#if GGUF_OVERLAP
    gguf_decode_overlap_init();
#endif
    tile_regs_acquire();
#if GGUF_OVERLAP
    gguf_decode_overlap_acquire();
#else
    gguf_zero_tile<thr>(0);
#endif
    page_to_dest(0, 2, 1);
#if GGUF_OVERLAP
    gguf_decode_overlap_page_ready();
#endif
    uint32_t set = 0;
    for (uint32_t kt = 0; kt < kt_count; kt++) {
        const uint32_t out = set ? 4 : 1;
        const uint32_t next_page = set ? 2 : 5;
        const uint32_t t = kt % L.group_tiles;
        const bool next = kt + 1 < kt_count;
        auto post_next = [&]() {
            if constexpr (codes) {
                dest_srca_mm_page_post_codes(next_page, set ? 1 : 4);
            } else {
                dest_srca_mm_page_post(next_page);
            }
        };
#if GGUF_OVERLAP
        gguf_decode_overlap_body_ready<GGUF_FMT, false>();
        if (GGUF_SPLIT && next) {
            post_next();
        }
        gguf_decode_overlap_wait_page();
        if (kt == 0) {
            gguf_zero_tile<thr>(0);
        }
        if (t == 0) {
            group(out, true);
        }
        gguf_decode_tile<GGUF_FMT, thr>(out, t);
        gguf_decode_overlap_tile_done();
#else
        if (t == 0) {
            group(out, true);
        }
        gguf_decode_tile<GGUF_FMT, thr>(out, t);
        // After the decode: the copy may not run beside a phase 2 with unarbitrated stores.
        if (GGUF_SPLIT && next) {
            post_next();
        }
#endif
        dest_srca_mm_unpack(cb_in0, kt);
        if (next) {
            if (GGUF_SPLIT) {
                if constexpr (codes) {
                    dest_srca_mm_page_unpack_codes(
                        cb_page,
                        kt + 1,
                        gguf::BFP_CODES,
                        gguf::BFP_CODE_EXP,
                        L.scale_offset,
                        gguf::bfp_scale_words<GGUF_FMT>(),
                        experimental::gguf::paired_words_shift<static_cast<experimental::gguf::Format>(GGUF_FMT)>(
                            kt + 1));
                } else {
                    dest_srca_mm_page_unpack(cb_page, kt + 1);
                }
                dest_srca_mm_page_wait(cb_page, next_page);
            } else {
                page_to_dest(kt + 1, next_page, set ? 1 : 4);
            }
#if GGUF_OVERLAP
            gguf_decode_overlap_page_ready();
#endif
        }
#if GGUF_OVERLAP
        gguf_decode_overlap_body_ready<GGUF_FMT, true>();
        gguf_decode_overlap_wait_tile();
#endif
        dest_srca_mm_math<GGUF_IN0_ROWS>(out, 0);
        set ^= 1;
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
#else
    // DEST tile 1 the decoded tile, 2 its page, 3 scratch; a half-sync DEST alternates its halves per K tile.
    for (uint32_t kt = 0; kt < kt_count; kt++) {
        tile_regs_acquire();
        page_to_dest(kt, 2, 1);
        group(1, false);
        gguf_decode_tile<GGUF_FMT, thr>(1, kt % L.group_tiles);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(1, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
#endif
    cb_pop_front(cb_page, kt_count);
    cb_pop_front(cb_in0, kt_count);
    dest_srca_mm_uninit();
}
