// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
// Blackhole-only: the GGUF decoders live only in the Blackhole LLK tree. The pack thread can run them too.
#if (defined(TRISC_MATH) || defined(TRISC_PACK)) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/llk_math_gguf_decode.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// The thread whose SFPU runs the GGUF decode. Pack: the decode of one tile overlaps the math thread's matmul of the
// one before (gguf_decode_overlap_*).
enum class GgufDecodeThread : std::uint8_t { Math, Pack };

#define GGUF_DECODE_ON(thr, ...)                   \
    if constexpr (thr == GgufDecodeThread::Math) { \
        MATH((__VA_ARGS__));                       \
    } else {                                       \
        PACK((__VA_ARGS__));                       \
    }

// clang-format off
/**
 * Sets up the SFPU of thread thr to decode GGUF weight tiles of format fmt (ckernel::gguf::Format,
 * experimental/ckernel_gguf_page.h). Call again after other SFPU code on that thread. Needs fp32_dest_acc_en.
 * beside_unpack: whether an unpack to DEST (e.g. the next page's copy) may run beside a decode; false selects faster
 * decoders that nothing may write DEST beside. The same value as the gguf_decode_tile calls'.
 */
// clang-format on
template <std::uint32_t fmt, GgufDecodeThread thr = GgufDecodeThread::Math, bool beside_unpack = true>
ALWI void gguf_decode_init() {
    static_assert(DST_ACCUM_MODE, "gguf_decode needs fp32_dest_acc_en: pages are UInt32 in DEST");
    GGUF_DECODE_ON(thr, llk_math_gguf_decode_init<fmt, beside_unpack>());
}

// Group step of the formats that convert a group's scales once (Q4_K, Q5_K, Q3_K, Q2_K; nothing for the others): call
// before the first tile of a group that gguf_decode_tile decodes at idst and rows, with any page of the group in tile
// idst + 1. Until the group ends, nothing else may write tile idst + 2 or tile idst + 1 past its page.
template <std::uint32_t fmt, GgufDecodeThread thr = GgufDecodeThread::Math>
ALWI void gguf_decode_group(const std::uint32_t idst, const std::uint32_t rows = 0) {
    GGUF_DECODE_ON(thr, llk_math_gguf_decode_group<fmt>(idst, rows));
}

// clang-format off
/**
 * Decodes one GGUF weight tile into a Float32 [K][N] tile in DEST, scales applied (ggml's dequantization). Q4_K,
 * Q5_K, Q3_K and Q2_K need gguf_decode_group for the tile's group first. On the pack thread only inside the overlap
 * protocol below.
 *
 * | Argument | Description                                                                                  | Type     | Valid Range              | Required |
 * |----------|----------------------------------------------------------------------------------------------|----------|--------------------------|----------|
 * | idst     | DEST tile of the output; its page must be in tile idst + 1 (UInt32), tile idst + 2 is scratch | uint32_t | DEST tiles               | True     |
 * | ti       | The tile's index in its group of K tiles (page_layout<fmt>().group_tiles)                    | uint32_t | 0 to group_tiles - 1     | True     |
 * | rows     | DEST row offset of the page inside tile idst + 1 (several pages copied as one tile)          | uint32_t | 0, 2 * 16 * j            | False    |
 */
// clang-format on
template <std::uint32_t fmt, GgufDecodeThread thr = GgufDecodeThread::Math, bool beside_unpack = true>
ALWI void gguf_decode_tile(const std::uint32_t idst, const std::uint32_t ti, const std::uint32_t rows = 0) {
    static_assert(thr == GgufDecodeThread::Math || beside_unpack, "the overlap protocol copies pages beside decodes");
    if constexpr (thr == GgufDecodeThread::Math) {
        MATH((llk_math_gguf_decode_tile<fmt, true, true, beside_unpack>(idst, ti, rows)));
    } else {
        // A phase 2 that stores through SFPLOADMACRO waits for the matmul of the tile before (such stores beside the
        // FPU's DEST writes are lost) and, where they are not arbitrated against the unpacker either, for the copy of
        // the next page (gguf_decode_overlap_body_ready). One with SFPSTOREs runs right after phase 1, but for Q8_0's,
        // which waits too (measured faster).
        PACK(({
            if constexpr (llk_math_gguf_decode_body_macro_stores<fmt>() || fmt == gguf::Q8_0) {
                llk_math_gguf_decode_tile<fmt, true, false>(idst, ti, rows);
                TTI_SEMWAIT(
                    p_stall::STALL_SFPU | p_stall::STALL_SYNC,
                    semaphore::t6_sem(semaphore::PACK_DONE),
                    p_stall::STALL_ON_ZERO);
                t6_semaphore_get<p_stall::NONE>(semaphore::PACK_DONE);
                llk_math_gguf_decode_tile<fmt, false, true>(idst, ti, rows);
            } else {
                llk_math_gguf_decode_tile<fmt>(idst, ti, rows);
            }
        }));
    }
}

// Float32 zeros in DEST tile idst (e.g. a dest_srca_mm accumulator), on the SFPU of thread thr.
template <GgufDecodeThread thr = GgufDecodeThread::Math>
ALWI void gguf_zero_tile(const std::uint32_t idst) {
    GGUF_DECODE_ON(thr, llk_math_gguf_zero_tile(idst));
}

#undef GGUF_DECODE_ON

// clang-format off
/**
 * Overlap of the pack thread's decode (GgufDecodeThread::Pack) with the math thread's dest_srca_mm, inside one
 * full-sync DEST acquire, with three Tensix semaphores, two as sdpa.h's FPU <-> SFPU handoff:
 *   PAGE = FPU_SFPU, math -> pack: page k is in DEST, and the DEST rows its decoded tile overwrites are free;
 *   TILE = UNPACK_MATH_DONE, pack -> math: decoded tile k is complete;
 *   BODY = PACK_DONE, math -> pack, where phase 2 stores through SFPLOADMACRO
 *   (llk_math_gguf_decode_body_macro_stores) and for Q8_0: phase 2 of tile k may start: the matmul of tile k - 1 is
 *   done, and so is the copy of page k + 1 where phase 2's stores are not arbitrated against the unpacker
 *   (llk_math_gguf_decode_body_unarbitrated).
 * Per K tile k, math:  overlap_body_ready<fmt, false>(); [dest_srca_mm_page_post(next)];
 *                      [page_wait(next); overlap_page_ready()]; overlap_body_ready<fmt, true>();
 *                      overlap_wait_tile(); dest_srca_mm_math(tile k)
 *            pack:     overlap_wait_page(); gguf_decode_tile<fmt, Pack>(k); overlap_tile_done()
 * Math hands page k + 1 over before it waits for tile k, so phase 1 of tile k + 1 overlaps the matmul of tile k. A
 * page the next tile also decodes from (several pages in one DEST tile) is handed over again after the matmul has
 * read the tile over it.
 * gguf_decode_overlap_acquire, right after tile_regs_acquire, marks every DEST row valid until the release, so the
 * same-cycle writes of the two threads cannot drop a row's valid bit.
 */
// clang-format on
ALWI void gguf_decode_overlap_init() {
    MATH((t6_semaphore_init(semaphore::FPU_SFPU, 0, 2)));
    MATH((t6_semaphore_init(semaphore::PACK_DONE, 0, 2)));
    PACK((t6_semaphore_init(semaphore::UNPACK_MATH_DONE, 0, 2)));
}

// MATH, after tile_regs_acquire: every DEST row valid (ZEROACC revert) until the packer clears them at the release.
ALWI void gguf_decode_overlap_acquire() { MATH((TTI_ZEROACC(p_zeroacc::CLR_ALL, 1, 1, ADDR_MOD_3, 0))); }

// MATH: the next page is in DEST and its output rows are free.
ALWI void gguf_decode_overlap_page_ready() { MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU))); }

// PACK: waits for a page; STALL_CFG holds the decoder's DEST offset write behind the wait too.
ALWI void gguf_decode_overlap_wait_page() {
    PACK((TTI_SEMWAIT(
        p_stall::STALL_SFPU | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::FPU_SFPU), p_stall::STALL_ON_ZERO)));
}

// PACK: the decoded tile is complete once the SFPU is done; hands it to the math thread.
ALWI void gguf_decode_overlap_tile_done() {
    PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
    PACK((t6_semaphore_post<p_stall::NONE>(semaphore::UNPACK_MATH_DONE)));
}

// MATH, twice per K tile: first (after_copy false) and after the copy of the next page, or where it would be (true).
// Where fmt's phase 2 waits for BODY (gguf_decode_tile), posts it at one of the two, once the math before it is done:
// at the second where its stores are not arbitrated against the unpacker.
template <std::uint32_t fmt, bool after_copy>
ALWI void gguf_decode_overlap_body_ready() {
    MATH(({
        if constexpr (
            (llk_math_gguf_decode_body_macro_stores<fmt>() || fmt == gguf::Q8_0) &&
            llk_math_gguf_decode_body_unarbitrated<fmt>() == after_copy) {
            t6_semaphore_post<p_stall::MATH>(semaphore::PACK_DONE);
        }
    }));
}

// MATH: waits for a decoded tile.
ALWI void gguf_decode_overlap_wait_tile() {
    MATH((TTI_SEMWAIT(
        p_stall::STALL_MATH | p_stall::STALL_SYNC,
        semaphore::t6_sem(semaphore::UNPACK_MATH_DONE),
        p_stall::STALL_ON_ZERO)));
    MATH((t6_semaphore_get<p_stall::NONE>(semaphore::UNPACK_MATH_DONE)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
