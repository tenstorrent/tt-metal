// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#if defined(ARCH_BLACKHOLE)
#include "experimental/ckernel_gguf_page.h"
#endif
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

// Sets up thread thr's SFPU to decode format fmt (fp32_dest_acc_en); beside_unpack: whether an unpack to DEST may run
// beside a decode (false: faster decoders), as in the gguf_decode_tile calls.
template <std::uint32_t fmt, GgufDecodeThread thr = GgufDecodeThread::Math, bool beside_unpack = true>
ALWI void gguf_decode_init() {
    static_assert(DST_ACCUM_MODE, "gguf_decode needs fp32_dest_acc_en: pages are UInt32 in DEST");
    GGUF_DECODE_ON(thr, llk_math_gguf_decode_init<fmt, beside_unpack>());
}

// Group step (Q4_K, Q5_K, Q3_K, Q2_K): before a group's first tile, a page of the group in tile idst + 1; tiles idst + 2
// and idst + 1 past the page stay untouched until the group ends. delta: also to the tiles at idst + delta.
template <std::uint32_t fmt, GgufDecodeThread thr = GgufDecodeThread::Math, int delta = 0>
ALWI void gguf_decode_group(const std::uint32_t idst, const std::uint32_t rows = 0) {
    GGUF_DECODE_ON(thr, llk_math_gguf_decode_group<fmt, delta>(idst, rows));
}

// In place of gguf_decode_group at idst + delta: copies the group step's results at idst there.
template <std::uint32_t fmt, int delta, GgufDecodeThread thr = GgufDecodeThread::Math>
ALWI void gguf_decode_group_copy(const std::uint32_t idst, const std::uint32_t rows = 0) {
    GGUF_DECODE_ON(thr, llk_math_gguf_decode_group_copy<fmt, delta>(idst, rows));
}

// Formats whose phase 1 is the same for every tile of a group: a tile may keep the params another tile left.
template <std::uint32_t fmt>
constexpr bool gguf_decode_params_per_group = fmt == gguf::Q1_0 || fmt == gguf::TQ1_0;

// clang-format off
/**
 * Decodes one GGUF weight tile into a Float32 [K][N] tile in DEST (ggml's dequantization; the K formats after
 * gguf_decode_group). On the pack thread only inside the overlap protocol below.
 *
 * | Argument | Description                                                                                  | Type     | Valid Range              | Required |
 * |----------|----------------------------------------------------------------------------------------------|----------|--------------------------|----------|
 * | idst     | DEST tile of the output; its page must be in tile idst + 1 (UInt32), tile idst + 2 is scratch | uint32_t | DEST tiles               | True     |
 * | ti       | The tile's index in its group of K tiles (page_layout<fmt>().group_tiles)                    | uint32_t | 0 to group_tiles - 1     | True     |
 * | rows     | DEST row offset of the page inside tile idst + 1 (several pages copied as one tile)          | uint32_t | 0, 2 * 16 * j            | False    |
 *
 * params = false skips phase 1 for a gguf_decode_params_per_group format whose tile idst + 2 still holds the params.
 */
// clang-format on
template <
    std::uint32_t fmt,
    GgufDecodeThread thr = GgufDecodeThread::Math,
    bool beside_unpack = true,
    bool params = true>
ALWI void gguf_decode_tile(const std::uint32_t idst, const std::uint32_t ti, const std::uint32_t rows = 0) {
    static_assert(thr == GgufDecodeThread::Math || beside_unpack, "the overlap protocol copies pages beside decodes");
    static_assert(params || gguf_decode_params_per_group<fmt>, "phase 1 differs between the tiles of a group");
    if constexpr (thr == GgufDecodeThread::Math) {
        MATH((llk_math_gguf_decode_tile<fmt, params, true, beside_unpack>(idst, ti, rows)));
    } else {
        // An SFPLOADMACRO phase 2 (and Q8_0's) waits for BODY (gguf_decode_overlap_body_ready): its stores would be
        // lost beside the FPU's DEST writes or the unpacker's.
        PACK(({
            if constexpr (llk_math_gguf_decode_body_macro_stores<fmt>() || fmt == gguf::Q8_0) {
                if constexpr (params) {
                    llk_math_gguf_decode_tile<fmt, true, false>(idst, ti, rows);
                }
                TTI_SEMWAIT(
                    p_stall::STALL_SFPU | p_stall::STALL_SYNC,
                    semaphore::t6_sem(semaphore::PACK_DONE),
                    p_stall::STALL_ON_ZERO);
                t6_semaphore_get<p_stall::NONE>(semaphore::PACK_DONE);
                llk_math_gguf_decode_tile<fmt, false, true>(idst, ti, rows);
            } else {
                llk_math_gguf_decode_tile<fmt, params, true>(idst, ti, rows);
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
 * Overlap of the pack thread's decode with the math thread's dest_srca_mm in one full-sync acquire: semaphores PAGE
 * (FPU_SFPU, math -> pack), TILE (UNPACK_MATH_DONE, pack -> math) and BODY (PACK_DONE, math -> pack); see the calls.
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

// MATH, twice per K tile (before and after the next page's copy): posts BODY where fmt's phase 2 waits for it, after
// the copy where its stores are not arbitrated against the unpacker.
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
