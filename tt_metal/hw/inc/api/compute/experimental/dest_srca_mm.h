// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
// Blackhole-only: the dest_srca_mm LLKs live only in the Blackhole llk_api / llk_lib trees.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_dest_srca_mm_api.h"
#endif
#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "llk_unpack_A_api.h"
#include "experimental/llk_unpack_dest_srca_mm_api.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// clang-format off
/**
 * Matmul whose in1 is a Float32 tile in DEST instead of a CB: D[dst_idst] += in0 * DEST tile src_idst, where in1 is
 * moved to SrcA by MOVD2A (truncated to TF32) and in0 comes from a CB. For an in1 produced in DEST, e.g. a GGUF weight
 * tile decoded by gguf_decode_tile (gguf_decode.h), without a round trip through L1.
 *
 * in0: [in0_rows, 32] tiles, bf16; in0_rows 8 or 16 take the top half of a [16, 32] or [32, 32] tile.
 * in1: one [32, 32] Float32 DEST tile. Output: a [in0_rows, 32] block of a Float32 DEST tile. Needs fp32_dest_acc_en.
 *
 * The page_* calls bring a 32-bit tile of a CB (e.g. a GGUF page, UInt32) into DEST unchanged. page_to_dest does it in
 * one step; page_post / page_unpack / page_wait split it so that math work can run while the unpacker copies, and a
 * kernel orders each of them on its own thread. A copy must not overlap dest_srca_mm: wait for it first.
 *
 * Call after compute_kernel_hw_startup(page_cb, in0_cb, out_cb); dest_srca_mm_uninit restores the math state.
 *
 * | Argument | Description                                                    | Type     | Valid Range | Required |
 * |----------|----------------------------------------------------------------|----------|-------------|----------|
 * | in0_rows | Rows of in0 the matmul computes                                | uint32_t | 8, 16, 32   | False    |
 * | in0_cb   | CB of the in0 tiles                                            | uint32_t | 0 to 31     | True     |
 * | page_cb  | CB of the 32-bit tiles copied to DEST                          | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <std::uint32_t in0_rows = 32>
ALWI void dest_srca_mm_init(const std::uint32_t in0_cb, const std::uint32_t page_cb) {
    static_assert(DST_ACCUM_MODE, "dest_srca_mm needs fp32_dest_acc_en");
    UNPACK(
        (llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(false, false, page_cb)));
    UNPACK((llk_unpack_dest_srca_mm_init(in0_cb)));
    MATH((llk_math_dest_srca_mm_init<in0_rows>()));
}

// Re-enables implied SrcA formats for other math after dest_srca_mm.
ALWI void dest_srca_mm_uninit() { MATH((llk_math_dest_srca_mm_uninit())); }

// MATH: once earlier math is done, hands DEST tile idst to the unpacker.
ALWI void dest_srca_mm_page_post(const std::uint32_t idst) { MATH((llk_math_dest_srca_mm_page_post(idst))); }

// UNPACK: tile tile_index of page_cb into the DEST tile posted by dest_srca_mm_page_post.
ALWI void dest_srca_mm_page_unpack(const std::uint32_t page_cb, const std::uint32_t tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(page_cb, tile_index)));
}

// MATH: waits for the copy into DEST tile idst.
ALWI void dest_srca_mm_page_wait(const std::uint32_t page_cb, const std::uint32_t idst) {
    MATH((llk_math_dest_srca_mm_page_wait(page_cb, idst)));
}

// Tile tile_index of page_cb into DEST tile idst, unchanged.
ALWI void dest_srca_mm_page_to_dest(
    const std::uint32_t page_cb, const std::uint32_t tile_index, const std::uint32_t idst) {
    dest_srca_mm_page_post(idst);
    dest_srca_mm_page_unpack(page_cb, tile_index);
    dest_srca_mm_page_wait(page_cb, idst);
}

// UNPACK half of dest_srca_mm: SrcA handed to the math thread, then in0 tile in0_tile_index of in0_cb to SrcB.
ALWI void dest_srca_mm_unpack(const std::uint32_t in0_cb, const std::uint32_t in0_tile_index) {
    UNPACK((llk_unpack_dest_srca_mm(in0_cb, in0_tile_index)));
}

// MATH half of dest_srca_mm. src_rows: DEST rows past tile src_idst where in1 starts.
template <std::uint32_t in0_rows = 32>
ALWI void dest_srca_mm_math(
    const std::uint32_t src_idst, const std::uint32_t dst_idst, const std::uint32_t src_rows = 0) {
    MATH((llk_math_dest_srca_mm<in0_rows>(src_idst, dst_idst, src_rows)));
}

// D[dst_idst] += in0 tile in0_tile_index of in0_cb * DEST tile src_idst (from src_rows rows past it).
template <std::uint32_t in0_rows = 32>
ALWI void dest_srca_mm(
    const std::uint32_t in0_cb,
    const std::uint32_t in0_tile_index,
    const std::uint32_t src_idst,
    const std::uint32_t dst_idst,
    const std::uint32_t src_rows = 0) {
    dest_srca_mm_unpack(in0_cb, in0_tile_index);
    dest_srca_mm_math<in0_rows>(src_idst, dst_idst, src_rows);
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
