// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_pack_untilize_api.h"  // legacy CB-id API + unified llk_pack_untilize_*_impl cores
#include "data_format_derive.h"
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

/*************************************************************************
 * LLK PACK UNTILIZE -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API (llk_pack_untilize_init / llk_pack_untilize / _uninit),
 * distinguished by taking an LLKMemDescriptor (the OUTPUT buffer L1 format + geometry) as the first NTTP.
 * The dest-register (pack src) format is derived HERE from DESC.format + the fp32-dest-acc flag and never
 * exposed above the LLK. All three forward to the same unified cores as the CB-id API. The runtime write
 * base pointer comes from the caller via cb_operand_helpers.h::cb_write_address (absolute/out-of-order).
 *
 * ADDRESSING ASSUMPTION (documented at the compute layer, 2_0/pack_untilize.h): the id-free op has no CB
 * handle, so the per-tile-row output stride (page_stride) is derived from the output descriptor via
 * tile_stride_words == one tile's L1 size. Correct for linear formats (geometry-exact) and block floats
 * (exp section included); the remaining edge is padded/multi-tile pages, which no shipping factory uses.
 *************************************************************************/

template <
    ckernel::experimental::LLKMemDescriptor OUT_DESC,
    bool is_fp32_dest_acc_en = false,
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
inline void llk_pack_untilize_init() {
    constexpr std::uint8_t RegFmt = ckernel::infer_pack_reg_fmt(OUT_DESC.format, is_fp32_dest_acc_en);
    llk_pack_untilize_init_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(
        RegFmt,
        static_cast<std::uint32_t>(OUT_DESC.format),
        OUT_DESC.shape.face_r_dim,
        OUT_DESC.shape.total_num_faces());
}

template <
    ckernel::experimental::LLKMemDescriptor OUT_DESC,
    bool is_fp32_dest_acc_en = false,
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense = false>
inline void llk_pack_untilize(
    std::uint32_t block_rt_dim,
    std::uint32_t base_ptr,
    const std::uint32_t block_c_index = 0,
    const std::uint32_t tile_dst_rt_offset = 0) {
    constexpr std::uint8_t RegFmt = ckernel::infer_pack_reg_fmt(OUT_DESC.format, is_fp32_dest_acc_en);
    // Per tile-row output stride, folded to a compile-time constant. Replaces the legacy
    // full_ct_dim * fifo_page_size; tile_stride_words is the one-tile L1 size (exp section included for BFP).
    constexpr std::uint32_t page_stride =
        full_ct_dim * ckernel::experimental::tile_stride_words(OUT_DESC.format, OUT_DESC.shape);
    llk_pack_untilize_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, tile_dst_ct_offset, dense>(
        block_rt_dim,
        base_ptr,
        RegFmt,
        static_cast<std::uint32_t>(OUT_DESC.format),
        page_stride,
        OUT_DESC.shape.face_r_dim,
        OUT_DESC.shape.total_num_faces(),
        block_c_index,
        tile_dst_rt_offset);
}

template <ckernel::experimental::LLKMemDescriptor OUT_DESC, bool is_fp32_dest_acc_en = false>
inline void llk_pack_untilize_uninit() {
    constexpr std::uint8_t RegFmt = ckernel::infer_pack_reg_fmt(OUT_DESC.format, is_fp32_dest_acc_en);
    llk_pack_untilize_uninit_impl(RegFmt);
}
