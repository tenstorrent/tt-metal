// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_pack_untilize_api.h"  // legacy CB-id API + unified llk_pack_untilize_*_impl cores
#include "data_format_derive.h"
#include "api/compute/experimental/2_0/llk_mem_descriptor.h"

/*************************************************************************
 * LLK PACK UNTILIZE -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API (llk_pack_untilize_init / llk_pack_untilize / _uninit),
 * distinguished by taking an LLKMemDescriptor (the OUTPUT buffer L1 format + geometry) as the first NTTP.
 * The dest-register (pack src) format is derived HERE from DESC.format + the fp32-dest-acc flag and never
 * exposed above the LLK. All three forward to the same unified cores as the CB-id API. The runtime write
 * base pointer comes from the caller via llk_mem_descriptor.h::cb_write_address (absolute/out-of-order).
 *
 * ADDRESSING ASSUMPTION (documented at the compute layer, 2_0/pack_untilize.h): the id-free op has no CB
 * handle, so the per-tile-row output stride (page_stride) is derived from the output descriptor and assumes
 * fifo_page_size == a single tile's size. Exact for linear formats (Float32 / Float16 / int); for block
 * formats or padded/multi-tile pages it diverges (SCALE_DATUM_SIZE omits the shared-exponent bytes). The
 * current test path is single tile-row (block_rt_dim == 1), where the stride is never applied.
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
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(
        ckernel::infer_pack_src_format(static_cast<DataFormat>(OUT_DESC.format), is_fp32_dest_acc_en));
    llk_pack_untilize_init_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense>(
        RegFmt, OUT_DESC.format, OUT_DESC.shape.face_r_dim, OUT_DESC.shape.total_num_faces());
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
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(
        ckernel::infer_pack_src_format(static_cast<DataFormat>(OUT_DESC.format), is_fp32_dest_acc_en));
    // Per tile-row output stride, folded to a compile-time constant. Replaces the legacy
    // full_ct_dim * fifo_page_size; assumes fifo_page_size == a single tile's size (see header note).
    constexpr std::uint32_t page_stride =
        full_ct_dim *
        (SCALE_DATUM_SIZE(static_cast<std::uint32_t>(OUT_DESC.format), OUT_DESC.shape.total_tensor_size()) >> 4);
    llk_pack_untilize_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, tile_dst_ct_offset, dense>(
        block_rt_dim,
        base_ptr,
        RegFmt,
        OUT_DESC.format,
        page_stride,
        OUT_DESC.shape.face_r_dim,
        OUT_DESC.shape.total_num_faces(),
        block_c_index,
        tile_dst_rt_offset);
}

template <ckernel::experimental::LLKMemDescriptor OUT_DESC, bool is_fp32_dest_acc_en = false>
inline void llk_pack_untilize_uninit() {
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(
        ckernel::infer_pack_src_format(static_cast<DataFormat>(OUT_DESC.format), is_fp32_dest_acc_en));
    llk_pack_untilize_uninit_impl(RegFmt);
}
