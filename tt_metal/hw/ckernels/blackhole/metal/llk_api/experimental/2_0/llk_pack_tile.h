// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_pack_tile_api.h"  // legacy CB-id API + unified llk_pack_init_impl / llk_pack_impl
#include "llk_pack_rows_api.h"  // legacy CB-id row pack + raw _llk_pack_rows_ / get_pack_dest_max_tiles
#include "data_format_derive.h"
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

/*************************************************************************
 * LLK PACK -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API (llk_pack_init / llk_pack), distinguished by taking an
 * LLKMemDescriptor as the first NTTP. DESC carries the output buffer L1 format + geometry; the Dest
 * register format is derived HERE from DESC.format + the fp32-dest-acc flag and never exposed above the
 * LLK. Both overloads forward to the same unified cores as the CB-id API. The runtime write address
 * comes from the caller via cb_operand_helpers.h::cb_write_address (absolute/out-of-order) -- no fifo bookkeeping.
 *************************************************************************/

template <
    ckernel::experimental::LLKMemDescriptor DESC,
    bool is_fp32_dest_acc_en = false,
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(const std::uint32_t num_tiles = 1) {
    static_assert(
        pack_mode != PackMode::Tilize,
        "single-descriptor llk_pack_init cannot do PackMode::Tilize -- the 8-bit tilize workaround needs the "
        "input operand's format; use the two-descriptor (OUT_DESC, IN_DESC) overload.");
    constexpr std::uint8_t RegFmt = ckernel::infer_pack_reg_fmt(DESC.format, is_fp32_dest_acc_en);
    // is_input_8bit_format only affects the tilize workaround; irrelevant for PackMode::Default datacopy.
    constexpr bool is_input_8bit_format = false;
    llk_pack_init_impl<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        RegFmt,
        DESC.shape.face_r_dim,
        DESC.shape.total_col_dim(),
        DESC.shape.total_num_faces(),
        num_tiles,
        is_input_8bit_format);
}

// Tilize pack init (id-free): the packer format/geometry come from the OUTPUT descriptor, but the
// 8-bit tilize workaround depends on the INPUT (untilized) operand's L1 format -- so this overload takes
// BOTH descriptors. Distinguished from the single-DESC llk_pack_init above by the second LLKMemDescriptor
// NTTP. Mirrors the CB-id llk_pack_init<PackMode::Tilize>(ocb, num_tiles, icb).
template <
    ckernel::experimental::LLKMemDescriptor OUT_DESC,
    ckernel::experimental::LLKMemDescriptor IN_DESC,
    bool is_fp32_dest_acc_en = false,
    PackMode pack_mode = PackMode::Tilize,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(const std::uint32_t num_tiles = 1) {
    constexpr std::uint8_t RegFmt = ckernel::infer_pack_reg_fmt(OUT_DESC.format, is_fp32_dest_acc_en);
    constexpr bool is_input_8bit_format = IS_8BIT_FORMAT(static_cast<std::uint32_t>(IN_DESC.format));
    llk_pack_init_impl<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        RegFmt,
        OUT_DESC.shape.face_r_dim,
        OUT_DESC.shape.total_col_dim(),
        OUT_DESC.shape.total_num_faces(),
        num_tiles,
        is_input_8bit_format);
}

// DESC is required only to disambiguate this overload from the CB-id llk_pack (same runtime arg count);
// the pack op itself needs only the runtime write address (format/geometry were set at llk_pack_init).
template <
    ckernel::experimental::LLKMemDescriptor DESC,
    bool is_fp32_dest_acc_en = false,
    bool out_of_order_output = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t base_ptr) {
    llk_pack_impl<is_fp32_dest_acc_en, pack_mode>(tile_index, base_ptr);
}

// Id-free row pack: packs the configured number of rows (set via the CB-id llk_pack_rows_init, which is
// format-free) from DST[dst_index] to the absolute L1 address base_ptr in row-major order. The DESC NTTP
// disambiguates this overload from the CB-id llk_pack_rows (same runtime arg count) and carries the output
// L1 format/geometry -- but the row-pack HW path needs NO format register (formats were programmed at
// hw_startup/pack_init), so DESC is not read here. Mirrors the CB-id llk_pack_rows minus the CB->address
// resolution (base_ptr is supplied absolute, e.g. cb_write_address) and the CB-array format debug assert
// (not reproducible id-free), keeping the retained DST-capacity bound check.
template <ckernel::experimental::LLKMemDescriptor DESC>
inline void llk_pack_rows(std::uint32_t dst_index, std::uint32_t base_ptr) {
    LLK_ASSERT(
        (dst_index < get_pack_dest_max_tiles<DST_SYNC_MODE>()),
        "Dst tile exceeds packer destination capacity for the configured W-stride.");
    _llk_pack_rows_(dst_index, base_ptr);
}

// Id-free packer data-format reconfigure: reprogram the packer's src/dst format registers + tile geometry
// from an output LLKMemDescriptor, mirroring the CB-id llk_pack_reconfig_data_format(new_output). Unlike
// llk_pack_init (which does NOT touch the format registers -- only addrmod/mop/strides), this programs the
// out/in data formats, dest-read control and exp/section sizes. Used by the id-free pack_untilize init so
// the packer formats are correct regardless of the prior op (the CB-id path calls reconfig for the same
// reason). tile_size is the one-tile L1 size derived from the descriptor via tile_stride_words (the id-free
// stand-in for fifo_page_size): geometry-exact for linear formats, exp section included for block floats.
template <ckernel::experimental::LLKMemDescriptor DESC, bool is_fp32_dest_acc_en = false>
inline void llk_pack_reconfig_data_format() {
    constexpr std::uint8_t RegFmt = ckernel::infer_pack_reg_fmt(DESC.format, is_fp32_dest_acc_en);
    // tile_size in 16B words (fifo_page_size units) == a single tile's L1 size.
    constexpr std::uint32_t tile_size = ckernel::experimental::tile_stride_words(DESC.format, DESC.shape);
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(
        RegFmt,
        static_cast<std::uint32_t>(DESC.format),
        tile_size,
        DESC.shape.total_col_dim(),
        DESC.shape.total_num_faces(),
        false /* partial_face */);
}
