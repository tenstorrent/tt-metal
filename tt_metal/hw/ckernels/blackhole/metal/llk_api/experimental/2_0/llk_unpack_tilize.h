// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_unpack_tilize_api.h"  // legacy CB-id API + unified llk_unpack_tilize_init_impl / llk_unpack_tilize_impl
#include "data_format_derive.h"
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

/*************************************************************************
 * LLK UNPACK TILIZE -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API (llk_unpack_tilize_init / llk_unpack_tilize), distinguished by
 * taking an LLKMemDescriptor as the first NTTP. DESC carries the input buffer L1 format + tile geometry;
 * the SrcA register (dst) format is derived HERE from DESC.format + the fp32-dest-acc flag and never
 * exposed above the LLK. narrow_tile is derived from the tile geometry (a single face-column wide tile
 * is narrow). The runtime base address comes from the caller (cb_operand_helpers.h::cb_read_address).
 *************************************************************************/

template <ckernel::experimental::LLKMemDescriptor DESC, bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_init(const std::uint32_t ct_dim) {
    constexpr std::uint8_t RegFmt = ckernel::infer_unpack_reg_fmt(DESC.format, is_fp32_dest_acc_en);
    constexpr bool narrow_tile = (DESC.shape.num_faces_c_dim == 1);
    llk_unpack_tilize_init_impl(
        static_cast<std::uint32_t>(DESC.format),
        RegFmt,
        ct_dim,
        DESC.shape.face_r_dim,
        narrow_tile,
        DESC.shape.total_num_faces());
}

template <ckernel::experimental::LLKMemDescriptor DESC, bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize(const std::uint32_t base_address, const std::uint32_t tile_index) {
    constexpr std::uint8_t RegFmt = ckernel::infer_unpack_reg_fmt(DESC.format, is_fp32_dest_acc_en);
    constexpr bool narrow_tile = (DESC.shape.num_faces_c_dim == 1);
    llk_unpack_tilize_impl(
        base_address,
        tile_index,
        static_cast<std::uint32_t>(DESC.format),
        RegFmt,
        DESC.shape.face_r_dim,
        DESC.shape.total_num_faces(),
        narrow_tile);
}

template <ckernel::experimental::LLKMemDescriptor DESC, bool is_fp32_dest_acc_en = false>
inline void llk_unpack_tilize_uninit() {
    constexpr std::uint8_t RegFmt = ckernel::infer_unpack_reg_fmt(DESC.format, is_fp32_dest_acc_en);
    llk_unpack_tilize_uninit_impl(RegFmt, DESC.shape);
}
