// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_assert.h"
#include "llk_pack_common_api.h"
#include "llk_pack_rows.h"
#include "tensor_shape.h"

/*************************************************************************
 * LLK PACK ROWS
 *************************************************************************/

// Reinit-guard id for the row-pack descriptor. It is not a real DFB, so a DFB pack execute after a row pack without
// its own pack init asserts, and so does a row-pack execute after another op's pack init.
inline constexpr std::uint32_t pack_rows_reinit_guard_id = dfb::NUM_DFBS;

/**
 * @brief Initialize packer0 to pack rows from the destination register to an arbitrary L1 address
 *
 * @param num_rows Number of 16-datum rows to pack, values = [1, 64]
 *
 * There is no output operand, so the L1 format is taken from the current pack BFD, i.e. the output of the last
 * pack init. Unlike WH/BH, pack_reconfig_data_format does not change it: call pack_init(output) first to switch the
 * row format. MX output formats are not supported. Allocates a BFD of its own, because each llk_pack_rows_to_addr
 * call rewrites the BFD's L1 base.
 */
inline void llk_pack_rows_init(const std::uint32_t num_rows) {
    const DataFormat l1_format =
        ckernel::trisc::_get_buf_desc_l1_format_(ckernel::trisc::bfd_current<pack_bfd_resource>());
    // MX formats pack two rows per exponent block and write a separate exponent section, so they would write outside
    // the requested rows.
    LLK_ASSERT(
        l1_format != DataFormat::MxFp8R && l1_format != DataFormat::MxFp8P && l1_format != DataFormat::MxFp6R &&
            l1_format != DataFormat::MxFp6P && l1_format != DataFormat::MxFp4 && l1_format != DataFormat::MxInt8 &&
            l1_format != DataFormat::MxInt4 && l1_format != DataFormat::MxInt2,
        "pack rows does not support MX output formats");
    // A 4-bit row is 8B, half of the 16B unit l1_addr is given in, so an odd row count ends inside a unit.
    LLK_ASSERT(
        l1_format != DataFormat::Int4 && l1_format != DataFormat::UInt4,
        "pack rows does not support 4-bit output formats");
    // The L1 base is set on every llk_pack_rows_to_addr call.
    const std::uint8_t bfd_id = ckernel::trisc::bfd_alloc_and_program<pack_bfd_resource>(
        ckernel::DEFAULT_TENSOR_SHAPE, 0 /*l1_base_addr*/, static_cast<std::uint32_t>(l1_format));
    LLK_REINIT_GUARD_NOTE_PROGRAMMED(pack_bfd_resource, pack_rows_reinit_guard_id);
    _llk_pack_rows_init_(bfd_id, num_rows);
}

/**
 * @brief Pack rows from the destination register to an arbitrary L1 address
 *
 * @param dst_index Index of the tile in the destination register to pack from
 * @param l1_addr L1 address to pack to, in 16B units
 *
 * Packs the rows set up by llk_pack_rows_init. Bypasses the DFB write pointer, so only the packed rows are written.
 * llk_pack_rows_init must be called again after any other pack init. These PACRs name no DFB, so they do not meet
 * the TEN-4746 ordering for a DFB push: when the rows patch a reserved DFB, call llk_pack_dummy(dfb) before its push.
 */
inline void llk_pack_rows_to_addr(const std::uint32_t dst_index, const std::uint32_t l1_addr) {
    LLK_REINIT_GUARD_ASSERT_MATCHES(
        pack_bfd_resource,
        pack_rows_reinit_guard_id,
        "llk_pack_rows_to_addr without llk_pack_rows_init after the last pack init");
    LLK_ASSERT(
        (dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
        "Dst tile index exceeds the destination register capacity");
    _llk_pack_rows_(ckernel::trisc::bfd_current<pack_bfd_resource>(), dst_index, l1_addr);
}

/**
 * @brief Uninitialize the pack rows operation
 *
 * Clears the packer counters the row pack leaves part way through a tile. Call a pack init before the next DFB pack.
 */
inline void llk_pack_rows_uninit() { _llk_pack_rows_uninit_(); }
