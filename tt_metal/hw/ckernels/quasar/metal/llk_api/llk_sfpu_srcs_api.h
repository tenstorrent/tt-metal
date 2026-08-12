// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_srcs.h"

// SrcS unary SFPU pipeline: UNP_S -> SrcS -> SFPU -> PACK1 -> L1, on the ISOLATE_SFPU TRISC.
// SrcS index convention: input in the first slice (SFPU_SRCS_BASE_ADDR), result in the third
// (SFPU_SRCS_BASE_ADDR + 2 * YDIM). A 32x32 tile spans slice_count(mode) slices, each YDIM >> 1
// SFPU rows (SFP_ROWS == 2).
//
// TODO: only supports per-slice ops for now. Add SFPLOADMACRO support once BinarySFPU (#47469) is merged.

// Buffer descriptor ids are LLK-internal: callers never hold or pass one. The SrcS pipeline uses
// two fixed slots from the TRISC3 partition [24, 32) defined by the BFD re-architecture (#52762).
// TODO(#52762): allocate via bfd_alloc_and_program<BfdResource::UnpS/Pack1> once the allocator lands.
constexpr std::uint8_t SFPU_SRCS_BUF_DESC_ID_UNPACK = 24;
constexpr std::uint8_t SFPU_SRCS_BUF_DESC_ID_PACK = 25;

/**
 * @brief Configure the unary SrcS SFPU pipeline: unpack (UNP_S), pack (PACK1) and SFPU.
 *
 * Builds the unpack/pack buffer descriptors, programs the SrcS auto-loop for one tile and inits the
 * SFPU. SrcS geometry is derived from @p unpack_S_dst_format, not passed by the caller.
 *
 * @tparam INSTRN_COUNT: Unpack/pack instructions per SrcS auto-loop (see llk_srcs.h).
 * @param l1_in_addr_16B: L1 input address (16B units).
 * @param unpack_S_src_format: L1 input format.
 * @param unpack_S_dst_format: SrcS format (also selects 32-bit SrcS mode).
 * @param l1_out_addr_16B: L1 output address (16B units).
 * @param pack_S_src_format: SrcS source format for the packer.
 * @param pack_S_dst_format: L1 output format.
 * @param implied_math_format: When false, disables implied SrcS math format for this TRISC.
 */
template <std::uint8_t INSTRN_COUNT = 1>
inline void llk_sfpu_srcs_init(
    const std::uint32_t l1_in_addr_16B,
    const DataFormat unpack_S_src_format,
    const DataFormat unpack_S_dst_format,
    const std::uint32_t l1_out_addr_16B,
    const DataFormat pack_S_src_format,
    const DataFormat pack_S_dst_format,
    const bool implied_math_format) {
    const bool srcs_32bit_mode = ckernel::trisc::_is_srcs_32bit_mode_(unpack_S_dst_format);
    const std::uint32_t ydim = ckernel::trisc::srcs_dims::ydim(srcs_32bit_mode);

    // One SrcS slice = ydim rows x XDIM datums, single face (x=16, y=ydim, z=1).
    const ckernel::TensorShape srcs_shape = ckernel::make_tensor_shape(
        static_cast<std::uint8_t>(ydim), static_cast<std::uint8_t>(ckernel::trisc::srcs_dims::XDIM), 1, 1);

    // Unpack BD: L1 input -> SrcS
    const tdma_descriptor_t td_unpack = ckernel::trisc::construct_tdma_desc(
        srcs_shape,
        l1_in_addr_16B,
        static_cast<std::uint32_t>(unpack_S_src_format),
        SFPU_SRCS_BUF_DESC_ID_UNPACK,
        static_cast<std::uint32_t>(unpack_S_dst_format));
    ckernel::trisc::_configure_buf_desc_table_(td_unpack.buf_desc_id, td_unpack.buf_desc);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(td_unpack);

    // Pack BD: SrcS -> L1 output
    const tdma_descriptor_t td_pack = ckernel::trisc::construct_tdma_desc(
        srcs_shape,
        l1_out_addr_16B,
        static_cast<std::uint32_t>(pack_S_dst_format),
        SFPU_SRCS_BUF_DESC_ID_PACK,
        static_cast<std::uint32_t>(pack_S_src_format));
    ckernel::trisc::_configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK1, false>(td_pack, ckernel::ReluConfig::none());

    cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + ckernel::TRISC_ID] = !implied_math_format;

    _llk_unpack_srcs_config_for_tile_<INSTRN_COUNT>(srcs_32bit_mode);
    _llk_pack_srcs_config_for_tile_<INSTRN_COUNT>(srcs_32bit_mode);
    _llk_math_eltwise_sfpu_init_();
}

/**
 * @brief Run a unary SFPU op over num_tiles tiles on the SrcS path.
 *
 * Per tile: unpack to SrcS, pack from SrcS, then per slice invoke @p sfpu_op and clear the SrcS
 * valids. The load/store base addresses and per-slice row count are computed here and passed to
 * @p sfpu_op, so the op carries no SrcS bookkeeping.
 *
 * @tparam INSTRN_COUNT: Must match the value passed to llk_sfpu_srcs_init.
 * @tparam SfpuOp: Callable sfpu_op(int load_base_addr, int store_base_addr, int num_sfpu_iterations).
 * @param num_tiles: Number of 32x32 tiles to process.
 * @param unpack_S_dst_format: SrcS format used to derive geometry.
 * @param sfpu_op: Per-slice SFPU computation.
 */
template <std::uint8_t INSTRN_COUNT = 1, typename SfpuOp>
inline void llk_sfpu_srcs(const std::uint32_t num_tiles, const DataFormat unpack_S_dst_format, SfpuOp&& sfpu_op) {
    const bool srcs_32bit_mode = ckernel::trisc::_is_srcs_32bit_mode_(unpack_S_dst_format);
    const std::uint32_t ydim = ckernel::trisc::srcs_dims::ydim(srcs_32bit_mode);
    const std::uint32_t slice_count = ckernel::trisc::srcs_dims::slice_count(srcs_32bit_mode);

    const int num_sfpu_iterations = static_cast<int>(ydim >> 1);  // SFP_ROWS == 2
    const int load_base_addr = ckernel::math::SFPU_SRCS_BASE_ADDR;
    const int store_base_addr = ckernel::math::SFPU_SRCS_BASE_ADDR + 2 * static_cast<int>(ydim);

    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        _llk_unpack_srcs_<INSTRN_COUNT>(SFPU_SRCS_BUF_DESC_ID_UNPACK, i * slice_count);  // Sets dvalid for SFPU to read

        // Pack is issued before the SFPU loop: the SFPU loop fills the instruction buffer and can
        // clog it, leading to hangs if the pack is queued after.
        _llk_pack_srcs_<INSTRN_COUNT>(SFPU_SRCS_BUF_DESC_ID_PACK, i * slice_count);  // Sets dvalid for SFPU to write

        for (std::uint32_t slice = 0; slice < slice_count; slice++) {
            sfpu_op(load_base_addr, store_base_addr, num_sfpu_iterations);
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, true>();  // Clears dvalid for SFPU read and write
        }
    }
}
