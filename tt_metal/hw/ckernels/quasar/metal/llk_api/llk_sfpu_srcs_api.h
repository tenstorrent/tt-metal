// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_bfd_alloc.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_srcs.h"

// SrcS SFPU pipeline: UNP_S -> SrcS -> SFPU -> PACK1 -> L1, on the ISOLATE_SFPU TRISC.
// SrcS index convention: inputs in the first slice (SFPU_SRCS_BASE_ADDR) and, for binary ops, the
// second (SFPU_SRCS_BASE_ADDR + YDIM); result in the third (SFPU_SRCS_BASE_ADDR + 2 * YDIM). A
// 32x32 tile spans slice_count(mode) slices, each YDIM >> 1 SFPU rows (SFP_ROWS == 2).
//
// Unary drives UNP_S from the auto-loop: one UNPACR2 per tile, replayed in HW. Issuing one per
// slice instead deadlocks the instruction buffer at 16 slices (32-bit SrcS). Binary has to issue
// per slice (auto-loop cannot alternate descriptors, tt-llk #1635) so it bakes both ids into a MOP.
// Either way this is one UNP_S engine with two table rows, not UNP_A+UNP_B.
//
// TODO(#52522): integrate the SFPLOADMACRO fast path (merged on main) into the wrapper. Only
// per-slice ops are supported for now.

// Allocates and programs the PACK1 buffer descriptor viewing the L1 output as SrcS slices.
inline void llk_sfpu_srcs_configure_pack_impl(
    const ckernel::TensorShape& srcs_shape,
    const std::uint32_t l1_addr_16B,
    const DataFormat srcs_format,
    const DataFormat l1_format) {
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack1>(
        srcs_shape, l1_addr_16B, static_cast<std::uint32_t>(l1_format));
    _llk_pack_hw_configure_<p_pacr::PACK1, false>(srcs_format, ckernel::ReluConfig::none());
}

// One SrcS slice = ydim rows x XDIM datums, single face (x=16, y=ydim, z=1).
inline ckernel::TensorShape llk_sfpu_srcs_slice_shape_impl(const std::uint32_t ydim) {
    return ckernel::make_tensor_shape(
        static_cast<std::uint8_t>(ydim), static_cast<std::uint8_t>(ckernel::trisc::srcs_dims::XDIM), 1, 1);
}

/**
 * @brief Configure the unary SrcS SFPU pipeline: unpack (UNP_S), pack (PACK1) and SFPU.
 *
 * Programs UNP_S / PACK1 buffer descriptors and auto-loops for one tile, and inits the SFPU. SrcS
 * geometry is derived from @p unpack_S_dst_format, not passed by the caller.
 *
 * @tparam INSTRN_COUNT: Pack instructions per SrcS auto-loop (see llk_srcs.h).
 * @param l1_in_addr_16B: L1 input address (16B units).
 * @param unpack_S_src_format: L1 input format.
 * @param unpack_S_dst_format: SrcS format (also selects 32-bit SrcS mode).
 * @param l1_out_addr_16B: L1 output address (16B units).
 * @param pack_S_src_format: SrcS source format for the packer.
 * @param pack_S_dst_format: L1 output format.
 * @param implied_math_format: When false, disables implied SrcS math format for this TRISC.
 */
template <std::uint8_t INSTRN_COUNT = 1>
inline void llk_sfpu_srcs_unary_init(
    const std::uint32_t l1_in_addr_16B,
    const DataFormat unpack_S_src_format,
    const DataFormat unpack_S_dst_format,
    const std::uint32_t l1_out_addr_16B,
    const DataFormat pack_S_src_format,
    const DataFormat pack_S_dst_format,
    const bool implied_math_format) {
    const bool srcs_32bit_mode = ckernel::trisc::_is_srcs_32bit_mode_(unpack_S_dst_format);
    const ckernel::TensorShape srcs_shape =
        llk_sfpu_srcs_slice_shape_impl(ckernel::trisc::srcs_dims::ydim(srcs_32bit_mode));

    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp2_Slice0>(
        srcs_shape, l1_in_addr_16B, static_cast<std::uint32_t>(unpack_S_src_format));
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(unpack_S_dst_format);
    llk_sfpu_srcs_configure_pack_impl(srcs_shape, l1_out_addr_16B, pack_S_src_format, pack_S_dst_format);

    cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + ckernel::TRISC_ID] = !implied_math_format;

    _llk_unpack_srcs_config_for_tile_<INSTRN_COUNT>(srcs_32bit_mode);
    _llk_pack_srcs_config_for_tile_<INSTRN_COUNT>(srcs_32bit_mode);
    _llk_math_eltwise_sfpu_init_();
}

/**
 * @brief Configure the binary SrcS SFPU pipeline: unpack (UNP_S), pack (PACK1) and SFPU.
 *
 * Like llk_sfpu_srcs_unary_init, for ops with two inputs sharing formats. Both inputs use the
 * single UNP_S engine (two BFD table rows). The unpack auto-loop cannot alternate descriptors
 * (tt-llk #1635), so ids are baked into a two-instruction MOP and llk_sfpu_srcs_binary runs that
 * MOP per slice.
 *
 * @tparam INSTRN_COUNT: Pack instructions per SrcS auto-loop (see llk_srcs.h).
 * @param l1_in0_addr_16B: L1 first input address (16B units).
 * @param l1_in1_addr_16B: L1 second input address (16B units).
 * @param unpack_S_src_format: L1 input format (both inputs).
 * @param unpack_S_dst_format: SrcS format (also selects 32-bit SrcS mode).
 * @param l1_out_addr_16B: L1 output address (16B units).
 * @param pack_S_src_format: SrcS source format for the packer.
 * @param pack_S_dst_format: L1 output format.
 * @param implied_math_format: When false, disables implied SrcS math format for this TRISC.
 */
template <std::uint8_t INSTRN_COUNT = 1>
inline void llk_sfpu_srcs_binary_init(
    const std::uint32_t l1_in0_addr_16B,
    const std::uint32_t l1_in1_addr_16B,
    const DataFormat unpack_S_src_format,
    const DataFormat unpack_S_dst_format,
    const std::uint32_t l1_out_addr_16B,
    const DataFormat pack_S_src_format,
    const DataFormat pack_S_dst_format,
    const bool implied_math_format) {
    const bool srcs_32bit_mode = ckernel::trisc::_is_srcs_32bit_mode_(unpack_S_dst_format);
    const ckernel::TensorShape srcs_shape =
        llk_sfpu_srcs_slice_shape_impl(ckernel::trisc::srcs_dims::ydim(srcs_32bit_mode));

    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp2_Slice0>(
        srcs_shape, l1_in0_addr_16B, static_cast<std::uint32_t>(unpack_S_src_format));
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp2_Slice1>(
        srcs_shape, l1_in1_addr_16B, static_cast<std::uint32_t>(unpack_S_src_format));
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(unpack_S_dst_format);
    llk_sfpu_srcs_configure_pack_impl(srcs_shape, l1_out_addr_16B, pack_S_src_format, pack_S_dst_format);

    cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + ckernel::TRISC_ID] = !implied_math_format;

    _llk_unpack_srcs_config_<1, 1>();
    _llk_unpack_srcs_binary_mop_config_(
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp2_Slice0>(),
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp2_Slice1>());
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
 * @tparam INSTRN_COUNT: Must match the value passed to llk_sfpu_srcs_unary_init.
 * @tparam SfpuOp: Callable sfpu_op(int load_base_addr, int store_base_addr, int num_sfpu_iterations).
 * @param num_tiles: Number of 32x32 tiles to process.
 * @param unpack_S_dst_format: SrcS format used to derive geometry.
 * @param sfpu_op: Per-slice SFPU computation.
 */
template <std::uint8_t INSTRN_COUNT = 1, typename SfpuOp>
inline void llk_sfpu_srcs_unary(const std::uint32_t num_tiles, const DataFormat unpack_S_dst_format, SfpuOp&& sfpu_op) {
    const bool srcs_32bit_mode = ckernel::trisc::_is_srcs_32bit_mode_(unpack_S_dst_format);
    const std::uint32_t ydim = ckernel::trisc::srcs_dims::ydim(srcs_32bit_mode);
    const std::uint32_t slice_count = ckernel::trisc::srcs_dims::slice_count(srcs_32bit_mode);

    const int num_sfpu_iterations = static_cast<int>(ydim >> 1);  // SFP_ROWS == 2
    const int load_base_addr = ckernel::math::SFPU_SRCS_BASE_ADDR;
    const int store_base_addr = ckernel::math::SFPU_SRCS_BASE_ADDR + 2 * static_cast<int>(ydim);

    const std::uint8_t bfd_unpack = ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp2_Slice0>();
    const std::uint8_t bfd_pack = ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack1>();

    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        _llk_unpack_srcs_<INSTRN_COUNT>(bfd_unpack, i * slice_count);  // Sets dvalid for SFPU to read

        // Pack before SFPU: the SFPU loop clogs the instruction buffer and hangs if pack comes after.
        _llk_pack_srcs_<INSTRN_COUNT>(bfd_pack, i * slice_count);  // Sets dvalid for SFPU to write

        for (std::uint32_t slice = 0; slice < slice_count; slice++) {
            sfpu_op(load_base_addr, store_base_addr, num_sfpu_iterations);
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, true>();  // Clears dvalid for SFPU read and write
        }
    }
}

/**
 * @brief Run a binary SFPU op over num_tiles tiles on the SrcS path.
 *
 * Per tile: pack from SrcS, then per slice unpack one slice of each input, invoke @p sfpu_op and
 * clear the SrcS valids. Simple direct-issue variant; the pipelined preload + replay variant
 * stays kernel-local (see tt-llk #1635).
 *
 * @tparam INSTRN_COUNT: Must match the value passed to llk_sfpu_srcs_binary_init.
 * @tparam SfpuOp: Callable sfpu_op(int in0_base_addr, int in1_base_addr, int store_base_addr, int num_sfpu_iterations).
 * @param num_tiles: Number of 32x32 tiles to process.
 * @param unpack_S_dst_format: SrcS format used to derive geometry.
 * @param sfpu_op: Per-slice SFPU computation.
 */
template <std::uint8_t INSTRN_COUNT = 1, typename SfpuOp>
inline void llk_sfpu_srcs_binary(
    const std::uint32_t num_tiles, const DataFormat unpack_S_dst_format, SfpuOp&& sfpu_op) {
    const bool srcs_32bit_mode = ckernel::trisc::_is_srcs_32bit_mode_(unpack_S_dst_format);
    const std::uint32_t ydim = ckernel::trisc::srcs_dims::ydim(srcs_32bit_mode);
    const std::uint32_t slice_count = ckernel::trisc::srcs_dims::slice_count(srcs_32bit_mode);

    const int num_sfpu_iterations = static_cast<int>(ydim >> 1);  // SFP_ROWS == 2
    const int in0_base_addr = ckernel::math::SFPU_SRCS_BASE_ADDR;
    const int in1_base_addr = ckernel::math::SFPU_SRCS_BASE_ADDR + static_cast<int>(ydim);
    const int store_base_addr = ckernel::math::SFPU_SRCS_BASE_ADDR + 2 * static_cast<int>(ydim);

    const std::uint8_t bfd_pack = ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack1>();

    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, i * slice_count);

        // Pack before SFPU: the SFPU loop clogs the instruction buffer and hangs if pack comes after.
        _llk_pack_srcs_<INSTRN_COUNT>(bfd_pack, i * slice_count);  // Sets dvalid for SFPU to write

        for (std::uint32_t slice = 0; slice < slice_count; slice++) {
            _llk_unpack_srcs_binary_();
            sfpu_op(in0_base_addr, in1_base_addr, store_base_addr, num_sfpu_iterations);
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true, true>();  // Clears dvalid for SFPU read and write
        }
    }
}
