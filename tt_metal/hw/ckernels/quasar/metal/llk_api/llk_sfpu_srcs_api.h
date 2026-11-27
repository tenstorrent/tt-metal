// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "ckernel_buf_desc.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_sfpu/ckernel_sfpu_dest.h"
#include "llk_sfpu/ckernel_sfpu_srcs.h"
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

/** @brief Point the buffer descriptor BFD_ID of the SrcS pipeline at an L1 buffer. */
template <std::uint8_t BFD_ID>
inline void llk_sfpu_srcs_set_l1(
    const DataFormat srcs_format, const std::uint32_t l1_addr_16B, const DataFormat l1_format) {
    // One SrcS slice = ydim rows x XDIM datums, single face (x=16, y=ydim, z=1).
    const std::uint32_t ydim = ckernel::trisc::srcs_dims::ydim(ckernel::trisc::_is_srcs_32bit_mode_(srcs_format));
    const ckernel::TensorShape slice_shape = ckernel::make_tensor_shape(
        static_cast<std::uint8_t>(ydim), static_cast<std::uint8_t>(ckernel::trisc::srcs_dims::XDIM), 1, 1);
    ckernel::trisc::program_bfd<BFD_ID>(slice_shape, l1_addr_16B, static_cast<std::uint32_t>(l1_format));
}

/**
 * @brief Configure the unary SrcS SFPU pipeline: unpack (UNP_S), pack (PACK1) and SFPU.
 *
 * @param unpack_S_dst_format: SrcS format (also selects 32-bit SrcS mode).
 * @param pack_S_src_format: SrcS source format for the packer.
 */
inline void llk_sfpu_srcs_unary_configure(const DataFormat unpack_S_dst_format, const DataFormat pack_S_src_format) {
    const bool srcs_32bit_mode = ckernel::trisc::_is_srcs_32bit_mode_(unpack_S_dst_format);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(unpack_S_dst_format);
    _llk_pack_hw_configure_<p_pacr::PACK1, false>(pack_S_src_format, ckernel::ReluConfig::none());
    _llk_unpack_srcs_config_for_tile_<1 /*INSTRN_COUNT*/>(srcs_32bit_mode);
    _llk_pack_srcs_config_for_tile_<1 /*INSTRN_COUNT*/>(srcs_32bit_mode);
    _llk_math_eltwise_sfpu_init_();
}

/**
 * @brief Configure the binary SrcS SFPU pipeline: unpack (UNP_S), pack (PACK1) and SFPU.
 *
 * @param unpack_S_dst_format: SrcS format (also selects 32-bit SrcS mode).
 * @param pack_S_src_format: SrcS source format for the packer.
 */
inline void llk_sfpu_srcs_binary_configure(const DataFormat unpack_S_dst_format, const DataFormat pack_S_src_format) {
    const bool srcs_32bit_mode = ckernel::trisc::_is_srcs_32bit_mode_(unpack_S_dst_format);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(unpack_S_dst_format);
    _llk_pack_hw_configure_<p_pacr::PACK1, false>(pack_S_src_format, ckernel::ReluConfig::none());
    _llk_unpack_srcs_config_<1 /*INSTRN_COUNT*/, 1 /*INSTRN_LOOP_COUNT*/>();
    _llk_unpack_srcs_binary_mop_config_(ckernel::trisc::BFD_ID_UNP2_SLICE0, ckernel::trisc::BFD_ID_UNP2_SLICE1);
    _llk_pack_srcs_config_for_tile_<1 /*INSTRN_COUNT*/>(srcs_32bit_mode);
    _llk_math_eltwise_sfpu_init_();
}

namespace ckernel::sfpu {

template <typename Op>
inline void SfpuSrcsOpCommon<Op>::pack(const std::uint32_t l1_addr_16B, const DataFormat l1_format) {
    llk_sfpu_srcs_set_l1<trisc::BFD_ID_PACK1>(Op::Layout::format, l1_addr_16B, l1_format);
}

// Op::init_op() runs after the SFPU init so its state (e.g. the LoadMacro replay) is not reset.
template <typename Op>
inline void SfpuSrcsOpBase<Op, 1>::init() {
    llk_sfpu_srcs_unary_configure(Op::Layout::format, Op::Layout::format);
    Op::init_op();
}

template <typename Op>
inline void SfpuSrcsOpBase<Op, 1>::init(
    const std::uint32_t l1_in_addr_16B,
    const DataFormat l1_in_format,
    const std::uint32_t l1_out_addr_16B,
    const DataFormat l1_out_format) {
    init();
    unpack(l1_in_addr_16B, l1_in_format);
    Op::pack(l1_out_addr_16B, l1_out_format);
}

template <typename Op>
inline void SfpuSrcsOpBase<Op, 1>::unpack(const std::uint32_t l1_addr_16B, const DataFormat l1_format) {
    llk_sfpu_srcs_set_l1<trisc::BFD_ID_UNP2_SLICE0>(Op::Layout::format, l1_addr_16B, l1_format);
}

template <typename Op>
inline void SfpuSrcsOpBase<Op, 1>::run(const std::uint32_t num_tiles) {
    for (std::uint32_t tile = 0; tile < num_tiles; ++tile) {
        const std::uint32_t l1_tile_idx = tile * Op::Layout::slices;
        _llk_unpack_srcs_<1 /*INSTRN_COUNT*/>(trisc::BFD_ID_UNP2_SLICE0, l1_tile_idx);  // dvalid for SFPU to read

        // Pack before SFPU: the SFPU loop clogs the instruction buffer and hangs if pack comes after.
        _llk_pack_srcs_<1 /*INSTRN_COUNT*/>(trisc::BFD_ID_PACK1, l1_tile_idx);  // dvalid for SFPU to write

        for (int slice = 0; slice < Op::Layout::slices; slice++) {
            Op::calculate();
            if constexpr (!Op::hw_clears_valids) {
                _llk_math_eltwise_sfpu_srcs_clear_vlds_<true /*SRCS_RD_DONE*/, true /*SRCS_WR_DONE*/>();
            }
        }
    }
}

template <typename Op>
inline void SfpuSrcsOpBase<Op, 2>::init() {
    llk_sfpu_srcs_binary_configure(Op::Layout::format, Op::Layout::format);
    Op::init_op();
}

template <typename Op>
inline void SfpuSrcsOpBase<Op, 2>::init(
    const std::uint32_t l1_in0_addr_16B,
    const std::uint32_t l1_in1_addr_16B,
    const DataFormat l1_in_format,
    const std::uint32_t l1_out_addr_16B,
    const DataFormat l1_out_format) {
    init();
    unpack(l1_in0_addr_16B, l1_in1_addr_16B, l1_in_format);
    Op::pack(l1_out_addr_16B, l1_out_format);
}

template <typename Op>
inline void SfpuSrcsOpBase<Op, 2>::unpack(
    const std::uint32_t l1_in0_addr_16B, const std::uint32_t l1_in1_addr_16B, const DataFormat l1_format) {
    llk_sfpu_srcs_set_l1<trisc::BFD_ID_UNP2_SLICE0>(Op::Layout::format, l1_in0_addr_16B, l1_format);
    llk_sfpu_srcs_set_l1<trisc::BFD_ID_UNP2_SLICE1>(Op::Layout::format, l1_in1_addr_16B, l1_format);
}

template <typename Op>
inline void SfpuSrcsOpBase<Op, 2>::run(const std::uint32_t num_tiles) {
    for (std::uint32_t tile = 0; tile < num_tiles; ++tile) {
        const std::uint32_t l1_tile_idx = tile * Op::Layout::slices;
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, l1_tile_idx);

        // Pack before SFPU: the SFPU loop clogs the instruction buffer and hangs if pack comes after.
        _llk_pack_srcs_<1 /*INSTRN_COUNT*/>(trisc::BFD_ID_PACK1, l1_tile_idx);  // dvalid for SFPU to write

        for (int slice = 0; slice < Op::Layout::slices; slice++) {
            _llk_unpack_srcs_binary_();
            Op::calculate();
            if constexpr (!Op::hw_clears_valids) {
                _llk_math_eltwise_sfpu_srcs_clear_vlds_<true /*SRCS_RD_DONE*/, true /*SRCS_WR_DONE*/>();
            }
        }
    }
}

template <typename MATH, sfpi::DataLayout LAYOUT, sfpi::DataLayout DEST_LAYOUT>
template <DstSync DST_SYNC, bool DST_ACCUM>
inline void SfpuSrcsDestOp<MATH, LAYOUT, DEST_LAYOUT>::run(const DstIndex dst_index) {
    constexpr trisc::DstTileShape tile_shape = trisc::DstTileShape::Tile32x32;
    const std::uint32_t index = static_cast<std::uint32_t>(dst_index);
    LLK_ASSERT(
        (index < trisc::get_dest_max_tiles<DST_SYNC, DST_ACCUM, tile_shape>()), "dst_index exceeds max dest tiles");
    // The Dest tile is an offset from the lent section's base, so no config is written here: a write issued after
    // the unpack below would wait for it, and it finishes only with the SFPU work.
    const int dest_tile = static_cast<int>(index << (trisc::get_dest_tile_size_log2(tile_shape) - 1));

    _llk_unpack_srcs_<1 /*INSTRN_COUNT*/>(trisc::BFD_ID_UNP2_SLICE0, 0 /*start_l1_tile_idx*/);
    _llk_pack_srcs_<1 /*INSTRN_COUNT*/>(trisc::BFD_ID_PACK1, 0 /*start_l1_tile_idx*/);
#pragma GCC unroll 16
    for (int slice = 0; slice < Layout::slices; slice++) {
        calculate(dest_tile + slice * Layout::ops);
        _llk_math_eltwise_sfpu_srcs_clear_vlds_<true /*SRCS_RD_DONE*/, true /*SRCS_WR_DONE*/>();
    }
}

}  // namespace ckernel::sfpu
