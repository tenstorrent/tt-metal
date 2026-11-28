// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Welford state-save helpers writing into tiles that the packer's ZEROACC left undefined.
//
// Per block, in SyncHalf, MATH does what the Welford layernorm and groupnorm kernels do: acquire a
// fresh Dst half, copy one input tile into INPUT_DST (standing in for transpose_tile), load the
// statistics into LREG4..7, then save them with the real helpers through the same
// _llk_math_welfords_sfpu_params_ wrapper the compute API uses, and release. PACK packs the three
// state tiles and releases normally, clearing the other half with ZEROACC on each release.
// On Blackhole, a store that executes in the same cycle as that ZEROACC loses its row-valid update,
// so a 4-row group written by only one store reads back as zero.
//
// The statistics are the input tile's own values, so every block has fresh per-lane values:
//   LREG4 (mean)   <- raw offset 0  of INPUT_DST (rows 0..3)
//   LREG5 (M2)     <- raw offset 4  (rows 4..7)
//   LREG6 (M2 b)   <- raw offset 8  (rows 8..11)
//   LREG7 (anchor) <- raw offset 12 (rows 12..15)
//
// params.DST_INDEX: bits 0..11 = NOPs between the input copy and the state save, which slide the
// save across the packer's ZEROACC; bit 12 selects the helpers:
//   0: _two_pass_store_mean_m2_to_dst_<true> + _two_pass_store_anchor_to_state_dst_ (layernorm)
//   1: _store_mean_m2_to_dst_group_(0) + _two_pass_store_anchor_to_dst_ into ANCHOR_DST
//      (groupnorm group 0 slot, and the single-acquire layernorm anchor tile)

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t MEAN_DST   = 0; // M2 is in MEAN_DST + 1
constexpr std::uint32_t ANCHOR_DST = 2;
constexpr std::uint32_t INPUT_DST  = 3;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), formats.unpack_A_src, formats.unpack_A_dst);
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            L1_ADDRESS(params.buffer_A[block]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_welfords_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"
#include "params.h"

using namespace ckernel;

// Statistics into LREG4..7 from the copied input tile (current Dst base = INPUT_DST).
sfpi_inline void load_stats_from_input()
{
    TTI_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 4);
    TTI_SFPLOAD(p_sfpu::LREG6, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 8);
    TTI_SFPLOAD(p_sfpu::LREG7, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, 12);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats             = params.formats;
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        tensor_shape, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_two_pass_sfpu_init_();

    const std::uint32_t nops = params.DST_INDEX & 0xfff;
    const bool group_helpers = (params.DST_INDEX & (1u << 12)) != 0;

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
            INPUT_DST, formats.math, formats.math, tensor_shape);
        _llk_math_welfords_sfpu_params_(load_stats_from_input, INPUT_DST);
        for (std::uint32_t k = 0; k < nops; ++k)
        {
            TTI_NOP;
        }
        if (group_helpers)
        {
            // welford_save_state(mean_dst, 0) and two_pass_stats_save_anchor(anchor_dst)
            _llk_math_welfords_sfpu_params_(ckernel::sfpu::_store_mean_m2_to_dst_group_, MEAN_DST, 0u);
            _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_store_anchor_to_dst_, ANCHOR_DST);
        }
        else
        {
            // two_pass_stats_save_state(mean_dst) and two_pass_stats_save_anchor_to_state(mean_dst)
            _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_store_mean_m2_to_dst_<true>, MEAN_DST);
            _llk_math_welfords_sfpu_params_(ckernel::sfpu::_two_pass_store_anchor_to_state_dst_, MEAN_DST);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t t = 0; t < 3; ++t)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(MEAN_DST + t, L1_ADDRESS(params.buffer_Res[3 * block + t]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
