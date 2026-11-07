// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Functional chain of pack programs in one kernel (Blackhole): each step packs its own four tiles of the input to its own four result tiles, then the next
// step runs with whatever pack state the previous one left, the way a fused kernel does. CHAIN_STEPS (pack_state_chain.py) lists the steps:
//   0 PT   per-tile _llk_pack_ (the default write: the byte-wise destination write under SyncFull)
//   1 PB1  _llk_pack_block_ (the plain block of llk_pack_block, programs the output)
//   2 PBW  W offset block: the output programmed as llk_pack_w_take does, then _llk_pack_block_w_
//   3 PBC  _llk_pack_block_closed_ (block float and padded pages)
//   4 PBCW closed block with the W offset: programmed as llk_pack_w_take does, then _llk_pack_block_closed_w_mop_
//   5 ROWS pack_rows of ROWS_R rows per tile, then the default pack init (pack_rows has no program restore, as on main)
//   6 UNT  pack_untilize of the four tiles as one row (init, block, uninit, the default pack init: the sequence pack_untilize_init/uninit make)
//   7 FU   fast untilize of the four tiles as one row; the pack uninit is skipped with FU_PACK_UNINIT off, the block packs of llk_pack_block close it
//   8 PTC  per-tile pack after the close (the per-tile fallback of llk_pack_block)
// The block steps run _llk_pack_close_fast_untilize_ where llk_pack_block runs it: before it programs the output, once per output, not on the W offset continuation.

#include <cstdint>

#include "ckernel.h"
#include "experimental/llk_fast_untilize_common.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
[[maybe_unused]] constexpr int K_PT = 0, K_PB1 = 1, K_PBW = 2, K_PBC = 3, K_PBCW = 4, K_ROWS = 5, K_UNT = 6, K_FU = 7, K_PTC = 8;
constexpr std::uint32_t NT = 4; // tiles per step
[[maybe_unused]] constexpr std::uint32_t CT = 4; // the untilize steps run one tile row of NT tiles
[[maybe_unused]] constexpr auto FU_SYNC = ckernel::FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE;
#ifndef DST_ACCUM_MODE
[[maybe_unused]] constexpr bool DST_ACCUM_MODE = is_fp32_dest_acc_en;
#endif
[[maybe_unused]] constexpr bool FU_BFP_IN = UNPACK_A_IN == ckernel::to_underlying(DataFormat::Bfp8_b) || UNPACK_A_IN == ckernel::to_underlying(DataFormat::Bfp4_b);
} // namespace

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_fast_untilize.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const Operand& buffer_A = params.buffer_A;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);

    for (int s = 0; s < NSTEPS; s++)
    {
        const std::uint32_t t0 = s * NT;
        if (CHAIN_STEPS[s] == K_FU)
        {
            ckernel::_llk_unpack_fast_untilize_init_<DST_ACCUM_MODE>(formats.unpack_A_src, formats.unpack_A_dst, FU_BFP_IN ? 1 : CT);
            if constexpr (FU_BFP_IN)
            {
                const std::uint32_t address         = L1_ADDRESS(buffer_A[t0]);
                const std::uint32_t tile_stride_16B = L1_ADDRESS(buffer_A[t0 + 1]) - address;
                ckernel::_llk_unpack_fast_untilize_bfp_block_(address, tile_stride_16B, CT);
            }
            else
            {
                ckernel::_llk_unpack_fast_untilize_block_(L1_ADDRESS(buffer_A[t0]), CT);
            }
            ckernel::_llk_unpack_fast_untilize_uninit_();
        }
        else
        {
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), formats.unpack_A_src, formats.unpack_A_dst);
            for (std::uint32_t t = 0; t < NT; t++)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                    L1_ADDRESS(buffer_A[t0 + t]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_fast_untilize_api.h"
#include "llk_lib_math_wrappers.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    for (int s = 0; s < NSTEPS; s++)
    {
        const int kind = CHAIN_STEPS[s];
        if (kind == K_FU)
        {
            _llk_math_pack_sync_init_<FU_SYNC, is_fp32_dest_acc_en>();
            llk_math_fast_untilize_init();
            _llk_math_wait_for_dest_available_<FU_SYNC>();
            llk_math_fast_untilize_block<is_fp32_dest_acc_en>(0, CT);
            _llk_math_dest_section_done_<FU_SYNC, is_fp32_dest_acc_en>();
            llk_math_fast_untilize_uninit();
            // the mode the kernel runs in, as fast_untilize_uninit does
            if constexpr (dest_sync != FU_SYNC)
            {
                _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        else
        {
            _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
            if (kind == K_UNT)
            {
                _llk_math_reconfig_remap_wrapper_(true);
            }
            _llk_math_wait_for_dest_available_<dest_sync>();
            for (std::uint32_t t = 0; t < NT; t++)
            {
                _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, false>(t, formats.math, formats.math, ckernel::DEFAULT_TENSOR_SHAPE);
            }
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            if (kind == K_UNT)
            {
                _llk_math_reconfig_remap_wrapper_(false);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_fast_untilize.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "llk_pack_rows.h"
#include "llk_pack_untilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const Operand& buffer_Res = params.buffer_Res;
    const std::uint32_t tile_stride_bytes = (L1_ADDRESS(buffer_Res[1]) - L1_ADDRESS(buffer_Res[0])) << 4;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM));
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    for (int s = 0; s < NSTEPS; s++)
    {
        const int kind          = CHAIN_STEPS[s];
        const std::uint32_t r0  = s * NT;
        const std::uint32_t a0  = L1_ADDRESS(buffer_Res[r0]);
        if (kind == K_FU)
        {
            _llk_pack_dest_init_<FU_SYNC, is_fp32_dest_acc_en>();
            ckernel::_llk_pack_fast_untilize_init_<ckernel::FAST_UNTILIZE_MAX_UNIT_DIM, CT>(formats.pack_src, formats.pack_dst);
            _llk_init_packer_dest_offset_registers_<FU_SYNC>();
            std::uint32_t prev_unit_dim = 0;
            _llk_packer_wait_for_math_done_();
            ckernel::_llk_pack_fast_untilize_block_<ckernel::FAST_UNTILIZE_MAX_UNIT_DIM>(a0, CT, prev_unit_dim);
            _llk_pack_dest_section_done_<FU_SYNC, is_fp32_dest_acc_en>();
            if constexpr (FU_PACK_UNINIT)
            {
                ckernel::_llk_pack_fast_untilize_uninit_(formats.pack_src);
                _llk_init_packer_dest_offset_registers_<dest_sync>();
            }
            continue;
        }
        if (kind == K_ROWS)
        {
            _llk_pack_rows_init_(ROWS_R);
        }
        else if (kind == K_UNT)
        {
            _llk_pack_untilize_init_wrapper_<CT, CT>(formats.pack_src, formats.pack_dst, FACE_R_DIM, 4);
            _llk_init_packer_dest_offset_registers_<dest_sync>();
        }
        _llk_packer_wait_for_math_done_();
        switch (kind)
        {
            case K_PTC:
                _llk_pack_close_fast_untilize_<dest_sync>();
                [[fallthrough]];
            case K_PT:
                for (std::uint32_t t = 0; t < NT; t++)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(t, L1_ADDRESS(buffer_Res[r0 + t]));
                }
                break;
            case K_PB1:
                _llk_pack_close_fast_untilize_<dest_sync>();
                _llk_pack_block_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(0, a0, NT);
                break;
            case K_PBW:
                _llk_pack_close_fast_untilize_<dest_sync>();
                _llk_pack_block_w_init_(a0, tile_stride_bytes);
                _llk_pack_block_w_<dest_sync, is_fp32_dest_acc_en>(0, 0, NT);
                break;
            case K_PBC:
                _llk_pack_close_fast_untilize_<dest_sync>();
                _llk_pack_block_closed_<dest_sync, is_fp32_dest_acc_en>(0, a0, NT, tile_stride_bytes, 4);
                break;
            case K_PBCW:
                _llk_pack_close_fast_untilize_<dest_sync>();
                _llk_pack_block_w_init_(a0, tile_stride_bytes);
                _llk_pack_block_closed_w_mop_<dest_sync, is_fp32_dest_acc_en>(0, 0, NT, 4);
                break;
            case K_ROWS:
                for (std::uint32_t t = 0; t < NT; t++)
                {
                    _llk_pack_rows_(t, L1_ADDRESS(buffer_Res[r0 + t]));
                }
                break;
            case K_UNT:
                _llk_pack_untilize_wrapper_<CT, CT, false, false, TILE_C_DIM, 0>(a0, formats.pack_dst, FACE_R_DIM, 4, 0);
                break;
            default: break;
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        if (kind == K_ROWS)
        {
            _llk_pack_rows_uninit_();
            _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
        }
        else if (kind == K_UNT)
        {
            _llk_pack_untilize_uninit_wrapper_(formats.pack_src, FACE_R_DIM);
            _llk_init_packer_dest_offset_registers_<dest_sync>();
            _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
        }
    }
}

#endif
