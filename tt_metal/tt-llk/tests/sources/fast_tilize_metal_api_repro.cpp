// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Should be identical to test 3a. Sanity check.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM = params.BLOCK_CT_DIM;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);

    // Delta 2: Metal API init_unit_dim = (full_dim >= 4) ? 4 : full_dim
    const std::uint32_t init_unit_dim = (KT_DIM >= 4) ? 4 : KT_DIM;
    _llk_unpack_fast_tilize_init_(formats.unpack_A_dst, KT_DIM, KT_DIM > 5 ? 4 : KT_DIM == 5 ? 2 : KT_DIM);

    // Base address is programmed inside _llk_unpack_fast_tilize_block_ via
    // _llk_unpack_configure_single_address_ (respects current cfg context).
    _llk_unpack_fast_tilize_reinit_xdim_(KT_DIM);
    _llk_unpack_fast_tilize_block_(L1_ADDRESS(params.buffer_A[0]), 0, formats.unpack_A_src, KT_DIM, 4, 0);

    _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_fast_tilize_init_<is_fp32_dest_acc_en>(formats.math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_fast_tilize_block_<is_fp32_dest_acc_en>(0, formats.math, 4);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Delta 1: odd section_done compensation (tilize.h does this)
    update_dest_offset_id();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());

    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(formats.math);
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_pack_fast_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM = params.BLOCK_CT_DIM;

    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
        formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM), FACE_R_DIM, TILE_C_DIM, 4, false);

    _llk_pack_fast_tilize_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, formats.pack_dst, KT_DIM, 4);

    _llk_packer_wait_for_math_done_();
    _llk_pack_fast_tilize_block_(0, L1_ADDRESS(params.buffer_Res[0]), KT_DIM, 4);
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Delta 1: odd section_done compensation
    flip_packer_dest_offset_id();
    select_packer_dest_registers<DstSync::SyncHalf>();

    // Delta 3: ZEROACC (tilize.h line 387-390)
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    TTI_ZEROACC(p_zeroacc::CLR_ALL, 0, 0, ADDR_MOD_1, 0);

    _llk_pack_fast_tilize_uninit_<DstSync::SyncHalf, is_fp32_dest_acc_en>(formats.pack_dst, FACE_R_DIM, 4);
}

#endif
