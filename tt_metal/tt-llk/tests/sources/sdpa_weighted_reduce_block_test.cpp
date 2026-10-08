// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Batched SDPA weighted reduce (api/compute/experimental/sdpa_weighted_reduce.h, weighted_reduce_block): NUM_CHUNKS qk
// tiles of two 16x16 faces each (buffer_B) against one weights tile (buffer_A) through one unpack context transaction
// (_llk_unpack_AB_sdpa_weighted_reduce_block_), the header's two MVMULs per chunk into DEST slot c (16 rows apart), then
// a standard pack of DEST tile 0. Slot c's first row is face c's first row of that tile, which the driver checks. With
// ROW_PACK the pack is weighted_reduce_pack_block's instead: chunk c's output row lands in row c of the result.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

using namespace ckernel;

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sdpa_weighted_reduce.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // SrcA <- qk (buffer_B, two faces per tile), SrcB <- weights (buffer_A).
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src,
        formats.unpack_A_src,
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        2 /* unpA_num_faces */,
        4 /* unpB_num_faces */,
        params.TILE_SIZE_UNPACK_B,
        params.TILE_SIZE_UNPACK_A);
    // weighted_reduce_init_short: no haloize, one 16x16 face per UNPACR on both unpackers.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TTI_SETADCXX(p_setadc::UNP_B, FACE_R_DIM * FACE_C_DIM - 1, 0x0);

    _llk_unpack_AB_sdpa_weighted_reduce_block_(L1_ADDRESS(params.buffer_B[0]), L1_ADDRESS(params.buffer_A[0]), 2 /* qk_num_faces */, NUM_CHUNKS);
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_matmul_custom_no_mop.h"
#include "llk_math_common.h"

// weighted_reduce_addrmod_init and weighted_reduce_math_impl of the api header (ADDR_MOD_3 stands in for
// sdpa_custom_mm_init's).
inline void weighted_reduce_addrmod_init_math()
{
    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_6);
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_3);
}

inline void weighted_reduce_math_impl(const std::uint32_t dst_slot)
{
    constexpr std::uint32_t weighted_dest_slot_rows = 16;
    math::reset_counters(p_setrwc::SET_ABD_F);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base() + dst_slot * weighted_dest_slot_rows);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_6, 0);
    TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_matmul_init_no_mop_<ckernel::MathFidelity::LoFi, 0>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, 0, 1, 1);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    weighted_reduce_addrmod_init_math();

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t c = 0; c < NUM_CHUNKS; c++)
    {
        weighted_reduce_math_impl(c);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_matmul_uninit_no_mop_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    if constexpr (ROW_PACK)
    {
        // weighted_reduce_pack_block: the DEST-read strides sdpa_custom_mm's pack init leaves (faces 8 rows apart, slots
        // 16), weighted_reduce_addrmod_init's ADDR_MOD_3, then one destination and two PACRs per chunk into rows 0 onwards.
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * 8 * 2);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * 8 * 2);
        addr_mod_pack_t {
            .y_src = {.incr = 0, .clr = 0, .cr = 0},
            .y_dst = {.incr = 1, .clr = 0, .cr = 0},
            .z_src = {.incr = 1, .clr = 0},
            .z_dst = {.incr = 0, .clr = 0},
        }
            .set(ADDR_MOD_3);
        _llk_packer_wait_for_math_done_();
        set_dst_write_addr(0);
        program_packer_destination(L1_ADDRESS(params.buffer_Res[0]));
        for (std::uint32_t i = 0; i < 2 * NUM_CHUNKS - 1; i++)
        {
            TTI_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_NORMAL_MODE,
                ADDR_MOD_3,
                p_pacr::ADDR_CNT_CTXT_0,
                p_pacr::P_ZERO_OUTPUT_DISABLED,
                p_pacr::SINGLE_INTF_ACTIVE,
                0,
                0,
                0,
                0,
                0);
        }
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_1,
            p_pacr::ADDR_CNT_CTXT_0,
            p_pacr::P_ZERO_OUTPUT_DISABLED,
            p_pacr::SINGLE_INTF_ACTIVE,
            0,
            0,
            0,
            0,
            1);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    }
    else
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
