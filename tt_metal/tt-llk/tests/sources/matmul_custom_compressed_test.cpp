// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// K splits over CUSTOM_MM_NUM_CALLS back-to-back calls that accumulate into one DEST; 1 is the single-call kernel.
// More than one call is for test_matmul_custom_compressed_multi_call, which exercises races at the call boundary,
// where the unpacker enters call c + 1 while math still works on call c: the both-bank SrcB clear (race and rearm
// cases) and the bfp2 -> bfp4/bfp8 format change (boundary cases). The result must not depend on that timing.
// Call c's compressed tiles sit in a bfp8-sized slot (68 16-byte words per tile), its metadata is a fresh
// 10-tiles-per-u32 stream that restarts on the previous-format sentinel, and its in0 tiles follow the last call's.
constexpr std::uint32_t num_calls           = CUSTOM_MM_NUM_CALLS;
constexpr std::uint32_t kt_per_call         = KT_DIM / num_calls;
constexpr std::uint32_t b_words_per_call    = kt_per_call * CT_DIM * 68;
constexpr std::uint32_t meta_bytes_per_call = ((kt_per_call * CT_DIM + 9) / 10) * sizeof(std::uint32_t);
static_assert(kt_per_call * num_calls == KT_DIM && kt_per_call % 2 == 0, "each call needs an even share of the K tiles");

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_compressed_custom_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src,
        formats.unpack_A_src,
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        params.in1_face_r_dim,
        params.in0_face_r_dim,
        params.num_faces_B,
        params.num_faces_A,
        params.TILE_SIZE_UNPACK_B,
        params.TILE_SIZE_UNPACK_A);

    if constexpr (num_calls > 1)
    {
        // Race and rearm cases only; the boundary cases' bfp2 stalls clear SrcA to 0 anyway. Leave a non-zero SrcA
        // clear value on unpacker 0: a both-bank SrcB clear only corrupts the SrcA writes it overlaps while unpacker
        // 0's last SrcA clear value is non-zero, and that value outlives kernels. Without this the race would pass
        // or fail depending on what last ran on the core.
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_1, p_unpacr_nop::CLR_SRC);
    }

    _llk_unpack_AB_compressed_custom_mm_init_<false /* transpose */, true /* clear_src */>(params.in0_face_r_dim);

    // An in0 tile is 2 faces x in0_face_r_dim rows x 16 Float16_b datums = 4 * in0_face_r_dim 16-byte words.
    for (std::uint32_t call = 0; call < num_calls; call++)
    {
        _llk_unpack_AB_compressed_custom_mm_(
            L1_ADDRESS(params.buffer_B[0]) + call * b_words_per_call,
            L1_ADDRESS(params.buffer_A[0]) + call * kt_per_call * 4 * params.in0_face_r_dim,
            params.buffer_C[0] + call * meta_bytes_per_call,
            kt_per_call,
            CT_DIM);
        if constexpr (CUSTOM_MM_REARM)
        {
            // Rearm cases only. Leave a -inf SrcA clear value after every call, as a max-reduce or top-k between
            // calls would. Each call ends on the end-of-call stall, a SrcA clear to 0, which would otherwise hide a
            // both-bank clear in the execute at every boundary after the first. Like the end stall it waits for its
            // bank, so it never clears a bank math still reads.
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1 /* Stall_Clr_Cntrl */, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_compressed_custom_mm.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_compressed_custom_mm_init_<false, false, true>(params.in0_face_r_dim);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    for (std::uint32_t call = 0; call < num_calls; call++)
    {
        if constexpr (num_calls > 1)
        {
            // Race, rearm and boundary cases. Hold math back before each call so the unpacker runs ahead and reaches
            // call c + 1 while math still holds call c's last banks. A both-bank SrcB clear then starts on the same
            // bank release as call c + 1's first SrcA writes, and call c's last bfp2 unpack is still waiting for its
            // bank when call c + 1 changes format. Without the delay math keeps up and neither boundary is contended.
            ckernel::wait(2000);
        }
        _llk_math_compressed_custom_mm_<false>(params.buffer_C[0] + call * meta_bytes_per_call, params.in0_face_r_dim, 0, kt_per_call, CT_DIM);
    }

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true);

    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        formats.pack_src, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);

    _llk_packer_wait_for_math_done_();

    for (std::uint32_t i = 0; i < CT_DIM; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }

    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif
