// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// Math leaves the Src zero flag at keep (as copy_tile_custom does) after hw_configure and then re-arms the matmul before
// running it (see the python side; params.h carries the harness define, so it must come first):
//   0 = no keep, 1 = _llk_math_matmul_reinit_no_mop_ (the SDPA inner-loop path), 2 = a second _llk_math_matmul_init_no_mop_.
#ifndef MATMUL_NO_MOP_REARM_AFTER_KEEP
#define MATMUL_NO_MOP_REARM_AFTER_KEEP 0
#endif

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        params.num_faces_A,
        params.num_faces_B,
        params.TILE_SIZE_UNPACK_A,
        params.TILE_SIZE_UNPACK_B);
    _llk_unpack_AB_matmul_init_<>(
        0 /* transpose */,
        params.CT_DIM,
        params.RT_DIM,
        params.KT_DIM,
        FACE_R_DIM,
        FACE_R_DIM,
        4 /* unpA_num_faces */,
        4 /* unpB_num_faces */,
        false /* unpA_partial_face */,
        false /* unpB_partial_face */);
    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(params.buffer_A[0]),
            L1_ADDRESS(params.buffer_B[0]),
            j,
            j * params.CT_DIM,
            params.TILE_SIZE_UNPACK_A,
            params.TILE_SIZE_UNPACK_B,
            false /* unpA_partial_face */,
            false /* unpB_partial_face */,
            params.CT_DIM,
            params.RT_DIM,
            params.KT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_matmul_custom_no_mop.h"
#include "llk_math_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_matmul_init_no_mop_<MATH_FIDELITY, THROTTLE_LEVEL>(
        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /* partial_face */, 0 /* transpose */, params.CT_DIM, params.RT_DIM);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

#ifdef ARCH_BLACKHOLE
    // The flag restore is Blackhole-only; the branch is compiled out on Wormhole, where the switch is never set.
    if constexpr (MATMUL_NO_MOP_REARM_AFTER_KEEP != 0)
    {
        // Leave the Src zero flag at keep, as a preceding copy_tile_custom does (hw_configure above applied the
        // default itself, so the init at the top never saw a keep), then re-arm the matmul. The re-arm must restore
        // the operand-driven value, or MVMUL keeps the denormal SrcB operands it should flush.
        math::_configure_preserve_zero_flag_state_();
        if constexpr (MATMUL_NO_MOP_REARM_AFTER_KEEP == 1)
        {
            // The inner-loop reinit path.
            _llk_math_matmul_reinit_no_mop_<MATH_FIDELITY, THROTTLE_LEVEL>(
                false /*transpose*/, TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /*partial_face*/);
        }
        else
        {
            // The full init, as a kernel that re-inits the matmul after a copy_tile_custom runs it.
            _llk_math_matmul_init_no_mop_<MATH_FIDELITY, THROTTLE_LEVEL>(
                TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /* partial_face */, 0 /* transpose */, params.CT_DIM, params.RT_DIM);
        }
    }
#endif

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_math_matmul_no_mop_<MATH_FIDELITY, THROTTLE_LEVEL>(0 /* dst_index */, params.CT_DIM, params.RT_DIM);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_matmul_uninit_no_mop_();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
