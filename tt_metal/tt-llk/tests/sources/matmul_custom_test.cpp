// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

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
    // in0 (buffer_A) feeds SrcB through unpacker B, in1 (buffer_B) feeds SrcA through unpacker A,
    // so the in1 geometry goes to the unpA_* arguments and the in0 geometry to unpB_* (as in math_matmul_test.cpp).
    const std::uint32_t unpA_face_r_dim = params.in1_tile_r_dim < FACE_R_DIM ? params.in1_tile_r_dim : FACE_R_DIM;
    const std::uint32_t unpB_face_r_dim = params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        unpA_face_r_dim,
        unpB_face_r_dim,
        params.num_faces_B, // in1
        params.num_faces_A, // in0
        params.TILE_SIZE_UNPACK_B,
        params.TILE_SIZE_UNPACK_A);
    _llk_unpack_AB_matmul_init_<>(
        params.UNPACK_TRANSPOSE_FACES,
        params.CT_DIM,
        params.RT_DIM,
        params.KT_DIM,
        unpA_face_r_dim,
        unpB_face_r_dim,
        params.num_faces_B,     // in1
        params.num_faces_A,     // in0
        params.PARTIAL_FACE_B,  // in1
        params.PARTIAL_FACE_A); // in0
    // Expected-assert case (transpose with a 32x16 in1, rejected by the math init under LLK asserts): leave nothing
    // pending for the next kernel on this core -- do not load SrcA/SrcB or set dvalid, math never consumes them.
    if (params.UNPACK_TRANSPOSE_FACES && params.in1_tile_r_dim == TILE_R_DIM && params.in1_tile_c_dim == FACE_C_DIM)
    {
        return;
    }
    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(params.buffer_A[0]),
            L1_ADDRESS(params.buffer_B[0]),
            j,
            j * params.CT_DIM,
            params.TILE_SIZE_UNPACK_A,
            params.TILE_SIZE_UNPACK_B,
            params.PARTIAL_FACE_B, // in1
            params.PARTIAL_FACE_A, // in0
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
        params.in0_tile_r_dim,
        params.in0_tile_c_dim,
        params.in1_tile_r_dim,
        params.in1_tile_c_dim,
        params.PARTIAL_FACE_MATH,
        params.UNPACK_TRANSPOSE_FACES,
        params.CT_DIM,
        params.RT_DIM);
    // Expected-assert case (transpose with a 32x16 in1): the init above rejects it under LLK asserts; the unpack thread
    // loads no sources for it, so without the asserts the MVMULs would stall on src valid -- do not run them.
    if (params.UNPACK_TRANSPOSE_FACES && params.in1_tile_r_dim == TILE_R_DIM && params.in1_tile_c_dim == FACE_C_DIM)
    {
        return;
    }
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_math_matmul_no_mop_<MATH_FIDELITY, THROTTLE_LEVEL>(
            0 /* dst_index */,
            params.CT_DIM,
            params.RT_DIM,
            params.in0_tile_r_dim,
            params.in0_tile_c_dim,
            params.in1_tile_r_dim,
            params.in1_tile_c_dim,
            params.PARTIAL_FACE_MATH);
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
    // The result tile has in0's row geometry: a 16x32 in0 tiny tile yields a 2-face result.
    const std::uint32_t pack_face_r_dim = params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, pack_face_r_dim, TILE_C_DIM, params.num_faces, params.PARTIAL_FACE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, pack_face_r_dim, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    bool math_done = true;
    if (params.UNPACK_TRANSPOSE_FACES && params.in1_tile_r_dim == TILE_R_DIM && params.in1_tile_c_dim == FACE_C_DIM)
    {
        // Expected-assert case: math rejects the geometry at init and never posts MATH_PACK. Poll the semaphore from
        // the RISC with a bound instead of SEMWAITing on it, because a SEMWAIT left in the Tensix pack FIFO outlives
        // the RISC soft reset and hangs the next kernel on this core. The bound stays inside the harness's 2 s
        // completion window.
        constexpr std::uint64_t math_done_bound_cycles = 1'500'000'000;
        const std::uint64_t start                      = ckernel::read_wall_clock();
        while (semaphore_read(semaphore::MATH_PACK) == 0 && ckernel::read_wall_clock() - start < math_done_bound_cycles)
        {
        }
        math_done = semaphore_read(semaphore::MATH_PACK) != 0;
    }

    if (math_done)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
