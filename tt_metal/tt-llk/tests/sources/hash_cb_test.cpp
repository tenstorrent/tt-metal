// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone LLK test for the experimental SFPU-backed CB hash (23-bit FNV
// variant). See tt_llk_blackhole/llk_lib/experimental/llk_math_hash_cb.h for
// the algorithm. The kernel reads TILE_CNT input tiles (INT32 format) from
// buffer_A, accumulates a per-lane hash in SFPU LReg state, reduces to lane 0,
// and packs a single output tile to buffer_Res[0] where byte[0..3] == hash.
//
// STATUS: draft. This test has not been run on hardware. The matching pytest
// is tests/python_tests/test_hash_cb.py with the NumPy golden.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define DEBUG_CB_HASH 1

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, 4, formats.unpack_A_src, formats.unpack_A_dst);

    // Unpack every input tile into DEST slot 0 (accumulator state lives in SFPU LRegs,
    // so reusing DEST row 0 per tile is the desired pattern).
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_defs.h"
#include "experimental/llk_math_hash_cb.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const bool is_int_fpu_en = true;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, is_int_fpu_en>(4, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);
#endif

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    // Initialise SFPU accumulator (LReg2) and prime constant (LReg1).
    _llk_math_hash_cb_init_();

    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        // Copy this input tile A -> DEST row 0 so SFPU can read it.
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0, formats.math, formats.math);

        // Accumulate this tile into the 32 per-lane hash accumulators.
        _llk_math_hash_cb_tile_(/*dst_tile_idx=*/0);
    }

    // Reduce 32 lanes -> lane 0 and write the final 23-bit hash to DEST[0][0][0].
    _llk_math_hash_cb_reduce_and_store_(/*dst_tile_idx=*/0);

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats.pack_src, formats.pack_dst, 16 * 16);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(formats.pack_src, formats.pack_dst, 16 * 16);
#endif

    _llk_pack_init_<false, false>(formats.pack_dst);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, false, false>();
#endif

    _llk_packer_wait_for_math_done_();

    // Pack the single hash-carrying tile at DEST slot 0 to buffer_Res[0].
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(params.buffer_Res[0]));

    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
