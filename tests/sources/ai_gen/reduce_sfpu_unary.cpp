// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

// -----------------------------------------------------------------------------
// Test description:
//   UNPACK   : _llk_unpack_AB    (Unpack thread)
//   MATH     : Reduce (row/col/scalar) + immediate SFPU unary (math thread)
//   PACK     : Regular pack (pack thread)
//
// All configurable parameters (formats, reduce dim, pool type, dst sync, unary
// op, math fidelity, etc.) are provided through the auto-generated params.h
// header created by the Python host-side test infrastructure.  This file only
// contains the LLK call-sequence itself.
// -----------------------------------------------------------------------------

#include "params.h"

// -----------------------------------------------------------------------------
// Helper constexpr driven by build defines coming from params.h
// -----------------------------------------------------------------------------
constexpr std::uint32_t within_face_16x16_transpose = (REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW) ? 1 : 0;
constexpr bool row_pool                             = (REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW);

// -----------------------------------------------------------------------------
// UNPACK THREAD (TRISC_UNPACK)
// -----------------------------------------------------------------------------
#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel()
{
    // Configure hardware for AB unpack (single tile per input)
    _llk_unpack_AB_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(
        formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst, FACE_R_DIM, within_face_16x16_transpose);

    // Initialise unpacker state machine
    _llk_unpack_AB_init_<>(FACE_R_DIM, 4, false, within_face_16x16_transpose);

    // Unpack the two input tiles (A & B) into the destination register file
    _llk_unpack_AB_<>(L1_ADDRESS(buffer_A[0]), L1_ADDRESS(buffer_B[0]));
}

#endif // LLK_TRISC_UNPACK

// -----------------------------------------------------------------------------
// MATH THREAD (TRISC_MATH) – Reduce + SFPU unary
// -----------------------------------------------------------------------------
#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_reduce.h"
#include "sfpu_operations.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel()
{
    //------------------------------------------------------------------
    // Synchronisation & HW configuration
    //------------------------------------------------------------------
    _llk_math_pack_sync_init_<DstSync::SyncFull, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<DstSync::SyncFull>();

    // row_pool tells HW if pooling is performed across rows (affects transpose path)
    _llk_math_hw_configure_<false, row_pool>(formats.math, formats.math);

    //------------------------------------------------------------------
    // 1) Reduce operation
    //------------------------------------------------------------------
    const bool is_int_fpu_en  = false; // Not using int-FPU path
    const bool fp32_transpose = false; // No fp32 transpose on reduce path

    constexpr uint32_t math_fid = 4;
    _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, math_fid>();
    _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, math_fid, is_int_fpu_en, fp32_transpose>(0);

    //------------------------------------------------------------------
    // 2) SFPU unary directly on the reduced result in dest regs
    //------------------------------------------------------------------
    _llk_math_eltwise_unary_sfpu_init_<SFPU_UNARY_OPERATION>();
    _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncFull>(0);

    test_utils::call_sfpu_operation_32(SFPU_UNARY_OPERATION);

    _llk_math_eltwise_unary_sfpu_done_();
    _llk_math_dest_section_done_<DstSync::SyncFull, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_MATH

// -----------------------------------------------------------------------------
// PACK THREAD (TRISC_PACK)
// -----------------------------------------------------------------------------
#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(formats.pack_dst);

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#endif

    _llk_pack_reduce_mask_config_<false, REDUCE_DIM>();

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_dest_init_<DstSync::SyncFull, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, false>();
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncFull, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncFull, is_fp32_dest_acc_en>();

    _llk_pack_reduce_mask_clear_();
}

#endif
