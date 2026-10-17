// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Driver for the Blackhole-only experimental SFPU op sdpa_reduce_row
 * (tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sdpa_reduce_row.h,
 * promoted through experimental/llk_sfpu/{ckernel,llk_math}_sfpu_sdpa_reduce_row.h and
 * consumed by hw/inc/api/compute/experimental/sdpa.h).
 *
 * WHERE IT RUNS
 * -------------
 * In ttnn's SDPA the reduce is issued from the PACK TRISC (see the PACK((...)) calls in
 * sdpa.h: _init_sdpa_reduce_max_row_8x32_replay_buffers_() then
 * llk_math_sfpu_sdpa_reduce_max_row<...>()). We follow that placement here: MATH does the
 * A2D datacopy of the input tile into Dest, and PACK runs the SFPU reduction in place and
 * then packs the tile out. This mirrors sources/sfpu_reduce_sdpa_test.cpp, which runs its
 * reduce in PACK the same way.
 *
 * WHAT THE OP COMPUTES (golden derivation)
 * ----------------------------------------
 * The op is a ROW reduction (ReduceDim::REDUCE_ROW). sdpa.h notes "Each tile is 8x32, which
 * is the same as a full 16x16 face": the SFPU treats one 16x16 face as an 8-row x 32-col
 * logical block and reduces each of those 8 rows across its 32 columns down to a single
 * scalar, written back into column 0 of that row.
 *
 * Concretely (ckernel_sfpu_sdpa_reduce_row.h):
 *   - reduce_row_8x32_instrs<pool_type>() folds the eight 4-row x 8-lane Dest chunks
 *     (SFPLOAD offsets 0,2,4,6,8,10,12,14) pairwise into LREG0 (rows 0-3) and LREG2
 *     (rows 4-7) via reduce_lregs_instr:
 *         MAX -> TTI_SFPSWAP(..., ALL_ROWS_MAX)   (per-lane max)
 *         SUM -> TTI_SFPADD(a, LCONST_1, b, a)    (per-lane add)
 *   - _sdpa_reduce_row_8x32_epilogue_() then log-reduces across the 8 lanes of each subvector
 *     with SFPSHFT2 SHFLSHR1 shuffles (shift 4x, 2x, 1x) folded by the same reduce_lregs_instr,
 *     and a final SHFLROR1 places the reduced scalar in lane 0.
 *   So every one of the 8 logical rows ends up holding the reduction of 32 input values:
 *         MAX row output = max over the row's 32 columns
 *         SUM row output = sum over the row's 32 columns
 *   block_width>1 additionally folds block_width horizontally-adjacent tiles into the same
 *   accumulators, i.e. reduces over block_width*32 columns. This test uses block_width == 1.
 *
 * GOLDEN (which Dest datums form a logical row)
 * ---------------------------------------------
 * An SFPLOAD at Dest offset o reads the aligned group of four rows (o & ~3), one row per 8-lane
 * subvector, taking the even columns when (o & 2) == 0 and the odd columns otherwise
 * (https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md).
 * The op folds offsets 0, 2, 8, 10 into LREG0 and 4, 6, 12, 14 into LREG2, so
 * subvector s of LREG0 holds all 16 columns of face rows s and s + 8, and subvector s of LREG2
 * all 16 columns of face rows 4 + s and 12 + s. The epilogue reduces each subvector into its
 * lane 0 (SFPSHFT2 SHFLSHR1 shifts + folds, then SHFLROR1), and the SFPSTOREs at offsets 0 and 4
 * put that lane into column 0 of face rows 0-3 and 4-7. In row-major 32x32 tile coordinates:
 *     logical row r (r = 0..7) = tile[r, 0:16] ++ tile[r + 8, 0:16]
 *     tile_out[r, 0]           = MAX / SUM over that row
 * The Python side (test_sdpa_reduce_row.py) computes exactly that golden from random stimuli
 * and asserts only those eight output lanes; every other lane is undefined (left as the A2D
 * datacopy wrote it, or holding the epilogue's partials) and is not checked.
 *
 * skip_signalling == true: standalone run with no FPU<->SFPU partner, so the op's
 * t6_semaphore FPU_SFPU handshake is compiled out (matching how sdpa.h's reduce_sum path
 * passes skip_signalling=true).
 */

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Mirrors SdpaReduceRowPool in helpers/llk_params.py: which pool the op reduces with.
constexpr int SDPA_REDUCE_POOL_MAX = 0; // calculate_sdpa_reduce_max_row
constexpr int SDPA_REDUCE_POOL_SUM = 1; // calculate_sdpa_reduce_sum_row

static_assert(SDPA_REDUCE_POOL == SDPA_REDUCE_POOL_MAX || SDPA_REDUCE_POOL == SDPA_REDUCE_POOL_SUM, "unhandled SDPA_REDUCE_POOL");

// Single 16x16-face-sized reduction. src and dst are the same Dest tile (in-place reduce).
constexpr std::uint32_t SDPA_REDUCE_BLOCK_WIDTH = 1;
constexpr std::uint32_t SDPA_SRC_INDEX          = 0;
constexpr std::uint32_t SDPA_DST_INDEX          = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        4 /* num_faces */, formats.math);
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_sfpu/ckernel_sfpu_sdpa_reduce_row.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();

    // Enable the SFPU (config reg + counter reset), then run the op's self-contained init:
    // _init_sdpa_reduce_row_8x32_ sets the ZERO / TILE_OFFSET addr_mods, and the replay-buffer
    // recorder lays down the pool-specific fold sequence. sdpa.h records the replay buffer for
    // exactly the pool it is about to compute, so we do the same here.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::reduce>();
    ckernel::sfpu::_init_sdpa_reduce_row_8x32_<DataFormat::Float16_b>();
    _llk_math_eltwise_sfpu_start_(SDPA_DST_INDEX);

    if constexpr (SDPA_REDUCE_POOL == SDPA_REDUCE_POOL_MAX)
    {
        ckernel::sfpu::_init_sdpa_reduce_max_row_8x32_replay_buffers_();
        ckernel::sfpu::_calculate_sdpa_reduce_max_row_8x32_<DataFormat::Float16_b, SDPA_REDUCE_BLOCK_WIDTH, true /* skip_signalling */>(
            SDPA_SRC_INDEX, SDPA_DST_INDEX, false /* prev_max */);
    }
    else
    {
        ckernel::sfpu::_init_sdpa_reduce_sum_row_8x32_replay_buffers_();
        ckernel::sfpu::_calculate_sdpa_reduce_sum_row_8x32_<DataFormat::Float16_b, SDPA_REDUCE_BLOCK_WIDTH, true /* skip_signalling */>(
            SDPA_SRC_INDEX, SDPA_DST_INDEX, false /* prev_sum */);
    }

    _llk_math_eltwise_sfpu_done_();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
