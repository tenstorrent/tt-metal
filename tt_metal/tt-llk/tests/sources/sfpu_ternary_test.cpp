// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Resolve the compile-time input format (UNPACK_A_IN) to the format used to
// configure unpack/math/pack. Supported float/int formats pass through; anything
// else falls back to a raw 16-bit (UInt16) move.
//
// Unlike the where kernel (raw LO16 loads), the addc kernels read Dest with a
// float-aware SFPLOAD (InstrModLoadStore::DEFAULT), so bf16 must stay Float16_b
// rather than being moved as raw UInt16.
static constexpr std::uint8_t resolve_ternary_format(std::uint8_t in)
{
    switch (static_cast<DataFormat>(in))
    {
        case DataFormat::Float32:
        case DataFormat::Bfp8_b:
        case DataFormat::Float16_b:
        case DataFormat::Int32:
            return in;
        default:
            return static_cast<std::uint8_t>(DataFormat::UInt16);
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t UNPACK_FMT = resolve_ternary_format(UNPACK_A_IN);

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), UNPACK_FMT, UNPACK_FMT);

    // Multi-tile: unpack the three operand tiles (a, b, c) for every input tile.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            const std::uint32_t input_tile = block * params.NUM_TILES_IN_BLOCK + tile;
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[input_tile]), UNPACK_FMT, UNPACK_FMT);
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_B[input_tile]), UNPACK_FMT, UNPACK_FMT);
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_C[input_tile]), UNPACK_FMT, UNPACK_FMT);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

// Named DstSync mode forwarded to the ternary SFPU dispatch template below.
static constexpr ckernel::DstSync DST_SYNC_MODE = ckernel::DstSync::SyncHalf;

#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t MATH_FMT = resolve_ternary_format(UNPACK_A_IN);

    // Compile-time math format for the SFPU template dispatch (mirrors UNPACK_A_IN).
    constexpr DataFormat MATH_FORMAT = static_cast<DataFormat>(UNPACK_A_IN);

    const bool is_int_fpu_en = false;

    _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FMT, MATH_FMT);

    // Multi-tile: each iteration copies one tile's three operands (a, b, c) into
    // Dest tiles 0, 1, 2 and runs the ternary SFPU op, writing the result to tile 0.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
            _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
                4 /* num_faces */, MATH_FMT);
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                0, MATH_FMT, MATH_FMT); // input a
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                1, MATH_FMT, MATH_FMT); // input b
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                2, MATH_FMT, MATH_FMT); // input c
            // Reset dest addressing before the sfpi-based ternary op (matches the binary
            // SFPU test): the addc kernels use sfpi dst_reg[...] which needs the dest RWC
            // at tile-0 base, not left advanced by the datacopies.
            _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

            // Ternary SFPU: out(tile 0) = f(a=0, b=1, c=2). VectorMode::RC drives 4 faces
            // (8 rows each) so the per-call ITERATIONS is 8, matching the production APIs.
            test_utils::call_ternary_sfpu_operation_init<SFPU_TERNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en>();
            test_utils::
                call_ternary_sfpu_operation<DST_SYNC_MODE, is_fp32_dest_acc_en, SFPU_TERNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, 8>(
                    0 /*DST_IN0*/, 1 /*DST_IN1*/, 2 /*DST_IN2*/, 0 /*DST_OUT*/, SFPU_TERNARY_SCALAR, VectorMode::RC);

            _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

static constexpr ckernel::DstSync DST_SYNC_MODE = ckernel::DstSync::SyncHalf;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t PACK_FMT = resolve_ternary_format(UNPACK_A_IN);

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(PACK_FMT, PACK_FMT, 16 * 16 * 4 /* tile_size */);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(PACK_FMT);

    _llk_pack_dest_init_wrapper_<DST_SYNC_MODE, is_fp32_dest_acc_en, PackMode::Default>();

    // Multi-tile: pack each result tile (always at Dest tile 0) to its L1 slot.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            const std::uint32_t result_tile = block * params.NUM_TILES_IN_BLOCK + tile;
            _llk_packer_wait_for_math_done_();
            _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[result_tile]));
            _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        }
    }
}

#endif
