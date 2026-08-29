// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// LLK SFPU typecast test kernel.
//
// Mirrors the production typecast compute kernel
//   ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp
// which does, per tile:  copy_tile (unpack A -> Dest) -> typecast_tile_init ->
// typecast_tile -> pack_tile.
//
// The numeric conversion is performed in-place in Dest by the SFPU
// `calculate_typecast_*` primitives, dispatched through the shared unary-SFPU
// entry points (call_unary_sfpu_operation in helpers/include/sfpu_operations.h)
// under SfpuType::typecast. Pairs realised purely by unpacker/packer format
// conversion run the same copy + pack path with no SFPU call.
//
// Compile-time configuration emitted by the Python harness:
//   TYPECAST_IN_FORMAT  : DataFormat of the L1 input  (typecast IN_DTYPE)
//   TYPECAST_OUT_FORMAT : DataFormat of the L1 output (typecast OUT_DTYPE)
//   APPROX_MODE         : SFPU approximation mode
//

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

using namespace ckernel;
using namespace ckernel::sfpu;

constexpr int iterations = 8;

// Fresh semantic-C++ discriminator for UInt16 -> Float16_b.  It deliberately
// names no physical LREG, raw instruction, replay range, or load-macro
// schedule.  The typed Dst layout is the architectural boundary; allocation,
// conversion scheduling, and any future macro formation belong to GCC.
//
// Default impl-1 form (Lane AV adoption, 2026-08-17): the body inlines into
// run_kernel and face traversal stays at typed setrwc boundaries, so the
// compiler's loop-scoped config-ownership window forms (one descriptor
// config per tile).  This is the probe form Lane AK proved on silicon
// (-DTYPECAST_TYPED_RWC_BOUNDARY, +3.27% vs hand, from +22.6% for the
// noinline form; silicon-promotions-20260817/item2), adopted as the default
// per the REDUCE_SDPA typed-boundary precedent.  The former noinline
// per-face-call form is kept reachable behind
// -DTYPECAST_NOINLINE_FACE_BOUNDARY as the measurable probe for the
// cross-function descriptor-sharing mechanism gap (IPA-lite; see
// corpus/TYPECAST_DESCRIPTOR_SHARING_BLOCKER.md).
template <int ITERATIONS>
#ifdef TYPECAST_NOINLINE_FACE_BOUNDARY
__attribute__((noinline)) void calculate_typecast_uint16_to_fp16b_semantic()
#else
__attribute__((always_inline)) inline void calculate_typecast_uint16_to_fp16b_semantic()
#endif
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vUInt16 input = sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>();
        sfpi::vFloat fp32 = sfpi::convert<sfpi::vFloat>(input, sfpi::RoundMode::Nearest);
        sfpi::vFloat16b fp16b = sfpi::convert<sfpi::vFloat16b>(fp32, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0].mode<sfpi::DataLayout::F16b>() = fp16b;
        sfpi::dst_reg++;
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Copy SrcA to Dest (datacopy A2D). The A2D copy preserves the raw datum
    // bits in Dest (the SFPU below interprets/converts them), so no separate
    // integer-FPU flag is needed here. dest_acc follows production
    // (fp32_dest_acc_en): integer pairs that production runs in 16-bit Dest stay
    // in 16-bit Dest here too.
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // Program the SFPU for this specific typecast pair (no-op for pairs handled
    // purely by unpacker/packer format conversion). Goes through the shared
    // unary-SFPU dispatch under SfpuType::typecast, with the (IN, OUT) format
    // pair supplied as the trailing template parameters.
    test_utils::call_unary_sfpu_operation_init<
        SFPU_UNARY_OPERATION,
        APPROX_MODE,
        is_fp32_dest_acc_en,
        iterations,
        false /* FAST_MODE */,
        false /* STABLE_SORT */,
        false /* CLAMP_NEGATIVE */,
        TYPECAST_IN_FORMAT,
        TYPECAST_OUT_FORMAT>();

    LLK_ASSERT(
        (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "NUM_TILES_IN_BLOCK exceeds max dest tiles");

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                block_tile, formats.math, formats.math);

            // In-place numeric typecast of the tile sitting in Dest.
            {
                START_PERF_MEASURE("TYPECAST_BODY")
            if constexpr (TYPECAST_IMPL == 1)
            {
#ifdef TYPECAST_NOINLINE_FACE_BOUNDARY
                // Legacy noinline form: the semantic body describes one
                // 8-row face behind an out-of-line call per face, so the
                // descriptor configuration re-executes per face — the
                // measurable probe for the cross-function descriptor-sharing
                // mechanism gap (IPA-lite).
                _llk_math_eltwise_unary_sfpu_params_(calculate_typecast_uint16_to_fp16b_semantic<iterations>, block_tile, VectorMode::RC);
#else
                // Default (adopted probe form): compiler-visible face
                // traversal at typed setrwc boundaries.  The whole tile is
                // one compiler-visible region, so descriptor ownership is
                // provable across the four architectural RC faces and the
                // loop-scoped config window forms once per tile.
                _llk_math_eltwise_sfpu_start_(block_tile);
#pragma GCC unroll 0
                for (int face = 0; face < 4; ++face)
                {
                    calculate_typecast_uint16_to_fp16b_semantic<iterations>();
                    lltt::setrwc<0, 4, 8, 0, 0, 4>();
                    lltt::setrwc<0, 4, 8, 0, 0, 4>();
                }
                _llk_math_eltwise_sfpu_done_();
#endif
            }
            else
            {
                test_utils::call_unary_sfpu_operation<
                DST_SYNC,
                is_fp32_dest_acc_en,
                SFPU_UNARY_OPERATION,
                APPROX_MODE,
                is_fp32_dest_acc_en,
                iterations,
                false /* FAST_MODE */,
                false /* STABLE_SORT */,
                false /* CLAMP_NEGATIVE */,
                TYPECAST_IN_FORMAT,
                TYPECAST_OUT_FORMAT>(block_tile);
            }
            }
        }
        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    LLK_ASSERT(
        (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "NUM_TILES_IN_BLOCK exceeds max dest tiles");

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                block_tile, L1_ADDRESS(params.buffer_Res[block_start * params.NUM_TILES_IN_BLOCK + block_tile]));
        }
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
