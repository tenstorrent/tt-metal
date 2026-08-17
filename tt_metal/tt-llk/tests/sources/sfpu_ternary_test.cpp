// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "counters.h"
#include "llk_defs.h"
#include "profiler.h"

#ifndef TTNN_WHERE_IMPL
#define TTNN_WHERE_IMPL 0
#endif

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

#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"
#include "fresh_cpp_operations.h"

#ifndef FRESH_CPP_IMPL
#define FRESH_CPP_IMPL 0
#endif

namespace ckernel::sfpu
{

// Test-only compiler-flow spelling of the production where operation.  Keep
// the loads and store bit-preserving: where chooses an operand, it does not
// numerically convert it.  The condition is loaded bitwise (U16/U32) and
// tested != 0 — the handwritten kernel's own protocol (ckernel_sfpu_where.h
// loads the condition raw LO16/INT32 and predicates on SFPSETCC LREG_EQ0).
// Condition-mode unification (Lane R3 / where-adjudication 2026-08-17): with
// every load and the store in one launch-sourced mod0, the macro planner
// derives the compact separator-absorbed select calendar (misc 0x770) for
// fp16b/Float32 too, instead of the separator-kept 4-slot form (misc 0x706)
// that silicon adjudication proved delivery-divergent.  Semantics note: raw
// !=0 differs from float truthiness only on -0.0 and (if HW flushed them)
// denormal conditions; the suite's condition stimuli (uniform [0,1), exact
// 0/1 patterns) contain neither, and the handwritten kernel has always used
// the bitwise protocol.
template <DataFormat FORMAT, int ITERATIONS>
sfpi_inline void calculate_where_generated(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2,
    const std::uint32_t dst_index_out)
{
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d)
    {
        if constexpr (FORMAT == DataFormat::Float16_b)
        {
            sfpi::vUInt condition  = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt true_bits = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt false_bits = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U16>();
            sfpi::vUInt result = false_bits;
            v_if (condition != 0u)
            {
                result = true_bits;
            }
            v_endif;
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::U16>() = result;
        }
        else if constexpr (FORMAT == DataFormat::Float32)
        {
            sfpi::vUInt condition  = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt true_bits = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt false_bits = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt result = false_bits;
            v_if (condition != 0u)
            {
                result = true_bits;
            }
            v_endif;
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::U32>() = result;
        }
        else
        {
            static_assert(FORMAT == DataFormat::Int32 || FORMAT == DataFormat::UInt32);
            sfpi::vInt condition = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
            sfpi::vUInt true_bits = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt false_bits = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U32>();
            sfpi::vUInt result = false_bits;
            v_if (condition != 0)
            {
                result = true_bits;
            }
            v_endif;
            sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::U32>() = result;
        }

        sfpi::dst_reg++;
    }
}

template <DstSync DST_SYNC_MODE, bool DST_ACCUM_MODE, DataFormat FORMAT, int ITERATIONS>
inline void call_where_generated(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2,
    const std::uint32_t dst_index_out,
    VectorMode vector_mode)
{
    SFPU_TERNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_where_generated,
        (FORMAT, ITERATIONS),
        dst_index_in0,
        dst_index_in1,
        dst_index_in2,
        dst_index_out,
        vector_mode);
}

} // namespace ckernel::sfpu

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t MATH_FMT = resolve_ternary_format(UNPACK_A_IN);

    // Compile-time math format for the SFPU template dispatch (mirrors UNPACK_A_IN).
    constexpr DataFormat MATH_FORMAT = static_cast<DataFormat>(UNPACK_A_IN);

    const bool is_int_fpu_en = false;

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FMT, MATH_FMT);

    // Multi-tile: each iteration copies one tile's three operands (a, b, c) into
    // Dest tiles 0, 1, 2 and runs the ternary SFPU op, writing the result to tile 0.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
                4 /* num_faces */, MATH_FMT);
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                0, MATH_FMT, MATH_FMT); // input a
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                1, MATH_FMT, MATH_FMT); // input b
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                2, MATH_FMT, MATH_FMT); // input c
            // Reset dest addressing before the sfpi-based ternary op (matches the binary
            // SFPU test): the addc kernels use sfpi dst_reg[...] which needs the dest RWC
            // at tile-0 base, not left advanced by the datacopies.
            _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

            // Ternary SFPU: out(tile 0) = f(a=0, b=1, c=2). VectorMode::RC drives 4 faces
            // (8 rows each) so the per-call ITERATIONS is 8, matching the production APIs.
            static_assert(TTNN_WHERE_IMPL <= 1, "Unknown TTNNWhere implementation selector");
            if constexpr (TTNN_WHERE_IMPL == 0)
            {
                test_utils::call_ternary_sfpu_operation_init<SFPU_TERNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en>();
            }
            else
            {
                SFPU_TERNARY_INIT(where);
            }
            {
                START_PERF_MEASURE("TTNN_WHERE_BODY")
                if constexpr (FRESH_CPP_IMPL == 1 && SFPU_TERNARY_OPERATION == SfpuType::addcmul)
                {
                    SFPU_TERNARY_CALL(
                        dest_sync,
                        is_fp32_dest_acc_en,
                        calculate_addcmul_fresh_cpp,
                        (is_fp32_dest_acc_en, MATH_FORMAT, 8),
                        0 /*DST_IN0*/,
                        1 /*DST_IN1*/,
                        2 /*DST_IN2*/,
                        0 /*DST_OUT*/,
                        VectorMode::RC,
                        SFPU_TERNARY_SCALAR);
                }
                else if constexpr (TTNN_WHERE_IMPL == 0)
                {
                    test_utils::call_ternary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, SFPU_TERNARY_OPERATION, APPROX_MODE, is_fp32_dest_acc_en, MATH_FORMAT, 8>(
                        0 /*DST_IN0*/, 1 /*DST_IN1*/, 2 /*DST_IN2*/, 0 /*DST_OUT*/, SFPU_TERNARY_SCALAR, VectorMode::RC);
                }
                else
                {
                    ckernel::sfpu::call_where_generated<dest_sync, is_fp32_dest_acc_en, MATH_FORMAT, 8>(
                        0 /*DST_IN0*/, 1 /*DST_IN1*/, 2 /*DST_IN2*/, 0 /*DST_OUT*/, VectorMode::RC);
                }
            }

            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t PACK_FMT = resolve_ternary_format(UNPACK_A_IN);

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(PACK_FMT, PACK_FMT, 16 * 16 * 4 /* tile_size */);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(PACK_FMT);

    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();

    // Multi-tile: pack each result tile (always at Dest tile 0) to its L1 slot.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            const std::uint32_t result_tile = block * params.NUM_TILES_IN_BLOCK + tile;
            _llk_packer_wait_for_math_done_();
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[result_tile]));
            _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}

#endif
