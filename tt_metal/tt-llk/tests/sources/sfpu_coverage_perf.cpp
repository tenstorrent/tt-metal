// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Perf twin of sfpu_coverage_test.cpp (lane EU LLK-coverage-expansion):
// TILE_LOOP / MATH_ISOLATE cycles-per-tile for the corpus-uncovered SFPU
// kernels vs their fresh semantic bodies.  COVERAGE_OP / COVERAGE_SUBOP /
// FRESH_CPP_IMPL select the raced arm exactly as in the correctness twin.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context                          = 0;
std::uint32_t pack_sync_tile_dst_ptr                   = 0;
std::uint32_t math_sync_tile_dst_index                 = 0;
static constexpr std::uint32_t MAX_TILES_DEST          = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;

#ifndef COVERAGE_OP
#define COVERAGE_OP 1
#endif
#ifndef COVERAGE_SUBOP
#define COVERAGE_SUBOP 0
#endif
#ifndef FRESH_CPP_IMPL
#define FRESH_CPP_IMPL 0
#endif

static constexpr bool COVERAGE_TWO_TILE      = (COVERAGE_OP == 8 || COVERAGE_OP == 9);
static constexpr std::uint32_t COVERAGE_STEP = COVERAGE_TWO_TILE ? 2 : 1;

// Fixed op parameters — must match sfpu_coverage_test.cpp and the goldens.
static constexpr std::uint32_t SKF_BANK_MASK             = 0x3F;
static constexpr std::uint32_t SKF_MY_BANK               = 5;
static constexpr std::uint32_t SKF_GLOBAL_BANK_SHIFT     = 10;
static constexpr std::uint32_t SKF_WITHIN_BANK_MASK      = 0x3FF;
static constexpr std::uint32_t SKF_OUT_SHIFT             = 0;
static constexpr int ZERO_PAD_VALID_ROWS                 = 24;
static constexpr int ZERO_PAD_TOTAL_ROWS                 = 32;
static constexpr float SMOOTHSTEP_EDGE0                  = -0.5f;
static constexpr float SMOOTHSTEP_EDGE1                  = 0.5f;
static constexpr float SMOOTHSTEP_INV_DELTA              = 1.0f;
static constexpr std::uint32_t SMOOTHSTEP_EDGE0_BITS     = 0xBF000000u; // -0.5f
static constexpr std::uint32_t SMOOTHSTEP_INV_DELTA_BITS = 0x3F800000u; // 1.0f

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);

        _llk_unpack_A_init_<BROADCAST_TYPE, false /* acc_to_dest */, reuse_dest_type, unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES,
            UNPACK_TRANSPOSE_WITHIN_FACE,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Only unavoidable hardware synchronization on the math-isolate leg.
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_loop_set_valid<
                    /* src A */ true,
                    /* src B */ is_fp32_dest_acc_en>(
                    /* iterations*/ num_faces * TILE_CNT * LOOP_FACTOR);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE) // UNPACK_ISOLATE, L1_TO_L1, L1_CONGESTION
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, false /* acc_to_dest */, reuse_dest_type, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, /* tile_idx */ i), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

// SFPU call/init macro layer FIRST: its init header forward-declares the
// production init functions, and -Wredundant-decls requires declarations to
// precede the kernel headers' definitions (own include block so clang-format
// keeps the order).
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
// The legacy Converter helper several production kernels expect in scope
// (the sfpu_operations.h include-order convention).
#include "sfpu/ckernel_sfpu_converter.h"
// Byte-untouched production kernels under race (hand arms).
#include "experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h"
#include "experimental/llk_sfpu/ckernel_sfpu_custom_add.h"
#include "experimental/llk_sfpu/ckernel_sfpu_smoothstep.h"
#include "llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h"
#include "llk_sfpu/ckernel_sfpu_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_copy_dest_values.h"
#include "llk_sfpu/ckernel_sfpu_int_sum.h"
#include "llk_sfpu/ckernel_sfpu_tiled_prod.h"
#include "sfpu/experimental/ckernel_sfpu_sparse_k_filter.h"
#include "sfpu/experimental/ckernel_sfpu_zero_pad.h"
// Fresh semantic bodies (sem arms; storm contract, fresh_cpp/README.md).
#include "fresh_cpp/addrsqrt.h"
#include "fresh_cpp/copydest.h"
#include "fresh_cpp/customadd.h"
#include "fresh_cpp/intsum.h"
#include "fresh_cpp/rotate90.h"
#include "fresh_cpp/smoothstep.h"
#include "fresh_cpp/sparsekfilter.h"
#include "fresh_cpp/tiledprod.h"
#include "fresh_cpp/unarybitwise.h"
#include "fresh_cpp/zeropad.h"

// One raced-op invocation at Dst tile `block_tile` (the selector chains of the
// correctness twin, kept in a single function so the perf loops stay uniform).
static inline void coverage_op_once(const std::uint32_t block_tile)
{
#if FRESH_CPP_IMPL == 0
#if COVERAGE_OP == 1
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_alt_complex_rotate90, (APPROX_MODE, 4), block_tile, VectorMode::RC);
#elif COVERAGE_OP == 2
#if COVERAGE_SUBOP == 0
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_unary_bitwise,
        (APPROX_MODE, sfpu::UnaryBitwiseOp::AND, DataFormat::Int32, 8),
        block_tile,
        VectorMode::RC,
        SFPU_UNARY_SCALAR);
#elif COVERAGE_SUBOP == 1
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_unary_bitwise,
        (APPROX_MODE, sfpu::UnaryBitwiseOp::OR, DataFormat::Int32, 8),
        block_tile,
        VectorMode::RC,
        SFPU_UNARY_SCALAR);
#else
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_unary_bitwise,
        (APPROX_MODE, sfpu::UnaryBitwiseOp::XOR, DataFormat::Int32, 8),
        block_tile,
        VectorMode::RC,
        SFPU_UNARY_SCALAR);
#endif
#elif COVERAGE_OP == 3
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_add_rsqrt,
        (APPROX_MODE, 8, is_fp32_dest_acc_en, false /* FAST_APPROX */),
        block_tile,
        VectorMode::RC,
        SFPU_UNARY_SCALAR);
#elif COVERAGE_OP == 4
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, is_fp32_dest_acc_en, smoothstep_tile_face, block_tile, VectorMode::RC, SMOOTHSTEP_EDGE0, SMOOTHSTEP_EDGE1, SMOOTHSTEP_INV_DELTA);
#elif COVERAGE_OP == 5
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tiled_prod, (APPROX_MODE, 8), block_tile, VectorMode::None);
#elif COVERAGE_OP == 6
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, _zero_pad_tile_, (is_fp32_dest_acc_en, ZERO_PAD_VALID_ROWS, ZERO_PAD_TOTAL_ROWS), block_tile, VectorMode::None);
#elif COVERAGE_OP == 7
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _sparse_k_filter_tile_,
        (32 /* ITERATIONS */, SKF_BANK_MASK, SKF_MY_BANK, SKF_GLOBAL_BANK_SHIFT, SKF_WITHIN_BANK_MASK, SKF_OUT_SHIFT),
        block_tile,
        VectorMode::None);
#elif COVERAGE_OP == 8
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, is_fp32_dest_acc_en, my_add_tile_face, block_tile, VectorMode::RC, 0 /* in0 */, 1 /* in1 */, 0 /* out */);
#elif COVERAGE_OP == 9
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        copy_dest_value,
        (DataFormat::Float16_b, APPROX_MODE, 8),
        block_tile,
        VectorMode::RC,
        0 /* in */,
        1 /* out */,
        0 /* unused */);
#elif COVERAGE_OP == 10
#if COVERAGE_SUBOP == 0
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sum_int_col, (APPROX_MODE), block_tile, VectorMode::None);
#else
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sum_int_row, (APPROX_MODE), block_tile, VectorMode::None);
#endif
#else
#error "unknown COVERAGE_OP"
#endif

#else // FRESH_CPP_IMPL == 1
#if COVERAGE_OP == 1
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_rotate90_fresh_cpp, (16), block_tile, VectorMode::None);
#elif COVERAGE_OP == 2
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_unary_bitwise_fresh_cpp, (COVERAGE_SUBOP, 32), block_tile, VectorMode::None, SFPU_UNARY_SCALAR);
#elif COVERAGE_OP == 3
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_add_rsqrt_fresh_cpp, (32), block_tile, VectorMode::None, SFPU_UNARY_SCALAR);
#elif COVERAGE_OP == 4
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_smoothstep_fresh_cpp,
        (32),
        block_tile,
        VectorMode::None,
        SMOOTHSTEP_EDGE0_BITS,
        SMOOTHSTEP_INV_DELTA_BITS);
#elif COVERAGE_OP == 5
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_tiled_prod_fresh_cpp, (9), block_tile, VectorMode::None);
#elif COVERAGE_OP == 6
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_zero_pad_fresh_cpp, (ZERO_PAD_VALID_ROWS, ZERO_PAD_TOTAL_ROWS), block_tile, VectorMode::None);
#elif COVERAGE_OP == 7
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sparse_k_filter_fresh_cpp,
        (SKF_BANK_MASK, SKF_MY_BANK, SKF_GLOBAL_BANK_SHIFT, SKF_WITHIN_BANK_MASK, SKF_OUT_SHIFT, 32),
        block_tile,
        VectorMode::None);
#elif COVERAGE_OP == 8
    sfpu::call_custom_add_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(block_tile, block_tile + 1, block_tile, VectorMode::RC);
#elif COVERAGE_OP == 9
    sfpu::call_copy_dest_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(block_tile, block_tile + 1, VectorMode::RC);
#elif COVERAGE_OP == 10
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_int_sum_fresh_cpp, (COVERAGE_SUBOP), block_tile, VectorMode::None);
#else
#error "unknown COVERAGE_OP"
#endif
#endif // FRESH_CPP_IMPL
}

// Hand-arm init frame (production inits are the hand kernels' own; the fresh
// arms are self-contained).
static inline void coverage_op_init()
{
#if FRESH_CPP_IMPL == 0
#if COVERAGE_OP == 1
    sfpu::alt_complex_rotate90_init();
#elif COVERAGE_OP == 2
#if COVERAGE_SUBOP == 0
    sfpu::bitwise_and_init();
#elif COVERAGE_SUBOP == 1
    sfpu::bitwise_or_init();
#else
    sfpu::bitwise_xor_init();
#endif
#elif COVERAGE_OP == 3
    sfpu::init_add_rsqrt<APPROX_MODE>();
#elif COVERAGE_OP == 5
    sfpu::tiled_prod_init();
#elif COVERAGE_OP == 9
    sfpu::copy_dest_value_init();
#elif COVERAGE_OP == 10
    sfpu::sum_int_init<APPROX_MODE>();
#endif
#endif
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const DataCopyType data_copy_type = DataCopyType::A2D;

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        SFPU_UNARY_INIT(unused);
        coverage_op_init();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    if constexpr (unpack_to_dest)
                    {
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            i % MAX_TILES_DEST, formats.math, formats.math);
                    }
                    else
                    {
                        _perf_math_loop_clear_valid<
                            /* src A */ true,
                            /* src B */ true>(
                            /* iterations*/ num_faces);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    int block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (int block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        if constexpr (unpack_to_dest)
                        {
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                        else
                        {
                            _perf_math_loop_clear_valid<
                                /* src A */ true,
                                /* src B */ true>(
                                /* iterations*/ num_faces);
                        }
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile += COVERAGE_STEP)
                    {
                        // When data is not unpacked to dest, math copies srcA into dest before the SFPU op.
                        if constexpr (!unpack_to_dest)
                        {
                            LLK_ASSERT(
                                (block_tile + COVERAGE_STEP - 1 < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "block_tile exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                            if constexpr (COVERAGE_TWO_TILE)
                            {
                                _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                    block_tile + 1, formats.math, formats.math);
                            }
                        }

                        coverage_op_once(block_tile);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile += COVERAGE_STEP)
                    {
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);
                        if constexpr (COVERAGE_TWO_TILE)
                        {
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile + 1, formats.math, formats.math);
                        }

                        coverage_op_once(block_tile);
                    }

                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);

        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
