// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sum + reduce-to-scalar LLK test (experimental, Blackhole only).
//
// Mirrors the ttnn compute-kernel flow of ckernel::sum_reduce_scalar_tile
// (api/compute/experimental/sum_reduce_scalar.h). That op is the sum-only
// counterpart of mul_reduce_scalar_tile: instead of an ELWMUL multiply phase
// with an all-ones second operand, it copies each input tile into DEST via
// datacopy (A2D), then runs the *identical* mul_reduce_scalar reduce tail.
//
//   1. Copy phase:   DEST[i] = A[i]  (datacopy A2D, no second operand).
//   2. Switch UNPACK -> reduce phase (DEST reused as SrcA/SrcB via MOVD2A/MOVD2B).
//   3. Fill SrcB with the scaler, clear DEST[0].
//   4. Column-reduce every tile with GAPOOL, accumulating into DEST[0].
//   5. Collapse DEST[0] to a single scalar (transpose + GAPOOL).
// The packer applies the REDUCE_SCALAR mask so only element [0] is emitted.
//
// GOLDEN (see sum_reduce_scalar.h):
//   result = sum(all elements of all tiles) * scaler^2   -> DEST[0]
// The scaler is loaded into SrcB once and applied by BOTH GAPOOL passes (the
// per-tile column accumulate and the final scalar collapse), so it lands on
// the result twice. This test uses scaler == 1.0 (the Compute API default), so
// scaler^2 == 1 and the golden is simply sum(A) over all tiles/elements.
// Only element [0] of the output is defined; every other lane is unspecified.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// The reduction collapses everything into DEST[0], so every DEST access in this
// kernel targets index 0. Name it once instead of repeating the bare literal.
static constexpr std::uint32_t DST_INDEX = 0;

// The tile geometry is parametrized by the Python test: the full 32x32 tile
// (2x2 faces, num_faces=4) plus the 16x32 (1x2, num_faces=2) and 16x16 (1x1,
// num_faces=1) "tiny tiles", and 1x32 to 8x32 tiles with short faces
// (TEST_FACE_R_DIM rows). Each thread rebuilds the shape from the runtime
// params, matching the reduce_test.cpp idiom.

// Poison op: fills every row of both SrcA banks with 100.0 (unpack A zeroes SrcB), then releases DEST unread.
// A short-face op that reads those stale rows fails the golden.
static constexpr std::uint32_t POISON_TILES = 2;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_mul_reduce_scalar.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    {
        START_PERF_MEASURE("INIT")
        // compute_kernel_hw_startup
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            DEFAULT_TENSOR_SHAPE.total_num_faces(),
            DEFAULT_TENSOR_SHAPE.total_num_faces());

        // Poison op (see the file header).
        const std::uint32_t poison_tile_bytes = SCALE_DATUM_SIZE(formats.unpack_A_src, DEFAULT_TENSOR_SHAPE.total_tensor_size());
        _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/, DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        for (std::uint32_t i = 0; i < POISON_TILES; ++i)
        {
            _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[params.TILE_CNT] + i * poison_tile_bytes), formats.unpack_A_src, formats.unpack_A_dst);
        }
        _llk_unpack_reconfig_tile_shape_srca_(0 /*tile_size*/, tensor_shape.face_r_dim, num_faces);

        // sum_reduce_scalar_init -> copy_init: unpack A only,
        // no broadcast/transpose (datacopy A2D source path).
        _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                // Copy phase: generic unpack_A NONE posts SrcA + dummy SrcB per face.
                _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(params.TILE_CNT * num_faces);
                _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(1); // switch_to_reduce dummy SrcA+SrcB
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                }
                _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_fill.h"
#include "params.h"

// Scaler multiplier applied to the reduction (matches the Compute API default).
// scaler == 1.0 => scaler^2 == 1.0, so the golden is a plain sum(A).
static constexpr float REDUCE_SCALER = 1.0f;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const std::uint32_t tile_cnt            = params.TILE_CNT;
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    {
        START_PERF_MEASURE("INIT")
        // compute_kernel_hw_startup
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_sfpu_init_once_();

        // Poison op (see the file header).
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(DEFAULT_TENSOR_SHAPE, formats.math);
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        for (std::uint32_t i = 0; i < POISON_TILES; ++i)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                i, formats.math, formats.math, DEFAULT_TENSOR_SHAPE);
        }
        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

        // sum_reduce_scalar_init -> copy_init: datacopy A2D.
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(tensor_shape, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(tile_cnt * tensor_shape.total_num_faces());
                _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(1);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC>();
                }

                for (std::uint32_t i = 0; i < tile_cnt; ++i)
                {
                    LLK_ASSERT(
                        (i < get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Copy tile index exceeds maximum destination tiles");
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        i, formats.math, formats.math, tensor_shape);
                }

                _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false /* enforce_fp32_accumulation */>();

                _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(DST_INDEX);
                _llk_math_eltwise_unary_sfpu_params_(
                    ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, REDUCE_SCALER);
                _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(DST_INDEX);
                _llk_math_eltwise_unary_sfpu_params_(
                    ckernel::sfpu::_calculate_fill_<false /* APPROX */, 2 /* ITERATIONS */>, DST_INDEX, VectorMode::RC_custom, 0.0f /* clear DEST[0] */);

                _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
                for (std::uint32_t i = 1; i < tile_cnt; ++i)
                {
                    _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i);
                    _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
                }

                _llk_math_mul_reduce_scalar_<MATH_FIDELITY>();
                _llk_math_mul_reduce_scalar_clear_dvalid_();

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face);

        _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
            formats.pack_src, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);

        _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>();

        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

        // Poison op (see the file header): release its DEST section without packing it.
        _llk_packer_wait_for_math_done_();
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(params.buffer_Res[0]));
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_pack_reduce_mask_clear_();
}

#endif
