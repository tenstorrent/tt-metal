// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the bcast dest-reuse family (Blackhole only):
//   experimental/llk_math_bcast_dest_reuse.h (_llk_math_bcast_dest_reuse[_init]_)
//   experimental/llk_unpack_A_bcast_dest_reuse.h (_llk_unpack_A_bcast_dest_reuse_init_)
//
// Supports both BroadcastType::SCALAR and BroadcastType::COL across ELWADD, ELWSUB, and
// ELWMUL, SyncHalf and SyncFull, full and partial faces, and unpack_full_transpose.
//
// Sequence:
//   seed   A plain A2D datacopy lands one tile in DEST[0] (element [0] for SCALAR or
//          Face 0 column 0 rows 0..15 for COL). The rest of the tile is non-uniform so a
//          MOVD2B reading the wrong row/face shows up as a mismatch.
//   op     _llk_math_bcast_dest_reuse_ with src_index == dst_index == 0. MOVD2B pulls the
//          broadcast operand from DEST[0] into SrcB, the optional ZEROACC clears DEST, then
//          the MOP walks BCAST_NUM_TILES tiles of SrcA (unpacked in one _llk_unpack_A_ call).
//   pack   BCAST_NUM_TILES tiles.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t SEED_NUM_FACES            = (BCAST_TYPE == ckernel::BroadcastType::COL) ? BCAST_NUM_FACES : ckernel::TILE_NUM_FACES;
static constexpr ckernel::TensorShape SEED_TENSOR_SHAPE  = ckernel::tensor_shape_from_num_faces(ckernel::FACE_R_DIM, SEED_NUM_FACES);
static constexpr ckernel::TensorShape BCAST_TENSOR_SHAPE = ckernel::tensor_shape_from_num_faces(ckernel::FACE_R_DIM, BCAST_NUM_FACES);

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_A_bcast_dest_reuse.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
    const Operand& buffer_B         = params.buffer_B;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, SEED_NUM_FACES, SEED_NUM_FACES);

        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, SEED_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

        _llk_unpack_A_bcast_dest_reuse_init_<BCAST_NUM_TILES, BCAST_TYPE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
            BCAST_UNPACK_FULL_TRANSPOSE /* transpose_of_faces */, BCAST_UNPACK_FULL_TRANSPOSE /* within_face_16x16_transpose */, FACE_R_DIM, BCAST_NUM_FACES);
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
                _perf_unpack_stream_set_valid<true, true, PERF_RUN_TYPE>(SEED_NUM_FACES); // refresh the seed tile
                _perf_unpack_set_valid(ckernel::SrcB);
                _perf_unpack_stream_set_valid<true /*set_a*/, false /*set_b*/, PERF_RUN_TYPE>(BCAST_NUM_TILES * BCAST_NUM_FACES);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, SEED_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);

                if constexpr (BCAST_CLOBBER_UNPA_X_END)
                {
                    // Only the following bcast_dest_reuse init should repair this one-row span.
                    config_unpacker_x_end<p_setadc::UNP_A>(1 /*face_r_dim*/);
                }
                _llk_unpack_A_bcast_dest_reuse_init_<BCAST_NUM_TILES, BCAST_TYPE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    BCAST_UNPACK_FULL_TRANSPOSE /* transpose_of_faces */,
                    BCAST_UNPACK_FULL_TRANSPOSE /* within_face_16x16_transpose */,
                    FACE_R_DIM,
                    BCAST_NUM_FACES);
                // ONE call: the MOP itself walks all BCAST_NUM_TILES tiles from this base address.
                _llk_unpack_A_<BCAST_TYPE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(L1_ADDRESS(buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_bcast_dest_reuse.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        if constexpr (BCAST_PRESERVE_SRC_ZERO_FLAG)
        {
            // Leave the Src zero flag at keep, as copy_init does; the bcast_dest_reuse init must restore the default.
            math::_configure_preserve_zero_flag_state_();
        }

        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(SEED_TENSOR_SHAPE, formats.math);

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _perf_math_stream_clear_valid<true, true, PERF_RUN_TYPE>(SEED_NUM_FACES); // seed tile
                _perf_math_clear_valid(ckernel::SrcB);
                _perf_math_stream_clear_valid<true /*clear_a*/, false /*clear_b*/, PERF_RUN_TYPE>(BCAST_NUM_TILES * BCAST_NUM_FACES);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                }
                _llk_math_eltwise_unary_datacopy_init_wrapper_<
                    DataCopyType::A2D,
                    is_fp32_dest_acc_en,
                    BroadcastType::NONE,
                    false /* is_int_fpu_en */,
                    PackMode::Default>(SEED_TENSOR_SHAPE, formats.math);
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    0 /* dst_index */, formats.math, formats.math);
                _llk_math_bcast_dest_reuse_init_<ELTWISE_BINARY_OP, BCAST_TYPE, BCAST_NUM_TILES, MATH_FIDELITY>(BCAST_TENSOR_SHAPE, 0 /* acc_to_dest */);
                _llk_math_bcast_dest_reuse_<ELTWISE_BINARY_OP, BCAST_TYPE, BCAST_NUM_TILES, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY, BCAST_CLEAR_DEST>(
                    0 /* src_index */, 0 /* dst_index */, BCAST_TENSOR_SHAPE);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * SEED_NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, SEED_NUM_FACES);
        _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();
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
                for (std::uint32_t tile = 0; tile < BCAST_NUM_TILES; ++tile)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[tile]));
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
