// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf kernel for the SFPU ops that fit neither the unary registry sweep nor the binary one: rand, dropout, mask
// (float and Int32), copy_dest_values, reshuffle_rows, softcap, situ_glu and clamped_silu_glu. The unpack and pack
// threads and the zones are those of eltwise_unary_sfpu_perf.cpp; the math thread runs the datacopy carrier and one
// body per tile in the compute API form (four faces of eight rows, VectorMode::RC). The two-operand bodies copy the
// tile into DEST tiles 0 and 1 and write tile 0; reshuffle_rows accumulates DEST tile 0 into tile 1 with a 32-byte
// index array the math thread writes into the unused input ring C once, in the INIT zone.
//   SFPU_MISC_OPERATION   : 0 rand, 1 dropout, 2 mask (float), 3 copy_dest_values, 4 reshuffle_rows, 5 softcap,
//                           6 situ_glu, 7 clamped_silu_glu, 8 mask (Int32)   (SFPU_MISC_OPERATIONS in
//                           python_tests/helpers/test_variant_parameters.py)
//   SFPU_MISC_PARAM       : rand 0 = scale 1.0 (the normalisation folded into the scale, 16 instructions per row),
//                           1 = scale 2^-100 (the per-row normalise, 17); reshuffle_rows 0 = identity, 1 = reversed,
//                           2 = every second row skipped
//   SFPU_MISC_INIT_PER_TILE: re-run the op's init before every tile (the ttnn unary kernel pattern)


#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
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

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;

    const std::uint32_t TILE_CNT = params.TILE_CNT;

    const bool UNPACK_TRANSPOSE_FACES       = params.UNPACK_TRANSPOSE_FACES;
    const bool UNPACK_TRANSPOSE_WITHIN_FACE = params.UNPACK_TRANSPOSE_WITHIN_FACE;

    const auto& buffer_A = params.buffer_A;
#endif
    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);

        _llk_unpack_A_init_<BROADCAST_TYPE, false, reuse_dest_type, unpack_to_dest>(
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
            if (!unpack_to_dest)
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
                    _llk_unpack_A_<BROADCAST_TYPE, false /* acc_to_dest (see init) */, reuse_dest_type, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
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
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_rand.h"
#include "llk_sfpu/ckernel_sfpu_dropout.h"
#include "llk_sfpu/ckernel_sfpu_mask.h"
#include "llk_sfpu/ckernel_sfpu_copy_dest_values.h"
#include "llk_sfpu/ckernel_sfpu_reshuffle_rows.h"
#include "llk_sfpu/ckernel_sfpu_softcap.h"
#include "llk_sfpu/ckernel_sfpu_situ_glu.h"
#include "llk_sfpu/ckernel_sfpu_clamped_silu_glu.h"

// The bodies that read two DEST tiles (0 and 1) and write tile 0.
static constexpr bool MISC_BINARY = (SFPU_MISC_OPERATION == 2 || SFPU_MISC_OPERATION == 3 || SFPU_MISC_OPERATION == 6 || SFPU_MISC_OPERATION == 7 || SFPU_MISC_OPERATION == 8);
// rand: from 0.0f; scale 1.0f (the normalisation folded into the scale, the 16-instruction row) or 2^-100 (SFPU_MISC_PARAM 1: the per-row normalise, 17)
static constexpr std::uint32_t RAND_FROM    = 0x00000000u;
static constexpr std::uint32_t RAND_SCALE   = (SFPU_MISC_PARAM == 1) ? 0x0D800000u : 0x3F800000u;
static constexpr std::uint32_t DROPOUT_PROBABILITY    = 0x3FFFFFFFu; // 0.5 of INT_MAX
static constexpr std::uint32_t DROPOUT_SCALE   = 0x40000000u; // 2.0f
static constexpr std::uint32_t SOFTCAP_BETA = 0x40800000u; // 4.0f
static constexpr std::uint32_t SOFTCAP_BETA_RECIP  = 0x3E800000u; // 0.25f
static constexpr std::uint32_t RESHUFFLE_INDEX_L1       = PERF_INPUT_C; // L1 byte address of the reshuffle index array (the unused input ring C); the body reads idx_addr + 16
static constexpr DataFormat MISC_MATH_FORMAT_RAW    = static_cast<DataFormat>(formats.math);
static constexpr DataFormat MISC_MATH_FORMAT        = (MISC_MATH_FORMAT_RAW == DataFormat::Tf32) ? DataFormat::Float32 : MISC_MATH_FORMAT_RAW;

inline void write_reshuffle_index()
{
    volatile std::uint8_t* p = reinterpret_cast<volatile std::uint8_t*>(RESHUFFLE_INDEX_L1 + 16);
    for (std::uint32_t r = 0; r < 32; ++r)
    {
        std::uint8_t v = static_cast<std::uint8_t>(r); // SFPU_MISC_PARAM 0: the identity permutation
        if constexpr (SFPU_MISC_PARAM == 1) { v = static_cast<std::uint8_t>(31 - r); }        // reversed
        if constexpr (SFPU_MISC_PARAM == 2) { v = (r & 1) ? 255 : static_cast<std::uint8_t>(r); } // every second row skipped
        p[r] = v;
    }
}

inline void misc_op_init()
{
    if constexpr (SFPU_MISC_OPERATION == 0) { _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>(); ckernel::sfpu::rand_init<false>(0x12345678u); }
    else if constexpr (SFPU_MISC_OPERATION == 1) { _llk_math_eltwise_unary_sfpu_init_<SfpuType::dropout>(); ckernel::sfpu::dropout_init<false>(0x12345678u); }
    else if constexpr (SFPU_MISC_OPERATION == 2 || SFPU_MISC_OPERATION == 8) { _llk_math_eltwise_unary_sfpu_init_<SfpuType::mask>(); ckernel::sfpu::mask_init(); }
    else if constexpr (SFPU_MISC_OPERATION == 3) { _llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>(); }
    else if constexpr (SFPU_MISC_OPERATION == 4) { _llk_math_eltwise_unary_sfpu_init_<SfpuType::reshuffle_rows>(); ckernel::sfpu::reshuffle_rows_init(); }
    else if constexpr (SFPU_MISC_OPERATION == 5) { _llk_math_eltwise_unary_sfpu_init_<SfpuType::softcap>(); ckernel::sfpu::softcap_init(); }
    else if constexpr (SFPU_MISC_OPERATION == 6) { _llk_math_eltwise_binary_sfpu_init_<SfpuType::situ_glu>(); ckernel::sfpu::situ_glu_init(); }
    else if constexpr (SFPU_MISC_OPERATION == 7) { _llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>(); ckernel::sfpu::clamped_silu_glu_init(); }
}

inline void misc_op_body(std::uint32_t tile)
{
    if constexpr (SFPU_MISC_INIT_PER_TILE) { misc_op_init(); }
    if constexpr (SFPU_MISC_OPERATION == 0) { _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::rand<false>, tile, VectorMode::RC, RAND_FROM, RAND_SCALE); }
    else if constexpr (SFPU_MISC_OPERATION == 1) { _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_dropout<false, 8>, tile, VectorMode::RC, DROPOUT_PROBABILITY, DROPOUT_SCALE); }
    else if constexpr (SFPU_MISC_OPERATION == 2) { _llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::calculate_mask<true, 8>, 0, 1, 0, VectorMode::RC); }
    else if constexpr (SFPU_MISC_OPERATION == 8) { _llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::calculate_int_mask<true, 8>, 0, 1, 0, VectorMode::RC); }
    else if constexpr (SFPU_MISC_OPERATION == 3) { _llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::copy_dest_value<MISC_MATH_FORMAT, false, 8>, 0, 1, 0, VectorMode::RC); }
    else if constexpr (SFPU_MISC_OPERATION == 4) { _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_reshuffle_rows<false>, tile, VectorMode::RC_custom, RESHUFFLE_INDEX_L1); }
    else if constexpr (SFPU_MISC_OPERATION == 5) { _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_softcap<false, is_fp32_dest_acc_en, 8>, tile, VectorMode::RC, SOFTCAP_BETA, SOFTCAP_BETA_RECIP); }
    else if constexpr (SFPU_MISC_OPERATION == 6) { _llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::calculate_situ_glu<is_fp32_dest_acc_en, 8>, 0, 1, 0, VectorMode::RC); }
    else if constexpr (SFPU_MISC_OPERATION == 7) { _llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::calculate_clamped_silu_glu<is_fp32_dest_acc_en, 8>, 0, 1, 0, VectorMode::RC); }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    const DataCopyType data_copy_type = DataCopyType::A2D;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        if constexpr (SFPU_MISC_OPERATION == 4) { write_reshuffle_index(); }
        misc_op_init();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
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
                        _perf_math_loop_clear_valid<true, true>(num_faces);
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
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        const std::uint32_t t = MISC_BINARY ? 0 : block_tile;
                        if constexpr (!unpack_to_dest)
                        {
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                t, formats.math, formats.math);
                        }
                        misc_op_body(t);
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
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        const std::uint32_t t = MISC_BINARY ? 0 : block_tile;
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            t, formats.math, formats.math);
                        misc_op_body(t);
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
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const auto& buffer_Res          = params.buffer_Res;
#endif
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

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
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
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
