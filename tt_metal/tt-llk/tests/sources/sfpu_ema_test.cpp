// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared functional/perf driver for the EMA SFPU entry (ckernel_sfpu_ema.h via
// llk_math_ema_sfpu_entry.h). The entry is a stateful, two-tile kernel:
//   * input tile is read from dst index 0,
//   * output tile is written to dst index 1,
//   * the running EMA (EMA_old) is held in LREG4 and carried across tiles.
// EMA_new = alpha * EMA_old + beta * input, computed per column down the 32 rows
// (32 columns processed in parallel). Consecutive input tiles continue the row
// (time) sequence, so a [TILE_CNT*32, 32] input is an EMA over TILE_CNT*32 steps
// for 32 parallel channels.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// The EMA entry hard-codes input at dst tile 0 and output at dst tile 1.
static constexpr std::uint32_t EMA_INPUT_DST_INDEX  = 0;
static constexpr std::uint32_t EMA_OUTPUT_DST_INDEX = 1;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

template <typename Params>
std::uint32_t unpack_input_address(const Params& params, std::uint32_t tile)
{
#ifdef LLK_PROFILER
    (void)params;
    return PERF_ADDRESS(PERF_INPUT_A, tile);
#else
    return L1_ADDRESS(params.buffer_A[tile]);
#endif
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
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
            // Math isolate keeps only the hardware handshake; one SrcA valid per face
            // must match math retirements exactly or either side hangs.
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_stream_set_valid</* src A */ true, /* src B */ is_fp32_dest_acc_en, PERF_RUN_TYPE>(num_faces * TILE_CNT * LOOP_FACTOR /* iterations */);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        unpack_input_address(params, tile), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/llk_math_ema_sfpu_entry.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS)
{
    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats.math);
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

        // Clear the state once so LREG4 carries the recurrence across every tile
        // and, for perf, every amortization loop.
        llk_math_ema_sfpu_init();
        llk_math_ema_sfpu_load_alpha_beta(EMA_ALPHA_BITS, EMA_BETA_BITS);
        llk_math_ema_sfpu_clear_previous_output();

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    if constexpr (unpack_to_dest)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            EMA_INPUT_DST_INDEX, formats.math, formats.math);
                    }
                    else
                    {
                        // unpack_A publishes a SrcB valid with every SrcA valid, so retire both.
                        _perf_math_loop_clear_valid</* clear A */ true, /* clear B */ true>(num_faces);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // The 16-bit datacopy retires the SrcA valids produced by unpack; replacing
            // it with a bare TTI_CLEARDVALID hangs the math thread.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    if constexpr (!unpack_to_dest)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            EMA_INPUT_DST_INDEX, formats.math, formats.math);
                    }
                    llk_math_ema_sfpu_tile(EMA_INPUT_DST_INDEX);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC>();
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        EMA_INPUT_DST_INDEX, formats.math, formats.math);

                    if constexpr (EMA_BINARY_INIT_BEFORE_EMA)
                    {
                        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(
                            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), 0 /* acc_to_dest */);
                    }

                    llk_math_ema_sfpu_tile(EMA_INPUT_DST_INDEX);

                    if constexpr (EMA_BINARY_INIT_BEFORE_EMA)
                    {
                        _llk_math_eltwise_unary_datacopy_init_wrapper_<
                            DataCopyType::A2D,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            false /* is_int_fpu_en */,
                            PackMode::Default>(ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats.math);
                    }

                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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

template <typename Params>
std::uint32_t pack_output_address(const Params& params, std::uint32_t tile)
{
#ifdef LLK_PROFILER
    (void)params;
    return PERF_ADDRESS(PERF_OUTPUT, tile);
#else
    return L1_ADDRESS(params.buffer_Res[tile]);
#endif
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(EMA_OUTPUT_DST_INDEX, pack_output_address(params, tile));
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(EMA_OUTPUT_DST_INDEX, pack_output_address(params, tile));
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif
