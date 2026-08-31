// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf driver for the EMA SFPU entry (ckernel_sfpu_ema.h via
// llk_math_ema_sfpu_entry.h). Mirrors sources/sfpu_ema_test.cpp, with the tile loop
// wrapped in the perf markers and repeated LOOP_FACTOR times to amortise profiler
// overhead.
//
// MATH_ISOLATE is the run type that matters here: the change under measurement is
// confined to the SFPU math block (_compute_ema_math_), so unpack and pack are
// unaffected and only the math pipe should move.
//
// The entry is a stateful, two-tile kernel:
//   * input tile is read from dst index 0,
//   * output tile is written to dst index 1,
//   * the running EMA (EMA_old) is held in LREG4 and carried across tiles.
// Because it always works through that same pair of dst tiles, there is no
// MAX_TILES_DEST blocking here: each iteration is one dst 0 -> dst 1 step.
//
// All parameters (formats, LOOP_FACTOR, TILE_CNT, num_faces, alpha/beta) are
// compile-time constants emitted into params.h, so nothing is read from params.

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
            // Math isolate wants no software sync from unpack to math, so only the
            // unavoidable hardware valid-bit handshake is driven here.
            // One valid per face: the datacopy on the math side of MATH_ISOLATE consumes
            // SrcA a face at a time, so this count has to include num_faces. It must stay
            // in step with whatever the math isolate path actually retires -- a mismatch
            // in either direction hangs the handshake.
            _perf_unpack_loop_set_valid</* src A */ true, /* src B */ is_fp32_dest_acc_en>(num_faces * TILE_CNT * LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, /* tile_idx */ i), formats.unpack_A_src, formats.unpack_A_dst);
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
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/llk_math_ema_sfpu_entry.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")

        // Copy input tile from SrcA into dst.
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(num_faces, formats.math);
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

        // EMA init: program the SFPU and load the smoothing weights. Clear the
        // running EMA once for the whole batch.
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
            _perf_math_loop_clear_valid</* clear A */ true, /* clear B */ false>(TILE_CNT * LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Isolates the math pipe: no dest handshake with pack, so what is left is
            // the datacopy plus the EMA kernel.
            //
            // Same shape as eltwise_unary_sfpu_perf.cpp's MATH_ISOLATE: the datacopy
            // stays in. It is what consumes the SrcA valid bits that unpack sets, so
            // dropping it and trying to retire them with a bare TTI_CLEARDVALID hangs the
            // math thread. The datacopy is therefore a fixed cost inside this marker, the
            // same way it is for every other unary SFPU op measured this way -- it is
            // constant across a before/after comparison of the SFPU block, so it cancels
            // in the delta.
            //
            // EMA always works through dst tile 0 (input) and dst tile 1 (output) via
            // compile-time offsets, so there is no MAX_TILES_DEST blocking here: every
            // iteration copies into tile 0 and the kernel writes tile 1.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        EMA_INPUT_DST_INDEX, formats.math, formats.math);
                    llk_math_ema_sfpu_tile(EMA_INPUT_DST_INDEX);
                }
            }
        }
        else // L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC>();

                    // Input into dst tile 0.
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        EMA_INPUT_DST_INDEX, formats.math, formats.math);

                    // EMA reads dst tile 0, writes dst tile 1, updates the LREG4 carry.
                    llk_math_ema_sfpu_tile(EMA_INPUT_DST_INDEX);

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
                    // The EMA output always lands in dst tile 1.
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(EMA_OUTPUT_DST_INDEX, PERF_ADDRESS(PERF_OUTPUT, tile));
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
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(EMA_OUTPUT_DST_INDEX, PERF_ADDRESS(PERF_OUTPUT, tile));
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif
