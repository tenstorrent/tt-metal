// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dedicated driver for the EMA SFPU entry (ckernel_sfpu_ema.h via
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
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
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

    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
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
#include "llk_sfpu/llk_math_ema_sfpu_entry.h"

// EMA_IMPL selects the racing arm (lane FK ema-fresh registration):
//   0 = production hand entry (raw-TTI ckernel_sfpu_ema.h, LREG4/5/6
//       cross-call carry/alpha/beta ABI) — the vehicle's original arm;
//   1 = fresh typed semantic body (fresh_cpp/ema.h), "fma" arithmetic
//       contract (two single-rounded SFPMADs per step — the hand dataflow);
//   2 = fresh typed semantic body, "mul_add" contract (every product and
//       sum individually rounded) — the lane-FB demand-golden alternative.
#ifndef EMA_IMPL
#define EMA_IMPL 0
#endif
#if EMA_IMPL >= 1
#include "fresh_cpp/ema.h"
#endif

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Copy input tile from SrcA into dst.
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    // EMA init: program the SFPU and load the smoothing weights. Clear the
    // running EMA once for the whole (single) batch.
    llk_math_ema_sfpu_init();
#if EMA_IMPL == 0
    llk_math_ema_sfpu_load_alpha_beta(EMA_ALPHA_BITS, EMA_BETA_BITS);
    llk_math_ema_sfpu_clear_previous_output();
#else
    // Fresh typed arm: the alpha/beta/carry state is a caller-held typed
    // quad (the typed spelling of the hand kernel's LREG4/5/6 cross-call
    // ABI), programmed and scrambled once before the tile loop — the same
    // amortized fixed cost as the hand arm's alpha/beta load + carry clear.
    sfpu::EmaFreshState ema_state;
    sfpu::ema_fresh_state_init(ema_state, EMA_ALPHA_BITS, EMA_BETA_BITS);
#endif

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        // Input into dst tile 0.
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            EMA_INPUT_DST_INDEX, formats.math, formats.math);

        // EMA reads dst tile 0, writes dst tile 1, updates the carry.
        {
            // Isolated device-profile zone (test_sfpu_ema_device_profile):
            // profiler-only L1 bookkeeping (no SFPU instruction, no LREG
            // touched — safe around the LREG4/5/6 carry ABI), compiles away
            // in functional builds. Alpha/beta load and carry clear stay
            // outside the zone, like the Welford state clear.
            START_PERF_MEASURE("EMA_BODY")
#if EMA_IMPL == 0
            llk_math_ema_sfpu_tile(EMA_INPUT_DST_INDEX);
#else
            _llk_math_eltwise_sfpu_start_(EMA_INPUT_DST_INDEX);
            sfpu::_calculate_ema_fresh_tile_<EMA_IMPL>(ema_state);
            _llk_math_eltwise_sfpu_done_();
#endif
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

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();
        // The EMA output always lands in dst tile 1.
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(EMA_OUTPUT_DST_INDEX, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
