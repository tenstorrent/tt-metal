// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST = ckernel::get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>();
// the operand A buffer (in0) goes through unpacker 1, whose tile size is TILE_SIZE_UNPACK_B: a mixed-format functional test keeps A in the
// format of unpacker 1 and strides it by that size (the perf rows keep the sizes by the python buffers)
#ifdef MATMUL_MIXED_STRIDE
#define MATMUL_TILE_SIZE_IN0 TILE_SIZE_UNPACK_B
#define MATMUL_TILE_SIZE_IN1 TILE_SIZE_UNPACK_A
#else
#define MATMUL_TILE_SIZE_IN0 TILE_SIZE_UNPACK_A
#define MATMUL_TILE_SIZE_IN1 TILE_SIZE_UNPACK_B
#endif

// reuse direction of the block (llk_matmul_forms.h matmul_reuse_dir), chosen by the test's python key from the operand formats
#ifdef MATMUL_REUSE_DIR
constexpr std::uint32_t matmul_reuse_dir_k = MATMUL_REUSE_DIR;
#else
constexpr std::uint32_t matmul_reuse_dir_k = 0;
#endif

#if defined(ARCH_BLACKHOLE) && defined(MATMUL_PHASE_ELIDE)
#include "llk_matmul_forms.h"
constexpr ckernel::MathFidelity MATMUL_FIDELITY = ckernel::matmul_phase_fidelity<MATH_FIDELITY, MATMUL_IN0_FORMAT, MATMUL_IN1_FORMAT>();
#else
constexpr ckernel::MathFidelity MATMUL_FIDELITY = MATH_FIDELITY;
#endif

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_ROW_LE_8)
constexpr std::uint32_t matmul_max_row_tiles = 8;
#elif defined(ARCH_BLACKHOLE)
constexpr std::uint32_t matmul_max_row_tiles = get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();
#endif
#if defined(MATMUL_FORCE_FORM)
// math MOP kind (0 tile, 1 rows, 2 row) and hoisted unpack of the cell, whatever the forms table says: MATMUL_FORCE_FORM = 2 * kind + hoist
constexpr bool matmul_hoist_rows = (MATMUL_FORCE_FORM % 2) == 1;
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORCE_FORM_CHAR)
constexpr bool matmul_hoist_rows = MATMUL_FORCE_FORM_CHAR == 'h' || MATMUL_FORCE_FORM_CHAR == 'R';
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
constexpr bool matmul_hoist_rows = unpack_AB_matmul_hoist_rows<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, MATMUL_KT_PER_SECTION, MATMUL_OUT_FORMAT>();
#elif defined(ARCH_BLACKHOLE)
constexpr bool matmul_hoist_rows = unpack_AB_matmul_hoist_rows<MATH_FIDELITY, dest_sync>();
#endif
#line 38
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t num_faces_A        = params.num_faces_A;
    const std::uint32_t num_faces_B        = params.num_faces_B;

    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
    const Operand& buffer_A           = params.buffer_A;
    const Operand& buffer_B           = params.buffer_B;
#endif

#ifdef ARCH_BLACKHOLE
#ifdef MATMUL_UNPACK_TTSYNC_EXCL
    constexpr std::uint32_t unpack_auto_ttsync = 2;
#elif defined(MATMUL_UNPACK_TTSYNC)
    constexpr std::uint32_t unpack_auto_ttsync = 1;
#else
    constexpr std::uint32_t unpack_auto_ttsync = 0;
#endif
    // the test chooses the 8-bit stream at compile time, as a kernel's compile-time formats do
#ifdef MATMUL_STREAM_NARROW
    constexpr bool stream_narrow = !is_fp32_dest_acc_en;
#else
    constexpr bool stream_narrow = false;
#endif
#ifdef MATMUL_STREAM_Z
    constexpr std::uint32_t stream_z = 4;
#else
    constexpr std::uint32_t stream_z = 0;
#endif
#define MATMUL_UNPACK_INIT_TEMPLATE_ARGS 0, 0, unpack_auto_ttsync, stream_narrow, stream_z, matmul_reuse_dir_k
#define MATMUL_UNPACK_TEMPLATE_ARGS      0, 0, unpack_auto_ttsync, MAX_TILES_DEST, matmul_hoist_rows, matmul_max_row_tiles, stream_z, matmul_reuse_dir_k
#else
#define MATMUL_UNPACK_INIT_TEMPLATE_ARGS
#define MATMUL_UNPACK_TEMPLATE_ARGS
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            num_faces_A,
            num_faces_B,
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B);
        _llk_unpack_AB_matmul_init_<MATMUL_UNPACK_INIT_TEMPLATE_ARGS>(
            UNPACK_TRANSPOSE_FACES,
            CT_DIM,
            RT_DIM,
            KT_DIM,
            FACE_R_DIM,
            FACE_R_DIM,
            num_faces_A,
            num_faces_B,
            false,
            false);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, 1, matmul_reuse_dir_k);
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_K_LOOP)
                {
                    const bool perf_ring       = LOOP_FACTOR > 1;
                    const std::uint32_t addr_a = perf_ring ? PERF_ADDRESS(PERF_INPUT_A, 0) : L1_ADDRESS(buffer_A[0]);
                    const std::uint32_t addr_b = perf_ring ? PERF_ADDRESS(PERF_INPUT_B, 0) : L1_ADDRESS(buffer_B[0]);
                    _llk_unpack_AB_matmul_k_<MAX_TILES_DEST, stream_z>(
                        addr_a,
                        addr_b,
                        0,
                        0,
                        MATMUL_TILE_SIZE_IN0,
                        MATMUL_TILE_SIZE_IN1,
                        /* partial face */ false,
                        /* partial face */ false,
                        CT_DIM,
                        RT_DIM,
                        KT_DIM,
                        KT_DIM,
                        CT_DIM);
                }
#else
                for (std::uint32_t j = 0; j < KT_DIM; j++)
                {
                    // Golden (LOOP_FACTOR==1) reads unique stimuli tiles. Perf
                    // (LOOP_FACTOR>1) programs the PERF_ADDRESS ring base with tile
                    // index 0. That only removes the caller-side offset;
                    // _llk_unpack_AB_matmul_ still MOP-strides the THCON base (A by
                    // kt per extra output row, B by 1 per extra output column). The
                    // sweep keeps that walk inside PERF_RING_TILES.
                    const bool perf_ring       = LOOP_FACTOR > 1;
                    const std::uint32_t addr_a = perf_ring ? PERF_ADDRESS(PERF_INPUT_A, 0) : L1_ADDRESS(buffer_A[0]);
                    const std::uint32_t addr_b = perf_ring ? PERF_ADDRESS(PERF_INPUT_B, 0) : L1_ADDRESS(buffer_B[0]);
                    const std::uint32_t tile_a = perf_ring ? 0 : j;
                    const std::uint32_t tile_b = perf_ring ? 0 : j * CT_DIM;
                    const std::uint32_t last_a = tile_a + (RT_DIM - 1) * KT_DIM;
                    const std::uint32_t last_b = tile_b + CT_DIM - 1;
                    if (perf_ring)
                    {
                        LLK_ASSERT(last_a < PERF_RING_TILES && last_b < PERF_RING_TILES, "unpack MOP walk exceeds the 16-tile PERF_ADDRESS ring");
                    }
                    else
                    {
                        LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_A[last_a])), "unpack A real-buffer top address is outside L1");
                        LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_B[last_b])), "unpack B real-buffer top address is outside L1");
                    }
                    _llk_unpack_AB_matmul_<MATMUL_UNPACK_TEMPLATE_ARGS>(
                        addr_a,
                        addr_b,
                        tile_a,
                        tile_b,
                        MATMUL_TILE_SIZE_IN0,
                        MATMUL_TILE_SIZE_IN1,
                        /* partial face */ false,
                        /* partial face */ false,
                        CT_DIM,
                        RT_DIM,
                        KT_DIM);
                }
#endif
            }
        }
        PROFILER_SYNC();
    }
}

#endif
// zone markers of the other threads keep their line numbers
#line 165

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#if defined(MATMUL_FORCE_FORM)
constexpr bool matmul_tile_mop = (MATMUL_FORCE_FORM / 2) == 0;
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORCE_FORM_CHAR)
constexpr bool matmul_tile_mop = !(MATMUL_FORCE_FORM_CHAR == 'r' || MATMUL_FORCE_FORM_CHAR == 'R');
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
constexpr bool matmul_tile_mop = !matmul_math_row_form<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, MATMUL_KT_PER_SECTION, MATMUL_OUT_FORMAT>();
#else
constexpr bool matmul_tile_mop = false;
#endif

// keys production selects the forms by (partials format, its row MOP rule) and no fused post flip: see matmul_prod_keys.py
#ifdef MATMUL_PROD_KEYS
constexpr bool matmul_prod_keys = true;
#else
constexpr bool matmul_prod_keys = false;
#endif
#ifdef MATMUL_ROW_END_FOLD
constexpr bool matmul_row_end_fold_k = true;
#else
constexpr bool matmul_row_end_fold_k = false;
#endif
#ifndef MATMUL_KEY_IN0
#define MATMUL_KEY_IN0 DataFormat::Invalid
#define MATMUL_KEY_IN1 DataFormat::Invalid
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS) && !defined(MATMUL_FORCE_FORM_CHAR)
constexpr bool matmul_half_end_fold_k = matmul_tile_mop && matmul_half_end_fold<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, MATMUL_KT_PER_SECTION, MATMUL_OUT_FORMAT, MATMUL_KEY_IN0, MATMUL_KEY_IN1>();
#else
constexpr bool matmul_half_end_fold_k = false;
#endif
// Wormhole and Quasar take main's template arguments: the tile MOP, reuse direction and fold parameters are Blackhole's
#ifdef ARCH_BLACKHOLE
#define MATMUL_DEST_DONE_ARGS dest_sync, is_fp32_dest_acc_en, fused_post_flip, next_op_sets_base
#else
#define MATMUL_DEST_DONE_ARGS dest_sync, is_fp32_dest_acc_en
#endif
#ifdef ARCH_BLACKHOLE
#define MATMUL_MATH_DEFAULT_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL, false, matmul_tile_mop, matmul_reuse_dir_k, matmul_row_end_fold_k
#define MATMUL_MATH_DEFAULT_INIT_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL, false, matmul_tile_mop, 0, false, matmul_half_end_fold_k, matmul_reuse_dir_k, matmul_row_end_fold_k
#else
#define MATMUL_MATH_DEFAULT_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL
#define MATMUL_MATH_DEFAULT_INIT_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL
#endif
#line 176
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_ROW_MOP)
#define MATMUL_MATH_TEMPLATE_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL, true
#define MATMUL_MATH_INIT_TEMPLATE_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL, true
#else
#define MATMUL_MATH_TEMPLATE_ARGS MATMUL_MATH_DEFAULT_ARGS
#define MATMUL_MATH_INIT_TEMPLATE_ARGS MATMUL_MATH_DEFAULT_INIT_ARGS
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR   = params.LOOP_FACTOR;
    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_matmul_init_<MATMUL_MATH_INIT_TEMPLATE_ARGS>(
            /* tile A */ TILE_R_DIM,
            /* tile A */ TILE_C_DIM,
            /* tile B */ TILE_R_DIM,
            /* tile B */ TILE_C_DIM,
            /* partial face */ false,
            UNPACK_TRANSPOSE_FACES,
            CT_DIM,
            RT_DIM);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, 1, matmul_reuse_dir_k);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                LLK_ASSERT(
                    (get_dest_max_matmul_tiles(0, CT_DIM, RT_DIM) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles for matmul");

                for (std::uint32_t j = 0; j < KT_DIM; j++)
                {
                    _llk_math_matmul_<MATMUL_MATH_TEMPLATE_ARGS>(/* dest_index */ 0, CT_DIM, RT_DIM);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                LLK_ASSERT(
                    (get_dest_max_matmul_tiles(0, CT_DIM, RT_DIM) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles for matmul");

                _llk_math_wait_for_dest_available_<dest_sync>();
                for (std::uint32_t j = 0; j < KT_DIM; j++)
                {
                    _llk_math_matmul_<MATMUL_MATH_TEMPLATE_ARGS>(/* dest_index */ 0, CT_DIM, RT_DIM);
                }
                #if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
                #ifdef MATMUL_UNFUSED_POST
                [[maybe_unused]] constexpr bool fused_post_flip = false;
                #else
                [[maybe_unused]] constexpr bool fused_post_flip = !matmul_prod_keys && (static_cast<std::uint32_t>(MATH_FIDELITY) >= 3 || MATMUL_OUT_FORMAT != DataFormat::Bfp8_b);
                #endif
                #else
                [[maybe_unused]] constexpr bool fused_post_flip = !matmul_prod_keys && (static_cast<std::uint32_t>(MATH_FIDELITY) >= 3);
                #endif
                #if defined(MATMUL_KEEP_FLIP_WRITE)
                    [[maybe_unused]] constexpr bool next_op_sets_base = false;
#else
                    [[maybe_unused]] constexpr bool next_op_sets_base = !is_fp32_dest_acc_en;
#endif
                    _llk_math_dest_section_done_<MATMUL_DEST_DONE_ARGS>();
            }
        }
        PROFILER_SYNC();
    }

    _llk_math_matmul_uninit_();
}

#endif

#ifdef LLK_TRISC_PACK
#line 264

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

#ifndef MATMUL_PACK_FORM
#define MATMUL_PACK_FORM 0
#endif

#ifndef MATMUL_L1_ACC
#define MATMUL_L1_ACC 0
#endif

#if defined(ARCH_BLACKHOLE)
// MATMUL_PACK_FORM 1 packs the block with one program run, 2 closes every tile (block-float outputs and small tiles).
inline void matmul_pack_out(std::uint32_t tile_index, std::uint32_t address, std::uint32_t num_tiles, std::uint32_t tile_stride, std::uint32_t num_faces = 4)
{
#if MATMUL_PACK_FORM != 0
    // The fast forms need four faces of 16 rows. At half sync the pack hides behind math or unpack, where the denser DEST reads only add contention: the closed form stays off and the plain one needs 16-bit DEST.
    constexpr bool fast_plain = !(is_fp32_dest_acc_en && dest_sync == DstSync::SyncHalf);
    // the closed block pack at full sync issues its destination words before the math wait, as pack_block_prepare does in the ttnn kernels
    constexpr bool fast_closed = false;
    if (MATMUL_PACK_FORM == 1 && fast_plain)
    {
        _llk_pack_block_fast_<dest_sync, is_fp32_dest_acc_en>(tile_index, address, num_tiles);
        return;
    }
    if (MATMUL_PACK_FORM == 2 && fast_closed)
    {
        _llk_pack_block_closed_fast_<dest_sync, is_fp32_dest_acc_en>(tile_index, address, num_tiles, tile_stride);
        return;
    }
#endif
    if constexpr (MATMUL_PACK_FORM == 1)
    {
        _llk_pack_block_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile_index, address, num_tiles);
    }
    else
    {
        _llk_pack_block_closed_<dest_sync, is_fp32_dest_acc_en>(tile_index, address, num_tiles, tile_stride, num_faces);
    }
}
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
    const std::uint32_t CT_DIM         = params.CT_DIM;
    const std::uint32_t RT_DIM         = params.RT_DIM;
    const Operand& buffer_Res          = params.buffer_Res;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_SIZE_PACK);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#if defined(ARCH_BLACKHOLE) && MATMUL_PACK_FORM != 0
        _llk_pack_block_fast_init_();
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
#if defined(ARCH_BLACKHOLE) && MATMUL_L1_ACC == 1
            reconfigure_packer_l1_acc(1);
#endif
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                {
                    const std::uint32_t tile_index = tile % MAX_TILES_DEST;
#if defined(ARCH_BLACKHOLE)
                    if constexpr (MATMUL_PACK_FORM != 0)
                    {
                        matmul_pack_out(tile_index, PERF_ADDRESS(PERF_OUTPUT, tile_index), CT_DIM * RT_DIM, TILE_SIZE_PACK << 4);
                        break;
                    }
#endif
                    LLK_ASSERT(
                        (tile_index < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles for matmul");
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile_index, PERF_ADDRESS(PERF_OUTPUT, tile_index));
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
#if defined(ARCH_BLACKHOLE) && MATMUL_PACK_FORM == 2
                if constexpr (dest_sync == DstSync::SyncFull)
                {
                    _llk_pack_block_closed_pre_(0, LOOP_FACTOR > 1 ? PERF_ADDRESS(PERF_OUTPUT, 0) : L1_ADDRESS(buffer_Res[0]));
                }
#endif
                _llk_packer_wait_for_math_done_();
#if defined(ARCH_BLACKHOLE) && MATMUL_L1_ACC == 1
                if (loop == 0)
                {
                    reconfigure_packer_l1_acc(1);
                }
#endif
                // Pack dest occupancy (RT×CT).
                for (std::uint32_t i = 0; i < CT_DIM * RT_DIM; i++)
                {
                    LLK_ASSERT((i < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "i exceeds max dest tiles");
                    // Golden packs unique result tiles. Perf K=32 dest-fill Float32
                    // places buffer_Res past L1; use the PERF_ADDRESS ring like matmul_perf.
                    const std::uint32_t addr = LOOP_FACTOR > 1 ? PERF_ADDRESS(PERF_OUTPUT, i) : L1_ADDRESS(buffer_Res[i]);
                    if (LOOP_FACTOR == 1)
                    {
                        LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_Res[i])), "pack result real-buffer address is outside L1");
                    }
#if defined(ARCH_BLACKHOLE)
                    if constexpr (MATMUL_PACK_FORM != 0)
                    {
                        if (LOOP_FACTOR == 1)
                        {
                            LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_Res[CT_DIM * RT_DIM - 1])), "pack result real-buffer address is outside L1");
                        }
#if MATMUL_PACK_FORM == 2
                        if constexpr (dest_sync == DstSync::SyncFull)
                        {
                            _llk_pack_block_closed_run_(CT_DIM * RT_DIM, TILE_SIZE_PACK << 4, 4);
                            break;
                        }
#endif
                        matmul_pack_out(i, addr, CT_DIM * RT_DIM, TILE_SIZE_PACK << 4);
                        break;
                    }
#endif
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, addr);
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
