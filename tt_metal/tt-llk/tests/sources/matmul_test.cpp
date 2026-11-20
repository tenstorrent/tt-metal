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

#ifndef MATMUL_FUNC_BLOCKS
#define MATMUL_FUNC_BLOCKS 0
#endif

#if defined(ARCH_BLACKHOLE) && defined(MATMUL_PRODUCTION)
// a ttnn kernel's plain matmul_block_init / matmul_block: none of the harness's unpack and math opt-ins, the compute API's direction
#undef MATMUL_REUSE_DIR
#undef MATMUL_PHASE_ELIDE
#undef MATMUL_FORM_KEYS
#undef MATMUL_TRUE_MAIN
#undef MATMUL_BLOCK
#undef MATMUL_KEY_IN0
#undef MATMUL_KEY_IN1
#undef MATMUL_ROW_LE_8
#undef MATMUL_UNFUSED_POST
#undef MATMUL_KEEP_FLIP_WRITE
#undef MATMUL_FORCE_FORM
#undef MATMUL_FORCE_FORM_CHAR
#undef MATMUL_ROW_MOP
#undef MATMUL_ROW_END_FOLD
#undef MATMUL_K_LOOP
#undef MATMUL_STREAM_NARROW
#undef MATMUL_STREAM_Z
#undef MATMUL_UNPACK_TTSYNC
#undef MATMUL_UNPACK_TTSYNC_EXCL
#undef MATMUL_TTSYNC_BLOCK
#undef MATMUL_PROD_KEYS
#define MATMUL_REUSE_DIR MATMUL_PROD_DIR
#ifndef RUNTIME_FORMATS
// matmul_api_reuse_dir's run-time direction (3) of a LoFi half-sync kernel with a 32-bit operand beside a fast one, as
// matmul_api_runtime_dir resolves it for the key's operands (in1 is unpacker A, in0 unpacker B)
#include "llk_matmul_forms.h"
// (a macro: a keyed form, MATMUL_FE_FORM_KT below, keeps direction 0)
#define MATMUL_PROD_RUNTIME_DIR                                                                                                                             \
    (MATH_FIDELITY == ckernel::MathFidelity::LoFi && dest_sync == ckernel::DstSync::SyncHalf && static_cast<DataFormat>(PACK_OUT) != DataFormat::Float32 && \
     ckernel::matmul_kernel_lofi_mix(MATMUL_KERNEL_FORMATS) && MATMUL_FE_FORM_KT == 0)
#undef MATMUL_REUSE_DIR
#define MATMUL_REUSE_DIR                                                 \
    (MATMUL_PROD_RUNTIME_DIR ? ckernel::matmul_reuse_dir(                \
                                   ckernel::MathFidelity::LoFi,          \
                                   static_cast<DataFormat>(UNPACK_B_IN), \
                                   static_cast<DataFormat>(UNPACK_A_IN), \
                                   MATMUL_PK_CT,                         \
                                   MATMUL_PK_RT,                         \
                                   false,                                \
                                   true,                                 \
                                   THROTTLE_LEVEL != 0)                  \
                             : MATMUL_PROD_DIR)
#else
#define MATMUL_PROD_RUNTIME_DIR false
#endif
#endif

// G2h: a production key of compile-time block keys whose class polls runs the ttsync unpack on its cell (llk_matmul_forms.h
// matmul_kernel_ttsync_block, as the front end's matmul_fe_ttsync_cell); in1 is the test's A operand, in0 its B. minimal_matmul
// (MATMUL_PACK_TILES) passes run-time dims and keeps the class rule
#if defined(MATMUL_PRODUCTION) && defined(MATMUL_PACK_KEYS) && !defined(MATMUL_PACK_TILES) && !defined(MATMUL_TTSYNC_BLOCK_PROD)
#define MATMUL_TTSYNC_BLOCK_PROD                                                                                                           \
    ckernel::matmul_kernel_ttsync_block(                                                                                                   \
        MATH_FIDELITY, dest_sync == ckernel::DstSync::SyncFull, is_fp32_dest_acc_en, MATMUL_KERNEL_FORMATS, MATMUL_PK_IN_B, MATMUL_PK_IN_A, \
        MATMUL_PK_RT, MATMUL_PK_CT, MATMUL_PK_KT, MATMUL_PK_THROTTLE, MATMUL_PK_L1ACC)
#elif !defined(MATMUL_TTSYNC_BLOCK_PROD)
#define MATMUL_TTSYNC_BLOCK_PROD false
#endif

// a minimal_matmul key (run-time ct and rt, constant kt): the forms table key of the front end's plain call (llk_matmul_forms.h
// matmul_fe_form_out over the kernel's CB formats; in1 is the test's A operand, in0 its B), Invalid and 0 where it keeps main's form
#if defined(MATMUL_PRODUCTION) && defined(MATMUL_PACK_KEYS) && defined(MATMUL_PACK_TILES)
#define MATMUL_FE_FORM_OUT                                                                                                                   \
    ckernel::matmul_fe_form_out(                                                                                                             \
        MATH_FIDELITY, dest_sync == ckernel::DstSync::SyncFull, is_fp32_dest_acc_en, MATMUL_PK_KT, MATMUL_PK_IN_B, MATMUL_PK_IN_A, MATMUL_KERNEL_FORMATS, \
        MATMUL_PK_THROTTLE)
#define MATMUL_FE_FORM_KT (MATMUL_FE_FORM_OUT == DataFormat::Invalid ? 0u : static_cast<std::uint32_t>(MATMUL_PK_KT))
#else
#define MATMUL_FE_FORM_OUT DataFormat::Invalid
#define MATMUL_FE_FORM_KT  0u
#endif

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

#ifndef MATMUL_KEY_IN0
#define MATMUL_KEY_IN0 DataFormat::Invalid
#define MATMUL_KEY_IN1 DataFormat::Invalid
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
#include "llk_matmul_forms.h"
#if defined(MATMUL_TRUE_MAIN)
// a cell that runs main's code (an exact main row, or init before every call): the table key is 0 whatever the block
constexpr std::uint32_t matmul_kt_key = 0;
#elif defined(MATMUL_BLOCK)
// the forms table key: main's form (0) for the blocks llk_matmul_forms.h keeps on it
constexpr std::uint32_t matmul_kt_key = ckernel::matmul_form_k<
    MATH_FIDELITY, dest_sync == ckernel::DstSync::SyncHalf ? 0 : 1, MATMUL_BLOCK_ACC, MATMUL_KT_PER_SECTION, MATMUL_BLOCK_KT, MATMUL_KEY_IN0, MATMUL_KEY_IN1, MATMUL_OUT_FORMAT,
    MATMUL_BLOCK_CT, MATMUL_BLOCK_RT, MATMUL_BLOCK_THROTTLE, MATMUL_BLOCK_L1_ACC, MATMUL_BLOCK_NB, MATMUL_BLOCK_R0, MATMUL_BLOCK_C1>();
#else
constexpr std::uint32_t matmul_kt_key = MATMUL_KT_PER_SECTION;
#endif
#ifdef MATMUL_FORCE_FORM_CHAR
// a forced form does not apply to a block that keeps main's form
constexpr bool matmul_force_form = !(matmul_kt_key == 0 && MATMUL_KT_PER_SECTION != 0);
#endif
#endif

// main's form: the table key is 0 (an exact main row, or init before every call); the cell then keeps main's pack init and unpack row mask
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
constexpr bool matmul_true_main_key = matmul_kt_key == 0;
#else
constexpr bool matmul_true_main_key = false;
#endif

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_PRODUCTION)
// the compute API's: 8 rows unkeyed, the DEST's tiles under a forms key
constexpr std::uint32_t matmul_max_row_tiles = MATMUL_FE_FORM_KT == 0 ? 8 : get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();
#elif defined(ARCH_BLACKHOLE) && (defined(MATMUL_ROW_LE_8) || defined(MATMUL_TRUE_MAIN))
constexpr std::uint32_t matmul_max_row_tiles = 8;
#elif defined(ARCH_BLACKHOLE)
constexpr std::uint32_t matmul_max_row_tiles = get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();
#endif
#if defined(MATMUL_FORCE_FORM)
// math MOP kind (0 tile, 1 rows, 2 row) and hoisted unpack of the cell, whatever the forms table says: MATMUL_FORCE_FORM = 2 * kind + hoist
constexpr bool matmul_hoist_rows = (MATMUL_FORCE_FORM % 2) == 1;
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORCE_FORM_CHAR)
constexpr bool matmul_hoist_rows = matmul_force_form ? (MATMUL_FORCE_FORM_CHAR == 'h' || MATMUL_FORCE_FORM_CHAR == 'R')
                                                     : unpack_AB_matmul_hoist_rows<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT>();
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
constexpr bool matmul_hoist_rows = unpack_AB_matmul_hoist_rows<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT>();
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_PRODUCTION) && defined(MATMUL_PACK_TILES)
// minimal_matmul's run-time calls take the compute API's unkeyed hoist (matmul_fe_unkeyed_hoist: subblock_h RT, subblock_w CT,
// K_block_tiles KT), and above LoFi only with a 16-bit DEST. The key has no M block or bias epilogue: MATMUL_PK_M_BLOCK and
// MATMUL_PK_BIAS (LLK_EXTRA_DEFINES) default to an M block below 8 without bias, so a 2x2 bfp8 LoFi fp32 K<3 key differs
// from a production kernel with M block 8 or bias unless they are set
#ifndef MATMUL_PK_M_BLOCK
#define MATMUL_PK_M_BLOCK 0
#endif
#ifndef MATMUL_PK_BIAS
#define MATMUL_PK_BIAS 0
#endif
constexpr bool matmul_kernel_bfp = []
{
    for (const std::uint8_t f : MATMUL_KERNEL_FORMATS)
    {
        if (f == static_cast<std::uint8_t>(DataFormat::Bfp8_b) || f == static_cast<std::uint8_t>(DataFormat::Bfp4_b))
        {
            return true;
        }
    }
    return false;
}();
constexpr bool matmul_unkeyed_hoist = []
{
    constexpr bool lofi = MATH_FIDELITY == ckernel::MathFidelity::LoFi;
    if (MATMUL_PK_RT == 1 && MATMUL_PK_CT == 1)
    {
        return matmul_kernel_bfp;
    }
    if (MATMUL_PK_RT == 1)
    {
        return false;
    }
    if (MATMUL_PK_CT == 1)
    {
        return matmul_kernel_bfp && lofi;
    }
    return !(matmul_kernel_bfp && lofi && is_fp32_dest_acc_en && MATMUL_PK_KT < 3 && MATMUL_PK_M_BLOCK < 8 && !MATMUL_PK_BIAS);
}();
constexpr bool matmul_hoist_rows = (matmul_unkeyed_hoist && (MATH_FIDELITY == ckernel::MathFidelity::LoFi || !is_fp32_dest_acc_en)) ||
                                   unpack_AB_matmul_hoist_rows<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, MATMUL_FE_FORM_KT, MATMUL_FE_FORM_OUT>();
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_PRODUCTION)
constexpr bool matmul_hoist_rows = unpack_AB_matmul_hoist_rows<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, MATMUL_FE_FORM_KT, MATMUL_FE_FORM_OUT>();
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

#if defined(MATMUL_PRODUCTION) && defined(MATMUL_PACK_KEYS)
    // the production arm: compile-time ct and rt, as production's kernels pass them
    constexpr std::uint32_t CT_DIM = MATMUL_PK_CT;
    constexpr std::uint32_t RT_DIM = MATMUL_PK_RT;
#else
    const std::uint32_t CT_DIM = params.CT_DIM;
    const std::uint32_t RT_DIM = params.RT_DIM;
#endif
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
    const Operand& buffer_A           = params.buffer_A;
    const Operand& buffer_B           = params.buffer_B;
#endif

#ifdef ARCH_BLACKHOLE
#if defined(MATMUL_PRODUCTION)
#ifdef MATMUL_PACK_TILES
    // minimal_matmul passes run-time ct and rt: its front end sees no block, so the LoFi class rule polls (llk_matmul_unpack_ttsync)
    constexpr bool matmul_fe_dims_unseen = true;
#else
    constexpr bool matmul_fe_dims_unseen = false;
#endif
    // a plain call's kernel rule, the function llk_unpack_AB_matmul_kernel_ttsync calls, and on a polling class the block rule of a
    // call with constant keys (matmul_fe_ttsync_cell; in1 is the test's A operand, in0 its B)
    constexpr std::uint32_t unpack_auto_ttsync =
        matmul_kernel_ttsync(MATH_FIDELITY, dest_sync == ckernel::DstSync::SyncFull, is_fp32_dest_acc_en, MATMUL_KERNEL_FORMATS) && matmul_reuse_dir_k == 0 &&
                !(MATH_FIDELITY == ckernel::MathFidelity::LoFi && (matmul_fe_dims_unseen || !(MATMUL_PK_RT == 1 && MATMUL_PK_CT == 1)))
            ? 1
        : MATMUL_TTSYNC_BLOCK_PROD ? 3
                                   : 0;
#else
#ifdef MATMUL_UNPACK_TTSYNC_EXCL
    constexpr std::uint32_t unpack_ttsync_req = 2;
#elif defined(MATMUL_UNPACK_TTSYNC)
    constexpr std::uint32_t unpack_ttsync_req = 1;
#else
    constexpr std::uint32_t unpack_ttsync_req = 0;
#endif
    // the opt-in runs only on the blocks that keep the ttsync unpack (llk_matmul_forms.h); a kernel without block keys keeps what it asks for
#if defined(MATMUL_TTSYNC_BLOCK)
    constexpr std::uint32_t unpack_auto_ttsync = !matmul_unpack_ttsync_cell(
                                                     MATH_FIDELITY,
                                                     dest_sync == ckernel::DstSync::SyncHalf,
                                                     is_fp32_dest_acc_en,
                                                     MATMUL_TTSYNC_IN1,
                                                     MATMUL_TTSYNC_IN0,
                                                     MATMUL_TTSYNC_OUT,
                                                     MATMUL_TTSYNC_RT,
                                                     MATMUL_TTSYNC_CT,
                                                     MATMUL_TTSYNC_KT)
                                                     ? 0
                                                     : unpack_ttsync_req;
#elif defined(MATMUL_BLOCK) && !defined(LLK_MAIN_ARM)
    constexpr std::uint32_t unpack_auto_ttsync = (unpack_ttsync_req != 0 && !matmul_unpack_ttsync_cell(
                                                                                MATH_FIDELITY,
                                                                                dest_sync == ckernel::DstSync::SyncHalf,
                                                                                is_fp32_dest_acc_en,
                                                                                MATMUL_KEY_IN1,
                                                                                MATMUL_KEY_IN0,
                                                                                MATMUL_OUT_FORMAT,
                                                                                MATMUL_BLOCK_RT,
                                                                                MATMUL_BLOCK_CT,
                                                                                MATMUL_BLOCK_KT))
                                                     ? 0
                                                     : unpack_ttsync_req;
#else
    constexpr std::uint32_t unpack_auto_ttsync = unpack_ttsync_req;
#endif
#endif
#line 69
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
#if defined(MATMUL_PRODUCTION) && defined(MATMUL_PACK_TILES)
    // minimal_matmul names its subblock: a 1x1 key's calls are one tile each (llk_unpack_AB_matmul_named_1x1)
    constexpr bool matmul_one_tile_calls = MATMUL_PK_RT == 1 && MATMUL_PK_CT == 1;
#else
    constexpr bool matmul_one_tile_calls = false;
#endif
#define MATMUL_UNPACK_TEMPLATE_ARGS \
    0, 0, unpack_auto_ttsync, MAX_TILES_DEST, matmul_hoist_rows, matmul_max_row_tiles, stream_z, matmul_reuse_dir_k, matmul_one_tile_calls
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
#ifdef MATMUL_STOCH_RND
        _llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();
#endif
#line 129
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_PRODUCTION)
        // llk_unpack_AB_matmul_init: the 8-bit stream where the streamed operand is narrow, as there a choice on the run-time block
        if (matmul_reuse_dir_k == 0 && _llk_unpack_AB_matmul_stream_narrow_(
                                           CT_DIM,
                                           RT_DIM,
                                           static_cast<std::uint32_t>(formats.unpack_A_src),
                                           static_cast<std::uint32_t>(formats.unpack_B_src),
                                           MAX_TILES_DEST,
                                           FACE_R_DIM,
                                           MATH_FIDELITY == ckernel::MathFidelity::HiFi4,
                                           MATH_FIDELITY == ckernel::MathFidelity::LoFi))
        {
            _llk_unpack_AB_matmul_init_<0, 0, unpack_auto_ttsync, true, 0, 0>(
                UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM, KT_DIM, FACE_R_DIM, FACE_R_DIM, num_faces_A, num_faces_B, false, false);
        }
        else
#endif
#line 129
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
            _perf_unpack_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, 1, matmul_reuse_dir_k);
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_K_LOOP)
                {
                    const bool perf_ring       = LOOP_FACTOR > 1 && !MATMUL_FUNC_BLOCKS;
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
                    const bool perf_ring       = LOOP_FACTOR > 1 && !MATMUL_FUNC_BLOCKS;
                    const std::uint32_t addr_a = perf_ring ? PERF_ADDRESS(PERF_INPUT_A, 0) : L1_ADDRESS(buffer_A[0]);
                    const std::uint32_t addr_b = perf_ring ? PERF_ADDRESS(PERF_INPUT_B, 0) : L1_ADDRESS(buffer_B[0]);
                    const std::uint32_t tile_a = perf_ring ? 0 : j;
                    const std::uint32_t tile_b = perf_ring ? 0 : j * CT_DIM;
                    const std::uint32_t last_a = tile_a + (RT_DIM - 1) * KT_DIM;
                    const std::uint32_t last_b = tile_b + CT_DIM - 1;
                    if (perf_ring)
                    {
#ifdef MATMUL_WIDE_WALK
                        LLK_ASSERT(
                            (last_a + 1) * MATMUL_TILE_SIZE_IN0 <= (PERF_COUNTERS_BASE_ADDR - PERF_INPUT_A) / 16 &&
                                (last_b + 1) * MATMUL_TILE_SIZE_IN1 <= (PERF_COUNTERS_BASE_ADDR - PERF_INPUT_B) / 16,
                            "unpack MOP walk leaves the L1 below the perf counters");
#else
#line 195
                        LLK_ASSERT(last_a < PERF_RING_TILES && last_b < PERF_RING_TILES, "unpack MOP walk exceeds the 16-tile PERF_ADDRESS ring");
#endif
#line 196
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
constexpr bool matmul_tile_mop = matmul_force_form ? !(MATMUL_FORCE_FORM_CHAR == 'r' || MATMUL_FORCE_FORM_CHAR == 'R')
                                                   : !matmul_math_row_form<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT>();
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
constexpr bool matmul_tile_mop = !matmul_math_row_form<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT>();
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_PRODUCTION)
// llk_math_matmul_tile_mop of a plain call: llk_math_matmul_kernel_has_32bit_cb over the production kernel's CB formats
// (MATMUL_KERNEL_FORMATS, partials included) and the test's in0, in1 and output
#ifdef RUNTIME_FORMATS
constexpr bool matmul_prod_may_pack_32bit = true;
#else
constexpr bool matmul_prod_kernel_has_32bit_cb()
{
    for (const std::uint8_t f : MATMUL_KERNEL_FORMATS)
    {
        if (f == ckernel::to_underlying(DataFormat::Float32) || f == ckernel::to_underlying(DataFormat::Tf32))
        {
            return true;
        }
    }
    return false;
}
constexpr bool matmul_prod_may_pack_32bit = matmul_prod_kernel_has_32bit_cb() ||
                                            ckernel::matmul_format_is_32bit(static_cast<DataFormat>(UNPACK_A_IN)) ||
                                            ckernel::matmul_format_is_32bit(static_cast<DataFormat>(UNPACK_B_IN)) ||
                                            ckernel::matmul_format_is_32bit(static_cast<DataFormat>(PACK_OUT));
#endif
// minimal_matmul (MATMUL_PACK_TILES) passes run-time dims: the front end sees no constant block
#ifdef MATMUL_PACK_TILES
constexpr bool matmul_prod_dims_unseen = true;
#else
constexpr bool matmul_prod_dims_unseen = false;
#endif
// a run-time direction call runs the tile MOP whatever direction it resolves to (matmul_block: llk_dir 3)
#if defined(MATMUL_PACK_TILES)
// minimal_matmul's plain call (matmul_block_llk): main's math (the tile MOP) where matmul_fe_tile_mop holds (HiFi2 with a named M block
// below 4, else not the plain row form), switching to the row form from matmul_fe_rowform_min_kt (LoFi kt 3; HiFi2 half sync 32-bit DEST
// in a kernel naming an M block of 4 or more, kt 2); otherwise the plain row form. MATMUL_PK_M_BLOCK: the key's production M_block_tiles
#ifndef MATMUL_PK_M_BLOCK
#define MATMUL_PK_M_BLOCK 0
#endif
constexpr bool matmul_prod_fe_tile =
    (MATH_FIDELITY == ckernel::MathFidelity::HiFi2 && MATMUL_PK_M_BLOCK != 0 && MATMUL_PK_M_BLOCK < 4) ||
    !matmul_plain_row_form<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en>(matmul_prod_may_pack_32bit);
constexpr std::uint32_t matmul_prod_rowform_min_kt =
    dest_sync != ckernel::DstSync::SyncHalf ? 0
    : MATH_FIDELITY == ckernel::MathFidelity::LoFi ? 3
    : (matmul_hifi2_half_wide(MATH_FIDELITY, true, is_fp32_dest_acc_en, matmul_prod_may_pack_32bit) && MATMUL_PK_M_BLOCK >= 4) ? 2
                                                                                                                              : 0;
constexpr bool matmul_tile_mop = (matmul_prod_fe_tile && !(matmul_prod_rowform_min_kt != 0 && MATMUL_PK_KT >= matmul_prod_rowform_min_kt)) ||
                                 MATMUL_PROD_RUNTIME_DIR;
#else
constexpr bool matmul_tile_mop =
    MATMUL_FE_FORM_KT == 0
        ? (!matmul_plain_row_form<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en>(matmul_prod_may_pack_32bit) &&
           !(matmul_hifi2_half_wide(MATH_FIDELITY, dest_sync == ckernel::DstSync::SyncHalf, is_fp32_dest_acc_en, matmul_prod_may_pack_32bit) &&
             MATMUL_PK_KT >= 2 && !matmul_prod_dims_unseen) &&
           !(MATH_FIDELITY == ckernel::MathFidelity::LoFi && !matmul_prod_dims_unseen && !(MATMUL_PK_RT == 1 && MATMUL_PK_CT == 1) && MATMUL_PK_KT != 1 &&
             (dest_sync == ckernel::DstSync::SyncHalf
                  ? !ckernel::matmul_kernel_lofi_mix(MATMUL_KERNEL_FORMATS)
                  : !ckernel::matmul_operands_reorder(
                        static_cast<DataFormat>(MATMUL_KERNEL_FORMATS[0]), static_cast<DataFormat>(MATMUL_KERNEL_FORMATS[1]), THROTTLE_LEVEL != 0))) &&
           !(MATH_FIDELITY == ckernel::MathFidelity::LoFi && dest_sync == ckernel::DstSync::SyncHalf && matmul_prod_dims_unseen && MATMUL_PK_KT >= 3)) ||
              MATMUL_PROD_RUNTIME_DIR
        : !matmul_math_row_form<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, MATMUL_FE_FORM_KT, MATMUL_FE_FORM_OUT>();
#endif
#else
constexpr bool matmul_tile_mop = false;
#endif

// main's form writes the address mods exactly as main does (the bias register too); the other forms skip that write
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
constexpr bool matmul_bias_write = matmul_kt_key == 0;
#else
constexpr bool matmul_bias_write = false;
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
#ifdef MATMUL_PACK_KEYS
constexpr bool matmul_half_end_fold_k = matmul_tile_mop && matmul_half_end_fold_block<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT, MATMUL_KEY_IN0, MATMUL_KEY_IN1, MATMUL_PK_RT, MATMUL_PK_CT, MATMUL_PK_KT>();
#else
constexpr bool matmul_half_end_fold_k = matmul_tile_mop && matmul_half_end_fold<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT, MATMUL_KEY_IN0, MATMUL_KEY_IN1>();
#endif
#line 198
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_PRODUCTION)
constexpr bool matmul_half_end_fold_k = matmul_tile_mop && matmul_half_end_fold<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, MATMUL_FE_FORM_KT, DataFormat::Invalid>();
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
#ifdef MATMUL_PACK_KEYS
#define MATMUL_INIT_KEYS MATMUL_PK_CT, MATMUL_PK_RT, matmul_bias_write
#else
#define MATMUL_INIT_KEYS 0, 0, matmul_bias_write
#endif
#ifdef MATMUL_PRODUCTION
// the compute API's fidelity for the CB formats of a call that names none (matmul_fe_phase_format), the test's CBs being in0, in1
// and the output; run-time formats keep the request
#if !defined(RUNTIME_FORMATS) && !defined(MATMUL_PROD_NO_ELIDE)
constexpr bool matmul_prod_no_32bit_cb = !ckernel::matmul_format_is_32bit(static_cast<DataFormat>(UNPACK_A_IN)) &&
                                         !ckernel::matmul_format_is_32bit(static_cast<DataFormat>(UNPACK_B_IN)) &&
                                         !ckernel::matmul_format_is_32bit(static_cast<DataFormat>(PACK_OUT));
constexpr ckernel::MathFidelity matmul_prod_fidelity = ckernel::matmul_phase_fidelity<
    MATH_FIDELITY,
    static_cast<DataFormat>(UNPACK_B_IN),
    static_cast<DataFormat>(UNPACK_A_IN),
    matmul_prod_no_32bit_cb ? DataFormat::Float16_b : DataFormat::Invalid,
    is_fp32_dest_acc_en,
    THROTTLE_LEVEL,
    true>();
#else
constexpr ckernel::MathFidelity matmul_prod_fidelity = MATH_FIDELITY;
#endif
// llk_math_matmul_init_impl: the block keys of a call whose ct and rt are constant (the test's pack keys; none at run-time dims), with
// them the end fold at full sync and no bias write; tiny 16-row tiles unless half sync with a 32-bit DEST
// minimal_matmul (MATMUL_PACK_TILES) passes run-time ct and rt: production's init takes no block keys (0, 0, bias write)
#if defined(MATMUL_PACK_KEYS) && !defined(MATMUL_PACK_TILES)
constexpr std::uint32_t matmul_prod_key_ct = MATMUL_PK_CT, matmul_prod_key_rt = MATMUL_PK_RT;
#else
constexpr std::uint32_t matmul_prod_key_ct = 0, matmul_prod_key_rt = 0;
#endif
constexpr bool matmul_prod_keyed    = matmul_prod_key_ct != 0 && matmul_prod_key_rt != 0;
constexpr bool matmul_prod_end_fold = (dest_sync == ckernel::DstSync::SyncFull && matmul_prod_keyed) || matmul_half_end_fold_k;
#define MATMUL_PROD_TINY_16_ROWS !(dest_sync == ckernel::DstSync::SyncHalf && is_fp32_dest_acc_en)
// llk_math_matmul_bias_class
constexpr bool matmul_prod_bias_write =
    !matmul_prod_keyed || (matmul_prod_fidelity == ckernel::MathFidelity::HiFi2 && dest_sync == ckernel::DstSync::SyncHalf && is_fp32_dest_acc_en &&
                           (matmul_prod_key_ct == 1 || matmul_prod_key_rt == 1));
#define MATMUL_PROD_KEYS         matmul_prod_key_ct, matmul_prod_key_rt, matmul_prod_bias_write
// llk_math_matmul's half_base: LoFi half sync, not a run-time direction call (no size mode in the harness); minimal_matmul's default
// cells below kt 3 run main's math (matmul_fe_main_math), which has none, and from kt 3 the row form, which takes it
constexpr bool matmul_prod_half_base = matmul_prod_fidelity == ckernel::MathFidelity::LoFi && dest_sync == ckernel::DstSync::SyncHalf &&
                                       !MATMUL_PROD_RUNTIME_DIR && (!matmul_prod_dims_unseen || MATMUL_PK_KT >= 3);
#define MATMUL_MATH_DEFAULT_ARGS matmul_prod_fidelity, THROTTLE_LEVEL, false, matmul_tile_mop, matmul_reuse_dir_k, false, matmul_prod_half_base
#define MATMUL_MATH_DEFAULT_INIT_ARGS \
    matmul_prod_fidelity, THROTTLE_LEVEL, false, matmul_tile_mop, 0, MATMUL_PROD_TINY_16_ROWS, matmul_prod_end_fold, matmul_reuse_dir_k, false, MATMUL_PROD_KEYS
#define MATMUL_MATH_THROTTLE_ARGS matmul_prod_fidelity, 5, false, true, matmul_reuse_dir_k, false, matmul_prod_half_base
#define MATMUL_MATH_THROTTLE_INIT_ARGS                                                                                                                         \
    matmul_prod_fidelity, 5, false, true, 0, MATMUL_PROD_TINY_16_ROWS, dest_sync == ckernel::DstSync::SyncFull &&matmul_prod_keyed, matmul_reuse_dir_k, false, \
        MATMUL_PROD_KEYS
#else
#define MATMUL_MATH_DEFAULT_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL, false, matmul_tile_mop, matmul_reuse_dir_k, matmul_row_end_fold_k
#define MATMUL_MATH_DEFAULT_INIT_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL, false, matmul_tile_mop, 0, false, matmul_half_end_fold_k, matmul_reuse_dir_k, matmul_row_end_fold_k, MATMUL_INIT_KEYS
#endif
#else
#define MATMUL_MATH_DEFAULT_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL
#define MATMUL_MATH_DEFAULT_INIT_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL
#endif

#if defined(ARCH_BLACKHOLE) && defined(MATMUL_PRODUCTION)
constexpr bool matmul_production = true;
// matmul_block_math_dynamic_throttle's FW word, MEM_L1_ARC_FW_SCRATCH (the harness links nothing below L1 0x20): odd = throttle
volatile __attribute__((rvtt_l1_ptr)) std::uint32_t* matmul_throttle_ptr = reinterpret_cast<volatile __attribute__((rvtt_l1_ptr)) std::uint32_t*>(16);
static std::uint32_t matmul_throttled_mop_status                         = 0;

// The compute API's fidelity phase count (llk_math_matmul_phases.h) for in0 = UNPACK_B_IN (SrcB), in1 = UNPACK_A_IN (SrcA): after an
// init of the form's count, the phases the operands need, except on a one-tile block of a block-float in0 and a wide in1
__attribute__((always_inline)) inline void matmul_production_phases(const std::uint32_t requested, const std::uint32_t ct_dim, const std::uint32_t rt_dim)
{
#if !defined(RUNTIME_FORMATS) && !defined(MATMUL_PROD_NO_ELIDE)
    constexpr auto in0 = static_cast<DataFormat>(UNPACK_B_IN);
    constexpr auto in1 = static_cast<DataFormat>(UNPACK_A_IN);
    constexpr auto out = static_cast<DataFormat>(PACK_OUT);
    constexpr bool in0_bfp = in0 == DataFormat::Bfp8_b || in0 == DataFormat::Bfp4_b;
    constexpr bool in1_bfp = in1 == DataFormat::Bfp8_b || in1 == DataFormat::Bfp4_b;
    constexpr bool elidable = (MATH_FIDELITY == ckernel::MathFidelity::HiFi3 || MATH_FIDELITY == ckernel::MathFidelity::HiFi4) && THROTTLE_LEVEL == 0 &&
                              (in0_bfp || in1_bfp || out == DataFormat::Bfp8_b || out == DataFormat::Bfp4_b);
    if constexpr (elidable)
    {
        const bool lo_a_zero          = in1 == DataFormat::Bfp4_b;
        const std::uint32_t needed    = (requested >= 4 && !lo_a_zero && !in0_bfp) ? 4 : (requested >= 3 && !in0_bfp) ? 3 : 2;
        const std::uint32_t p         = needed < requested ? needed : requested;
        const bool keep_one_tile      = in0_bfp && !in1_bfp && ct_dim == 1 && rt_dim == 1;
        if (p != requested && !keep_one_tile)
        {
            reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE)[1] = p;
        }
    }
#endif
}

// matmul_block_init's math: the init, then the unthrottled MOP state
__attribute__((always_inline)) inline void matmul_production_init(
    const std::uint32_t in0_tile_r_dim,
    const std::uint32_t in0_tile_c_dim,
    const std::uint32_t in1_tile_r_dim,
    const std::uint32_t in1_tile_c_dim,
    const bool partial_face,
    const std::uint32_t transpose,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim)
{
    _llk_math_matmul_init_<MATMUL_MATH_DEFAULT_INIT_ARGS>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, ct_dim, rt_dim);
    matmul_throttled_mop_status = 0;
    if constexpr (ckernel::math::is_high_fidelity(matmul_prod_fidelity))
    {
        matmul_production_phases(static_cast<std::uint32_t>(matmul_prod_fidelity), ct_dim, rt_dim);
    }
}

// matmul_block's math: matmul_block_math_dynamic_throttle, re-initing on a change of the FW throttle
__attribute__((always_inline)) inline void matmul_production_math(const std::uint32_t ct_dim, const std::uint32_t rt_dim, const std::uint32_t transpose)
{
    if (*matmul_throttle_ptr % 2)
    {
        if (matmul_throttled_mop_status != 1)
        {
            _llk_math_matmul_init_<MATMUL_MATH_THROTTLE_INIT_ARGS>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, transpose, ct_dim, rt_dim);
            matmul_throttled_mop_status = 1;
        }
        _llk_math_matmul_<MATMUL_MATH_THROTTLE_ARGS>(0, ct_dim, rt_dim);
    }
    else
    {
        if (matmul_throttled_mop_status != 0)
        {
            _llk_math_matmul_init_<MATMUL_MATH_DEFAULT_INIT_ARGS>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, transpose, ct_dim, rt_dim);
            matmul_throttled_mop_status = 0;
            if constexpr (ckernel::math::is_high_fidelity(matmul_prod_fidelity))
            {
                matmul_production_phases(static_cast<std::uint32_t>(matmul_prod_fidelity), ct_dim, rt_dim);
            }
        }
        _llk_math_matmul_<MATMUL_MATH_DEFAULT_ARGS>(0, ct_dim, rt_dim);
    }
}

#define MATMUL_MATH_INIT                 matmul_production_init
#define MATMUL_MATH_CALL(ct_dim, rt_dim) matmul_production_math(ct_dim, rt_dim, UNPACK_TRANSPOSE_FACES)
#else
constexpr bool matmul_production = false;
#define MATMUL_MATH_INIT                 _llk_math_matmul_init_<MATMUL_MATH_INIT_TEMPLATE_ARGS>
#define MATMUL_MATH_CALL(ct_dim, rt_dim) _llk_math_matmul_<MATMUL_MATH_TEMPLATE_ARGS>(/* dest_index */ 0, ct_dim, rt_dim)
#endif
#line 176
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_ROW_MOP)
#define MATMUL_MATH_TEMPLATE_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL, true
#define MATMUL_MATH_INIT_TEMPLATE_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL, true, false, 0, false, false, 0, false, MATMUL_INIT_KEYS
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
#if defined(MATMUL_PRODUCTION) && defined(MATMUL_PACK_KEYS)
    // the production arm: compile-time ct and rt, as production's kernels pass them
    constexpr std::uint32_t CT_DIM = MATMUL_PK_CT;
    constexpr std::uint32_t RT_DIM = MATMUL_PK_RT;
#else
    const std::uint32_t CT_DIM = params.CT_DIM;
    const std::uint32_t RT_DIM = params.RT_DIM;
#endif
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        MATMUL_MATH_INIT(
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
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, 1, matmul_reuse_dir_k);
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
                    MATMUL_MATH_CALL(CT_DIM, RT_DIM);
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
                    MATMUL_MATH_CALL(CT_DIM, RT_DIM);
                }
// main's form ends its section as main does: the fused post and the skipped base write are for the cells the table gives another form
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
                [[maybe_unused]] constexpr bool matmul_main_math_form =
                    ckernel::matmul_default_form<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT>() == '.';
#else
                [[maybe_unused]] constexpr bool matmul_main_math_form = false;
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS) && defined(MATMUL_BLOCK)
                // the cells that keep main's whole section end: separate post and the base write of the DEST flip (llk_matmul_forms.h)
                [[maybe_unused]] constexpr bool matmul_main_flip_cell = ckernel::matmul_main_flip(
                    MATH_FIDELITY, dest_sync == ckernel::DstSync::SyncHalf, MATMUL_BLOCK_ACC, MATMUL_KEY_IN1, MATMUL_KEY_IN0, MATMUL_OUT_FORMAT, MATMUL_BLOCK_RT, MATMUL_BLOCK_CT,
                    MATMUL_BLOCK_KT, MATMUL_BLOCK_THROTTLE, MATMUL_BLOCK_L1_ACC, MATMUL_BLOCK_NB);
#else
                [[maybe_unused]] constexpr bool matmul_main_flip_cell = false;
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
#ifdef MATMUL_UNFUSED_POST
                [[maybe_unused]] constexpr bool fused_post_flip = false;
#else
#ifdef MATMUL_BLOCK
                // the cells where main's separate post reads faster under packer L1 accumulate (llk_matmul_forms.h)
                [[maybe_unused]] constexpr bool matmul_unfused_cell = matmul_main_flip_cell || ckernel::matmul_unfused_post(
                    MATH_FIDELITY, dest_sync == ckernel::DstSync::SyncHalf, MATMUL_BLOCK_ACC, MATMUL_KEY_IN1, MATMUL_KEY_IN0, MATMUL_OUT_FORMAT, MATMUL_BLOCK_RT, MATMUL_BLOCK_CT,
                    MATMUL_BLOCK_KT, MATMUL_BLOCK_THROTTLE, MATMUL_BLOCK_L1_ACC, MATMUL_BLOCK_NB);
#else
                [[maybe_unused]] constexpr bool matmul_unfused_cell = false;
#endif
                [[maybe_unused]] constexpr bool fused_post_flip = !matmul_unfused_cell && !matmul_main_math_form && !matmul_prod_keys &&
                                                                  (static_cast<std::uint32_t>(MATH_FIDELITY) >= 3 || MATMUL_OUT_FORMAT != DataFormat::Bfp8_b);
#endif
                #else
                [[maybe_unused]] constexpr bool fused_post_flip = !matmul_prod_keys && !matmul_production && (static_cast<std::uint32_t>(MATH_FIDELITY) >= 3);
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS) && defined(MATMUL_BLOCK)
                // the keep rule of the LLK (llk_matmul_forms.h): the python define is set only where the rule keeps the write (the modules also clear it for the blocks that keep main's form)
                static_assert(
                    ckernel::matmul_keep_flip_write(
                        MATH_FIDELITY, dest_sync == ckernel::DstSync::SyncHalf, MATMUL_BLOCK_ACC, MATMUL_KEY_IN1, MATMUL_KEY_IN0, MATMUL_OUT_FORMAT, MATMUL_BLOCK_RT, MATMUL_BLOCK_CT, MATMUL_BLOCK_KT,
                        MATMUL_BLOCK_THROTTLE, MATMUL_BLOCK_L1_ACC, MATMUL_BLOCK_NB) ||
#ifdef MATMUL_KEEP_FLIP_WRITE
                        false,
#else
                        true,
#endif
                    "the python keep_flip_write keeps a write that matmul_keep_flip_write drops");
                #endif
                #if defined(MATMUL_KEEP_FLIP_WRITE)
                    [[maybe_unused]] constexpr bool next_op_sets_base = false;
#else
                [[maybe_unused]] constexpr bool next_op_sets_base =
                    !matmul_production && !is_fp32_dest_acc_en && !matmul_main_flip_cell && !matmul_main_math_form;
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

// The pack is production's. Compile-time formats (as every JIT kernel has them): the compute API's pack_block through
// pack_api_shim.h over a CB of the result buffer, built with production's -flto=1 -flto-partition=one (python kernel_flags):
// llk_pack_block<n> for a constant count on the PR's LLK, llk_matmul_pack's per-tile loop for a run-time count, main's
// per-tile loop on main's LLK. Run-time formats (no descriptor arrays, so no block forms): both arms pack per tile; the PR
// arm makes llk_pack_tile_of_output's choice on the run-time output format (that format test is the harness's own, see
// matmul_pack_tile_rt).
#if defined(ARCH_BLACKHOLE) && !defined(RUNTIME_FORMATS)
#if __has_include("llk_pack_block.h")
// the API traits of a ttnn matmul kernel (the PR's pack contexts; main has none): a matmul header, and compute_kernel_api.h's pack-thread SFPU init
#define PACK_API_SHIM_MATMUL
#define PACK_API_SHIM_SFPU
#endif
#include "pack_api_shim.h"
// a run-time-block key (MATMUL_PACK_TILES: minimal_matmul, all_gather) sets no pack shape mark, as production's run-time ct / rt set none
#if defined(MATMUL_PRODUCTION) && defined(MATMUL_PACK_KEYS) && defined(LLK_PACK_SHAPE_MARKS) && !defined(MATMUL_PACK_TILES)
// production's plain matmul_block_init with the constant block marks its shape for the pack (matmul.h MATMUL_FE_PACK_SHAPE)
inline constexpr auto matmul_pack_shape_mark = sizeof(ckernel::packer::llk_pack_shape_set<(MATMUL_PK_RT >= 4 && MATMUL_PK_CT <= 2) ? 6 : 5>);
inline constexpr auto matmul_pack_row_mark = sizeof(ckernel::packer::llk_pack_shape_set<(MATMUL_PK_RT == 1 && MATMUL_PK_CT >= 2) ? 9 : 10>);
inline constexpr auto matmul_pack_kt1_mark   = sizeof(ckernel::packer::llk_pack_shape_set < MATMUL_PK_KT == 1 ? 11 : 12 >);
#endif
#define MATMUL_PACK_SHIM 1
#define MATMUL_PR_LLK    PACK_API_SHIM_PR
#else
#define MATMUL_PACK_SHIM 0
#if defined(ARCH_BLACKHOLE) && __has_include("llk_pack_block.h")
#define MATMUL_PR_LLK 1
#else
#define MATMUL_PR_LLK 0
#endif
#endif

#ifndef MATMUL_PACK_FORM
#define MATMUL_PACK_FORM 0
#endif
#if defined(ARCH_BLACKHOLE)
#include "llk_matmul_forms.h"
#ifdef MATMUL_PACK_KEYS
// the pack form of the block: the LLK decision, which the python mirror (the MATMUL_PACK_FORM define) must equal
constexpr int matmul_pack_form_key = ckernel::matmul_pack_form_k(
    MATMUL_PK_FID, MATMUL_PK_HALF, MATMUL_PK_THROTTLE, MATMUL_PK_KT, MATMUL_PK_RT, MATMUL_PK_CT, MATMUL_PK_IN_A, MATMUL_PK_IN_B, MATMUL_PK_OUT, MATMUL_PK_R0, MATMUL_PK_C1, MATMUL_PK_DENSE,
    MATMUL_PK_L1ACC);
static_assert(matmul_pack_form_key == MATMUL_PACK_FORM, "the python matmul_pack_form differs from matmul_pack_form_k");
#else
constexpr int matmul_pack_form_key = MATMUL_PACK_FORM;
#endif
#else
constexpr int matmul_pack_form_key = MATMUL_PACK_FORM;
#endif
#ifdef MATMUL_PACK_KEYS
constexpr DataFormat matmul_pack_out_format = MATMUL_PK_OUT;
#else
constexpr DataFormat matmul_pack_out_format = DataFormat::Invalid;
#endif
// the python mirror's form of the block (MATMUL_PACK_FORM), checked against the LLK above; the pack itself is production's (below)
constexpr int matmul_pack_form_v = matmul_pack_form_key;
#line 271

#ifndef MATMUL_L1_ACC
#define MATMUL_L1_ACC 0
#endif
// arm of the shim's pack_block: 0 the form tables (production), 'T', 'B' or 'W' one form
#ifndef MATMUL_PACK_ARM
#define MATMUL_PACK_ARM 0
#endif

#if MATMUL_PACK_SHIM
constexpr std::uint32_t matmul_pack_out = PACK_API_SHIM_OUT;
#if defined(MATMUL_PACK_KEYS) && !(defined(MATMUL_PACK_NOK) && MATMUL_PACK_NOK)
constexpr std::uint32_t matmul_pack_kt = MATMUL_PK_KT;
#else
constexpr std::uint32_t matmul_pack_kt = 0;
#endif

// The perf CB: the whole blocks that fit the 64 KB PERF_OUTPUT ring, so a block continues the last one's W offset except
// at the wrap, as in production's output CB. Python gives the pages with constant dims (MATMUL_PERF_CB_PAGES in build.h).
inline std::uint32_t matmul_ring_pages([[maybe_unused]] const std::uint32_t block_tiles, [[maybe_unused]] const std::uint32_t page_words)
{
#ifdef MATMUL_PERF_CB_PAGES
    return MATMUL_PERF_CB_PAGES;
#else
    const std::uint32_t blocks = PERF_RING_TILES * 4096 / 16 / (block_tiles * page_words);
    return block_tiles * (blocks > 0 ? blocks : 1);
#endif
}

// production's pack_block(dst, cb, n) with a run-time count: llk_matmul_pack's per-tile loop
template <bool is_fp32_dest_acc_en>
inline void matmul_pack_runtime_count(const std::uint32_t start_tile_index, const std::uint32_t output, const std::uint32_t ntiles)
{
#if MATMUL_PR_LLK
    const std::uint8_t output_id = get_output_id(output);
    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++)
    {
        const std::uint32_t pack_tile_addr = get_output_tile_address<false, PackMode::Default>(output_id, 0);
        llk_pack_tile_of_output<is_fp32_dest_acc_en, PackMode::Default>(tile_index, pack_tile_addr, output_id);
    }
#else
    pack_api_shim::main_pack_block<is_fp32_dest_acc_en>(start_tile_index, output, ntiles);
#endif
}

// one DEST block of the output: pack_block<n> (constant count) or pack_block(dst, cb, n)
template <bool constant_count, std::uint32_t ntiles>
__attribute__((always_inline)) inline void matmul_pack_block(const std::uint32_t num_tiles)
{
#if MATMUL_PR_LLK
    if constexpr (constant_count)
    {
        pack_api_shim::pack_block<ntiles, is_fp32_dest_acc_en, matmul_pack_kt, MATMUL_PACK_ARM>(0, matmul_pack_out);
    }
    else
#endif
    {
        matmul_pack_runtime_count<is_fp32_dest_acc_en>(0, matmul_pack_out, num_tiles);
    }
}
#else
// run-time formats: one tile; the PR arm as llk_pack_tile_of_output, whose Float32-output test reads the run-time format here
template <bool is_fp32_dest_acc_en>
__attribute__((always_inline)) inline void matmul_pack_tile_rt(const std::uint32_t tile_index, const std::uint32_t address, [[maybe_unused]] const std::uint32_t pack_dst)
{
#if MATMUL_PR_LLK
    if constexpr (dest_sync == DstSync::SyncFull && !is_fp32_dest_acc_en)
    {
        if (pack_dst == static_cast<std::uint32_t>(DataFormat::Float32))
        {
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default, false, false, true>(tile_index, address);
            return;
        }
    }
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default, false, dest_sync == DstSync::SyncFull, true>(tile_index, address);
#else
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(tile_index, address);
#endif
}
#endif

#line 307
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
#if defined(MATMUL_PACK_CONST)
#ifndef MATMUL_CONST_DIMS
#error MATMUL_PACK_CONST needs MATMUL_CONST_DIMS
#endif
    // ttnn's matmul kernels pass pack_block a compile-time tile count (out_subblock_num_tiles)
    constexpr std::uint32_t CT_DIM = MATMUL_CONST_CT;
    constexpr std::uint32_t RT_DIM = MATMUL_CONST_RT;
#ifdef MATMUL_PACK_KEYS
    static_assert(CT_DIM == MATMUL_PK_CT && RT_DIM == MATMUL_PK_RT, "pack keys and tile dims disagree");
#endif
#else
    const std::uint32_t CT_DIM         = params.CT_DIM;
    const std::uint32_t RT_DIM         = params.RT_DIM;
#endif
#line 318
    const Operand& buffer_Res          = params.buffer_Res;
#endif
#if defined(MATMUL_PACK_CONST)
    [[maybe_unused]] constexpr bool matmul_const_count = true;
    [[maybe_unused]] constexpr std::uint32_t matmul_ntiles = MATMUL_CONST_CT * MATMUL_CONST_RT;
#else
    [[maybe_unused]] constexpr bool matmul_const_count = false;
    [[maybe_unused]] constexpr std::uint32_t matmul_ntiles = 0;
#endif
    // perf runs pack every block to the same place (the PERF_OUTPUT ring), golden runs to consecutive result tiles
    [[maybe_unused]] const bool perf_ring = LOOP_FACTOR > 1 && !MATMUL_FUNC_BLOCKS;

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_SIZE_PACK);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
#if MATMUL_PR_LLK
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#endif
#if MATMUL_PACK_SHIM
#if MATMUL_PR_LLK
        // as matmul_block_init marks the pack thread
        ckernel::packer::llk_pack_math_context(1);
#endif
        // the output CB over the result tiles, in the 16 byte words of the metal CB pointers: the perf ring, or every block (golden)
        {
            const std::uint32_t base  = (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || perf_ring) ? PERF_ADDRESS(PERF_OUTPUT, 0) : L1_ADDRESS(buffer_Res[0]);
            const std::uint32_t pages = (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || perf_ring) ? matmul_ring_pages(CT_DIM * RT_DIM, TILE_SIZE_PACK)
                                                                                                      : CT_DIM * RT_DIM * (MATMUL_FUNC_BLOCKS ? LOOP_FACTOR : 1);
            pack_api_shim::cb_bind(matmul_pack_out, base + 1, TILE_SIZE_PACK, pages);
        }
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }
        else
        {
#if defined(ARCH_BLACKHOLE) && MATMUL_L1_ACC == 1
            if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
            {
                reconfigure_packer_l1_acc(1);
            }
#endif
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
#if defined(MATMUL_PACK_STATE) && MATMUL_PACK_STATE == 1
                // every block the first of its output behind a pack reconfigure: production's reconfigure sets the state
                _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, TILE_SIZE_PACK);
#elif defined(MATMUL_PACK_STATE) && MATMUL_PACK_STATE == 2 && MATMUL_PACK_SHIM
                // every block the first of its output behind a pack init: the state a set_packer_strides leaves (one store the harness adds)
                ckernel::packer::llk_pack_state_set_half(0);
#endif
                if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
                {
                    _llk_packer_wait_for_math_done_();
                }
#if defined(ARCH_BLACKHOLE) && MATMUL_L1_ACC == 1
                if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
                {
#if defined(MATMUL_NB) && MATMUL_NB > 1
                    // one output over MATMUL_NB k blocks: the first block packs without, the next ones with L1 accumulation
                    if (loop % MATMUL_NB == 0)
                    {
                        reconfigure_packer_l1_acc(0);
                    }
                    else if (loop % MATMUL_NB == 1)
                    {
                        reconfigure_packer_l1_acc(1);
                    }
#else
                    if (loop == 0)
                    {
                        reconfigure_packer_l1_acc(1);
                    }
#endif
                }
#endif
#if MATMUL_PACK_SHIM
                matmul_pack_block<matmul_const_count, matmul_ntiles>(CT_DIM * RT_DIM);
#else
                for (std::uint32_t i = 0; i < CT_DIM * RT_DIM; i++)
                {
                    LLK_ASSERT((i < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "i exceeds max dest tiles");
                    const std::uint32_t tile_index = PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE ? i % MAX_TILES_DEST : i;
                    const std::uint32_t addr       = (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || perf_ring)
                                                         ? PERF_ADDRESS(PERF_OUTPUT, tile_index)
                                                         : L1_ADDRESS(buffer_Res[(MATMUL_FUNC_BLOCKS ? loop * CT_DIM * RT_DIM : 0) + i]);
                    matmul_pack_tile_rt<is_fp32_dest_acc_en>(tile_index, addr, formats.pack_dst);
                }
#endif
                if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
                {
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
#if MATMUL_PACK_SHIM
                // as a ttnn kernel: tile_regs_release, then cb_push_back
                pack_api_shim::cb_push_back(matmul_pack_out, CT_DIM * RT_DIM);
#endif
            }
        }
        PROFILER_SYNC();
    }
}

#endif
