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
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#if defined(ARCH_BLACKHOLE) && defined(MATMUL_PHASE_ELIDE)
#include "llk_matmul_forms.h"
constexpr ckernel::MathFidelity MATMUL_FIDELITY = ckernel::matmul_phase_fidelity<MATH_FIDELITY, MATMUL_IN0_FORMAT, MATMUL_IN1_FORMAT>();
#else
constexpr ckernel::MathFidelity MATMUL_FIDELITY = MATH_FIDELITY;
#endif

// reuse direction of the block (llk_matmul_forms.h matmul_reuse_dir), chosen by the test's python key from the operand formats
#ifdef MATMUL_REUSE_DIR
constexpr std::uint32_t matmul_reuse_dir_k = MATMUL_REUSE_DIR;
#else
constexpr std::uint32_t matmul_reuse_dir_k = 0;
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
// main's form writes the address mods exactly as main does (the bias register too); the other forms skip that write
constexpr bool matmul_bias_write = matmul_kt_key == 0;
#else
constexpr bool matmul_bias_write = false;
#endif

// main's form: the table key is 0 (an exact main row, or init before every call); the cell then keeps main's pack init and unpack row mask
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
constexpr bool matmul_true_main_key = matmul_kt_key == 0;
#else
constexpr bool matmul_true_main_key = false;
#endif

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#if defined(ARCH_BLACKHOLE) && (defined(MATMUL_ROW_LE_8) || defined(MATMUL_TRUE_MAIN))
constexpr std::uint32_t matmul_max_row_tiles = 8;
#elif defined(ARCH_BLACKHOLE)
constexpr std::uint32_t matmul_max_row_tiles = get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>();
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORCE_FORM_CHAR)
constexpr bool matmul_hoist_rows = MATMUL_FORCE_FORM_CHAR == 'h' || MATMUL_FORCE_FORM_CHAR == 'R';
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
constexpr bool matmul_hoist_rows = unpack_AB_matmul_hoist_rows<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT>();
#elif defined(ARCH_BLACKHOLE)
constexpr bool matmul_hoist_rows = unpack_AB_matmul_hoist_rows<MATH_FIDELITY, dest_sync>();
#endif
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim = params.in0_tile_r_dim;
    const std::uint32_t in1_tile_r_dim = params.in1_tile_r_dim;

    const bool PARTIAL_FACE_A = params.PARTIAL_FACE_A;
    const bool PARTIAL_FACE_B = params.PARTIAL_FACE_B;

    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t num_faces_A        = params.num_faces_A;
    const std::uint32_t num_faces_B        = params.num_faces_B;

    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const int NUM_BLOCKS              = params.NUM_BLOCKS;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
    const Operand& buffer_A           = params.buffer_A;
    const Operand& buffer_B           = params.buffer_B;
#endif

#ifdef ARCH_BLACKHOLE
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
#define MATMUL_UNPACK_INIT_TEMPLATE_ARGS 0, 0, false, stream_narrow, stream_z, matmul_reuse_dir_k
#define MATMUL_UNPACK_TEMPLATE_ARGS \
    0, 0, false, get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>(), matmul_hoist_rows, matmul_max_row_tiles, stream_z, matmul_reuse_dir_k
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
            in1_tile_r_dim < FACE_R_DIM ? in1_tile_r_dim : FACE_R_DIM,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            num_faces_B, // in1
            num_faces_A, // in0
#ifdef MATMUL_OPERAND_SIZES
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B);
#else
            TILE_SIZE_UNPACK_B,
            TILE_SIZE_UNPACK_A);
#endif
#line 114
        _llk_unpack_AB_matmul_init_<MATMUL_UNPACK_INIT_TEMPLATE_ARGS>(
            UNPACK_TRANSPOSE_FACES,
            CT_DIM,
            RT_DIM,
            KT_DIM,
            in1_tile_r_dim < FACE_R_DIM ? in1_tile_r_dim : FACE_R_DIM,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            num_faces_B,     // in1
            num_faces_A,     // in0
            PARTIAL_FACE_B,  // in1
            PARTIAL_FACE_A); // in0
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
            _perf_unpack_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, NUM_BLOCKS, matmul_reuse_dir_k);
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_K_LOOP)
                    _llk_unpack_AB_matmul_k_<get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>(), stream_z>(
                        L1_ADDRESS(buffer_A[0]),
                        L1_ADDRESS(buffer_B[0]),
                        0,
                        0,
                        TILE_SIZE_UNPACK_B,
                        TILE_SIZE_UNPACK_A,
                        PARTIAL_FACE_B,
                        PARTIAL_FACE_A,
                        CT_DIM,
                        RT_DIM,
                        KT_DIM,
                        KT_DIM,
                        CT_DIM);
#else
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_unpack_AB_matmul_<MATMUL_UNPACK_TEMPLATE_ARGS>(
                            L1_ADDRESS(buffer_A[0]),
                            L1_ADDRESS(buffer_B[0]),
                            j,
                            j * CT_DIM,
                            TILE_SIZE_UNPACK_B,
                            TILE_SIZE_UNPACK_A,
                            PARTIAL_FACE_B,
                            PARTIAL_FACE_A,
                            CT_DIM,
                            RT_DIM,
                            KT_DIM);
                    }
#endif
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
// zone markers of the other threads keep their line numbers
#ifdef MATMUL_ROW_END_FOLD
constexpr bool matmul_row_end_fold_k = true;
#else
constexpr bool matmul_row_end_fold_k = false;
#endif
// Wormhole and Quasar take main's template arguments: the row MOP, in1 order, reuse direction and fold parameters are Blackhole's
#ifdef ARCH_BLACKHOLE
#define MATMUL_DEST_DONE_ARGS dest_sync, is_fp32_dest_acc_en, fused_post_flip, next_op_sets_base
#else
#define MATMUL_DEST_DONE_ARGS dest_sync, is_fp32_dest_acc_en
#endif
#ifdef ARCH_BLACKHOLE
#define MATMUL_MATH_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL, matmul_row_mop, matmul_tile_mop, matmul_reuse_dir_k, matmul_row_end_fold_k
#define MATMUL_MATH_INIT_ARGS                                                                                                                     \
    MATMUL_FIDELITY, THROTTLE_LEVEL, matmul_row_mop, matmul_tile_mop, matmul_in1_order, !(dest_sync == DstSync::SyncHalf && is_fp32_dest_acc_en), \
        (dest_sync == DstSync::SyncFull && !matmul_bias_write) || matmul_half_end_fold_k, matmul_reuse_dir_k, matmul_row_end_fold_k, 0, 0, matmul_bias_write
#else
#define MATMUL_MATH_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL
#define MATMUL_MATH_INIT_ARGS MATMUL_FIDELITY, THROTTLE_LEVEL
#endif
#line 146

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#ifdef MATMUL_SFPU_AFTER
// an SFPU op after the matmul block in one kernel (init once at the start, as Metal's full inits do): relu on every tile of the block
#include "ckernel_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_relu.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORCE_FORM_CHAR)
constexpr bool matmul_tile_mop = !(MATMUL_FORCE_FORM_CHAR == 'r' || MATMUL_FORCE_FORM_CHAR == 'R');
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
constexpr bool matmul_tile_mop = !matmul_math_row_form<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT>();
#else
constexpr bool matmul_tile_mop = false;
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
#line 164
#else
constexpr bool matmul_half_end_fold_k = false;
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_IN1_ALT)
constexpr int matmul_in1_order = 1;
#else
constexpr int matmul_in1_order = 0;
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_ROW_MOP)
constexpr bool matmul_row_mop = true;
#else
constexpr bool matmul_row_mop = false;
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim = params.in0_tile_r_dim;
    const std::uint32_t in0_tile_c_dim = params.in0_tile_c_dim;
    const std::uint32_t in1_tile_r_dim = params.in1_tile_r_dim;
    const std::uint32_t in1_tile_c_dim = params.in1_tile_c_dim;

    const bool PARTIAL_FACE_MATH = params.PARTIAL_FACE_MATH;

    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const int DST_INDEX                    = params.DST_INDEX;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;

    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const int NUM_BLOCKS              = params.NUM_BLOCKS;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
#ifdef MATMUL_SFPU_AFTER
        _llk_math_eltwise_unary_sfpu_init_once_();
#endif
        _llk_math_matmul_init_<MATMUL_MATH_INIT_ARGS>(
            in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, PARTIAL_FACE_MATH, UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, NUM_BLOCKS, matmul_reuse_dir_k);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_math_matmul_<MATMUL_MATH_ARGS>(DST_INDEX, CT_DIM, RT_DIM);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                    LLK_ASSERT(
                        (NUM_TILES_IN_BLOCK <= get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Matmul block exceeds destination capacity");
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_math_matmul_<MATMUL_MATH_ARGS>(DST_INDEX, CT_DIM, RT_DIM);
                    }
#ifdef MATMUL_SFPU_AFTER
                    for (std::uint32_t sfpu_tile = 0; sfpu_tile < NUM_TILES_IN_BLOCK; sfpu_tile++)
                    {
                        _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_lrelu_<false>, DST_INDEX + sfpu_tile, VectorMode::RC, 8, 0u);
                    }
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS) && !defined(MATMUL_UNFUSED_POST)
#else
                    [[maybe_unused]] constexpr bool matmul_main_math_form = false;
#endif
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS) && defined(MATMUL_UNFUSED_POST)
                    [[maybe_unused]] constexpr bool fused_post_flip = false;
#elif defined(ARCH_BLACKHOLE) && defined(MATMUL_FORM_KEYS)
                    // main's form ends its section as main does: the fused post and the skipped base write are for the cells the table gives another form
                    [[maybe_unused]] constexpr bool matmul_main_math_form =
                        ckernel::matmul_default_form<MATH_FIDELITY, dest_sync, is_fp32_dest_acc_en, matmul_kt_key, MATMUL_OUT_FORMAT>() == '.';
                    [[maybe_unused]] constexpr bool fused_post_flip =
                        !matmul_main_math_form && (static_cast<std::uint32_t>(MATH_FIDELITY) >= 3 || MATMUL_OUT_FORMAT != DataFormat::Bfp8_b);
#else
                    [[maybe_unused]] constexpr bool fused_post_flip = static_cast<std::uint32_t>(MATH_FIDELITY) >= 3;
#endif
#if defined(MATMUL_KEEP_FLIP_WRITE)
                    [[maybe_unused]] constexpr bool next_op_sets_base = false;
#else
                    [[maybe_unused]] constexpr bool next_op_sets_base = !is_fp32_dest_acc_en && !matmul_main_math_form;
#endif
                    _llk_math_dest_section_done_<MATMUL_DEST_DONE_ARGS>();
                }
            }
        }
        PROFILER_SYNC();
    }

    _llk_math_matmul_uninit_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

// The pack is production's, as in matmul_test.cpp: with compile-time formats the compute API's pack_block through
// pack_api_shim.h over a CB of the result tiles (production's LTO), with run-time formats per tile in both arms. The
// output geometry is a run-time parameter here: python gives the shim's descriptors the same values (PACK_API_SHIM_NUM_FACES,
// PACK_API_SHIM_FACE_R_DIM). A partial-face output packs by block once the shim describes it (pack_partial_face from
// PACK_API_SHIM_PARTIAL_FACE), per tile in both arms until then.
#if defined(ARCH_BLACKHOLE) && !defined(RUNTIME_FORMATS)
#if !defined(PACK_API_SHIM_NUM_FACES) || !defined(PACK_API_SHIM_FACE_R_DIM)
#error "math_matmul_test's shim pack needs the output geometry from python (PACK_API_GEOMETRY)"
#endif
#if __has_include("llk_pack_block.h")
// the API traits of a ttnn matmul kernel (the PR's pack contexts; main has none): a matmul header, and compute_kernel_api.h's pack-thread SFPU init
#define PACK_API_SHIM_MATMUL
#define PACK_API_SHIM_SFPU
#endif
#include "pack_api_shim.h"
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
// the python mirror's form of the block (MATMUL_PACK_FORM), checked against the LLK above; the pack itself is production's (below)
constexpr int matmul_pack_form_v = matmul_pack_form_key;
#line 279

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
#if defined(PACK_API_SHIM_PARTIAL_FACE) && PACK_API_SHIM_PARTIAL_FACE
// the shim sets pack_partial_face when it has the knob
constexpr bool matmul_pack_by_block = pack_partial_face[PACK_API_SHIM_OUT] != 0;
#else
constexpr bool matmul_pack_by_block = true;
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
__attribute__((always_inline)) inline void matmul_pack_block(const std::uint32_t start_tile_index, const std::uint32_t num_tiles)
{
#if MATMUL_PR_LLK
    if constexpr (constant_count && matmul_pack_by_block)
    {
        pack_api_shim::pack_block<ntiles, is_fp32_dest_acc_en, matmul_pack_kt, MATMUL_PACK_ARM>(start_tile_index, matmul_pack_out);
    }
    else
#endif
    {
        matmul_pack_runtime_count<is_fp32_dest_acc_en>(start_tile_index, matmul_pack_out, num_tiles);
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

#line 320
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t in0_tile_r_dim     = params.in0_tile_r_dim;
    const bool PARTIAL_FACE_PACK           = params.PARTIAL_FACE_PACK;
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK     = params.TILE_SIZE_PACK;
    const int num_faces                    = params.num_faces;
    const int DST_INDEX                    = params.DST_INDEX;
#if defined(MATMUL_PACK_CONST)
#ifndef MATMUL_CONST_DIMS
#error MATMUL_PACK_CONST needs MATMUL_CONST_DIMS
#endif
    // ttnn's matmul kernels pass pack_block a compile-time tile count (out_subblock_num_tiles)
    [[maybe_unused]] constexpr std::uint32_t CT_DIM = MATMUL_CONST_CT;
    [[maybe_unused]] constexpr std::uint32_t RT_DIM = MATMUL_CONST_RT;
#ifdef MATMUL_PACK_KEYS
    static_assert(CT_DIM == MATMUL_PK_CT && RT_DIM == MATMUL_PK_RT, "pack keys and tile dims disagree");
#endif
#else
    [[maybe_unused]] const std::uint32_t CT_DIM = params.CT_DIM;
    [[maybe_unused]] const std::uint32_t RT_DIM = params.RT_DIM;
#endif
#line 335
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
#if defined(MATMUL_PACK_CONST)
    constexpr std::uint32_t NUM_TILES_IN_BLOCK = MATMUL_CONST_NTB;
#else
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
#line 337
    const Operand& buffer_Res              = params.buffer_Res;
#endif
#if defined(MATMUL_PACK_CONST)
    [[maybe_unused]] constexpr bool matmul_const_count     = true;
    [[maybe_unused]] constexpr std::uint32_t matmul_ntiles = MATMUL_CONST_NTB;
#else
    [[maybe_unused]] constexpr bool matmul_const_count     = false;
    [[maybe_unused]] constexpr std::uint32_t matmul_ntiles = 0;
#endif
#ifdef MATMUL_L1_FUNC
    // packer L1 accumulate: the blocks of a group pack onto the same result tiles, the first writes and the later ones accumulate (as the ttnn bmm kernels spill partials)
    constexpr std::uint32_t result_block_stride = 0;
#else
    const std::uint32_t result_block_stride = NUM_TILES_IN_BLOCK;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src,
            formats.pack_dst,
            TILE_SIZE_PACK,
            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM,
            TILE_C_DIM,
            num_faces,
            PARTIAL_FACE_PACK);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
            formats.pack_dst, in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM, TILE_C_DIM, num_faces);
#if MATMUL_PR_LLK
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en, !matmul_true_main_key>();
#else
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#endif
#if MATMUL_PACK_SHIM
#if MATMUL_PR_LLK
        // as matmul_block_init marks the pack thread
        ckernel::packer::llk_pack_math_context(1);
#endif
        // the output CB, in the 16 byte words of the metal CB pointers, pages of a result tile: perf runs the PERF_OUTPUT ring (as
        // matmul_test.cpp), golden runs and L1 accumulation groups the result tiles of the NUM_BLOCKS blocks (one block)
        {
            const bool ring           = LOOP_FACTOR > 1 && result_block_stride != 0;
            const std::uint32_t page  = L1_ADDRESS(buffer_Res[1]) - L1_ADDRESS(buffer_Res[0]);
            const std::uint32_t base  = ring ? PERF_ADDRESS(PERF_OUTPUT, 0) : L1_ADDRESS(buffer_Res[0]);
            const std::uint32_t pages = ring ? matmul_ring_pages(NUM_TILES_IN_BLOCK, page) : (result_block_stride == 0 ? NUM_TILES_IN_BLOCK : NUM_TILES_IN_BLOCK * NUM_BLOCKS);
            pack_api_shim::cb_bind(matmul_pack_out, base + 1, page, pages);
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
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
#if defined(MATMUL_PACK_STATE) && MATMUL_PACK_STATE == 1
                    // every block the first of its output behind a pack reconfigure: production's reconfigure sets the state
                    _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(
                        formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM, TILE_C_DIM, num_faces, PARTIAL_FACE_PACK);
#elif defined(MATMUL_PACK_STATE) && MATMUL_PACK_STATE == 2 && MATMUL_PACK_SHIM && MATMUL_PR_LLK
                    // every block the first of its output behind a pack init: the state a set_packer_strides leaves (one store the harness adds)
                    ckernel::packer::llk_pack_state_set_half(0);
#endif
#line 391
                    if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
#ifdef MATMUL_L1_FUNC
                    reconfigure_packer_l1_acc(block % MATMUL_L1_FUNC == 0 ? 0 : 1);
#endif
#if MATMUL_PACK_SHIM
                    matmul_pack_block<matmul_const_count, matmul_ntiles>(DST_INDEX, NUM_TILES_IN_BLOCK);
#else
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        matmul_pack_tile_rt<is_fp32_dest_acc_en>(DST_INDEX + tile, L1_ADDRESS(buffer_Res[block * result_block_stride + tile]), formats.pack_dst);
                    }
#endif
                    if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
                    {
                        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
#if MATMUL_PACK_SHIM
                    // as a ttnn kernel: tile_regs_release, then cb_push_back
                    pack_api_shim::cb_push_back(matmul_pack_out, NUM_TILES_IN_BLOCK);
#endif
                }
            }
#ifdef MATMUL_L1_FUNC
            reconfigure_packer_l1_acc(0);
#endif
        }
        PROFILER_SYNC();
    }
}

#endif
