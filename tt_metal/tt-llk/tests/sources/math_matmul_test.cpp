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

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#if defined(ARCH_BLACKHOLE) && defined(MATMUL_ROW_LE_8)
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
            num_faces_B,    // in1
            num_faces_A,    // in0
            PARTIAL_FACE_B, // in1
            PARTIAL_FACE_A);
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
            _perf_unpack_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, NUM_BLOCKS, matmul_reuse_dir_k);
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
        dest_sync == DstSync::SyncFull || matmul_half_end_fold_k, matmul_reuse_dir_k, matmul_row_end_fold_k, 0, 0, matmul_bias_write
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
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM, NUM_BLOCKS, matmul_reuse_dir_k);
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

#ifndef MATMUL_PACK_FORM
#define MATMUL_PACK_FORM 0
#endif
#if defined(ARCH_BLACKHOLE)
#include "llk_matmul_forms.h"
#ifdef MATMUL_PACK_KEYS
// the pack form of the block: the LLK decision, which the python mirror (the MATMUL_PACK_FORM define) must equal
constexpr int matmul_pack_form_v = ckernel::matmul_pack_form_k(
    MATMUL_PK_FID, MATMUL_PK_HALF, MATMUL_PK_THROTTLE, MATMUL_PK_KT, MATMUL_PK_RT, MATMUL_PK_CT, MATMUL_PK_IN_A, MATMUL_PK_IN_B, MATMUL_PK_OUT, MATMUL_PK_R0, MATMUL_PK_C1, MATMUL_PK_DENSE,
    MATMUL_PK_L1ACC);
static_assert(matmul_pack_form_v == MATMUL_PACK_FORM, "the python matmul_pack_form differs from matmul_pack_form_k");
#else
constexpr int matmul_pack_form_v = MATMUL_PACK_FORM;
#endif
#else
constexpr int matmul_pack_form_v = MATMUL_PACK_FORM;
#endif
#include "matmul_pack_production.h"
#ifdef MATMUL_PACK_KEYS
constexpr DataFormat matmul_pack_out_format = MATMUL_PK_OUT;
#else
constexpr DataFormat matmul_pack_out_format = DataFormat::Invalid;
#endif
#line 279
// The fast block packs need four faces of 16 rows.
#if defined(ARCH_BLACKHOLE)
#define MATMUL_FAST_PACK_ARG , num_faces == 4 && !PARTIAL_FACE_PACK
#else
#define MATMUL_FAST_PACK_ARG
#endif

#if defined(ARCH_BLACKHOLE)
// MATMUL_PACK_FORM 1 and 2: the block pack llk_pack_block runs (matmul_pack_production.h); 4 runs small tiles back to back with one close.
inline void matmul_pack_out(std::uint32_t tile_index, std::uint32_t address, std::uint32_t num_tiles, std::uint32_t tile_stride, std::uint32_t num_faces, [[maybe_unused]] std::uint32_t face_r_dim, [[maybe_unused]] bool fast = false)
{
    if constexpr (matmul_pack_form_v == 4)
    {
        _llk_pack_block_dense_<dest_sync, is_fp32_dest_acc_en>(tile_index, address, num_tiles, face_r_dim, num_faces);
    }
    else
    {
        matmul_pack_production::block<dest_sync, is_fp32_dest_acc_en, matmul_pack_form_v, matmul_pack_out_format>(tile_index, address, num_tiles, tile_stride, num_faces);
    }
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
    const std::uint32_t CT_DIM             = params.CT_DIM;
    const std::uint32_t RT_DIM             = params.RT_DIM;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res              = params.buffer_Res;
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
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#if defined(ARCH_BLACKHOLE)
        if constexpr (matmul_pack_form_v == 4) { _llk_pack_block_dense_init_(in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM, num_faces); }
#endif
        // the W base of the block pack is programmed by its second call, as in llk_pack_block
        //
        //
        //
        //
        //
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
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                    {
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));
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
                    _llk_packer_wait_for_math_done_();
#ifdef MATMUL_L1_FUNC
                    // packer L1 accumulate: the blocks of a group pack onto the same result tiles, the first writes and the later ones accumulate (as the ttnn bmm kernels spill partials)
                    reconfigure_packer_l1_acc(block % MATMUL_L1_FUNC == 0 ? 0 : 1);
                    constexpr std::uint32_t result_block_stride = 0;
#else
                    const std::uint32_t result_block_stride = NUM_TILES_IN_BLOCK;
#endif
#if defined(ARCH_BLACKHOLE)
                    if constexpr (matmul_pack_form_v != 0)
                    {
                        matmul_pack_out(
                            DST_INDEX,
                            L1_ADDRESS(buffer_Res[block * result_block_stride]),
                            NUM_TILES_IN_BLOCK,
                            (L1_ADDRESS(buffer_Res[1]) - L1_ADDRESS(buffer_Res[0])) << 4,
                            num_faces,
                            in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM MATMUL_FAST_PACK_ARG);
                    }
                    else
#endif
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            const std::uint32_t result_tile = block * result_block_stride + tile;
                            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX + tile, L1_ADDRESS(buffer_Res[result_tile]));
                        }
                    }
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
