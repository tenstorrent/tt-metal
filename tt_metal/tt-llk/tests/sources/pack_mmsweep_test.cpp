// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// pk-ctx sweep: contexts 12/13 add a math-thread / pack-thread SFPU tail of tail_op to the matmul. pack_ctx_test.cpp plus matmul contexts (7 matmul, 8 matmul +
// SFPU abs, 9 matmul + SFPU gelu) and mm_prefix (the matmul inits before the context's). llk_pack_block's forms behind the other unpack/math contexts
// (pk-limit3 context sweep): the pack side is pack_block_sweep8_test.cpp's, unpack and math are one of: 0 eltwise add, 1 eltwise mul (both operands from L1), 2
// unpack tilize, 3 reduce row, 4 unpack transpose, 5 datacopy then a unary SFPU op (abs), 6 reduce col. mm_ctx, POOL_TYPE and REDUCE_DIM come from the
// PACK_CTX template parameter.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr int CTX_ADD = 0, CTX_MUL = 1, CTX_TILIZE = 2, CTX_RROW = 3, CTX_TRANS = 4, CTX_SFPU = 5, CTX_RCOL = 6, CTX_MM = 7, CTX_MMSFPU = 8, CTX_MMGELU = 9,
              CTX_MMPABS = 10, CTX_MMPGELU = 11, CTX_TAILM = 12, CTX_TAILP = 13;
constexpr bool ctx_pack_sfpu                = mm_ctx == CTX_MMPABS || mm_ctx == CTX_MMPGELU || mm_ctx == CTX_TAILP;
constexpr bool ctx_matmul                   = mm_ctx >= CTX_MM;
constexpr bool ctx_reduce                   = mm_ctx == CTX_RROW || mm_ctx == CTX_RCOL;
static constexpr bool IS_SWAPPED_REDUCE_ROW = (REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW) && (POOL_TYPE != ckernel::PoolType::MAX);

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common.h"
#include "params.h"

template <int C>
inline void ctx_unpack_init(
    const FormatConfig& formats, const ckernel::TensorShape& ts, const std::uint32_t face_r, const std::uint32_t num_faces, const std::uint32_t block_tiles)
{
    if constexpr (mm_prefix || ctx_matmul)
    {
        if constexpr (ctx_matmul)
        {
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r, face_r, num_faces, num_faces);
        }
        _llk_unpack_AB_matmul_init_<>(0, block_tiles, 1, ctx_matmul ? mm_kt : 1, face_r, face_r, num_faces, num_faces, false, false);
        if constexpr (ctx_matmul)
        {
            return;
        }
    }
    if constexpr (C == CTX_ADD || C == CTX_MUL)
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r, face_r, num_faces, num_faces);
        _llk_unpack_AB_init_<BroadcastType::NONE>(ts, ts, ckernel::Transpose::None);
    }
    else if constexpr (C == CTX_RROW || C == CTX_RCOL)
    {
        if constexpr (IS_SWAPPED_REDUCE_ROW)
        {
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                formats.unpack_B_src, formats.unpack_A_src, formats.unpack_B_dst, formats.unpack_A_dst, face_r, face_r, num_faces, num_faces);
        }
        else
        {
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r, face_r, num_faces, num_faces);
        }
        _llk_unpack_AB_reduce_init_<POOL_TYPE, REDUCE_DIM>(ts, ts);
    }
    else if constexpr (C == CTX_TILIZE)
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r, face_r, num_faces, num_faces);
        _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, block_tiles, face_r, false);
    }
    else
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r, face_r, num_faces, num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            C == CTX_TRANS ? 1 : 0, C == CTX_TRANS ? 1 : 0, ts, formats.unpack_A_src, formats.unpack_A_dst);
    }
}

template <int C>
inline void ctx_unpack_tile(
    const FormatConfig& formats,
    const Operand& a,
    const Operand& b,
    const std::uint32_t block,
    const std::uint32_t tile,
    const std::uint32_t block_tiles,
    const std::uint32_t face_r)
{
    const std::uint32_t t = block * block_tiles + tile;
    if constexpr (C == CTX_ADD || C == CTX_MUL)
    {
        _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(a[t]), L1_ADDRESS(b[t]));
    }
    else if constexpr (C == CTX_RROW || C == CTX_RCOL)
    {
        _llk_unpack_AB_reduce_<POOL_TYPE, REDUCE_DIM>(L1_ADDRESS(a[t]), L1_ADDRESS(b[0]));
    }
    else if constexpr (C == CTX_TILIZE)
    {
        _llk_unpack_tilize_wrapper_(L1_ADDRESS(a[block * block_tiles]), tile, formats.unpack_A_src, formats.unpack_A_dst, 0, face_r, 4, false);
    }
    else
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(a[t]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_A                = params.buffer_A;
    const Operand& buffer_B                = params.buffer_B;
    const std::uint32_t face_r             = params.TEST_FACE_R_DIM;
    const ckernel::TensorShape ts          = ckernel::make_tensor_shape_from_legacy(face_r, num_faces);
    {
        START_PERF_MEASURE("INIT")
        ctx_unpack_init<mm_ctx>(formats, ts, face_r, num_faces, NUM_TILES_IN_BLOCK);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(NUM_BLOCKS); ++block)
                {
                    if constexpr (ctx_matmul)
                    {
                        for (std::uint32_t k = 0; k < mm_kt; ++k)
                        {
                            _llk_unpack_AB_matmul_<>(
                                L1_ADDRESS(buffer_A[0]),
                                L1_ADDRESS(buffer_B[0]),
                                0,
                                0,
                                params.TILE_SIZE_UNPACK_A,
                                params.TILE_SIZE_UNPACK_B,
                                false,
                                false,
                                NUM_TILES_IN_BLOCK,
                                1,
                                mm_kt);
                        }
                    }
                    else
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            ctx_unpack_tile<mm_ctx>(formats, buffer_A, buffer_B, block, tile, NUM_TILES_IN_BLOCK, face_r);
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    if constexpr (mm_ctx == CTX_TILIZE)
    {
        _llk_unpack_tilize_uninit_wrapper_(formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_matmul.h"
#include "llk_math_reduce.h"
#include "params.h"
#include "sfpu_operations.h"

using namespace ckernel;
constexpr SfpuType tail_op = SfpuType::PK_TAIL_OP;

template <int C>
inline void ctx_math_init(const FormatConfig& formats, const ckernel::TensorShape& ts, const std::uint32_t block_tiles)
{
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    if constexpr (mm_prefix || ctx_matmul)
    {
        _llk_math_matmul_init_<MATH_FIDELITY>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, 0, block_tiles, 1);
    }
    if constexpr (C == CTX_MMSFPU)
    {
        test_utils::call_unary_sfpu_operation_init<SfpuType::abs, false, is_fp32_dest_acc_en, 32>(formats.math);
    }
    else if constexpr (C == CTX_MMGELU)
    {
        test_utils::call_unary_sfpu_operation_init<SfpuType::gelu, false, is_fp32_dest_acc_en, 32>(formats.math);
    }
    else if constexpr (C == CTX_TAILM)
    {
        test_utils::call_unary_sfpu_operation_init<tail_op, false, is_fp32_dest_acc_en, 32>(formats.math);
    }
    else if constexpr (ctx_matmul)
    {
    }
    else if constexpr (C == CTX_ADD)
    {
        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::LoFi>(ts, 0);
    }
    else if constexpr (C == CTX_MUL)
    {
        _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY>(ts, 0);
    }
    else if constexpr (C == CTX_RROW || C == CTX_RCOL)
    {
        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(ts);
    }
    else if constexpr (C == CTX_TILIZE)
    {
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, llk_test_pack_mode_v<false, true>>(
            ts, formats.math);
    }
    else
    {
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(ts, formats.math);
        if constexpr (C == CTX_SFPU)
        {
            test_utils::call_unary_sfpu_operation_init<SfpuType::abs, false, is_fp32_dest_acc_en, 32>(formats.math);
        }
    }
}

template <int C>
inline void ctx_math_tile(const FormatConfig& formats, const ckernel::TensorShape& ts, const std::uint32_t dst)
{
    if constexpr (C == CTX_ADD)
    {
        _llk_math_eltwise_binary_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, dest_sync, is_fp32_dest_acc_en, MathFidelity::LoFi>(ts, dst, false);
    }
    else if constexpr (C == CTX_MUL)
    {
        _llk_math_eltwise_binary_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY>(ts, dst, false);
    }
    else if constexpr (C == CTX_RROW || C == CTX_RCOL)
    {
        _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, false>(dst, ts);
    }
    else
    {
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            dst, formats.math, formats.math, ts);
        if constexpr (C == CTX_SFPU)
        {
            test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, SfpuType::abs, false, is_fp32_dest_acc_en, 32>(dst, formats.math);
        }
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX & 0xff;
    const ckernel::TensorShape ts          = ckernel::make_tensor_shape_from_legacy(params.TEST_FACE_R_DIM, num_faces);
    {
        START_PERF_MEASURE("INIT")
        ctx_math_init<mm_ctx>(formats, ts, NUM_TILES_IN_BLOCK);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(NUM_BLOCKS); ++block)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                    if constexpr (ctx_matmul)
                    {
                        for (std::uint32_t k = 0; k < mm_kt; ++k)
                        {
                            _llk_math_matmul_<MATH_FIDELITY>(DST_INDEX, NUM_TILES_IN_BLOCK, 1);
                        }
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            if constexpr (mm_ctx == CTX_MMSFPU)
                            {
                                test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, SfpuType::abs, false, is_fp32_dest_acc_en, 32>(
                                    DST_INDEX + tile, formats.math);
                            }
                            else if constexpr (mm_ctx == CTX_MMGELU)
                            {
                                test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, SfpuType::gelu, false, is_fp32_dest_acc_en, 32>(
                                    DST_INDEX + tile, formats.math);
                            }
                            else if constexpr (mm_ctx == CTX_TAILM)
                            {
                                test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, tail_op, false, is_fp32_dest_acc_en, 32>(
                                    DST_INDEX + tile, formats.math);
                            }
                        }
                    }
                    else
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            ctx_math_tile<mm_ctx>(formats, ts, DST_INDEX + tile);
                        }
                    }
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
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_pack_common.h"
#include "params.h"
#include "sfpu_operations.h"

// after params.h: the shim reads the test's formats and PACK_API_SHIM_* settings; after sfpu_operations.h, whose sanitizer/api.h
// defines SAN_HOOK before the shim's fallback would
#include "pack_api_shim.h"
#include "pack_api_sweep.h"
constexpr SfpuType tail_op = SfpuType::PK_TAIL_OP;

template <bool init>
inline void pack_sfpu_tail(const std::uint32_t dst, const std::uint32_t math_format)
{
    if constexpr (mm_ctx == CTX_MMPABS)
    {
        if constexpr (init)
        {
            test_utils::call_unary_sfpu_operation_init<SfpuType::abs, false, is_fp32_dest_acc_en, 32>(math_format);
        }
        else
        {
            test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, SfpuType::abs, false, is_fp32_dest_acc_en, 32>(dst, math_format);
        }
    }
    else if constexpr (mm_ctx == CTX_MMPGELU)
    {
        if constexpr (init)
        {
            test_utils::call_unary_sfpu_operation_init<SfpuType::gelu, false, is_fp32_dest_acc_en, 32>(math_format);
        }
        else
        {
            test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, SfpuType::gelu, false, is_fp32_dest_acc_en, 32>(dst, math_format);
        }
    }
    else if constexpr (mm_ctx == CTX_TAILP)
    {
        if constexpr (init)
        {
            test_utils::call_unary_sfpu_operation_init<tail_op, false, is_fp32_dest_acc_en, 32>(math_format);
        }
        else
        {
            test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, tail_op, false, is_fp32_dest_acc_en, 32>(dst, math_format);
        }
    }
}

// Crossover sweep of llk_pack_block's forms through the compute API's code (pack_api_shim.h, pack_api_sweep.h), one ELF per
// arm: the state (steady, an init before every block, outputs A and B alternating with a reconfigure), the forms of A and
// B (PACK_API, PACK_API_SWEEP) and A's CB depth.

// The pack init of the context, production's llk_pack_init: tilize_init packs in the Tilize mode on Blackhole,
// reduce_init adds the edge masks of the reduce.
inline void ctx_pack_init(const std::uint32_t out, [[maybe_unused]] const std::uint32_t face_r, [[maybe_unused]] const std::uint32_t num_faces)
{
    if constexpr (mm_ctx == CTX_TILIZE)
    {
        pack_api_sweep::init<pack_api_arm, PackMode::Tilize>(out);
    }
    else
    {
        pack_api_sweep::init<pack_api_arm, PackMode::Default>(out);
    }
    if constexpr (ctx_reduce)
    {
        _llk_pack_reduce_mask_config_<REDUCE_DIM>(ckernel::make_tensor_shape_from_legacy(face_r, num_faces));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const int NUM_BLOCKS            = params.NUM_BLOCKS;
    const int DST_INDEX             = params.DST_INDEX & 0xff;
    const int RELU_CONFIG           = params.RELU_CONFIG;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
#if defined(ARCH_BLACKHOLE)
    constexpr std::uint32_t out_a = PACK_API_SHIM_OUT;
    constexpr std::uint32_t out_b = PACK_API_SHIM_OUT2;
    const std::uint32_t dst       = pack_api_sweep::dest_start(DST_INDEX);
    const std::uint32_t face_r    = params.TEST_FACE_R_DIM;
    const std::uint32_t tile_c    = params.in0_tile_c_dim;
    const std::uint32_t page_a    = pack_api_sweep::page_bytes(formats.pack_dst, face_r, num_faces);
    const std::uint32_t page_b    = pack_api_sweep::page_bytes(PACK_API_SHIM_OUT2_FORMAT, face_r, num_faces);
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
            formats.pack_src, formats.pack_dst, page_a, face_r, tile_c, num_faces, false, false, RELU_CONFIG);
        pack_api_sweep::bind<pack_api_tiles, pack_sweep_state, pack_sweep_cbd, out_a, out_b>(L1_ADDRESS(buffer_Res[0]), NUM_BLOCKS, page_a, page_b);
        ctx_pack_init(out_a, face_r, num_faces);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        if constexpr (ctx_matmul)
        {
            // as matmul_block_init marks the pack thread
            pack_api_sweep::mark_matmul();
        }
        if constexpr (ctx_pack_sfpu)
        {
            // as a pack-thread SFPU init marks it behind the matmul
            pack_api_sweep::mark_sfpu();
            pack_sfpu_tail<true>(0, formats.math);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        pack_api_sweep::run<pack_api_tiles, is_fp32_dest_acc_en, pack_api_kt, pack_sweep_state, pack_api_arm, pack_sweep_arm_b, out_a, out_b>(
            LOOP_FACTOR,
            NUM_BLOCKS,
            dst,
            [&] { ctx_pack_init(out_a, face_r, num_faces); },
            [&]
            {
                if constexpr (ctx_pack_sfpu)
                {
                    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset());
                    for (std::uint32_t tile = 0; tile < pack_api_tiles; ++tile)
                    {
                        pack_sfpu_tail<false>(dst + tile, formats.math);
                    }
                    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
                }
            });
        PROFILER_SYNC();
    }
    if constexpr (ctx_reduce)
    {
        _llk_pack_reduce_mask_clear_();
    }
#endif
}
#endif
