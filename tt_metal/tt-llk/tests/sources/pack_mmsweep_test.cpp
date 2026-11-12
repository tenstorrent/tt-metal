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
#include "pack_sweep_helpers.h"
#include "params.h"
#include "sfpu_operations.h"
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

// Crossover sweep of llk_pack_block's forms (mirrors tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h). The state and the form come in the
// high bits of DST_INDEX so that every arm of a cell is the same ELF: state 0 steady (one output, no init or reconfigure between blocks), 1 an init before
// every block, 2 two outputs alternating with a reconfigure before each block. Form 0 per-tile packs, 1 the plain or closed block that programs the output,
// 2 the W offset block (steady only; the output programmed once before the loop). Reconfigure state: 0 both outputs per tile, 1 A block, 2 both block, 3 B
// block.
#if defined(ARCH_BLACKHOLE)
// the llk_pack_block gate: the default destination write of the per-tile and block packs is the byte-wise one under full sync
constexpr bool sw_rmw = dest_sync == DstSync::SyncFull;

static inline std::uint32_t sw_page_bytes(const std::uint32_t fmt, const std::uint32_t r, const std::uint32_t faces)
{
    switch (static_cast<DataFormat>(fmt))
    {
        case DataFormat::Float32:
        case DataFormat::Int32:
        case DataFormat::UInt32:
        case DataFormat::Tf32:
            return faces * r * 16 * 4;
        case DataFormat::Int8:
        case DataFormat::UInt8:
        case DataFormat::Lf8:
        case DataFormat::Fp8_e4m3:
            return faces * r * 16;
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp8:
            return faces * (r * 16 + 16);
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp4:
            return faces * (r * 8 + 16);
        case DataFormat::Bfp2_b:
        case DataFormat::Bfp2:
            return faces * (r * 4 + 16);
        default:
            return faces * r * 16 * 2;
    }
}

template <bool fp32>
inline void sw_pack_tile(const std::uint32_t fmt, const std::uint32_t tile, const std::uint32_t addr)
{
    if (!fp32 && fmt == static_cast<std::uint32_t>(DataFormat::Float32))
    {
        _llk_pack_<dest_sync, fp32, ckernel::PackMode::Default, false, false>(tile, addr);
    }
    else
    {
        _llk_pack_<dest_sync, fp32, ckernel::PackMode::Default, false, sw_rmw>(tile, addr);
    }
}

// The circular buffer interface llk_pack_block reads and advances (fifo_wr_ptr, fifo_wr_tile_ptr, fifo_page_size, in 16 byte words): the per-call work of the
// API around the packs, so that the forms are compared with their RISC side included.
struct sw_cb_t
{
    std::uint32_t fifo_wr_ptr;
    std::uint32_t fifo_wr_tile_ptr;
    std::uint32_t fifo_page_size;
};

static volatile sw_cb_t sw_cb[2];
#endif

// The pack init of the context: tilize_init packs in the Tilize mode on Blackhole, reduce_init adds the edge masks of the reduce.
inline void ctx_pack_init(
    const std::uint32_t src, const std::uint32_t dst, const std::uint32_t face_r, const std::uint32_t tile_c, const std::uint32_t num_faces, const bool skip_tw)
{
    if constexpr (mm_ctx == CTX_TILIZE)
    {
        _llk_pack_init_with_src_wrapper_<llk_test_pack_mode_v<false, true>, false>(src, dst, face_r, tile_c, num_faces, false, false, 1, skip_tw);
    }
    else
    {
        _llk_pack_init_with_src_wrapper_<llk_test_pack_mode_v<false, false>, false>(src, dst, face_r, tile_c, num_faces);
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
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX & 0xff;
    const int RELU_CONFIG                  = params.RELU_CONFIG;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
#if defined(ARCH_BLACKHOLE)
    const std::uint32_t face_r   = params.TEST_FACE_R_DIM;
    const std::uint32_t tile_c   = params.in0_tile_c_dim;
    const std::uint32_t fmt_a    = formats.pack_dst;
    const std::uint32_t fmt_b    = static_cast<std::uint32_t>(DataFormat::Float16_b);
    const std::uint32_t page_a   = sw_page_bytes(fmt_a, face_r, num_faces);
    const std::uint32_t page_b   = sw_page_bytes(fmt_b, face_r, num_faces);
    const std::uint32_t tiles_a  = ((static_cast<std::uint32_t>(NUM_BLOCKS) + 1) / 2) * NUM_TILES_IN_BLOCK;
    const std::uint32_t base_a   = L1_ADDRESS(buffer_Res[0]);
    const std::uint32_t base_b   = base_a + ((tiles_a * page_a + 1023) & ~1023u) / 16;
    const std::uint32_t state_rt = (static_cast<std::uint32_t>(params.DST_INDEX) >> 12) & 3;
    const std::uint32_t form_rt  = (static_cast<std::uint32_t>(params.DST_INDEX) >> 8) & 3;
    // pk-limit5: DST_INDEX bit 14 = main's per-tile pack in the state (no close, no byte-wise RMW)
    const std::uint32_t main_rt = (static_cast<std::uint32_t>(params.DST_INDEX) >> 14) & 1;
    // pk-limit5: DST_INDEX bit 15 = the W offset under cb0ac's rule in a CB of two blocks: W continues only right behind a W block of the output at the
    // next address, otherwise programs the base first; the CB wraps every two blocks.
    const std::uint32_t wrule_rt = (static_cast<std::uint32_t>(params.DST_INDEX) >> 15) & 1;
    const bool skip_tw           = _llk_pack_skip_bh_tilize_workaround_wrapper_(formats.unpack_A_src);
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
            formats.pack_src, fmt_a, page_a, face_r, tile_c, num_faces, false, false, RELU_CONFIG);
        ctx_pack_init(formats.pack_src, fmt_a, face_r, tile_c, num_faces, skip_tw);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        if constexpr (sw_rmw)
        {
            _llk_pack_rmw_dest_init_<true>();
        }
        if constexpr (ctx_pack_sfpu)
        {
            pack_sfpu_tail<true>(0, formats.math);
        }
        PROFILER_SYNC();
    }
    // read through volatiles so every arm of a cell is the same loop
    const volatile std::uint32_t state_v = state_rt;
    const volatile std::uint32_t form_v  = form_rt;
    const std::uint32_t state            = state_v;
    const std::uint32_t form_a           = form_v;
    const volatile std::uint32_t main_v  = main_rt;
    const bool main_arm                  = main_v != 0;
    const volatile std::uint32_t wrule_v = wrule_rt;
    const bool wrule                     = wrule_v != 0;
    bool w_prev_w                        = false;
    if (state == 0 && form_a == 2 && !wrule)
    {
        _llk_pack_block_w_init_(base_a, page_a);
        pack_w_valid = true;
        llk_pack_state_set_half(1u | LLK_PACK_STATE_W);
        pack_w_page = page_a / 16;
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
                std::uint32_t cur         = 0;
                sw_cb[0].fifo_wr_ptr      = base_a;
                sw_cb[0].fifo_wr_tile_ptr = 1;
                sw_cb[0].fifo_page_size   = page_a / 16;
                sw_cb[1].fifo_wr_ptr      = base_b;
                sw_cb[1].fifo_wr_tile_ptr = 1;
                sw_cb[1].fifo_page_size   = page_b / 16;
                if (!wrule)
                {
                    pack_w_next  = base_a;
                    pack_w_tiles = 0;
                }
                for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(NUM_BLOCKS); ++block)
                {
                    if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    if constexpr (ctx_pack_sfpu)
                    {
                        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset());
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            pack_sfpu_tail<false>(DST_INDEX + tile, formats.math);
                        }
                        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
                    }
                    const std::uint32_t out  = state == 2 ? (block & 1) : 0u;
                    const std::uint32_t fmt  = out ? fmt_b : fmt_a;
                    const std::uint32_t page = out ? page_b : page_a;
                    if (out != cur)
                    {
                        _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(formats.pack_src, fmt, page, face_r, tile_c, num_faces);
                        cur = out;
                    }
                    if (state == 1)
                    {
                        ctx_pack_init(formats.pack_src, fmt_a, face_r, tile_c, num_faces, skip_tw);
                        if constexpr (sw_rmw)
                        {
                            if (!main_arm)
                            {
                                _llk_pack_rmw_dest_init_<true>();
                            }
                        }
                    }
                    // reconfigure state: form 0 both outputs per tile, 1 A block, 2 both blocks, 3 B block (B is a Float16_b output)
                    const std::uint32_t form =
                        state != 2 ? form_a : (out ? ((form_a == 2 || form_a == 3) ? 1u : 0u) : ((form_a == 1 || form_a == 2) ? 1u : 0u));
                    const bool closed    = out ? sweep_b_closed : sweep_closed;
                    volatile sw_cb_t& cb = sw_cb[out];
                    if ((state == 0 && form == 3) || main_arm)
                    {
                        // main per-tile pack: the full destination write, no byte-wise RMW
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            const std::uint32_t a = cb.fifo_wr_ptr + cb.fifo_wr_tile_ptr - 1;
                            cb.fifo_wr_tile_ptr   = cb.fifo_wr_tile_ptr + cb.fifo_page_size;
                            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default, false, false>(DST_INDEX + tile, a);
                        }
                    }
                    else if (form == 0)
                    {
                        _llk_pack_close_fast_untilize_<dest_sync>();
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            const std::uint32_t a = cb.fifo_wr_ptr + cb.fifo_wr_tile_ptr - 1;
                            cb.fifo_wr_tile_ptr   = cb.fifo_wr_tile_ptr + cb.fifo_page_size;
                            sw_pack_tile<is_fp32_dest_acc_en>(fmt, DST_INDEX + tile, a);
                        }
                    }
                    else
                    {
                        if (wrule && (block & 1) == 0)
                        {
                            cb.fifo_wr_tile_ptr = 1;
                        }
                        const std::uint32_t pg = cb.fifo_page_size;
                        const std::uint32_t a  = cb.fifo_wr_ptr + cb.fifo_wr_tile_ptr - 1;
                        cb.fifo_wr_tile_ptr    = cb.fifo_wr_tile_ptr + pg * NUM_TILES_IN_BLOCK;
                        if (form == 1)
                        {
                            if (closed)
                            {
                                _llk_pack_block_closed_<dest_sync, is_fp32_dest_acc_en, sw_rmw>(DST_INDEX, a, NUM_TILES_IN_BLOCK, page, num_faces);
                            }
                            else
                            {
                                _llk_pack_block_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default, sw_rmw>(DST_INDEX, a, NUM_TILES_IN_BLOCK);
                            }
                        }
                        else
                        {
                            // llk_pack_w_take: the block continues the one before
                            std::uint32_t out_tile = 0;
                            if (wrule)
                            {
                                if (!(w_prev_w && pack_w_valid && pack_w_page == pg && pack_w_next == a &&
                                      pack_w_tiles + NUM_TILES_IN_BLOCK <= (page_a > 2048 ? 64u : 127u)))
                                {
                                    pack_sweep_w_reprogram(a, pg << 4);
                                    pack_w_valid = true;
                                    pack_w_page  = pg;
                                    pack_w_next  = a;
                                    pack_w_tiles = 0;
                                }
                                out_tile = pack_w_tiles;
                                pack_w_tiles += NUM_TILES_IN_BLOCK;
                                pack_w_next += pg * NUM_TILES_IN_BLOCK;
                                w_prev_w = true;
                            }
                            else if (
                                llk_pack_state_half() == ((out + 1u) | LLK_PACK_STATE_W) && pack_w_valid && pack_w_page == pg && pack_w_next == a &&
                                pack_w_tiles + NUM_TILES_IN_BLOCK <= (page_a > 2048 ? 64u : 127u))
                            {
                                out_tile = pack_w_tiles;
                                pack_w_tiles += NUM_TILES_IN_BLOCK;
                                pack_w_next += pg * NUM_TILES_IN_BLOCK;
                            }
                            if (closed)
                            {
                                _llk_pack_block_closed_w_mop_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX, out_tile, NUM_TILES_IN_BLOCK, num_faces);
                            }
                            else
                            {
                                _llk_pack_block_w_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX, out_tile, NUM_TILES_IN_BLOCK);
                            }
                        }
                    }
                    if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
                    {
                        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    if constexpr (ctx_reduce)
    {
        _llk_pack_reduce_mask_clear_();
    }
#endif
}
#endif
