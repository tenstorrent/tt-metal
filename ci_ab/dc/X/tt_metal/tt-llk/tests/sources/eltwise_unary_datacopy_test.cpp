
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"
// DATACOPY_SYNC_FULL (dc-prod census only, never committed): every DEST sync of the kernel is SyncFull
#ifndef DC_DEST_SYNC
#ifdef DATACOPY_SYNC_FULL
#define DC_DEST_SYNC DstSync::SyncFull
#else
#define DC_DEST_SYNC DstSync::SyncHalf
#endif
#endif

// UNPACK_BLOCK (driver template): one _llk_unpack_A_block_ call per DEST block instead of one _llk_unpack_A_ per tile (Blackhole only)
#ifndef UNPACK_BLOCK
#define UNPACK_BLOCK 0
#endif
// DATACOPY_TILES_BACK_TO_BACK (driver template): the block's tiles are uncompressed and back to back in L1
#ifndef DATACOPY_TILES_BACK_TO_BACK
#define DATACOPY_TILES_BACK_TO_BACK 0
#endif
// DATACOPY_EARLY_POLL (dc-prod census only, never committed): the per tile call polls the context early, as copy_tile does for 16-bit input and DEST
#ifndef DATACOPY_EARLY_POLL
#define DATACOPY_EARLY_POLL 0
#endif

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
    const Operand& buffer_A                = params.buffer_A;
#endif
    const std::uint32_t num_tiles           = NUM_BLOCKS * NUM_TILES_IN_BLOCK;
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * _perf_src_handshake_iters_(tilize_en, num_tiles, num_faces);

    {
        START_PERF_MEASURE("INIT")
        if constexpr (!tilize_en)
        {
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                0 /* transpose_of_faces */,
                0 /* within_face_16x16_transpose */,
                ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
                formats.unpack_A_src,
                formats.unpack_A_dst);
        }
        else
        {
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
            _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, FACE_R_DIM, false /* narrow_tile */, num_faces);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // MATH_ISOLATE must fake the dvalids unpack would post. Blackhole tilize
            // always posts SrcB (UNP_ZEROSRC) per tile regardless of dest acc;
            // non-tilize unpack_A posts SrcB only for FP32 dest acc. Leaving unpack
            // idle here deadlocks math.
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_stream_set_valid</* src A */ true, /* src B */ tilize_en || is_fp32_dest_acc_en, PERF_RUN_TYPE>(src_handshake_iters);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (!tilize_en)
                {
#if UNPACK_BLOCK && defined(ARCH_BLACKHOLE) && 0  // main: no block LLK (copy_block runs per tile)
                    const std::uint32_t tile_stride_16B = (buffer_A[1] - buffer_A[0]) >> 4;
                    for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                    {
                        _llk_unpack_A_block_<
                            BroadcastType::NONE,
                            false,
                            EltwiseBinaryReuseDestType::NONE,
                            unpack_to_dest,
                            is_fp32_dest_acc_en,
                            DATACOPY_TILES_BACK_TO_BACK != 0>(
                            L1_ADDRESS(buffer_A[block_num * NUM_TILES_IN_BLOCK]),
                            NUM_TILES_IN_BLOCK,
                            tile_stride_16B,
                            formats.unpack_A_src,
                            formats.unpack_A_dst,
                            num_faces);
                    }
#else
                    for (std::uint32_t i = 0; i < num_tiles; ++i)
                    {
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(  // main: no early poll
                            L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                    }
#endif
                }
                else
                {
                    for (std::uint32_t i = 0; i < BLOCK_RT_DIM; i++)
                    {
                        const std::uint32_t read_offset = i * BLOCK_CT_DIM;
                        for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
                        {
                            _llk_unpack_tilize_wrapper_(
                                L1_ADDRESS(buffer_A[read_offset]),
                                j,
                                formats.unpack_A_src,
                                formats.unpack_A_dst,
                                0 /* block_ct_dim */,
                                FACE_R_DIM,
                                num_faces,
                                false /* narrow_tile */);
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    if constexpr (tilize_en)
    {
        _llk_unpack_tilize_uninit_wrapper_(formats.unpack_A_dst, num_faces);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"
#include "params.h"
// DATACOPY_SYNC_FULL (dc-prod census only, never committed): every DEST sync of the kernel is SyncFull
#ifndef DC_DEST_SYNC
#ifdef DATACOPY_SYNC_FULL
#define DC_DEST_SYNC DstSync::SyncFull
#else
#define DC_DEST_SYNC DstSync::SyncHalf
#endif
#endif

using namespace ckernel;

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
    const int DST_INDEX                    = params.DST_INDEX;
#endif
    const std::uint32_t num_tiles           = NUM_BLOCKS * NUM_TILES_IN_BLOCK;
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * _perf_src_handshake_iters_(tilize_en, num_tiles, num_faces);

    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            is_int_fpu_en,
            llk_test_pack_mode_v<false, tilize_en>>(tensor_shape, formats.math);
        _llk_math_pack_sync_init_<DC_DEST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            if constexpr (unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                    {
#if UNPACK_BLOCK && defined(ARCH_BLACKHOLE) && 0  // main: no block LLK (copy_block runs per tile)
                        _llk_math_eltwise_unary_datacopy_block_<DataCopyType::A2D, DC_DEST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            DST_INDEX, NUM_TILES_IN_BLOCK, formats.math, formats.math, num_faces);
#else
                        for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                        {
                            _llk_math_eltwise_unary_datacopy_wrapper_<
                                DataCopyType::A2D,
                                DC_DEST_SYNC,
                                is_fp32_dest_acc_en,
                                BroadcastType::NONE,
                                unpack_to_dest>(DST_INDEX + tile_num, formats.math, formats.math, tensor_shape);
                        }
#endif
                    }
                }
            }
            else
            {
                _perf_math_stream_clear_valid</* src A */ true, /* src B */ true, PERF_RUN_TYPE>(src_handshake_iters);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Unpack-to-dest fills dest in unpack; math has no source consumer
            // and must not wait on unpack dvalids.
            if constexpr (!unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                    {
                        for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                        {
                            LLK_ASSERT(
                                (DST_INDEX + tile_num < get_dest_max_tiles<DC_DEST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "tile_num exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_wrapper_<
                                DataCopyType::A2D,
                                DC_DEST_SYNC,
                                is_fp32_dest_acc_en,
                                BroadcastType::NONE,
                                unpack_to_dest>(DST_INDEX + tile_num, formats.math, formats.math, tensor_shape);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                {
                    _llk_math_wait_for_dest_available_<DC_DEST_SYNC>();
#if UNPACK_BLOCK && defined(ARCH_BLACKHOLE) && 0  // main: no block LLK (copy_block runs per tile)
                    if constexpr (unpack_to_dest)
                    {
                        _llk_math_eltwise_unary_datacopy_block_<DataCopyType::A2D, DC_DEST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            DST_INDEX, NUM_TILES_IN_BLOCK, formats.math, formats.math, num_faces);
                    }
                    else
#endif
                    {
                        for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                        {
                            LLK_ASSERT(
                                (DST_INDEX + tile_num < get_dest_max_tiles<DC_DEST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "tile_num exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_wrapper_<
                                DataCopyType::A2D,
                                DC_DEST_SYNC,
                                is_fp32_dest_acc_en,
                                BroadcastType::NONE,
                                unpack_to_dest>(DST_INDEX + tile_num, formats.math, formats.math, tensor_shape);
                        }
                    }
                    _llk_math_dest_section_done_<DC_DEST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h" // room-dc: MATH_FIDELITY (build.h) before the head pack headers
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"
// DATACOPY_SYNC_FULL (dc-prod census only, never committed): every DEST sync of the kernel is SyncFull
#ifndef DC_DEST_SYNC
#ifdef DATACOPY_SYNC_FULL
#define DC_DEST_SYNC DstSync::SyncFull
#else
#define DC_DEST_SYNC DstSync::SyncHalf
#endif
#endif

// DATACOPY_PACK_BLOCK (driver template): one block pack per DEST block under the block unpack (Blackhole only)
#ifndef DATACOPY_PACK_BLOCK
#define DATACOPY_PACK_BLOCK 0
#endif

// dc census 11-19: production's pack (compute API pack_block<n> / pack_tile through pack_api_shim.h over a CB of the result buffer)
#if defined(ARCH_BLACKHOLE) && !defined(RUNTIME_FORMATS) && defined(DC_PACK_NT)
#define DST_SYNC_MODE DC_DEST_SYNC
// the JIT gives the input CB (0) pack formats too: its own format, and the packer input of that format
#define PACK_API_SHIM_OUT2 0
#define PACK_API_SHIM_OUT2_FORMAT UNPACK_A_IN
#ifndef DC_CB0_PACK_SRC
#define DC_CB0_PACK_SRC PACK_IN
#endif
#define PACK_API_SHIM_OUT2_SRC DC_CB0_PACK_SRC
#include "pack_api_shim.h"
#define DC_PACK_SHIM 1
template <std::uint32_t nt>
__attribute__((always_inline)) inline void dc_pack_prod(const std::uint32_t dst, const std::uint32_t ntiles)
{
#if PACK_API_SHIM_PR
    if constexpr (DATACOPY_PACK_BLOCK && nt > 1)
    {
        pack_api_shim::pack_block<nt, is_fp32_dest_acc_en>(dst, PACK_API_SHIM_OUT);
    }
    else
    {
        const std::uint8_t output_id = get_output_id(PACK_API_SHIM_OUT);
        for (std::uint32_t i = 0; i < ntiles; ++i)
        {
            llk_pack_tile_of_output<is_fp32_dest_acc_en, PackMode::Default, true, false>(
                dst + i, get_output_tile_address<false, PackMode::Default>(output_id, 0), output_id);
        }
    }
#else
    pack_api_shim::main_pack_block<is_fp32_dest_acc_en>(dst, PACK_API_SHIM_OUT, ntiles);
#endif
}
#else
#define DC_PACK_SHIM 0
#endif

#if DATACOPY_PACK_BLOCK && defined(ARCH_BLACKHOLE)
template <typename Res>
inline void pack_dest_block(
    const std::uint32_t pack_dst,
    const std::uint32_t dst_index,
    const Res& buffer_Res,
    const std::uint32_t first,
    const std::uint32_t num_tiles,
    const std::uint32_t num_faces)
{
    // Block-float outputs and tiles of fewer than four faces close every tile
    if (IS_BFP_FORMAT(pack_dst) || num_faces != 4)
    {
        _llk_pack_block_closed_<DC_DEST_SYNC, is_fp32_dest_acc_en>(
            dst_index, L1_ADDRESS(buffer_Res[first]), num_tiles, (L1_ADDRESS(buffer_Res[first + 1]) - L1_ADDRESS(buffer_Res[first])) << 4, num_faces);
    }
    else
    {
        _llk_pack_block_<DC_DEST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(dst_index, L1_ADDRESS(buffer_Res[first]), num_tiles);
    }
}
#endif

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
    const int DST_INDEX                    = params.DST_INDEX;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_init_wrapper_<llk_test_pack_mode_v<false, tilize_en>, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DC_DEST_SYNC, is_fp32_dest_acc_en>();
#if DC_PACK_SHIM
#if PACK_API_SHIM_PR
        ckernel::packer::llk_pack_math_context(0); // copy_init: PACK_MATH_CONTEXT(false)
#endif
        {
            const std::uint32_t n    = static_cast<std::uint32_t>(NUM_BLOCKS) * NUM_TILES_IN_BLOCK;
            const std::uint32_t page = n > 1 ? L1_ADDRESS(buffer_Res[1]) - L1_ADDRESS(buffer_Res[0]) : params.TILE_SIZE_PACK;
            pack_api_shim::cb_bind(PACK_API_SHIM_OUT, L1_ADDRESS(buffer_Res[0]) + 1, page, n);
        }
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                {
#if DATACOPY_PACK_BLOCK && defined(ARCH_BLACKHOLE)
                    pack_dest_block(formats.pack_dst, DST_INDEX, buffer_Res, block_num * NUM_TILES_IN_BLOCK, NUM_TILES_IN_BLOCK, num_faces);
#else
                    for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                    {
                        LLK_ASSERT(
                            (DST_INDEX + tile_num < get_dest_max_tiles<DC_DEST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "tile_num exceeds max dest tiles");
                        _llk_pack_<DC_DEST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default, false, (DC_DEST_SYNC == DstSync::SyncFull || !(is_fp32_dest_acc_en && unpack_to_dest)), true>(
                            DST_INDEX + tile_num, L1_ADDRESS(buffer_Res[block_num * NUM_TILES_IN_BLOCK + tile_num]));
                    }
#endif
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                {
                    _llk_packer_wait_for_math_done_();
#if DC_PACK_SHIM
                    dc_pack_prod<DC_PACK_NT>(0, DC_PACK_NT); // production: pack_block<n>(0, cb) / pack_tile(i, cb), constant DEST index and count
#elif DATACOPY_PACK_BLOCK && defined(ARCH_BLACKHOLE)
                    pack_dest_block(formats.pack_dst, DST_INDEX, buffer_Res, block_num * NUM_TILES_IN_BLOCK, NUM_TILES_IN_BLOCK, num_faces);
#else
                    for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                    {
                        LLK_ASSERT(
                            (DST_INDEX + tile_num < get_dest_max_tiles<DC_DEST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "tile_num exceeds max dest tiles");
                        _llk_pack_<DC_DEST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default, false, (DC_DEST_SYNC == DstSync::SyncFull || !(is_fp32_dest_acc_en && unpack_to_dest)), true>(
                            DST_INDEX + tile_num, L1_ADDRESS(buffer_Res[block_num * NUM_TILES_IN_BLOCK + tile_num]));
                    }
#endif
                    _llk_pack_dest_section_done_<DC_DEST_SYNC, is_fp32_dest_acc_en>();
#if DC_PACK_SHIM
                    pack_api_shim::cb_push_back(PACK_API_SHIM_OUT, NUM_TILES_IN_BLOCK);
#endif
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif
