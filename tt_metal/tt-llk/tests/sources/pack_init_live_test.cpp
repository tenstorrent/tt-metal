// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Default pack inits through the compute API's path (pack_api_shim.h) between packs of two outputs of different format
// and width (Blackhole): the address modes, strides, X counter and MOP hw_configure left live, pack_strides_width and
// pack_mop_dirty.

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

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

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
    const int num_total_tiles               = NUM_TILES_IN_BLOCK * NUM_BLOCKS;
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * num_total_tiles * num_faces;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_stream_set_valid</* src A */ true, /* src B */ is_fp32_dest_acc_en, PERF_RUN_TYPE>(src_handshake_iters);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int tile = 0; tile < num_total_tiles; ++tile)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
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
#include "params.h"

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
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * NUM_BLOCKS * NUM_TILES_IN_BLOCK * num_faces;

    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
            tensor_shape, formats.math);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
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
                    for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(NUM_BLOCKS); ++block)
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                DST_INDEX + tile, formats.math, formats.math, tensor_shape);
                        }
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
            if constexpr (!unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(NUM_BLOCKS); ++block)
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            LLK_ASSERT(
                                ((DST_INDEX + tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "Block tile index exceeds maximum destination tiles");
                            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                DST_INDEX + tile, formats.math, formats.math, tensor_shape);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(NUM_BLOCKS); ++block)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        LLK_ASSERT(
                            ((DST_INDEX + tile) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            DST_INDEX + tile, formats.math, formats.math, tensor_shape);
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
#include "llk_pack_common.h"
#include "params.h"
// Outputs A (CB 16, the test's output format) and B (CB 17, alt_pack_dst from alt_pack_src) as the JIT's arrays describe them.
#define PACK_API_SHIM_OUT2        17
#define PACK_API_SHIM_OUT2_FORMAT ckernel::to_underlying(alt_pack_dst)
#define PACK_API_SHIM_OUT2_SRC    ckernel::to_underlying(alt_pack_src)
#include "pack_api_shim.h"

// The compute API's default pack inits between packs of two outputs of different widths, after hw_configure kept the
// default program live: init_mode 'n' none, 'l' the one-tile init, 'c' a counted init of one tile, 'z' a zero-output
// init then the one-tile init, 'm' a two-tile init then the one-tile init (the last two leave pack_mop_dirty set), 's' a
// counted init that skips the address modes and strides.
#if defined(ARCH_BLACKHOLE) && PACK_API_SHIM_PR
static inline std::uint32_t alt_page_bytes(const std::uint32_t fmt)
{
    return fmt == static_cast<std::uint32_t>(DataFormat::Float32) ? 4096 : (fmt == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ? 1088 : 2048);
}

inline void live_init(const std::uint32_t cb)
{
    if constexpr (init_mode == 'l')
    {
        pack_api_shim::pack_init(cb);
    }
    else if constexpr (init_mode == 'c')
    {
        pack_api_shim::pack_init(cb, 1);
    }
    else if constexpr (init_mode == 'z')
    {
        pack_api_shim::pack_init<PackMode::Default, true>(cb);
        pack_api_shim::pack_init(cb);
    }
    else if constexpr (init_mode == 'm')
    {
        pack_api_shim::pack_init(cb, 2);
        pack_api_shim::pack_init(cb);
    }
    else if constexpr (init_mode == 's')
    {
        // SDPA streaming's MOP-only refresh: the strides the reconfigure wrote stay.
        pack_api_shim::pack_init<PackMode::Default, false, true, true>(cb, 1);
    }
}
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
#if defined(ARCH_BLACKHOLE) && PACK_API_SHIM_PR
    constexpr std::uint32_t cb_a = PACK_API_SHIM_OUT, cb_b = PACK_API_SHIM_OUT2;
    const std::uint32_t fmt_a   = formats.pack_dst;
    const std::uint32_t fmt_b   = static_cast<std::uint32_t>(alt_pack_dst);
    const std::uint32_t src_b   = static_cast<std::uint32_t>(alt_pack_src);
    const std::uint32_t page_a  = alt_page_bytes(fmt_a);
    const std::uint32_t page_b  = alt_page_bytes(fmt_b);
    const std::uint32_t tiles_a = ((static_cast<std::uint32_t>(NUM_BLOCKS) + 1) / 2) * NUM_TILES_IN_BLOCK;
    const std::uint32_t base_a  = L1_ADDRESS(buffer_Res[0]);
    // output B starts at the next 1 KB after the tiles of A (the python side reads it from the same offset)
    const std::uint32_t base_b = base_a + ((tiles_a * page_a + 1023) & ~1023u) / 16;
    {
        START_PERF_MEASURE("INIT")
        pack_api_shim::pack_hw_configure<is_fp32_dest_acc_en>(cb_a, page_a / 16);
        pack_api_shim::pack_init(cb_a);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            std::uint32_t cur = 0;
            for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(NUM_BLOCKS); ++block)
            {
                _llk_packer_wait_for_math_done_();
                const std::uint32_t out = block & 1;
                const std::uint32_t fmt = out ? fmt_b : fmt_a;
                if (out != cur)
                {
                    _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(out ? src_b : static_cast<std::uint32_t>(formats.pack_src), fmt, out ? page_b : page_a);
                    live_init(out ? cb_b : cb_a);
                    cur = out;
                }
                const std::uint32_t page  = out ? page_b : page_a;
                const std::uint32_t first = (block >> 1) * NUM_TILES_IN_BLOCK;
                const std::uint32_t addr0 = (out ? base_b : base_a) + first * (page / 16);
                for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default, false, llk_pack_rmw_per_tile<is_fp32_dest_acc_en, PackMode::Default>>(
                        DST_INDEX + tile, addr0 + tile * (page / 16));
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
#endif
}
#endif
