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

// pack_block on output A alternating with pack_block on output B, the two outputs of different format and page size (the gathered bmm kernels pack the output
// and the partials this way). Every block is the first block of its output (another output was packed in between), so llk_pack_block never takes the W offset
// here: it takes the per-tile pack, the plain block or the closed block. alt_head_form 0: main's form, every tile packed one by one.
#if defined(ARCH_BLACKHOLE)
static inline std::uint32_t alt_page_bytes(const std::uint32_t fmt)
{
    return fmt == static_cast<std::uint32_t>(DataFormat::Float32) ? 4096 : (fmt == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ? 1088 : 2048);
}

// llk_pack_block writes the destination by bytes under full sync (llk_pack_rmw_gate)
constexpr bool alt_rmw = dest_sync == DstSync::SyncFull;

template <bool fp32>
inline void alt_pack_tile(const std::uint32_t fmt, const std::uint32_t tile, const std::uint32_t addr)
{
    if (!fp32 && fmt == static_cast<std::uint32_t>(DataFormat::Float32))
    {
        _llk_pack_<dest_sync, fp32, ckernel::PackMode::Default, false, false>(tile, addr);
    }
    else
    {
        _llk_pack_<dest_sync, fp32, ckernel::PackMode::Default, false, alt_rmw>(tile, addr);
    }
}

// The form llk_pack_block takes for the first block of an output after a packer reconfigure: the rows of llk_pack_reconf_form (tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h,
// mirrored in hardware_free/kernels/test_pack_block_form.py) for the three formats of the pairs. Columns: half sync 16 bit DEST, half 32 bit, full 16 bit, full 32 bit.
constexpr char alt_reconf_form[2][3][4][17] = {
    {
        {"TTBTBTBTBBBBBBBB", "TTBTBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"}, // Bfp8_b
        {"TBBBBBBTBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"}, // Float32
        {"TBBBBTBTBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"}, // Float16_b
    },
    {
        {"TTBTBTBTBBBBBBBB", "TTTTBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"}, // Bfp8_b
        {"TBBBBBBTBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"}, // Float32
        {"TBBBBTBTBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"}, // Float16_b
    },
};

inline std::uint32_t alt_form(const std::uint32_t fmt, const std::uint32_t ntiles, const std::uint32_t head, const bool src32)
{
    const bool bfp = fmt == static_cast<std::uint32_t>(DataFormat::Bfp8_b);
    const bool wide = fmt == static_cast<std::uint32_t>(DataFormat::Float32);
    const std::uint32_t column = (dest_sync == DstSync::SyncFull ? 2 : 0) + (is_fp32_dest_acc_en ? 1 : 0);
    if (!head || alt_reconf_form[src32 ? 1 : 0][bfp ? 0 : (wide ? 1 : 2)][column][(ntiles < 16 ? ntiles : 16) - 1] == 'T')
    {
        return 0; // per tile
    }
    return bfp ? 2 : 1; // closed : plain
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
    const int RELU_CONFIG                  = params.RELU_CONFIG;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
#if defined(ARCH_BLACKHOLE)
    const std::uint32_t fmt_a   = formats.pack_dst;
    const std::uint32_t fmt_b   = static_cast<std::uint32_t>(alt_pack_dst);
    const std::uint32_t page_a  = alt_page_bytes(fmt_a);
    const std::uint32_t page_b  = alt_page_bytes(fmt_b);
    const std::uint32_t tiles_a = ((static_cast<std::uint32_t>(NUM_BLOCKS) + 1) / 2) * NUM_TILES_IN_BLOCK;
    const std::uint32_t base_a  = L1_ADDRESS(buffer_Res[0]);
    // output B starts at the next 1 KB after the tiles of A (the python side reads it from the same offset)
    const std::uint32_t base_b = base_a + ((tiles_a * page_a + 1023) & ~1023u) / 16;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
            formats.pack_src, fmt_a, page_a, FACE_R_DIM, TILE_C_DIM, num_faces, false, false, RELU_CONFIG);
        _llk_pack_init_wrapper_<llk_test_pack_mode_v<false, tilize_en>, false>(fmt_a, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        if constexpr (alt_rmw)
        {
            _llk_pack_rmw_dest_init_<true>();
        }
        PROFILER_SYNC();
    }
    // the form of each output is decided once, as a kernel with constant tile counts has it folded
    // read through a volatile so the main arm compiles to the same loop as the head arm and the two differ in data only
    // a 32 bit output CB next to a 32 bit DEST is the family llk_pack_block takes the forms no sweep read slower for
    const bool alt_src32 = is_fp32_dest_acc_en && (fmt_a == static_cast<std::uint32_t>(DataFormat::Float32) || fmt_b == static_cast<std::uint32_t>(DataFormat::Float32));
    const volatile std::uint32_t head_v = alt_head_form ? 1 : 0;
    const std::uint32_t form_a          = alt_form(fmt_a, NUM_TILES_IN_BLOCK, 1, alt_src32) & (0u - head_v);
    const std::uint32_t form_b          = alt_form(fmt_b, NUM_TILES_IN_BLOCK, 1, alt_src32) & (0u - head_v);
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                std::uint32_t cur = 0;
                for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(NUM_BLOCKS); ++block)
                {
                    if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    const std::uint32_t out  = block & 1;
                    const std::uint32_t fmt  = out ? fmt_b : fmt_a;
                    const std::uint32_t page = out ? page_b : page_a;
                    if (out != cur)
                    {
                        _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(formats.pack_src, fmt, page);
                        cur = out;
                    }
                    if constexpr (alt_l1_acc)
                    {
                        _llk_pack_reconfig_l1_acc_(out);
                    }
                    const std::uint32_t first = (block >> 1) * NUM_TILES_IN_BLOCK;
                    const std::uint32_t base  = out ? base_b : base_a;
                    const std::uint32_t form  = out ? form_b : form_a;
                    const std::uint32_t addr0 = base + first * (page / 16);
                    if (form == 0)
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            alt_pack_tile<is_fp32_dest_acc_en>(fmt, DST_INDEX + tile, addr0 + tile * (page / 16));
                        }
                    }
                    else if (form == 1)
                    {
                        _llk_pack_block_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default, alt_rmw>(DST_INDEX, addr0, NUM_TILES_IN_BLOCK);
                    }
                    else
                    {
                        _llk_pack_block_closed_<dest_sync, is_fp32_dest_acc_en, alt_rmw>(DST_INDEX, addr0, NUM_TILES_IN_BLOCK, page);
                    }
                    if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
                    {
                        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
                if constexpr (alt_l1_acc)
                {
                    _llk_pack_reconfig_l1_acc_(0);
                }
            }
        }
        PROFILER_SYNC();
    }
#endif
}
#endif
