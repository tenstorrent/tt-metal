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
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            params.TEST_FACE_R_DIM,
            params.TEST_FACE_R_DIM,
            num_faces,
            num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(params.TEST_FACE_R_DIM, num_faces),
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
    const int DST_INDEX                    = params.DST_INDEX & 0xff;
#endif
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * NUM_BLOCKS * NUM_TILES_IN_BLOCK * num_faces;

    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(params.TEST_FACE_R_DIM, num_faces);

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
#include "pack_sweep_helpers.h"
#include "params.h"

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
    // pk-limit5: DST_INDEX bit 14 = main's per-tile pack in the state (no fast-untilize close, no dispatch, no byte-wise RMW)
    const std::uint32_t main_rt = (static_cast<std::uint32_t>(params.DST_INDEX) >> 14) & 1;
    // pk-limit5: DST_INDEX bit 15 = the W offset under cb0ac's rule in a CB of two blocks: W continues only right behind a W block of the output at the
    // next address, otherwise programs the base first; the CB wraps every two blocks.
    const std::uint32_t wrule_rt = (static_cast<std::uint32_t>(params.DST_INDEX) >> 15) & 1;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
            formats.pack_src, fmt_a, page_a, face_r, tile_c, num_faces, false, false, RELU_CONFIG);
        _llk_pack_init_with_src_wrapper_<llk_test_pack_mode_v<false, tilize_en>, false>(formats.pack_src, fmt_a, face_r, tile_c, num_faces);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        if constexpr (sw_rmw)
        {
            _llk_pack_rmw_dest_init_<true>();
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
                        _llk_pack_init_with_src_wrapper_<llk_test_pack_mode_v<false, tilize_en>, false>(formats.pack_src, fmt_a, face_r, tile_c, num_faces);
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
                    if (main_arm)
                    {
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
#endif
}
#endif
