
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

constexpr std::uint32_t RACE_PROBE_L1_ADDR = 0xA0000; // past the debug stimuli/result buffers at 0x70000

// Race probe: after each 32-bit unpack to Dest, reconfigure the unpacker (RISC stores in
// configure_unpack_AB, gated only by wait_for_idle on UNPACK_SYNC) for Float16_b, drain, and
// check the unpacker-0 ch1 Z-stride. If unpack_to_dest_tile_done's Z-stride WRCFG (queued behind
// the SEMGET and a STALLWAIT on the unpacker) landed after configure's store, the register holds
// the Float32 canonical stride instead of the Float16_b one. Also checks unpacker 0's input and
// output format words, which the Wormhole dummy unpack restores the same way.
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
    const std::uint32_t num_tiles          = NUM_BLOCKS * NUM_TILES_IN_BLOCK;

    constexpr std::uint32_t probe_fmt = to_underlying(DataFormat::Float16_b);
    const std::uint32_t expected_z    = FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(probe_fmt);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    std::uint32_t stale = 0;
    std::uint32_t last_seen = 0;
    std::uint32_t stale_fmt = 0;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
    {
        for (std::uint32_t i = 0; i < num_tiles; ++i)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);

            // Reconfigure right away, as compute_kernel_hw_startup would.
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(probe_fmt, probe_fmt, probe_fmt, probe_fmt, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
            tensix_sync(); // every queued Tensix config write has landed after this
            const std::uint32_t z = (cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] & UNP0_ADDR_CTRL_ZW_REG_1_Zstride_MASK) >> UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT;
            if (z != expected_z)
            {
                stale++;
                last_seen = z;
            }
            if ((cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & 0xf) != probe_fmt || (cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] & 0xf) != probe_fmt)
            {
                stale_fmt++;
            }

            // Restore the test's own configuration for the next unpack to Dest.
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats.unpack_A_src, formats.unpack_A_dst);
        }
    }
    // Leave the counts where the host can read them.
    *reinterpret_cast<volatile std::uint32_t*>(RACE_PROBE_L1_ADDR)      = stale;
    *reinterpret_cast<volatile std::uint32_t*>(RACE_PROBE_L1_ADDR + 4)  = last_seen;
    *reinterpret_cast<volatile std::uint32_t*>(RACE_PROBE_L1_ADDR + 8)  = LOOP_FACTOR * num_tiles;
    *reinterpret_cast<volatile std::uint32_t*>(RACE_PROBE_L1_ADDR + 12) = 0xC0FFEE00;
    *reinterpret_cast<volatile std::uint32_t*>(RACE_PROBE_L1_ADDR + 16) = stale_fmt;
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
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            if constexpr (unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                    {
                        for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                        {
                            _llk_math_eltwise_unary_datacopy_wrapper_<
                                DataCopyType::A2D,
                                DstSync::SyncHalf,
                                is_fp32_dest_acc_en,
                                BroadcastType::NONE,
                                unpack_to_dest>(DST_INDEX + tile_num, formats.math, formats.math, tensor_shape);
                        }
                    }
                }
            }
            else
            {
                _perf_math_loop_clear_valid</* src A */ true, /* src B */ true>(src_handshake_iters);
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
                                (DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "tile_num exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_wrapper_<
                                DataCopyType::A2D,
                                DstSync::SyncHalf,
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
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                    {
                        LLK_ASSERT(
                            (DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "tile_num exceeds max dest tiles");
                        _llk_math_eltwise_unary_datacopy_wrapper_<
                            DataCopyType::A2D,
                            DstSync::SyncHalf,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            unpack_to_dest>(DST_INDEX + tile_num, formats.math, formats.math, tensor_shape);
                    }
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                {
                    for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                    {
                        LLK_ASSERT(
                            (DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "tile_num exceeds max dest tiles");
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            DST_INDEX + tile_num, L1_ADDRESS(buffer_Res[block_num * NUM_TILES_IN_BLOCK + tile_num]));
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
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                    {
                        LLK_ASSERT(
                            (DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "tile_num exceeds max dest tiles");
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            DST_INDEX + tile_num, L1_ADDRESS(buffer_Res[block_num * NUM_TILES_IN_BLOCK + tile_num]));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif
