// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MXFP4 stored as relabelled bfloat4_b, restored by the SFPU carry decode.
//
// Unpack A (Bfp4_b) -> datacopy to Dest -> SFPU on Pack: bits += 3 * (bits & 0x00200000) on the FP32
// LReg image -> pack. A value with the mantissa's 0.25 bit set (1.25 * 2^e, 1.75 * 2^e) drops that bit
// and gains one exponent step (2^(e+1), 1.5 * 2^(e+1)), which maps the stored codes 5 and 7 (in units of
// half the MXFP4 scale) back to the e2m1 values 8 and 12.
// The decoders are the ones Blaze runs, in tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mxfp4_decode.h;
// MXFP4_DECODER picks one: 0 the sfpi loop, 1 the 3-issue SFPLOADMACRO decoder, 2 the 2-load one, 3 the 2-load
// one replayed (what the K3 MoE decode runs).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;

// Math only stages the tile into Dest; the decode runs on Pack, as in the Blaze kernel.
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        ckernel::DEFAULT_TENSOR_SHAPE, formats.math);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0 /* dst_index */, formats.math, formats.math);

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_pack_common.h"
#include "sfpi.h"

#include "ckernel_sfpu_mxfp4_decode.h"

#ifndef MXFP4_DECODER
#define MXFP4_DECODER 0
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    ckernel::sfpu::mxfp4_sfpu_thread_init<MXFP4_DECODER>();
    ckernel::sfpu::mxfp4_sfpu_configure<MXFP4_DECODER>();

    // Timing: wall-clock cycles around each tile's decode, both ends after the thread has drained
    // (tensix_sync), and the same bracket around nothing to calibrate the bracket itself.
    std::uint64_t decode_cycles = 0;
    std::uint64_t empty_cycles  = 0;
    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();

        ckernel::tensix_sync();
        const std::uint64_t e0 = ckernel::read_wall_clock();
        ckernel::tensix_sync();
        const std::uint64_t e1 = ckernel::read_wall_clock();
        ckernel::tensix_sync();
        const std::uint64_t t0 = ckernel::read_wall_clock();
        // The per-tile dispatch Blaze calls: the sfpu start / done around the decode are in the timed span.
        ckernel::sfpu::mxfp4_sfpu_tile<MXFP4_DECODER>(0 /* dst_index */);
        ckernel::tensix_sync();
        const std::uint64_t t1 = ckernel::read_wall_clock();
        decode_cycles += t1 - t0;
        empty_cycles += e1 - e0;

        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
    // The tile after the last result tile holds the timing words (read back as raw BF16 pairs).
    ckernel::tensix_sync();
    volatile std::uint32_t* timing = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[params.TILE_CNT]);
    timing[0]                      = static_cast<std::uint32_t>(decode_cycles);
    timing[1]                      = static_cast<std::uint32_t>(empty_cycles);
    timing[2]                      = params.TILE_CNT;
    timing[3]                      = 0x4D584650; // "MXFP" marker
}

#endif
