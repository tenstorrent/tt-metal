// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Packer data-format reconfig across an 8-bit integer output with a 16-bit Dest.
//
// Run 0 (CONFIGURE_TEST_RUN_IDX == 0): the pack thread configures the packer directly for the
// output format. Run 1: it configures the "previous" format first (the format a preceding op
// would have left behind) and then reconfigures to the output format with
// _llk_pack_reconfig_data_format_. Both runs pack the same Dest contents and must produce the
// same bytes: with Dest in 16-bit mode the packer's Dest-read control decides how a datum is
// read (8-bit integer raw read for Int8/UInt8, rounding read for the float formats), the full
// configure programs it from the format, and the reconfig has to program the same field set.
//
// Two legs, selected by the output format:
//   UInt8 output:     previous format Float16_b. Per tile the Float16_b input is A2D-copied into
//                     Dest and an in-place SFPU pass rounds every datum to an integer in 0..65535
//                     (SFP_STOCH_RND FP32_TO_UINT16) and stores its low 10 magnitude bits back as
//                     an "integer 8" Dest datum (SFPSTORE INT8), which an int8 pack reads. The pass
//                     uses the load / store address modes the SFPU typecast init programs.
//   Float16_b output: previous format UInt8. Plain datacopy; the stimuli carry bf16 denormals and
//                     NaN, which the full-configure read treats differently from a raw read.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

inline bool is_int8_output(const std::uint32_t pack_dst_format)
{
    return ckernel::masked_data_format(pack_dst_format) == ckernel::to_underlying(DataFormat::Int8);
}

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
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

using namespace ckernel;

// Address modes of the SFPU pass: the typecast init programs a no-increment slot for the load and a
// Dest += 2 slot for the store (ADDR_MOD_7 and ADDR_MOD_6). Wormhole SFPLOAD / SFPSTORE carry a 2-bit
// address-mode field that _llk_math_eltwise_sfpu_start_ rebases onto slots 4..7, so the kernel names
// ADDR_MOD_3 / ADDR_MOD_2 there; Blackhole's field is 3 bits wide and names the slots directly.
#ifdef ARCH_BLACKHOLE
constexpr std::uint8_t LOAD_ADDR_MOD  = ADDR_MOD_7;
constexpr std::uint8_t STORE_ADDR_MOD = ADDR_MOD_6;
#else
constexpr std::uint8_t LOAD_ADDR_MOD  = ADDR_MOD_3;
constexpr std::uint8_t STORE_ADDR_MOD = ADDR_MOD_2;
#endif

// Round the 16-bit Dest datums of the current face to integers in 0..65535 and store their low
// 10 magnitude bits back as "integer 8" Dest datums, 8 load / store pairs per face.
inline void round_face_to_int8_datums_in_place()
{
#pragma GCC unroll 8
    for (int d = 0; d < 8; ++d)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, LOAD_ADDR_MOD, 0);
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
        TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT8, STORE_ADDR_MOD, 0);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const bool int8_out = is_int8_output(formats.pack_dst);

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    if (int8_out)
    {
        // SFPU config plus the typecast load (no increment) and store (Dest += 2) address modes.
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::typecast>();
    }

    LLK_ASSERT(
        (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "NUM_TILES_IN_BLOCK exceeds max dest tiles");

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();
        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                block_tile, formats.math, formats.math);

            if (int8_out)
            {
                _llk_math_eltwise_unary_sfpu_params_(round_face_to_int8_datums_in_place, block_tile, VectorMode::RC);
            }
        }
        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr std::uint32_t tile_size = FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES;

    // Format the full configure programs before the reconfig under test: the other side of the
    // int8 / float boundary from the output.
    const std::uint32_t prev_format =
        is_int8_output(formats.pack_dst) ? ckernel::to_underlying(DataFormat::Float16_b) : ckernel::to_underlying(DataFormat::UInt8);

    if (params.CONFIGURE_TEST_RUN_IDX == 0)
    {
        // Reference: full configure straight onto the output format.
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, tile_size);
    }
    else
    {
        // Under test: full configure for the previous format, then the reconfig onto the output.
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(prev_format, prev_format, tile_size);
        _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, tile_size);
    }
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    LLK_ASSERT(
        (params.NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "NUM_TILES_IN_BLOCK exceeds max dest tiles");

    for (int block_start = 0; block_start < params.NUM_BLOCKS; block_start++)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t block_tile = 0; block_tile < params.NUM_TILES_IN_BLOCK; ++block_tile)
        {
            _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                block_tile, L1_ADDRESS(params.buffer_Res[block_start * params.NUM_TILES_IN_BLOCK + block_tile]));
        }
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
