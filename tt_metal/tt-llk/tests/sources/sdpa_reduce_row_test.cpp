// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ADVANCE TEST: covers demo-fork experimental LLK sdpa_reduce_row (tt-metal#47554 / tt-blaze#1971), pending promotion.
// Include path (shadow -I) repoint on promotion. Primitive differs from tt-blaze only in FPU<->SFPU signalling cadence
// (orthogonal to this numerical golden).
//
// What the op does (compute_sdpa_chunk in the demo tree, models/demos/deepseek_v3_b1/.../compute_kernel_api/sdpa.h):
// a row-wise MAX reduce, run on the SFPU, over `block_width` consecutive 8x32 tiles held in DEST. Each 8x32 tile is a
// full 16x16 face's worth of lanes (8 rows x 32 cols). For each of the 8 rows the op maxes across all block_width*32
// columns and writes the row-max into column 0 of the destination tile (a SFPLOAD max tree + a SFPSHFT2 8->4->2->1
// within-row epilogue). This advance test exercises the MAX (reduce-max) instantiation on a single 8x32 tile
// (block_width == 1), which is the softmax running-max step of flash attention.
//
// PRIMITIVE symbols under test (NOT the forked llk_math_sdpa_reduce_row.h wrapper / compute_kernel_api sdpa.h entry,
// both of which are demo-forked and rename the primitives). Targeted directly via the shadow -I:
//   ckernel::sfpu::_init_sdpa_reduce_row_8x32_<format>()                  -- SFPU config reg + addr_mods
//   ckernel::sfpu::_init_sdpa_reduce_max_row_8x32_replay_buffers_()       -- record the MAX reduce replay buffer
//   ckernel::sfpu::_calculate_sdpa_reduce_max_row_8x32_<format, block_width, skip_signalling>(src, dst, prev_max)
//
// FPU<->SFPU handshake fake: the primitive posts/waits on semaphore::FPU_SFPU whenever skip_signalling == false, which
// would deadlock this isolated single-thread kernel (there is no MATH producer). We instantiate with
// skip_signalling == true: the ENTIRE signalling block — and therefore the entire DEMO-vs-tt-blaze delta (the
// signal_granularity template param + the in-loop t6_semaphore_get/wait_on_zero) — lives inside
// `if constexpr (!skip_signalling)`, so pinning it true both avoids the deadlock and makes the compiled instruction
// stream byte-identical between the DEMO and tt-blaze headers. The numerical golden below therefore covers the shared
// (non-signalling) compute path.
//
// The op writes results via SFPU LREG stores to DEST addressed by TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset...), so we
// drive it from the thread that owns DEST. Mirroring the analog sfpu_reduce_sdpa_test.cpp, the SFPU reduce runs on the
// PACK thread with its own MATH/SFPU init, fed a Float16_b 8x32 (single-face) input tile seeded into DEST by an A2D
// datacopy on the MATH thread.
//
// Blackhole-only. The golden is verified on Blackhole silicon (p100a), not compile-green only.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Single 8x32 "tiny" tile = one 16x16 face's worth of lanes (num_faces == 1 for the SFPU reduce span). The row source
// lives in DEST[SRC_INDEX]; the per-row max is written back to DEST[DST_INDEX]. Reuse-in-place: SRC and DST are the
// same tile. block_width == 1 => a single tile is reduced (no cross-tile accumulation), the minimal instantiation.
static constexpr std::uint32_t SRC_INDEX   = 0;
static constexpr std::uint32_t DST_INDEX   = 0;
static constexpr std::uint32_t BLOCK_WIDTH = 1;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Configure unpacker for Float16_b and stream the input tile(s) into SrcA (the MATH thread A2D-copies them into
    // DEST, which the SFPU reduce then reads).
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
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
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(4 /* num_faces */, formats.math);
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, formats.math, formats.math);
        }

        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

// ckernel_sfpu_load_config.h must precede the primitive header: _init_sdpa_reduce_row_8x32_ is a template that calls
// _init_sfpu_config_reg() with no dependent arguments, so -Wtemplate-body wants the declaration visible at the point
// the template is parsed. Sorts ahead of the primitive anyway, so clang-format keeps this valid.
#include "ckernel_sfpu_load_config.h" // _init_sfpu_config_reg
// PRIMITIVE symbol under test (NOT the forked llk_math_sdpa_reduce_row.h wrapper / compute_kernel_api sdpa.h entry).
// On promotion, repoint the -I in test_config.py so this resolves to the canonical header and this line is unchanged.
#include "ckernel_sfpu_sdpa_reduce_row.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h" // _llk_math_eltwise_unary_sfpu_init_ / _llk_math_eltwise_sfpu_start_ / _done_
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Configure packer hardware
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);

    // Initialize destination for packing
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();

        // SFPU row-max reduce over the DEST tile. init programs the SFPU config reg + the ZERO/TILE_OFFSET addr_mods;
        // the replay-buffer record captures the SFPLOAD max tree; calculate runs the reduce and stores the per-row max
        // into column 0 of DEST[DST_INDEX]. skip_signalling == true elides the FPU<->SFPU semaphore handshake (see the
        // file banner) so the isolated PACK-thread kernel does not deadlock and the compute path is byte-identical to
        // tt-blaze.
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::reduce>();
        ckernel::sfpu::_init_sdpa_reduce_row_8x32_<DataFormat::Float16_b>();
        ckernel::sfpu::_init_sdpa_reduce_max_row_8x32_replay_buffers_();
        _llk_math_eltwise_sfpu_start_(0);
        ckernel::sfpu::_calculate_sdpa_reduce_max_row_8x32_<DataFormat::Float16_b, BLOCK_WIDTH, true /* skip_signalling */>(
            SRC_INDEX, DST_INDEX, false /* prev_max */);
        _llk_math_eltwise_sfpu_done_();

        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            const std::uint32_t result_tile = block * params.NUM_TILES_IN_BLOCK + tile;
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[result_tile]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
