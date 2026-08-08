// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// LLK SFPU rand test kernel.
//
// Mirrors the production rand compute path (ttnn uniform / randn / bernoulli):
//   init_sfpu -> rand_tile_init(seed) -> rand_tile(0, from, scale) -> pack_tile.
// rand is the SFPU PRNG generator in
//   tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h
// which fills the whole Dest tile with uniform floats in [from, from + scale).
//
// The kernel copies a zeroed input tile into Dest (so any Dest row the generator
// fails to write stays 0 and is caught by the range check), then overwrites the
// whole tile with rand. The store width must track the Dest accumulation width;
// this test runs both fp32_dest_acc modes so a store that writes the wrong Dest
// view (leaving half the tile unwritten) is caught on-card.
//
// Compile-time configuration emitted by the Python harness:
//   UNPACK_A_IN / MATH_FORMAT / PACK_OUT : DataFormat of the tile
//   APPROX_MODE          : SFPU approximation mode
//   is_fp32_dest_acc_en  : 32-bit vs 16-bit Dest accumulation
//   unpack_to_dest       : whether the input is unpacked straight into Dest
//

#include <cstdint>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// rand parameters as raw fp32 bit patterns. `from` is kept well above zero and
// the range narrow so that any unwritten (zero) Dest row, or a value collapsed
// out of range, is unambiguously detected by the Python range check.
constexpr std::uint32_t RAND_FROM  = 0x44800000u;  // 1024.0f
constexpr std::uint32_t RAND_SCALE = 0x43800000u;  //  256.0f  -> range [1024, 1280)
constexpr std::uint32_t RAND_SEED  = 0x00001234u;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        UNPACK_A_IN, UNPACK_A_IN, UNPACK_A_IN, UNPACK_A_IN, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, FACE_R_DIM, 4 /* num_faces */, UNPACK_A_IN, UNPACK_A_IN);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[0]), UNPACK_A_IN, UNPACK_A_IN);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

static constexpr bool DST_ACCUM_MODE = is_fp32_dest_acc_en;

#include "llk_sfpu/ckernel_sfpu_rand.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

void run_kernel(RUNTIME_PARAMETERS)
{
    const bool is_int_fpu_en = false;

    // Seed the zeroed input tile into Dest tile 0, then overwrite it with rand.
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FORMAT, MATH_FORMAT);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
        4 /* num_faces */, MATH_FORMAT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, MATH_FORMAT, MATH_FORMAT);
    // Reset dest addressing so the SFPU op starts at the tile-0 base.
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    // rand_tile_init(seed): configure the SFPU addr mode and seed the PRNG.
    SFPU_UNARY_INIT_FN_ARGS(unused, sfpu::rand_init, (APPROX_MODE), RAND_SEED);

    // rand_tile(0, from, scale): fill the whole tile. VectorMode::RC drives 4
    // faces (8 rows each), matching the production rand_tile API.
    SFPU_UNARY_CALL(
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        rand,
        (APPROX_MODE, is_fp32_dest_acc_en),
        0 /* dst_index */,
        VectorMode::RC,
        RAND_FROM,
        RAND_SCALE);

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(PACK_OUT, PACK_OUT, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(PACK_OUT);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
