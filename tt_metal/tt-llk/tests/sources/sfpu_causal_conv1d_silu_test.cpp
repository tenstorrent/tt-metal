// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// This test hand-rolls its own unpack/math/pack sequence (like ttnn_where_test.cpp) instead of
// going through the generic SfpuType-templated ternary dispatch, since the new op takes 8
// input tiles + 2 computed outputs -- more operands than that shared 3-in/1-out dispatch
// supports. See ckernel_sfpu_causal_conv1d_silu.h for the op itself.
//
// Layout (all full 32x32 tiles):
//   buffer_A: [wa, wb, wc, wd]   -- 4 per-channel conv weights (tiles 0-3)
//   buffer_B: [x, y, z, w]      -- 3 cache entries + 1 matmul-produced sample (tiles 0-3)
//   buffer_Res: [new_cache, x, y, silu_out] -- updated 3-wide cache (new_cache,x,y) + SiLU output
//
// DST tile indices 0-7 hold the 8 inputs (wa,wb,wc,wd,x,y,z,w). The two computed outputs reuse
// wa's slot (0) and wb's slot (1) -- both fully consumed into registers by
// ckernel_sfpu_causal_conv1d_silu.h before its first store -- so the whole op stays inside
// DstSync::SyncHalf's proven 8-tile budget, matching the exact `dst_index_out == dst_index_in0`
// reuse `calculate_addcmul()`/`calculate_where()` already rely on in production.
//
// NOTE: an earlier revision of this test failed against golden on ttsim (x/y passthrough, which
// this op never touches, always matched; only the two computed outputs were wrong). The root
// cause was that _calculate_causal_conv1d_silu_() was invoked directly, once, covering only
// ITERATIONS=8 rows -- a single 16x16 face's worth, not the full 32x32 tile. Every other SFPU op
// in this codebase runs through _llk_math_eltwise_sfpu_apply_vector_mode_ (VectorMode::RC), which
// calls the op 4 times -- once per face -- advancing the dest face address between calls; this
// test now does that explicitly below.

static constexpr std::uint8_t resolve_format(std::uint8_t in)
{
    switch (static_cast<DataFormat>(in))
    {
        case DataFormat::Float32:
        case DataFormat::Float16_b:
            return in;
        default:
            return static_cast<std::uint8_t>(DataFormat::Float16_b);
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t UNPACK_FMT = resolve_format(UNPACK_A_IN);

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, UNPACK_FMT, UNPACK_FMT);

    // weights a,b,c,d
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[0]), UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[1]), UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[2]), UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[3]), UNPACK_FMT, UNPACK_FMT);

    // cache x,y,z + matmul-produced sample w
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_B[0]), UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_B[1]), UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_B[2]), UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_B[3]), UNPACK_FMT, UNPACK_FMT);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "ckernel_sfpu_causal_conv1d_silu.h"
#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

static constexpr ckernel::DstSync DST_SYNC_MODE = ckernel::DstSync::SyncHalf;

#include "llk_math_eltwise_unary_sfpu.h"

void run_kernel(RUNTIME_PARAMETERS)
{
    const std::uint8_t MATH_FMT  = resolve_format(UNPACK_A_IN);
    constexpr bool is_int_fpu_en = false;

    _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FMT, MATH_FMT);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
        4 /* num_faces */, MATH_FMT);
    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();

    // Copy the 8 unpacked input tiles (SrcA) into DST tiles 0-7, in the same order they were
    // unpacked: 0=wa 1=wb 2=wc 3=wd 4=x 5=y 6=z 7=w.
    for (std::uint32_t dst_index = 0; dst_index < 8; ++dst_index)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            dst_index, MATH_FMT, MATH_FMT);
    }
    // _llk_math_eltwise_unary_datacopy_uninit_() is a no-op (see llk_math_eltwise_unary_datacopy.h);
    // the real "start a fresh sfpi dst_reg addressing pass at tile-0 base" reset -- with the
    // required STALLWAIT for the preceding datacopies to drain -- is _llk_math_eltwise_sfpu_start_,
    // the same primitive the generic ternary/unary SFPU dispatch (llk_math_eltwise_*_sfpu_params.h)
    // uses ahead of any sfpi-based compute.
    _llk_math_eltwise_sfpu_start_(0);

    // A 32x32 tile is 4 faces of 16x16; each call below advances only ITERATIONS=8 rows within
    // the current face's addressing window, so it must run once per face (matching how
    // _llk_math_eltwise_sfpu_apply_vector_mode_(..., VectorMode::RC) drives every other SFPU op
    // in this codebase), advancing the dest face address between calls.
    constexpr int kIterations = 8;
#pragma GCC unroll 4
    for (std::uint32_t face = 0; face < TILE_NUM_FACES; ++face)
    {
        ckernel::sfpu::_calculate_causal_conv1d_silu_<false /*APPROXIMATION_MODE*/, is_fp32_dest_acc_en, kIterations>(
            0 /*wa*/,
            1 /*wb*/,
            2 /*wc*/,
            3 /*wd*/,
            4 /*x*/,
            5 /*y*/,
            6 /*z*/,
            7 /*w*/,
            0 /*cache_out, reuses dead wa slot*/,
            1 /*silu_out, reuses dead wb slot*/);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();

    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t PACK_FMT = resolve_format(UNPACK_A_IN);

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(PACK_FMT, PACK_FMT, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(PACK_FMT);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    // Res[0]=new_cache (dst 0, == wa's slot), Res[1]=x unchanged (dst 4), Res[2]=y unchanged
    // (dst 5), Res[3]=silu_out (dst 1, == wb's slot) -- matches the issue's outB=[new_cache,x,y],
    // outA=SiLU(new_cache).
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(4, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(5, L1_ADDRESS(params.buffer_Res[2]));
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1, L1_ADDRESS(params.buffer_Res[3]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
