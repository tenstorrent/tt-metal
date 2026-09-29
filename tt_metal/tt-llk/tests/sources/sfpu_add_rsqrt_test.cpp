// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the add_rsqrt SFPU functor
// (hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h),
// promoted out of the deepseek_v3_b1 demo tree by tt-metal #52709.
//
// Computes, per DEST element:  result = rsqrt(x + addend)
//
// The addend is the RMSNorm epsilon in production (rsqrt(variance + eps)), which is
// why the fused form exists at all: the add happens inside the SFPU slot, so the
// variance never round-trips through DEST at bf16.
//
// This mirrors the call the compute API makes
// (api/compute/experimental/add_rsqrt.h -> add_rsqrt_tile<fast_and_approx>):
//
//     calculate_add_rsqrt<APPROX, ITERATIONS, DST_ACCUM_MODE, fast_and_approx>(addend)
//
// dispatched at VectorMode::RC with ITERATIONS=8, i.e. 8 SFPU slots covering all four
// faces of the tile. Both template axes the compute API exposes are swept here:
//
//   APPROX_MODE      picks the LUT-only SQRT_10-bits body vs the SQRT_23-bits
//                    Newton refinement (ckernel_sfpu_sqrt.h _calculate_sqrt_body_).
//   SFPU_FAST_APPROX drops the `v_if(x < 0) -> NaN` guard at the end of that body.
//                    That guard is the ONLY difference the flag makes here, so the
//                    python test drives a negative (x + addend) to tell the two apart;
//                    with a non-negative domain the flag is unobservable.
//
// init_add_rsqrt<APPROX> forwards to sqrt_init<APPROX>, which programs vConstIntPrgm0
// and vConstFloatPrgm1 (plus vConstFloatPrgm2 on the !APPROX path) — the seed constants
// the body reads. It is called through the standard LLK SFPU init so the invariant SFPU
// config + ADDR_MOD_7 are in place first.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
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

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_converter.h"

// ckernel_sfpu_add_rsqrt.h decodes the addend with Converter::as_float and pulls in
// ckernel_sfpu_rsqrt.h -> ckernel_sfpu_sqrt.h, which read bare APPROX / DST_ACCUM_MODE
// in the metal SFPU macro environment. Define them before the include, as the sampling
// driver does for its own metal-tree SFPU header.
#define DST_ACCUM_MODE is_fp32_dest_acc_en
constexpr bool APPROX = APPROX_MODE;
#include "experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h"
#undef DST_ACCUM_MODE

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    // Invariant SFPU config + ADDR_MOD_7, then the functor's own seed constants.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::init_add_rsqrt<APPROX_MODE>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, formats.math, formats.math);

    // Kept for init/uninit symmetry only -- it does NOT reset DEST addressing. The body of
    // _llk_math_eltwise_unary_datacopy_uninit_ is empty on Blackhole (and on Wormhole); what
    // actually rebases DEST to the tile-0 base is _llk_math_eltwise_unary_sfpu_params_ below,
    // via _llk_math_eltwise_sfpu_start_ -> math::set_dst_write_addr<Tile32x32, SrcRegs>(0).
    // The same paired call appears in sfpu_binop_scalar_test.cpp, sfpu_ternary_test.cpp and
    // sfpu_binary_test.cpp for the same reason.
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    // ITERATIONS=8 with VectorMode::RC is exactly what add_rsqrt_tile dispatches.
    _llk_math_eltwise_unary_sfpu_params_(
        [] { ckernel::sfpu::calculate_add_rsqrt<APPROX_MODE, 8 /* ITERATIONS */, is_fp32_dest_acc_en, SFPU_FAST_APPROX>(SFPU_UNARY_SCALAR); },
        0 /* dst_index */,
        VECTOR_MODE);

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
