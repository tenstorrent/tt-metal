// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the unary scalar-divisor fmod / remainder SFPU kernels
// (hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_fmod.h and
// ckernel_sfpu_remainder.h), with the divisor chosen by the test instead of the fixed 2.0 the
// generic unary suite uses. A power-of-two divisor has an exact reciprocal and an exact
// quotient, so it never exercises the quotient-rounding and residual-correction path; this
// driver exists so non-power-of-two divisors (3, 7, 0.003, ...) are covered on hardware.
//
// Mirrors the compute-API call (api/compute/eltwise_unary/fmod.h, remainder.h):
//
//     fmod_tile_init(divisor_bits, reciprocal_bits);   // vConstFloatPrgm0/1 (+ Prgm2)
//     fmod_tile(idst);                                 // calculate_fmod<APPROX, 8>() at VectorMode::RC
//
// SFPU_UNARY_SCALAR carries the divisor as raw fp32 bits and SFPU_UNARY_THRESHOLD the host's
// fl(1/divisor), exactly as ttnn passes them (unary_op_utils.cpp: std::bit_cast(1.0f / param0)).
// SFPU_UNARY_OPERATION (SfpuType::fmod / SfpuType::remainder) selects the kernel.

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

// The metal SFPU headers are written for the metal macro environment; provide the two names
// ckernel_sfpu_recip.h (pulled in by both kernels) may read, as the other metal-tree drivers do.
#define DST_ACCUM_MODE is_fp32_dest_acc_en
constexpr bool APPROX = APPROX_MODE;
#include "llk_sfpu/ckernel_sfpu_fmod.h"
#include "llk_sfpu/ckernel_sfpu_remainder.h"
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

    // Invariant SFPU config + ADDR_MOD_7, then the kernel's own constants (divisor, reciprocal,
    // rounding magic) exactly as fmod_tile_init / remainder_tile_init program them.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    if constexpr (SFPU_UNARY_OPERATION == SfpuType::fmod)
    {
        ckernel::sfpu::init_fmod<APPROX_MODE>(SFPU_UNARY_SCALAR, SFPU_UNARY_THRESHOLD);
    }
    else
    {
        ckernel::sfpu::init_remainder<APPROX_MODE>(SFPU_UNARY_SCALAR, SFPU_UNARY_THRESHOLD);
    }

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, formats.math, formats.math);

    // Kept for init/uninit symmetry only; DEST is rebased by _llk_math_eltwise_unary_sfpu_params_
    // below (see sfpu_add_rsqrt_test.cpp for the full note).
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    // ITERATIONS=8 with VectorMode::RC is exactly what fmod_tile / remainder_tile dispatch.
    _llk_math_eltwise_unary_sfpu_params_(
        []
        {
            if constexpr (SFPU_UNARY_OPERATION == SfpuType::fmod)
            {
                ckernel::sfpu::calculate_fmod<APPROX_MODE, 8 /* ITERATIONS */>();
            }
            else
            {
                ckernel::sfpu::calculate_remainder<APPROX_MODE, 8 /* ITERATIONS */>();
            }
        },
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
