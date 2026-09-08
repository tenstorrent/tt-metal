// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Deterministic regression for tt-llk#1120: relu_min reading an unloaded LREG2.
//
// Wormhole's _relu_min_impl_ takes its threshold from LREG2 as an *implicit input* --
// the body is raw TTI and copies LREG2 into LREG1 before the SFPSWAP, so the register
// is part of the calling contract but appears nowhere in the signature. The float
// branch of the _relu_min_ wrapper used to assign a local sfpi vector instead of
// loading LREG2, which compiled clean and then ran relu_min against whatever the
// previously executed SFPU kernel happened to leave in that register.
//
// The sweeps cannot pin this down on their own: they only fail when some *other* op
// runs first and dirties LREG2, so the failure is a property of test ordering rather
// than of the kernel. This driver removes the ordering: it writes a known poison value
// into LREG2 as the instruction immediately preceding the _relu_min_ call, inside the
// same SFPU block, so nothing can come between them.
//
//   fixed kernel:  _relu_min_ loads LREG2 itself   -> result = max(x, threshold)
//   broken kernel: the poison survives             -> result = max(x, poison)
//
// The python side drives a poison far outside the golden's range, so the two outcomes
// are separated by much more than any tolerance. Wormhole only -- Blackhole's
// _relu_min_ is a plain sfpi predicated form that genuinely consumes its parameter,
// so there is no register to poison and the test would pass vacuously.
//
// SFPU_UNARY_SCALAR carries the poison as raw fp32 bits; SFPU_UNARY_THRESHOLD carries
// the threshold the same way, so the C++ and the torch golden agree bit for bit.

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
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpu/ckernel_sfpu_relu.h"

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

    // relu_min needs only the invariant SFPU config + ADDR_MOD_7 and a dest RWC reset,
    // which is what the production relu_min_tile_init reduces to -- so the bare init is
    // the faithful one here, exactly as sfpu_operations.h routes relu_min.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, formats.math, formats.math);

    // Kept for init/uninit symmetry only -- what actually rebases DEST to the tile-0 base
    // is _llk_math_eltwise_unary_sfpu_params_ below. Same paired call as
    // sfpu_add_rsqrt_test.cpp and sfpu_binop_scalar_test.cpp, for the same reason.
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    // Both halves live in one SFPU block on purpose: the poison must be the last thing
    // written to LREG2 before _relu_min_ runs, with no init, no sync and no other SFPU
    // kernel in between. ITERATIONS=8 at VectorMode::RC is what relu_min_tile dispatches.
    _llk_math_eltwise_unary_sfpu_params_(
        []
        {
            ckernel::sfpu::_sfpu_load_imm32_(p_sfpu::LREG2, SFPU_UNARY_SCALAR);
            ckernel::sfpu::_relu_min_<sfpi::vFloat, APPROX_MODE, 8 /* ITERATIONS */, float>(__builtin_bit_cast(float, SFPU_UNARY_THRESHOLD));
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
