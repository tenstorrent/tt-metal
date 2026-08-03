// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the sampling SFPU helpers
// (hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sampling.h).
//
// Despite the ckernel_sfpu_* name and its content (raw sfpi vector code), that
// header sits in the metal llk_api tree rather than in tt-llk, and it has no
// llk_api wrapper and no compute-API caller. It is an SFPU LLK by content, so it
// is tested here alongside the tt-llk SFPU kernels.
//
// Entry points, and the DEST region each one touches (all inside face 0):
//
//   recip_scalar<legacy_compat>()          one SFPU slot  -> rows 0-3
//   clamp_max_scalar(max)                  one SFPU slot  -> rows 0-3
//   mul_unary_scalar_first_column(k)       4 slots, +4     -> rows 0-15
//   binary_comp_first_column<le|lt|ge>()   4 slots, +4     -> rows 0-15
//   {add,sub,mul}_binary_first_column()    4 slots, +4     -> rows 0-15
//
// An SFPU slot is 4 DEST rows x 8 of the face's 16 columns, so each of these
// covers half the columns of the rows it walks -- the callers only care about
// column 0 ("first column"), hence the names. The python golden pins down which
// half, and also asserts a column-order-independent invariant so a mapping
// mismatch is distinguishable from a wrong result.
//
// Buffer layout (all three input tiles come from buffer_A so no second operand
// buffer is needed):
//   tile 0 -> DEST tile 0 : in0
//   tile 1 -> DEST tile 1 : in1
//   tile 2 -> DEST tile 2 : zeros (deterministic background for the binary ops,
//                                  which only write a sub-region of their output)
// All three DEST tiles are packed back out, so the golden can also assert that
// the untouched tiles are unchanged.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// DEST tile indices.
static constexpr std::uint32_t SAMPLING_IN0_TILE  = 0;
static constexpr std::uint32_t SAMPLING_IN1_TILE  = 1;
static constexpr std::uint32_t SAMPLING_OUT_TILE  = 2;
static constexpr std::uint32_t SAMPLING_NUM_TILES = 3;

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

    for (std::uint32_t tile = 0; tile < SAMPLING_NUM_TILES; ++tile)
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
#include "llk_math_eltwise_unary_sfpu_params.h"

// ckernel_sfpu_sampling.h reads bare APPROX / DST_ACCUM_MODE (it is written
// against the metal SFPU macro environment), so define them before including it.
#define DST_ACCUM_MODE is_fp32_dest_acc_en
constexpr bool APPROX = false;
#include "experimental/llk_sfpu/ckernel_sfpu_sampling.h"
#undef DST_ACCUM_MODE

using namespace ckernel;

namespace
{
inline void run_sampling_op()
{
#if defined(SAMPLING_OP_RECIP_SCALAR)
    ckernel::sfpu::calculate_sampling_recip_scalar<SAMPLING_LEGACY_COMPAT>();
#elif defined(SAMPLING_OP_CLAMP_MAX_SCALAR)
    ckernel::sfpu::calculate_sampling_clamp_max_scalar(SFPU_UNARY_SCALAR);
#elif defined(SAMPLING_OP_MUL_UNARY_SCALAR)
    ckernel::sfpu::calculate_sampling_mul_unary_scalar_first_column(SFPU_UNARY_SCALAR);
#elif defined(SAMPLING_OP_LE)
    ckernel::sfpu::calculate_sampling_binary_comp_first_column<SfpuType::le>(SAMPLING_IN0_TILE, SAMPLING_IN1_TILE, SAMPLING_OUT_TILE);
#elif defined(SAMPLING_OP_LT)
    ckernel::sfpu::calculate_sampling_binary_comp_first_column<SfpuType::lt>(SAMPLING_IN0_TILE, SAMPLING_IN1_TILE, SAMPLING_OUT_TILE);
#elif defined(SAMPLING_OP_GE)
    ckernel::sfpu::calculate_sampling_binary_comp_first_column<SfpuType::ge>(SAMPLING_IN0_TILE, SAMPLING_IN1_TILE, SAMPLING_OUT_TILE);
#elif defined(SAMPLING_OP_ADD)
    ckernel::sfpu::calculate_sampling_add_binary_first_column(SAMPLING_IN0_TILE, SAMPLING_IN1_TILE, SAMPLING_OUT_TILE);
#elif defined(SAMPLING_OP_SUB)
    ckernel::sfpu::calculate_sampling_sub_binary_first_column(SAMPLING_IN0_TILE, SAMPLING_IN1_TILE, SAMPLING_OUT_TILE);
#elif defined(SAMPLING_OP_MUL)
    ckernel::sfpu::calculate_sampling_mul_binary_first_column(SAMPLING_IN0_TILE, SAMPLING_IN1_TILE, SAMPLING_OUT_TILE);
#else
#error "No SAMPLING_OP_* selected -- pass helpers.test_variant_parameters.SAMPLING_OP"
#endif
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    // The non-legacy reciprocal path reads vConstFloatPrgm0, so program it; the
    // legacy_compat path carries its own constants and needs no setup. Everything
    // else needs only the invariant SFPU config + ADDR_MOD_7 from the LLK init.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::sfpu_reciprocal_init<false /* APPROXIMATE */>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    for (std::uint32_t tile = 0; tile < SAMPLING_NUM_TILES; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    // The sampling helpers do their own DEST tile addressing off dst_reg, so the
    // window is opened at tile 0 and the vector mode is a single pass.
    _llk_math_eltwise_sfpu_start_(0 /* dst_index */);
    run_sampling_op();
    _llk_math_eltwise_sfpu_done_();

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
    for (std::uint32_t tile = 0; tile < SAMPLING_NUM_TILES; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
