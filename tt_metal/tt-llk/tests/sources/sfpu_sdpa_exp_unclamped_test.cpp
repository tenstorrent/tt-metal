// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the upper-unclamped exp SFPU entry
// (tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h).
//
// The header only exposes leaf helpers that map a vFloat to a vFloat; there is
// no dst_reg loop and no _init_ entry, so this test supplies the loop itself
// (`calculate_sdpa_exp_unclamped`, one SFPU slot per iteration, 8 iterations per
// face) and drives it through the standard VectorMode::RC wrapper.
//
// What is under test: `_ckernel_sfpu_exp_accurate_upper_unclamped_` is a copy of
// the accurate exp path with the *upper* input clamp removed. The clamped variant
// saturates xlog2 = val/ln2 + 127 at its upper bound, which is dead code for the
// SDPA use case where val <= 0 always. So:
//   * for val <= 0 the two variants must agree, and both must match exp(val);
//   * for val > 0 (past the clamp point) only the unclamped variant keeps
//     tracking exp(val).
// The python side sweeps both domains.
//
// NOTE: the LLK header has an inverted dependency -- it does `#include
// "ckernel_sfpu_exp.h"`, and there is no such file in tt-llk: the exp kernels
// live one layer up, in the metal llk_api tree
// (hw/ckernels/blackhole/metal/llk_api/llk_sfpu/). That unqualified spelling only
// resolves because the tt-llk test build now also puts llk_api/llk_sfpu on the
// include path (see setup_compilation_options in helpers/test_config.py). The
// explicit include below just makes the dependency visible at the use site.

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
#include "llk_math_eltwise_unary_sfpu_params.h"

// The experimental header reaches for the metal-tree exp helpers; pull them in
// under their on-path spelling first (see file header).
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h"

using namespace ckernel;

namespace
{
// One SFPU slot (4 dest rows x 8 dest columns) per iteration, 8 iterations to
// cover a full 16x16 face. VectorMode::RC repeats this over the four faces.
constexpr int SDPA_EXP_ITERATIONS = 8;

template <bool scale_en>
inline void calculate_sdpa_exp_unclamped(const std::uint32_t scale_bits)
{
    for (int d = 0; d < SDPA_EXP_ITERATIONS; ++d)
    {
        const sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0]       = ckernel::sfpu::_ckernel_sfpu_exp_accurate_upper_unclamped_<scale_en, is_fp32_dest_acc_en>(val, scale_bits);
        sfpi::dst_reg++;
    }
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

    // The upper-unclamped exp shares the accurate exp's SFPU setup.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::exp_init<false /* APPROXIMATION_MODE */, 0x3F800000 /* base scale = 1.0f */, true /* fast_and_approx off */, is_fp32_dest_acc_en>();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0 /* dst_tile_index */, formats.math, formats.math);

        _llk_math_eltwise_unary_sfpu_params_(calculate_sdpa_exp_unclamped<SFPU_SCALE_EN>, 0 /* dst_index */, VectorMode::RC, SFPU_UNARY_SCALAR);

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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* dst_index */, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
