// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test: the rank-stamped TopK merge must not change the SFPU's shared -1.0 register
// (LREG11, sfpi::vConstNeg1). It programs a lo16 clear mask into a programmable constant register
// and never restores it, so the mask must not land on LREG11.
//
// Probe = x - 1 computed against LCONST_neg1. Run the probe on a fresh input tile before the merge
// (reference) and again after it; the two must agree bit for bit, whatever LREG11 held at start.
// Control: rank_stamped=False keeps the merge but compiles the mask write out. Needs 32-bit DEST.

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

    for (int i = 0; i < 2; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"

// Set before the TopK LLK API header, as topk_test.cpp does.
#define DST_SYNC_MODE  DST_SYNC
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "llk_sfpu/ckernel_sfpu_topk.h"
#undef DST_SYNC_MODE
#undef DST_ACCUM_MODE

using namespace ckernel;

constexpr bool APPROX              = false;
constexpr bool NETWORK_STABLE_SORT = TOPK_STABLE_SORT && !TOPK_FUSED_STABLE;
static_assert(!(TOPK_RANK_STAMPED && TOPK_STABLE_SORT), "rank-stamped and comparator stable modes are mutually exclusive");
static_assert(!(TOPK_RANK_STAMPED && TOPK_FUSED_STABLE), "rank-stamped and fused-key modes are mutually exclusive");
static_assert(!TOPK_RANK_STAMPED || is_fp32_dest_acc_en, "rank-stamped stable topk requires 32-bit DEST (dest_acc)");
static_assert(!TOPK_FUSED_STABLE, "the poison step is the rank-stamped merge; fused-key modes are out of scope");

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    // Programs ADDR_MOD_7, which the merge and the probe both use. Also reloads the
    // constant file, so it must stay ahead of the poison.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    // Raw TTI, not `x - 1.0f`: the compiler may otherwise materialize the -1.0 elsewhere and
    // never read LCONST_neg1. Same idiom as _floor_body_.
    constexpr auto probe = []
    {
        constexpr int PROBE_ITERATIONS = 8;
        for (int d = 0; d < PROBE_ITERATIONS; d++)
        {
            sfpi::vFloat x                  = sfpi::dst_reg[0];
            sfpi::l_reg[sfpi::LRegs::LReg0] = x;
            TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LCONST_neg1, p_sfpu::LREG0, 0);
            sfpi::vFloat x_minus_1 = sfpi::l_reg[sfpi::LRegs::LReg0];
            sfpi::dst_reg[0]       = x_minus_1;
            sfpi::dst_reg++;
        }
    };

    // Reference: the probe before the merge.
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_params_(probe, 0 /* dst_index */, VectorMode::RC);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

    // Poison: one merge, called only for its effect on the constant registers; its DEST output is
    // discarded. Safe without a preceding phases-steps call: the merge records and replays nothing.
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_sfpu_params_(
        []
        {
            if constexpr (TOPK_RANK_STAMPED)
            {
                ckernel::sfpu::_init_topk_rank_stamped_<TOPK_TAG_BITS>();
            }
            else
            {
                ckernel::sfpu::_init_topk();
            }

            ckernel::sfpu::calculate_bitonic_topk_merge<
                APPROX,
                is_fp32_dest_acc_en,
                TOPK_SORT_DIRECTION,
                NETWORK_STABLE_SORT,
                TOPK_FUSED_STABLE,
                TOPK_RANK_STAMPED,
                ckernel::sfpu::TopkTieOrder::Unset,
                TOPK_TAG_BITS>(0 /* m_iter */, TOPK_K);
        },
        0 /* dst_index */,
        VectorMode::None);

    // The probe again, on a fresh copy of the input.
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_params_(probe, 0 /* dst_index */, VectorMode::RC);
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

    for (int i = 0; i < 2; ++i)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[i]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
