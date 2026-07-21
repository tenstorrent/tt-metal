// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Experimental generalized_moe_gate end-to-end LLK test (Blackhole + Wormhole B0).
//
// Mirrors the single-block (<=256 experts) non-sigmoid compute-kernel flow of the
// ttnn generalized_moe_gate op (unified_kernels/generalized_moe_gate.hpp, num_blocks==1),
// expanded from the Compute API (api/compute/experimental/generalized_moe_gate.h) down to
// the raw _llk_* / SFPU-functor layer so it runs inside the tt-llk harness.
//
// Three fused phases in one kernel:
//   1. Eltwise value+bias (FPU ELWADD, COPY mode) -> bias-corrected scores in DEST.
//   2. In-place single-face (16x16) dest transposes (step0/step1[/step1_hi]/step2).
//   3. Single-face bitonic top-2 / top-4-group / top-8-expert selection (SFPU), then normalize.
//
// GMG_UNGROUPED_TOP8 (compile-time, injected by the Python driver): 1 = ungrouped global
// top-k, 0 = grouped DeepSeek. Both are covered.
//
// Input is a single 32x32 tile (face 0 = the 256 experts): logits (bf16, buffer_A),
// bias (bf16 transposed, buffer_B), and expert indices (uint16, arange transposed, buffer_C).
// Output DEST tile 0 = normalized top-8 scores, DEST tile 1 = their expert indices.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// 32x32 bf16 tile with 4 faces; the gate operates on face 0.
static constexpr std::uint32_t NUM_FACES = 4;

// uint16 device DataFormat underlying value (indices are copied bit-exact); derived from the
// named enum rather than hardcoded, matching the other index-carrying test sources (topk_test.cpp).
static constexpr std::uint32_t UINT16_FORMAT = ckernel::to_underlying(DataFormat::UInt16);

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    // compute_kernel_hw_startup
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, NUM_FACES, NUM_FACES);

    // copy_tile(input_indices_cb, 0, 1): unpack the uint16 index tile into SrcA (bit-exact).
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(
        0 /* transpose_of_faces */, 0 /* within_face */, tensor_shape, UINT16_FORMAT, UINT16_FORMAT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>(L1_ADDRESS(params.buffer_C[0]), UINT16_FORMAT, UINT16_FORMAT);

    // generalized_moe_gate_init (non-sigmoid): unpack A (logits) and B (bias) with Transpose::Both,
    // matching the Compute API (llk_unpack_AB_init<NONE>(icb0, icb1, Transpose::Both)). This is why the
    // bias tile is uploaded transposed (the unpacker's within-face transpose undoes it).
    _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, ckernel::Transpose::Both);

    // generalized_moe_gate (non-sigmoid): copy-add -> stream logits (SrcA) and bias (SrcB).
    _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]));

    // Set srcb dummy valid for the transpose-dest / SFPU phases.
    _llk_unpack_set_srcb_dummy_valid_();
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_generalized_moe_gate_eltwise_binary.h"
#include "experimental/llk_math_generalized_moe_gate_transpose_dest_single_face.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "params.h"
#include "sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"

static constexpr bool APPROX   = false;
static constexpr bool IS_32BIT = false; // gate is bf16-only (topk/transpose static_assert !is_32bit)

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t eps   = params.GMG_EPS;
    const std::uint32_t scale = params.GMG_SCALE;

    // compute_kernel_hw_startup
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // The harness bypasses compute_kernel_hw_startup's one-time SFPU config, so run the
    // idempotent once-init before any SFPU op.
    _llk_math_eltwise_unary_sfpu_init_once_();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // copy_tile(input_indices_cb, 0, 1): datacopy the uint16 index tile to DEST tile 1.
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(NUM_FACES, UINT16_FORMAT);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE>(1 /* dst_index */, UINT16_FORMAT, UINT16_FORMAT);

    // ---- generalized_moe_gate_init (non-sigmoid) ----
    _llk_math_generalized_moe_gate_eltwise_binary_init_<EltwiseBinaryType::ELWADD, GeneralizedMoeGateEltwiseBinaryMode::COPY, MATH_FIDELITY>(
        NUM_FACES, 0 /* acc_to_dest */);
    _llk_math_generalized_moe_gate_transpose_dest_single_face_common_init_<IS_32BIT>();
    // topk init (SFPU): the metal wrapper resets the Dst RWC counter then calls the (no-op)
    // lib init; replicate both here since the harness has no wrapper layer.
    math::reset_counters(p_setrwc::SET_ABD_F);
    ckernel::sfpu::_init_generalized_moe_gate_topk<APPROX, is_fp32_dest_acc_en>();

    // ---- generalized_moe_gate (non-sigmoid) ----
    // Copy-add (FPU): bias-corrected scores.
    _llk_math_generalized_moe_gate_eltwise_binary_<EltwiseBinaryType::ELWADD, DST_SYNC, is_fp32_dest_acc_en, MATH_FIDELITY>(NUM_FACES, 0 /* dst_index */);

    // Sum top2 (SFPU).
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_generalized_moe_gate_sum_top2<APPROX, is_fp32_dest_acc_en>, 0 /* dst_index */, VectorMode::RC_custom);

    // Transpose dest step 0 (FPU) — puts each group g at DEST row g.
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init_<IS_32BIT>();
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, IS_32BIT>();

#if GMG_UNGROUPED_TOP8
    // TRUE GLOBAL TOP-8 over all 256 experts (ungrouped).
    // Save groups 4-7 (rows 4-7) -> rows 8-11.
    _llk_math_generalized_moe_gate_copy4rows_init_<4, 8, IS_32BIT, 16>();
    _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, IS_32BIT>();
    // topA = top8(groups 0-3): step1_hi<d2b_dst=0> -> run rows 0-7 -> merge -> topA at {0,2}.
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_<0, 0, IS_32BIT>();
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_<is_fp32_dest_acc_en, IS_32BIT>();
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_gmg_merge4_top8<is_fp32_dest_acc_en, 0, 0, 2>, 0 /* dst_index */, VectorMode::RC_custom);
    // Park topA (rows 0-3) -> rows 12-15; restore groups 4-7 (rows 8-11) -> rows 4-7.
    _llk_math_generalized_moe_gate_copy4rows_init_<0, 12, IS_32BIT, 20>();
    _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, IS_32BIT>();
    _llk_math_generalized_moe_gate_copy4rows_init_<8, 4, IS_32BIT, 24>();
    _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, IS_32BIT>();
    // topB = top8(groups 4-7): step1_hi<d2b_dst=4> -> run rows 0-7 -> merge -> topB at {4,6}.
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_<4, 0, IS_32BIT>();
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_<is_fp32_dest_acc_en, IS_32BIT>();
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_gmg_merge4_top8<is_fp32_dest_acc_en, 0, 4, 6>, 0 /* dst_index */, VectorMode::RC_custom);
    // Restore topA (rows 12-15) -> rows 0-3; now topA@{0,2}, topB@{4,6}.
    _llk_math_generalized_moe_gate_copy4rows_init_<12, 0, IS_32BIT, 28>();
    _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, IS_32BIT>();
    // Single <=256 block: full bitonic sort of topA{0,2}+topB{4,6} -> global top-8, keep top-topk + normalize.
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_generalized_moe_gate_finalize_ungrouped<APPROX, is_fp32_dest_acc_en, GMG_TOPK, false>,
        0 /* dst_index */,
        VectorMode::RC_custom,
        eps,
        scale);
#else
    // Grouped DeepSeek gate: sort_top4 selects top-4 groups, step1 lays them out, top8 merges.
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_generalized_moe_gate_sort_top4_groups<APPROX, is_fp32_dest_acc_en>, 0 /* dst_index */, VectorMode::RC_custom);
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init_<IS_32BIT>();
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, IS_32BIT>();
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_generalized_moe_gate_top8<APPROX, is_fp32_dest_acc_en>, 0 /* dst_index */, VectorMode::RC_custom, eps, scale);
#endif // GMG_UNGROUPED_TOP8

    // Transpose dest step 2 (FPU) — final output layout.
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_<IS_32BIT>();
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, IS_32BIT>();

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    const bool narrow_tile        = tensor_shape.num_faces_c_dim == 1;

    // compute_kernel_hw_startup
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);

    _llk_packer_wait_for_math_done_();
    // DEST tile 0 = normalized scores, DEST tile 1 = expert indices.
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* dst_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1 /* dst_index */, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
