// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reconfig-escape / empty-uninit test for the experimental LLKs (Blackhole + Wormhole B0).
//
// Several experimental LLKs have an empty or no-op `uninit` even though their `init`
// clobbers persistent HW state (ADDR_MODs, cfg regs such as THCON_SEC0_REG2_Haloize_mode).
// This test PINS that no such state leaks between an experimental op and a canonical op,
// in both directions, by running a two-op sequence and validating the SECOND op:
//
//   run 0: packed to a scratch buffer and discarded.
//   run 1: packed to buffer_Res[0] and validated against its golden.
//
// SEQUENCE selects which two ops run, and therefore which invariant is pinned:
//   0 = FORWARD (X -> datacopy): experimental op X then a canonical A2D datacopy of a fresh,
//                 known tile (buffer_A[1]). Pins that X's empty uninit leaves no escape the
//                 datacopy's own init doesn't re-establish (ADDR_MODs, Haloize, and under
//                 fp32 DEST also ALU_ACC_CTRL_Zero_Flag_disabled_src / ALU_FORMAT_SPEC).
//   1 = REVERSE (datacopy -> X): a canonical datacopy first, then experimental op X, validating
//                 X. Pins that X's own init is self-sufficient and does not silently rely on
//                 leftover clean state from a preceding canonical op.
//   2 = REPEAT  (X -> X): experimental op X twice, validating the second. Pins that X's empty
//                 uninit is safe for back-to-back reuse (the fused-kernel usage pattern).
//
// REVERSE / REPEAT validate the experimental op itself, so they only run for the polluters with
// a clean single-tile golden — matmul_custom_no_mop (0) and the SDPA sub+bcast-col pair (2).
// reduce_block_max_row (1) is FORWARD-only (its validated-op behavior is already covered by the
// fuser reduce test); this is enforced in the Python parametrization.
//
// POLLUTER selects which experimental op X is:
//   0 = matmul_custom_no_mop     (MATH: at THROTTLE_LEVEL 0 clobbers ADDR_MODs — BH full-tile path
//                                 writes 0/1/2/4/5, WH writes 0..5; ADDR_MOD_6 is programmed only at
//                                 nonzero throttling — and loads a replay image; empty uninit)
//   1 = reduce_block_max_row     (MATH: clobbers ADDR_MOD_1/2/3/6 + programs a MOP template; empty uninit.
//                                 UNPACK: RMWs Haloize_mode, and under fp32 DEST also
//                                 ALU_ACC_CTRL_Zero_Flag_disabled_src + ALU_FORMAT_SPEC SrcA, none restored
//                                 by its uninit — armed only when the sweep runs dest_acc=Yes.)
//   2 = sdpa_sub_bcast_col       (the paired SDPA fused sub+bcast-col op, exercising BOTH empty uninits:
//                                 UNPACK unpack_AB_sub_bcast_col_custom (RMW THCON_SEC0_REG2_Haloize_mode
//                                 + SETADCXX x_end) and MATH eltwise_binary_custom (ADDR_MOD_7 +
//                                 CLR_DVALID_SrcA disable). These two ops only run correctly as a pair.)

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// POLLUTER (which experimental op) and SEQUENCE (op ordering) are #defined by the generated
// build.h, pulled in via params.h inside each per-thread block below.
#define SEQ_FORWARD 0
#define SEQ_REVERSE 1
#define SEQ_REPEAT  2

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// 32x32 tile, 4 faces of 16x16 — the geometry every op uses.
static constexpr std::uint32_t NUM_FACES = 4;

// Scratch L1 address for the discarded run-0 packed output.
static constexpr std::uint32_t buffer_polluter_scratch = 0xA0000;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_reduce_custom.h"
#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "params.h"

// Unpack side of the experimental (POLLUTER-selected) op. Reads buffer_A[0]/buffer_B[0].
static inline void unpack_experimental_(RUNTIME_PARAMETERS params, [[maybe_unused]] const ckernel::TensorShape& tensor_shape)
{
#if POLLUTER == 2
    // sdpa_sub_bcast_col: RMW Haloize_mode=0 (transpose within face) + set both unpacker x_end.
    // Loads srcB once + ct_dim srcA tiles, paired with the MATH bcast-col reuse scaffold.
    _llk_unpack_AB_sub_bcast_col_init_custom_(tensor_shape);
    _llk_unpack_AB_sub_bcast_col_custom_(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]), 1 /* ct_dim */);
    _llk_unpack_AB_sub_bcast_col_uninit_custom_();
#elif POLLUTER == 1
    // reduce_block_max_row: SrcA + SrcB (scaler) streamed via the paired custom AB-reduce unpack.
    // Its init RMWs Haloize_mode, and under fp32 DEST also ALU_ACC_CTRL_Zero_Flag_disabled_src +
    // ALU_FORMAT_SPEC SrcA (for MOVB2D hi16/lo16); its uninit restores none of them. The victim
    // datacopy's own format reconfig + init must reset them — pinned only in the dest_acc=Yes sweep.
    _llk_unpack_AB_reduce_block_max_row_init_<1 /* block_ct_dim */, is_fp32_dest_acc_en>(tensor_shape);
    _llk_unpack_AB_reduce_block_max_row_(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]));
    _llk_unpack_AB_reduce_block_max_row_uninit_();
#else
    // matmul_custom_no_mop (0): in0 -> SrcB, in1 -> SrcA via the regular matmul unpacker
    // (the no-mop MATH replay consumes SrcA/SrcB in the matmul dvalid pattern).
    const std::uint32_t mm_tile_size = FACE_R_DIM * FACE_C_DIM * NUM_FACES / (is_fp32_dest_acc_en ? 1 : 2);
    _llk_unpack_AB_matmul_init_<>(0 /* transpose */, 1 /* ct_dim */, 1 /* rt_dim */, 1 /* kt_dim */);
    _llk_unpack_AB_matmul_<>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]), 0, 0, mm_tile_size, mm_tile_size);
#endif
}

// Unpack side of a canonical A2D datacopy of the fresh, known tile buffer_A[1].
// Runs ONLY its own standard init; if a prior op left the unpacker in a stale state
// (e.g. a leaked Haloize transpose), this datacopy would read buffer_A[1] wrong.
static inline void unpack_datacopy_(RUNTIME_PARAMETERS params, const FormatConfig& formats)
{
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
        L1_ADDRESS(params.buffer_A[1]), formats.unpack_A_src, formats.unpack_A_dst);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    // compute_kernel_hw_startup
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, NUM_FACES, NUM_FACES);

    // ---- Run 0 (discarded to scratch) ----
#if SEQUENCE == SEQ_REVERSE
    unpack_datacopy_(params, formats); // canonical op first
#else
    unpack_experimental_(params, tensor_shape); // FORWARD / REPEAT: experimental first
#endif

    // Wait until the run-0 packer has drained before touching unpacker state for run 1.
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // ---- Run 1 (validated, packed to buffer_Res[0]) ----
#if SEQUENCE == SEQ_FORWARD
    unpack_datacopy_(params, formats); // canonical victim datacopy
#else
    unpack_experimental_(params, tensor_shape); // REVERSE / REPEAT: experimental op is the victim
#endif
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_binary_custom.h"
#include "experimental/llk_math_matmul_custom_no_mop.h"
#include "experimental/llk_math_reduce_custom.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

// Math side of the experimental (POLLUTER-selected) op.
static inline void math_experimental_(RUNTIME_PARAMETERS params, [[maybe_unused]] const ckernel::TensorShape& tensor_shape)
{
#if POLLUTER == 0
    // matmul_custom_no_mop: this THROTTLE_LEVEL 0 full-tile init clobbers ADDR_MODs (BH writes
    // 0/1/2/4/5 — no ADDR_MOD_3 in the full-tile custom path; WH writes 0..5) and records the
    // replay image; empty uninit. ADDR_MOD_6 is only programmed at nonzero throttling.
    _llk_math_matmul_init_no_mop_<MATH_FIDELITY, 0>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, 0, 1, 1);
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_matmul_no_mop_<MATH_FIDELITY, 0>(0 /* dst_index */, 1, 1);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_matmul_uninit_no_mop_();
#elif POLLUTER == 1
    // reduce_block_max_row: init sets ADDR_MOD_1/2/3/6 + a MOP template; empty uninit.
    _llk_math_reduce_block_max_row_init_<1 /* block_ct_dim */, is_fp32_dest_acc_en>(tensor_shape);
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_reduce_block_max_row_<1 /* block_ct_dim */, is_fp32_dest_acc_en>(0 /* dst_index */, tensor_shape);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_reduce_block_max_row_uninit_<is_fp32_dest_acc_en>();
#elif POLLUTER == 2
    // eltwise_binary_custom paired with the sub_bcast_col unpacker (the SDPA fused SUB op).
    // init sets ADDR_MOD_7 + CLR_DVALID_SrcA disable; empty uninit. The SUB bcast-col reuse
    // scaffold is the only eltwise_binary_custom entry point common to BH and WH.
    _llk_math_eltwise_binary_init_custom_<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(NUM_FACES);
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_sub_bcast_cols_reuse_custom_(1 /* ct_dim */, tensor_shape, 0 /* dst_index */);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_eltwise_binary_uninit_custom_();
#endif
}

// Math side of a canonical A2D datacopy. Its own init reprograms the ADDR_MODs it consumes
// (ADDR_MOD_0/2/3) and (via the format reconfig) the ALU format/acc-ctrl state it depends on.
static inline void math_datacopy_(RUNTIME_PARAMETERS params, const FormatConfig& formats)
{
    _llk_math_reconfig_data_format_<is_fp32_dest_acc_en, false>(formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */>(NUM_FACES, formats.math);
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, false /* unpack_to_dest */>(
        0 /* dst_index */, formats.math, formats.math);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {FACE_R_DIM, FACE_C_DIM, 2 /* num_faces_r_dim */, 2 /* num_faces_c_dim */};

    // compute_kernel_hw_startup
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // ---- Run 0 (discarded) ----
#if SEQUENCE == SEQ_REVERSE
    math_datacopy_(params, formats); // canonical op first
#else
    math_experimental_(params, tensor_shape); // FORWARD / REPEAT: experimental first
#endif

    // ---- Run 1 (validated) ----
#if SEQUENCE == SEQ_FORWARD
    math_datacopy_(params, formats); // canonical victim datacopy
#else
    math_experimental_(params, tensor_shape); // REVERSE / REPEAT: experimental op is the victim
#endif
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

#if POLLUTER == 1
    // reduce_block_max_row (FORWARD-only, so it is always run 0) emits a masked (REDUCE_ROW)
    // scalar; mask so the discarded polluter output packs without asserting, then clear it
    // before the victim pack.
    _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_ROW>();
#endif

    // ---- Run 0: pack the discarded result to scratch ----
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* dst_index */, L1_ADDRESS(buffer_polluter_scratch));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();

#if POLLUTER == 1
    _llk_pack_reduce_mask_clear_();
#endif

    // Signal the unpacker that run-0 has fully drained.
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // ---- Run 1: pack the validated result to the output buffer ----
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* dst_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
