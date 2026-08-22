// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Test for the Blackhole-only experimental LLK
//   _llk_math_sdpa_bcast_col_srcb_reuse_
// (headers:
//   tt_llk_blackhole/llk_lib/experimental/llk_math_sdpa_bcast_col_srcb_reuse.h
//   hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_sdpa_bcast_col_srcb_reuse_api.h)
//
// -----------------------------------------------------------------------------
// WHAT THE HEADER ACTUALLY COMPUTES (how the golden is derived)
// -----------------------------------------------------------------------------
// This op is a column-broadcast elementwise binary in which the *broadcast*
// operand (SrcB) is not unpacked from L1 but is *reused out of DEST*:
//
//   * preamble (_llk_math_sdpa_bcast_col_srcb_reuse_preamble_):
//       Four TTI_MOVD2B move DEST rows (SRC_ZERO_OFFSET, i.e. the tile at
//       DEST index 0) into SrcB, 4 rows at a time. So whatever sits in DEST
//       tile 0 *before* the op becomes the SrcB operand.  clear_dest defaults
//       to false, so DEST tile 0 is NOT zeroed here — we rely on that to seed B.
//   * main op (_llk_math_sdpa_bcast_col_srcb_reuse_):
//       The programmed MOP issues TT_OP_ELW{ADD,SUB,MUL} with
//       broadcast_type = p_elwise::SRCB_BCAST_COL.  On Tensix, ELWSUB computes
//       SrcA - SrcB, ELWADD computes SrcA + SrcB, ELWMUL computes SrcA * SrcB.
//       SRCB_BCAST_COL broadcasts column 0 of each SrcB face across all 16
//       columns of that face (per BroadcastGolden._broadcast_column). SrcA is
//       the operand freshly unpacked from L1 for this op.
//   * postamble: TTI_SETRWC(CLR_B ...) clears SrcB.
//
//   Result written to DEST[dst_index] = A  <op>  bcast_col(B)
//
//   where A is the fresh L1 operand and B is the operand that was seeded into
//   DEST tile 0 (here via an A2D datacopy, see kernel below).
//
//   configure_mop() LLK_ASSERTs num_faces == 2, so this op runs on the top two
//   faces (faces 0 and 1 = rows 0..15 x cols 0..31, one 16x32 half-tile). Only
//   those 2 faces are defined; the Python golden validates only those lanes.
//
// GOLDEN (mirrors the above):
//   golden = EltwiseBinary( mathop,
//                           A_tilized,
//                           BroadcastColumn(B_tilized, num_faces=2) )
//   validated over the first 2 faces only.
// =============================================================================

#include <cstdint>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "operand.h"
#include "params.h"
#include "tensix_types.h"
#include "tensor_shape.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// L1 layout (mirrors the manual-stimuli convention used by sdpa_reinits_test.cpp).
// The Python side writes tilized A to BUFFER_A_ADDR and tilized B to BUFFER_B_ADDR
// and reads the result from BUFFER_RES_ADDR.
constexpr std::uint32_t BUFFER_A_ADDR   = 0x1a000;
constexpr std::uint32_t BUFFER_B_ADDR   = 0x1a800;
constexpr std::uint32_t BUFFER_RES_ADDR = 0x1b000;
constexpr std::uint32_t TILE_SIZE_BYTES = 2048; // Float16_b 32x32 tile

// The op requires exactly 2 active faces (configure_mop asserts num_faces == 2).
constexpr std::uint32_t OP_NUM_FACES = 2;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_A_sdpa.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS)
{
    const std::uint32_t src_format = ckernel::to_underlying(DataFormat::Float16_b);
    const std::uint32_t dst_format = ckernel::to_underlying(DataFormat::Float16_b);

    const Operand buffer_A(BUFFER_A_ADDR, TILE_SIZE_BYTES);
    const Operand buffer_B(BUFFER_B_ADDR, TILE_SIZE_BYTES);

    _llk_unpack_hw_configure_<false>(src_format, src_format, dst_format, dst_format, FACE_R_DIM, FACE_R_DIM, OP_NUM_FACES, OP_NUM_FACES);

    // Step 1: unpack B into SrcA so the math thread can datacopy it into DEST[0]
    // (this becomes the broadcast operand that the preamble MOVD2Bs into SrcB).
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, OP_NUM_FACES), src_format, dst_format);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(L1_ADDRESS(buffer_B[0]), src_format, dst_format);

    // Step 2: set the dummy SrcB dvalid the preamble's MOVD2B assumes (SrcB is
    // filled from DEST, not from L1, so the unpacker must fake its data-valid).
    _llk_unpack_A_sdpa_set_srcb_dummy_valid_();

    // Step 3: unpack A into SrcA for the actual bcast-col op.
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, OP_NUM_FACES), src_format, dst_format);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(L1_ADDRESS(buffer_A[0]), src_format, dst_format);
}

#endif

#ifdef LLK_TRISC_MATH

// NOTE: sdpa_bcast_col_srcb_reuse_configure_mop() in the experimental header below declares
// three helper locals (addr_mod/innerloop/outerloop) that are unused in every instantiated
// branch, which trips -Werror=unused-variable when the op is instantiated. -Wunused-variable is
// keyed to the variable's source location (the header line), so suppressing it for the region
// that lexically contains that line silences the warning at every later instantiation. The
// header is a promoted experimental LLK we must not edit here, so we suppress only around it.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "experimental/llk_math_sdpa_bcast_col_srcb_reuse.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS)
{
    const std::uint32_t math_format = ckernel::to_underlying(DataFormat::Float16_b);
    constexpr DstSync dest_sync     = DstSync::SyncHalf;
    constexpr bool fp32_dest_acc    = false;

    _llk_math_hw_configure_<fp32_dest_acc>(math_format, math_format);
    _llk_math_pack_sync_init_<dest_sync, fp32_dest_acc>();

    _llk_math_wait_for_dest_available_<dest_sync>();

    // Step 1: seed DEST[0] with B (A2D datacopy of the B operand unpacked into SrcA).
    // This is the operand that will be broadcast-across-columns and reused via SrcB.
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, fp32_dest_acc, BroadcastType::NONE, false /* is_int_fpu_en */>(OP_NUM_FACES, math_format);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, fp32_dest_acc, BroadcastType::NONE, false>(
        0 /* dst_index */, math_format, math_format, OP_NUM_FACES);

    // Step 2: init + preamble + op + postamble for the sdpa bcast-col srcB-reuse.
    // NUM_TILES template must match the MOP the math programs; a single half-tile op.
    constexpr std::uint32_t NUM_TILES = 1;
    _llk_math_sdpa_bcast_col_srcb_reuse_init_<ELTWISE_BINARY_OP, NUM_TILES, MATH_FIDELITY, false /* dense */>(OP_NUM_FACES, 0 /* acc_to_dest */);

    // preamble: MOVD2B DEST[0] -> SrcB.  clear_dest=false keeps the seeded B in DEST[0].
    _llk_math_sdpa_bcast_col_srcb_reuse_preamble_<dest_sync, fp32_dest_acc, false /* clear_dest */>();

    // main op: DEST[0] = A <op> bcast_col(B)
    _llk_math_sdpa_bcast_col_srcb_reuse_<ELTWISE_BINARY_OP, NUM_TILES, dest_sync, fp32_dest_acc, MATH_FIDELITY, false>(0 /* dst_index */);

    _llk_math_sdpa_bcast_col_srcb_reuse_postamble_();

    _llk_math_dest_section_done_<dest_sync, fp32_dest_acc>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS)
{
    const std::uint32_t pack_src_format = ckernel::to_underlying(DataFormat::Float16_b);
    const std::uint32_t pack_dst_format = ckernel::to_underlying(DataFormat::Float16_b);
    constexpr DstSync dest_sync         = DstSync::SyncHalf;
    constexpr bool fp32_dest_acc        = false;

    const Operand buffer_Res(BUFFER_RES_ADDR, TILE_SIZE_BYTES);

    _llk_pack_hw_configure_wrapper_<fp32_dest_acc, PackMode::Default>(pack_src_format, pack_dst_format, TILE_SIZE_BYTES, FACE_R_DIM, TILE_C_DIM, OP_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(pack_dst_format, FACE_R_DIM, TILE_C_DIM, OP_NUM_FACES);
    _llk_pack_dest_init_wrapper_<dest_sync, fp32_dest_acc, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<dest_sync, fp32_dest_acc, ckernel::PackMode::Default>(0, L1_ADDRESS(buffer_Res[0]));
    _llk_pack_dest_section_done_<dest_sync, fp32_dest_acc>();
}

#endif
