// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Cross-op unpacker-state restore test for the SDPA `_llk_unpack_bcastA_B_` path.
//
// PR #45127 changed `_llk_unpack_bcastA_B_uninit_` so its SrcA Y-stride restore
// went from a hardcoded `FACE_R_DIM*2` (=32) to `canonical_unpA_y_stride(dst_format)`
// (=16 for bf16), and it added the matching Y-stride (re)write to
// `_llk_unpack_reconfig_data_format_srca_impl_` (the FACE_ROW_MAJOR branch).
// NO existing C++ test calls `_llk_unpack_bcastA_B_uninit_` at all, and the matmul
// tilize test only ever calls the reconfig with `p_dim_stride_target::IGNORE`, so
// the new Y-stride write is unexercised.
//
// `_llk_unpack_bcastA_B_init_` mutates the SrcA Y-stride to 32. This test runs:
//   Run 0: `_llk_unpack_bcastA_B_` (which leaves Y-stride = 32) + an eltwise-binary
//          math op to drain the pipeline.
//   Restore under test (selected by `RESTORE_VIA_RECONFIG`):
//     * false -> `_llk_unpack_bcastA_B_uninit_(dst)`            (C1, Phase 4)
//     * true  -> `_llk_unpack_reconfig_data_format_srca_impl_<.., FACE_ROW_MAJOR>` (C2, Phase 3)
//   Run 1: a plain `_llk_unpack_A_` datacopy of the ORIGINAL operand-A tile
//          (`buffer_A[0]`), with NO other state reset in between.
//
// If the restore leaves the SrcA Y-stride at the bcast-specific value (32) instead
// of the canonical baseline (16 for bf16), the run-1 datacopy reads SrcA with the
// wrong row stride and the result diverges from `DataCopyGolden(src_A)`.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Scratch L1 address that holds the run-0 (bcast) result. Run 1 does NOT read it;
// it exists only so the run-0 packer has a destination and the pipeline drains.
constexpr std::uint32_t buffer_bcast_scratch = 0xA0000;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    constexpr std::uint32_t num_faces = 4;
    // Datum count for one 32x32 tile; only used to seed the tile-size GPR (the
    // single-tile run-1 datacopy reads tile index 0, so the exact value is moot).
    constexpr std::uint32_t tile_size_datums = num_faces * FACE_R_DIM * FACE_C_DIM;

    // ---- Run 0: SDPA row-broadcast of operand A against operand B ----
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats_array[0].unpack_A_src,
        formats_array[0].unpack_B_src,
        formats_array[0].unpack_A_dst,
        formats_array[0].unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        num_faces,
        num_faces);
    _llk_unpack_bcastA_B_init_();
    _llk_unpack_bcastA_B_(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]), params.SRCA_REUSE_COUNT);

    // Wait until the run-0 packer has drained before touching unpacker state.
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // ---- Restore under test ----
    if constexpr (RESTORE_VIA_RECONFIG)
    {
        // C2 (Phase 3): the reconfig FACE_ROW_MAJOR branch re-commits the canonical
        // SrcA Y-stride (and Z-stride / Tile_x_dim / Z-dim). This is the only place
        // the new Y-stride write is exercised; the matmul tilize test uses IGNORE.
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, false>(
            formats_array[1].unpack_A_src, formats_array[1].unpack_A_dst, tile_size_datums, FACE_R_DIM, num_faces);
    }
    else
    {
        // C1 (Phase 4): the bcast-specific uninit restores the canonical Y-stride
        // from the dst format. NOTE: intentionally NO reconfig here, so this uninit
        // is the sole reset of the unpacker state before the next op.
        _llk_unpack_bcastA_B_uninit_(formats_array[1].unpack_A_dst);
    }

    // ---- Run 1: plain datacopy of the ORIGINAL operand-A tile (no reconfig) ----
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats_array[1].unpack_A_src, formats_array[1].unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats_array[1].unpack_A_src, formats_array[1].unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    constexpr std::uint32_t num_faces = 4;
    const bool is_int_fpu_en          = false;
    const std::uint32_t res_dst_idx   = 0;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats_array[0].math, formats_array[0].math);

    // ---- Run 0: eltwise-binary consuming the broadcast operands ----
    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, ckernel::MathFidelity::LoFi>(params.SRCA_REUSE_COUNT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_binary_(res_dst_idx);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // ---- Run 1: plain datacopy ----
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, ckernel::PackMode::Default>(
        num_faces, formats_array[1].math);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        res_dst_idx, formats_array[1].math, formats_array[1].math, num_faces);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    constexpr std::uint32_t num_faces = 4;
    const std::uint32_t res_dst_idx   = 0;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats_array[0].pack_src, formats_array[0].pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces /* tile_size */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats_array[0].pack_dst);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    // ---- Run 0: pack the bcast result to scratch (drains the pipeline) ----
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(res_dst_idx, L1_ADDRESS(buffer_bcast_scratch));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Signal the unpacker that run-0 has fully drained.
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // ---- Run 1: pack the datacopy result to the output buffer ----
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(res_dst_idx, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
