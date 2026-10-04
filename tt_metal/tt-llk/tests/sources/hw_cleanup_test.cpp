// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile/smoke test for the Blackhole-only experimental hardware-teardown LLK
// family "hw_cleanup" (compute_kernel_hw_cleanup.h ->
// llk_{unpack,math,pack}_hw_cleanup.h + shared llk_hw_cleanup.h).
//
// hw_cleanup is a TEARDOWN family with NO numeric output of its own: it drains
// the three TRISCs, rendezvouses T0/T1/T2 through hardware mailboxes
// (Unpack/Pack -> READY; Math grants CONFIGURE in T0->T1->T2 order; then
// CONFIGURED -> CLEANUP_DONE, see llk_hw_cleanup.h start()/finish()), and
// reprograms both cfg banks to a canonical Float16_b 32x32 / four-face / 2048B
// tile geometry, leaving cfg bank 0 selected. It deliberately poisons pack MOP /
// strides / PAC X, so a following op must re-init pack before packing.
//
// GOLDEN: two identity datacopies (the eltwise_unary_datacopy A2D path) with the
// cleanup between them. Each thread finishes the first tile, the pack included,
// before it enters the cleanup, as the API requires (MATH_PACK and UNPACK_SYNC
// drained); every thread then re-inits, as a following op must, and copies the
// second tile. Both packed tiles must equal their inputs, and nothing may hang.
//
// The per-thread cleanup calls are the same entry points compute_kernel_hw_cleanup()
// dispatches (UNPACK -> _llk_unpack_hw_cleanup_canonical_<DST_ACCUM_MODE>,
// MATH  -> _llk_math_hw_cleanup_canonical_<DST_SYNC_MODE, DST_ACCUM_MODE>,
// PACK  -> _llk_pack_hw_cleanup_canonical_<DST_SYNC_MODE, DST_ACCUM_MODE>);
// this test drives each thread's canonical directly since the LLK test harness
// builds one kernel per TRISC.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// One config: two 32x32 tiles, four faces, SyncHalf.
static constexpr ckernel::DstSync HW_CLEANUP_DST_SYNC = ckernel::DstSync::SyncHalf;
constexpr std::uint32_t HW_CLEANUP_NUM_FACES          = 4;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_hw_cleanup.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

inline void unpack_init(const FormatConfig& formats)
{
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, HW_CLEANUP_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        HW_CLEANUP_NUM_FACES,
        HW_CLEANUP_NUM_FACES);
    unpack_init(formats);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);

    _llk_unpack_hw_cleanup_canonical_<is_fp32_dest_acc_en>();

    // The cleanup poisons the unpack MOP.
    unpack_init(formats);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[1]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_hw_cleanup.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"

using namespace ckernel;

inline void datacopy_init(const FormatConfig& formats)
{
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        false /* is_int_fpu_en */,
        ckernel::PackMode::Default>(HW_CLEANUP_NUM_FACES, formats.math);
}

inline void datacopy_tile(const FormatConfig& formats)
{
    _llk_math_wait_for_dest_available_<HW_CLEANUP_DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, HW_CLEANUP_DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, formats.math, formats.math, HW_CLEANUP_NUM_FACES);
    _llk_math_dest_section_done_<HW_CLEANUP_DST_SYNC, is_fp32_dest_acc_en>();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    datacopy_init(formats);
    _llk_math_pack_sync_init_<HW_CLEANUP_DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    datacopy_tile(formats);

    _llk_math_hw_cleanup_canonical_<HW_CLEANUP_DST_SYNC, is_fp32_dest_acc_en>();

    // The cleanup zeroes ADDR_MOD_0..7 and poisons the math MOP.
    datacopy_init(formats);
    datacopy_tile(formats);
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_hw_cleanup.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

inline void pack_init(const FormatConfig& formats)
{
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * HW_CLEANUP_NUM_FACES /* tile_size */, FACE_R_DIM, ckernel::TILE_C_DIM, HW_CLEANUP_NUM_FACES);
    _llk_pack_init_<ckernel::PackMode::Default, false /* zero_output */>(
        formats.pack_dst, FACE_R_DIM, ckernel::TILE_C_DIM, HW_CLEANUP_NUM_FACES, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
    _llk_pack_dest_init_<HW_CLEANUP_DST_SYNC, is_fp32_dest_acc_en>();
}

inline void pack_dest_tile(const std::uint32_t address)
{
    _llk_packer_wait_for_math_done_();
    _llk_pack_<HW_CLEANUP_DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, address);
    _llk_pack_dest_section_done_<HW_CLEANUP_DST_SYNC, is_fp32_dest_acc_en>();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    pack_init(formats);
    // The math side of the cleanup waits for MATH_PACK to drain, so the section is released first.
    pack_dest_tile(L1_ADDRESS(params.buffer_Res[0]));

    _llk_pack_hw_cleanup_canonical_<HW_CLEANUP_DST_SYNC, is_fp32_dest_acc_en>();

    // The cleanup poisons the pack MOP, strides and PAC X.
    pack_init(formats);
    pack_dest_tile(L1_ADDRESS(params.buffer_Res[1]));
}

#endif
