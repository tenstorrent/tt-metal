// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone LLK test for the SFPU-backed CB hash (23-bit "FNV23"). See
// tt_llk_{blackhole,wormhole_b0}/llk_lib/debug/llk_math_hash_cb.h.
//
// Data path:
//   UNPACK: unpacks TILE_CNT INT32 tiles from buffer_A into DEST slot 0.
//   MATH:   datacopies each tile into DEST, folds it into 32 per-lane FNV23
//           accumulators, then writes the accumulators back into DEST (row 0;
//           the rest of the tile is zeroed).
//   PACK:   packs the DEST tile to buffer_Res[0].
//
// The host XOR-folds the whole result tile; the zeroed rows cancel, leaving
// XOR(32 accumulators). Uses only the proven INT32 DEST -> PACK -> L1 path.
//
// HASH_CB_SFPU_OP_BEFORE_INIT / HASH_CB_SFPU_OP_BEFORE_STORE additionally run a
// whole-tile SFPU op on DEST slot HASH_CB_SFPU_OP_DST_INDEX inside the same DEST
// window: before the hash init (the production order of hash_cb_sfpu, where a
// user op precedes the hash) or between the fold loop and the store-back (a
// standalone caller). Either op leaves the math DEST offset at
// base + 64 * dst_index (its _done_ only resets the RWC), which is the state the
// hash must be immune to: the unpack-to-dest datacopy writes slot 0 at an
// absolute address, so the fold and the store-back must still target slot 0.
//
// HASH_CB_DEST_BANK selects the half-sync bank the hash runs in: the kernel first
// completes that many whole math/pack windows (datacopy the tiles into slot 0,
// pack them to buffer_Res[1]), each _llk_math_dest_section_done_ flipping
// dest_offset_id, so with HASH_CB_DEST_BANK = 1 the hash window runs with
// dest_offset_id == 1 and its bank base is DEST_REGISTER_HALF_SIZE, not 0 --
// the state hash_cb_sfpu sees every other window in production.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define DEBUG_CB_HASH 1

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    // One pass per prior window (HASH_CB_DEST_BANK of them) plus the hash window.
    for (std::uint32_t window = 0; window <= HASH_CB_DEST_BANK; window++)
    {
        for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_defs.h"
#include "ckernel_sfpu.h"
#include "debug/llk_math_hash_cb.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/ckernel_sfpu_negative.h"
#include "params.h"

using namespace ckernel::sfpu;

// A face is FACE_R_DIM x FACE_C_DIM = 256 datums and the SFPU moves 32 lanes per
// dst_reg step, so one face is 8 steps.
constexpr int FACE_SFPU_STEPS = (FACE_R_DIM * FACE_C_DIM) / 32;

constexpr bool HASH_CB_SFPU_OP_ENABLED = HASH_CB_SFPU_OP_BEFORE_INIT || HASH_CB_SFPU_OP_BEFORE_STORE;

// A whole-tile SFPU op on another slot in the same DEST window (what an eltwise
// SFPU op at that dst_index does): start_ points the math DEST offset at
// base + 64 * dst_index and done_ leaves it there. Slot 0 is not touched. The
// negate keeps one live vFloat so it stays off the hash's live LRegs (checked
// in the ELF disassembly: only LReg0, which the hash reloads on every step).
inline void sfpu_op_on_slot(std::uint32_t dst_index)
{
    _llk_math_eltwise_unary_sfpu_params_(_calculate_negative_<false, FACE_SFPU_STEPS>, dst_index, VectorMode::RC);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const bool is_int_fpu_en = true;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);
    if constexpr (HASH_CB_SFPU_OP_ENABLED)
    {
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    }

    // Prior windows: datacopy the tiles into slot 0 and hand the bank to the
    // packer, so the hash window below runs in bank HASH_CB_DEST_BANK.
    for (std::uint32_t window = 0; window < HASH_CB_DEST_BANK; window++)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                0, formats.math, formats.math);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    if constexpr (HASH_CB_SFPU_OP_BEFORE_INIT)
    {
        // Production order: a user SFPU op runs before hash_cb_sfpu's init.
        sfpu_op_on_slot(HASH_CB_SFPU_OP_DST_INDEX);
    }

    _llk_math_hash_cb_init_();

    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        // Datacopy A → DEST slot 0 so the SFPU can read it, then fold it in.
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0, formats.math, formats.math);
        _llk_math_hash_cb_tile_(/*dst_tile_idx=*/0);
    }

    if constexpr (HASH_CB_SFPU_OP_BEFORE_STORE)
    {
        // Standalone-caller order: an SFPU op between the fold and the store-back,
        // after the last _llk_math_hash_cb_tile_ has already pinned the offset.
        sfpu_op_on_slot(HASH_CB_SFPU_OP_DST_INDEX);
    }

    // Write the per-lane accumulators back into DEST for the packer.
    _llk_math_hash_cb_store_to_dest_();

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Raw INT32 bitcast pack (pack_src == pack_dst == Int32): the packer copies
    // DEST datums without a numeric (FP32) conversion.
    constexpr std::uint32_t int32_fmt = static_cast<std::uint32_t>(DataFormat::Int32);
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
        int32_fmt, int32_fmt, 32 * 32 * 4 /* INT32 tile size in bytes */, FACE_R_DIM, 4 /* num_faces */);
    // The hw-configure above established the packer strides, so the wrapper skips re-programming them;
    // it still programs the X (datum) counter, which configure_pack does not touch.
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(int32_fmt, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Prior windows: pack the datacopied slot 0 to buffer_Res[1] (the raw input).
    for (std::uint32_t window = 0; window < HASH_CB_DEST_BANK; window++)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[1]));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
