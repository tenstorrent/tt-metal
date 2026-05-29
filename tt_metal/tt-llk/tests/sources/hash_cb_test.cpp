// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Standalone LLK test for the SFPU-backed CB hash (23-bit "FNV23"). See
// tt_llk_{blackhole,wormhole_b0}/llk_lib/debug/llk_math_hash_cb.h for the
// per-arch SFPU implementations.
//
// Test data path mirrors the production hash_cb_sfpu orchestration:
//   UNPACK: unpacks TILE_CNT INT32 tiles from buffer_A into DEST slot 0.
//   MATH:   inits SFPU state, folds each tile into 32 per-lane accumulators,
//           reduces to lane 0, then writes a single u32 to the MEM_LLK_DEBUG
//           hash slot in L1 plus a ready flag.
//   UNPACK: polls the ready flag, reads the hash out of the debug L1 slot,
//           and copies it into buffer_Res[0] so the host-side test can pick
//           it up via the standard result-buffer path.
//
// No output CB and no PACK trip are involved — the production debug API uses
// the same L1 slot to hand the hash from MATH to UNPACK for DPRINT.
//
// STATUS: WH SFPU sequence is hardware-validated (WH B0 n150) via the
// tt-metal gtest in tests/tt_metal/tt_metal/llk/test_cb_hash.cpp. BH SFPU
// sequence is still pending hardware bring-up.
// The matching pytest is tests/python_tests/test_hash_cb.py.

#include <cstdint>

#include "ckernel.h"
#include "dev_mem_map.h"  // MEM_LLK_DEBUG_BASE
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define DEBUG_CB_HASH 1

// Must match DEBUG_HASH_L1_OFFSET in tt_metal/hw/inc/api/compute/debug/cb_hash.h.
static constexpr std::uint32_t DEBUG_HASH_L1_OFFSET     = 64;
static constexpr std::uint32_t DEBUG_HASH_L1_HASH_ADDR  = MEM_LLK_DEBUG_BASE + DEBUG_HASH_L1_OFFSET;
static constexpr std::uint32_t DEBUG_HASH_L1_READY_ADDR = MEM_LLK_DEBUG_BASE + DEBUG_HASH_L1_OFFSET + 4;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Clear the ready flag *before* the unpack loop so MATH's post-compute
    // write is the only one we'll observe in the poll below.
    *reinterpret_cast<volatile std::uint32_t*>(DEBUG_HASH_L1_READY_ADDR) = 0u;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, 4, formats.unpack_A_src, formats.unpack_A_dst);

    // Unpack each input tile into DEST slot 0; SFPU LRegs hold the running
    // accumulator state, so reusing slot 0 across iterations is intentional.
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }

    // Wait for MATH to publish the reduced hash to L1.
    volatile std::uint32_t* const ready_ptr =
        reinterpret_cast<volatile std::uint32_t*>(DEBUG_HASH_L1_READY_ADDR);
    while (*ready_ptr == 0u)
    {
#if defined(ARCH_BLACKHOLE)
        asm("fence");  // invalidate_l1_cache() — BH has a write-through L1 cache
#endif
    }

    // Hand the hash off to the host-side test through the result buffer.
    const std::uint32_t h = *reinterpret_cast<volatile std::uint32_t*>(DEBUG_HASH_L1_HASH_ADDR);
    *reinterpret_cast<volatile std::uint32_t*>(L1_ADDRESS(params.buffer_Res[0])) = h;
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_defs.h"
#include "debug/llk_math_hash_cb.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const bool is_int_fpu_en = true;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);
#endif

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    _llk_math_hash_cb_init_();

    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        // Datacopy A → DEST slot 0 so SFPU can read it.
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0, formats.math, formats.math);

        _llk_math_hash_cb_tile_(/*dst_tile_idx=*/0);
    }

    // Reduce + drain SFPU + read DEST + publish hash and ready flag to L1.
    _llk_math_hash_cb_finish_to_l1_(DEBUG_HASH_L1_HASH_ADDR, DEBUG_HASH_L1_READY_ADDR);

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

// PACK is not used by hash_cb_sfpu — the hash takes the MATH → L1 → UNPACK
// path. The framework still expects a PACK entrypoint, so provide a stub
// that does nothing (UNPACK writes buffer_Res[0] directly).

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/) {}

#endif
