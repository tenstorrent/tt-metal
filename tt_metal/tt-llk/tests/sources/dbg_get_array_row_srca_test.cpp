// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// dbg_get_array_row(dbg_array_id::SRCA, ...) cannot dump SrcA directly, so it borrows a dest
// row: it saves that row into SFPU registers, stages the SrcA row there via MOVDBGA2D,
// reads it out over the debug bus, then restores the row it borrowed. Two contracts follow,
// and this driver checks both:
//
//   * The borrowed row must come back bit-exact (SRCA_ROW_CHECK false; Blackhole, which has
//     the RISC-V debug window into DEST):
//       1. write a known tile into DEST through the window
//       2. call dbg_get_array_row on SRCA, which borrows and must restore the row
//       3. read DEST back and compare against what went in
//     The row is 32 datums wide and the SFPU moves it in two halves, so the save needs two
//     registers -- holding both halves in one loses the first half, and the restore writes the
//     surviving half over both. The save must also move bits rather than convert them: the
//     kernel pins the FPU and SFPU views of dest to the test's DEST width, and the host plants
//     Int32 values a converting save would change. SrcA is never unpacked into here, so the row
//     staged in step 2 is whatever SrcA happens to hold; only dest's survival is asserted.
//
//   * The dump must be the SrcA row (SRCA_ROW_CHECK true; both architectures): unpack a tile
//     into SrcA and datacopy it into dest tile DUMP_DST_INDEX, in the upper dest half when
//     DUMP_UPPER_HALF, so the math thread's dest offset sits where a real kernel leaves it, then
//     dump SrcA row 0 and hand the raw debug-bus words to the host, which knows the tile and the
//     dest bit layout. The datacopy consumed the faces in order, so the bank the dump reads holds
//     the last face. The helper stages and restores through the thread's dest offset but reads
//     back a physical row, so unless it pins the offset first, a dump taken with the offset off
//     zero returns a stale row instead of the SrcA row. On Blackhole the tile the helper borrows
//     from is planted from buffer_B and read back as well.
//
// The dump is bracketed by dbg_thread_halt / dbg_thread_unhalt, as every in-tree caller does
// (dbg_halt / dbg_unhalt, the DPRINT dest dump). The debug array read leaves the packer needing
// the soft reset MATH's unhalt performs; without it the core is left wedged and the NEXT test
// scheduled on it hangs in its boot-time ZEROACC -- a hang that lands on whatever test follows,
// not on this one.
//
// Result layout (constants shared with the host through DBG_SRCA_DUMP): tile 0 of the result buffer carries the
// raw debug-bus words of the SrcA dump, then the thread's dest offset read before the dump, after it, and the
// value the datacopy left it at; tile 1 is the DEST tile read back on Blackhole.

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_debug.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t DEST_ROWS_PER_TILE = (ckernel::TILE_HEIGHT * ckernel::TILE_WIDTH) / ckernel::FACE_WIDTH; // a dest row is one face wide

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    if constexpr (SRCA_ROW_CHECK)
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, false /*unpack_to_dest*/>(
            0 /*transpose_of_faces*/,
            0 /*within_face_16x16_transpose*/,
            make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        _llk_unpack_A_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, false /*unpack_to_dest*/>(
            L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    }

    // Unpack's side of the debug rendezvous: idle until MATH's unhalt releases it.
    dbg_thread_halt<UnpackThreadId>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    [[maybe_unused]] const DataFormat l1_fmt = static_cast<DataFormat>(formats.unpack_A_src);

    if constexpr (SRCA_ROW_CHECK)
    {
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /*is_int_fpu_en*/, PackMode::Default>(
            TILE_NUM_FACES, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        if constexpr (DUMP_UPPER_HALF)
        {
            math::dest_section_flip();
        }
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        // Leaves the thread's dest offset at tile DUMP_DST_INDEX of the current half, as any op targeting it does.
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false /*unpack_to_dest*/>(
            DUMP_DST_INDEX, formats.math, formats.math, TILE_NUM_FACES);
    }
    else
    {
        // Pin dest to the test's DEST width for both the FPU (which sets the width MOVDBGA2D writes) and the SFPU
        // (which the save/restore goes through), the pairing _llk_math_hw_configure_ uses, so both modes are checked.
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
    }

    volatile std::uint32_t* words = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[0]);

    dbg_thread_halt<MathThreadId>();

    if constexpr (SRCA_ROW_CHECK)
    {
        // The offset the datacopy left (the halt drained its SETC16), and where the datacopy should have left it.
        words[DEST_OFFSET_WORD + 0] = dbg_read_dest_offset<MathThreadId>();
        words[DEST_OFFSET_WORD + 2] = DUMP_DST_INDEX * DEST_ROWS_PER_TILE + (DUMP_UPPER_HALF ? DEST_REGISTER_HALF_SIZE : 0);
    }

#ifdef ARCH_BLACKHOLE
    // The tile the helper borrows from is the first tile of the current dest half. The debug window addresses
    // physical tiles, so the upper half starts DEST_REGISTER_HALF_SIZE rows in whatever the dest width.
    const std::uint32_t borrowed_tile = DUMP_UPPER_HALF ? DEST_REGISTER_HALF_SIZE / DEST_ROWS_PER_TILE : 0;
    dbg_copy_dest_tile<DbgDestTileOp::Write, MathThreadId>(l1_fmt, borrowed_tile, reinterpret_cast<void*>(params.buffer_B[0]));
#endif

    // The call under test.
    std::uint32_t srca_row[WORDS_PER_ROW] = {0};
    dbg_get_array_row(dbg_array_id::SRCA, 0 /* row_addr */, srca_row);
    for (std::uint32_t i = 0; i < WORDS_PER_ROW; ++i)
    {
        words[i] = srca_row[i];
    }

#ifdef ARCH_BLACKHOLE
    dbg_copy_dest_tile<DbgDestTileOp::Read, MathThreadId>(l1_fmt, borrowed_tile, reinterpret_cast<void*>(params.buffer_Res[RESULT_TILE_DEST]));
#endif

    dbg_thread_unhalt<MathThreadId>();

    if constexpr (SRCA_ROW_CHECK)
    {
        // The offset the dump left behind (the unhalt drained its restore); it must be what the datacopy set.
        words[DEST_OFFSET_WORD + 1] = dbg_read_dest_offset<MathThreadId>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif
