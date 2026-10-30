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
//   * The dump must be the SrcA row (SRCA_ROW_CHECK true; both architectures): clear DEST, unpack
//     a tile into SrcA and datacopy it into dest tile DUMP_DST_INDEX only, in the upper dest half
//     when DUMP_UPPER_HALF, so the math thread's dest offset sits where a real kernel leaves it
//     (moved DUMP_ROW_OFFSET rows into that tile when nonzero, with the Blackhole DEST remap on
//     when DUMP_REMAP, so the remap and 32-bit swizzle terms of the read-back row are exercised),
//     then dump SrcA row 0 and hand the raw debug-bus words to the host, which knows the tile and
//     the dest bit layout. The datacopy consumed the faces in order, so the bank the dump reads
//     holds the last face. The helper stages and restores through the thread's dest offset but
//     reads back a physical row, so a dump taken with the offset off zero returns a stale row
//     unless the read-back follows the offset. The same run checks the helper's footprint: the
//     rows it borrows (the aligned group of four rows holding the offset's row) read back unchanged over
//     the debug bus, the dest offset is handed back untouched, and the packer packs tile 0 of the
//     lower half afterwards -- rows the datacopy never wrote must still pack as zero, which they
//     do not if the helper borrowed rows the caller never populated (an SFPLOAD of a row left
//     undefined by ZEROACC is undefined behaviour, and its restore then defines the row with
//     junk).
//
// The dump is bracketed by dbg_thread_halt / dbg_thread_unhalt, as every in-tree caller does
// (dbg_halt / dbg_unhalt, the DPRINT dest dump). The debug array read leaves the packer needing
// the soft reset MATH's unhalt performs; without it the core is left wedged and the NEXT test
// scheduled on it hangs in its boot-time ZEROACC -- a hang that lands on whatever test follows,
// not on this one.
//
// Result layout (constants shared with the host through DBG_SRCA_DUMP): tile 0 of the result buffer carries the
// raw debug-bus words of the SrcA dump, the thread's dest offset read before the dump, after it, and the value the
// datacopy left it at, then the rows the dump borrows (the aligned group of four rows holding the row at the offset,
// and the rows 8 further on that hold the low halves with a 32-bit DEST) read over the debug bus before and after the dump, then the SrcA
// format selection (SrcA_val and SrcA_override) read on the RISC before and after the dump; with PRESET_SRCA_OVERRIDE
// the test sets it to a non-default value first, as a caller inside a format-switching sequence would. Tile 1 is the
// DEST tile handed back: the planted tile read through the RISC-V window (restore check, Blackhole), or tile 0 of
// the lower dest half packed by the packer (SrcA check).

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

// Physical row of a logical dest row within the current half, as the debug bus addresses it (Dst.md): on Blackhole
// with DUMP_REMAP the untilize remap permutes bits 3-5 and, with a 32-bit DEST, the swizzle_32b permutation
// (which _llk_math_reconfig_remap_ enables with it) permutes bits 2-4; a 32-bit DEST then spreads each logical row
// over two physical rows (the read returns the high halves). Built from the test's compile-time knobs, not from
// the config the helper reads, so a helper that misreads the config shows up here.
constexpr std::uint32_t physical_dest_row(std::uint32_t row)
{
    if constexpr (DUMP_REMAP)
    {
        row = (row & 0x3c7) ^ ((row & 0x030) >> 1) ^ ((row & 0x008) << 2);
    }
    if constexpr (is_fp32_dest_acc_en)
    {
        if constexpr (DUMP_REMAP)
        {
            row = (row & 0x3f3) ^ ((row & 0x018) >> 1) ^ ((row & 0x004) << 1);
        }
        row = ((row & 0x1f8) << 1) | (row & 0x207);
    }
    return row;
}

// The dest row the dump stages at: tile DUMP_DST_INDEX of the current half, DUMP_ROW_OFFSET rows in.
constexpr std::uint32_t DUMP_FIRST_ROW = DUMP_DST_INDEX * DEST_ROWS_PER_TILE + DUMP_ROW_OFFSET;
// The SFPU save/restore moves aligned groups of four rows (SFPLOAD.md), so the rows the dump borrows are the group
// holding the staged row.
constexpr std::uint32_t BORROWED_FIRST_ROW = DUMP_FIRST_ROW & ~3u;
static_assert((DUMP_ROW_OFFSET & ~3u) + 4 <= DEST_ROWS_PER_TILE, "the borrowed rows must stay inside the datacopied tile");

// The SrcA format selection the test leaves active across the dump (PRESET_SRCA_OVERRIDE): override on, BF16.
constexpr std::uint32_t SRCA_FMT_SELECT_MASK = ALU_FORMAT_SPEC_REG_SrcA_val_MASK | ALU_FORMAT_SPEC_REG_SrcA_override_MASK;

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
#ifdef ARCH_BLACKHOLE
        // Set the DEST remap explicitly either way: earlier tests on the core may have left it on.
        _llk_math_reconfig_remap_(DUMP_REMAP);
#else
        static_assert(!DUMP_REMAP, "the DEST remap is Blackhole-only");
#endif
        // Every row not written below is undefined and packs as zero; the helper must not define any of them.
        zeroacc();
        if constexpr (DUMP_UPPER_HALF)
        {
            math::dest_section_flip();
        }
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        // Leaves the thread's dest offset at tile DUMP_DST_INDEX of the current half, as any op targeting it does.
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false /*unpack_to_dest*/>(
            DUMP_DST_INDEX, formats.math, formats.math, TILE_NUM_FACES);
        if constexpr (DUMP_ROW_OFFSET != 0)
        {
            // A sub-tile dest offset, set the way the custom matmul and SDPA paths set theirs.
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
            TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base() + DUMP_FIRST_ROW);
        }
        if constexpr (PRESET_SRCA_OVERRIDE)
        {
            // A caller mid-way through a format-switching sequence: the dump must hand this selection back.
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, SRCA_FMT_SELECT_MASK>(
                ALU_FORMAT_SPEC_REG_SrcA_override_MASK | (to_underlying(DataFormat::Float16_b) << ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT));
        }
    }
    else
    {
        // Pin dest to the test's DEST width for both the FPU (which sets the width MOVDBGA2D writes) and the SFPU
        // (which the save/restore goes through), the pairing _llk_math_hw_configure_ uses, so both modes are checked.
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
        // Pin the math thread's dest offset to tile 0, where the tile is planted: the harness resets only the
        // software half index, so the hardware offset is whatever the previous kernel left.
        math::set_dest_section_base<DstStart::StartZero>();
        tensix_sync();
    }

    volatile std::uint32_t* words = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[0]);

    dbg_thread_halt<MathThreadId>();

    if constexpr (SRCA_ROW_CHECK)
    {
        // The offset the datacopy left (the halt drained its SETC16), and where the datacopy should have left it.
        words[DEST_OFFSET_BEFORE_WORD] = dbg_read_dest_offset();
        words[DEST_OFFSET_SET_WORD]    = DUMP_FIRST_ROW + (DUMP_UPPER_HALF ? DEST_REGISTER_HALF_SIZE : 0);
        words[SRCA_FMT_BEFORE_WORD]    = cfg_read(ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32) & SRCA_FMT_SELECT_MASK;
    }

    if constexpr (!SRCA_ROW_CHECK)
    {
#ifdef ARCH_BLACKHOLE
        // Plant a known tile where the helper borrows its row: the tile at the math thread's dest offset, tile 0.
        dbg_copy_dest_tile<DbgDestTileOp::Write, MathThreadId>(l1_fmt, 0 /*tile_id*/, reinterpret_cast<void*>(params.buffer_B[0]));
#endif
    }

    // The rows the dump borrows, as the debug bus sees them: the aligned group of four rows holding the dest
    // offset's row (the SFPU save/restore footprint) and the rows 8 further on (the low halves with a 32-bit DEST).
    // dbg_get_array_row adds the current half's base to a DEST row itself.
    auto read_borrowed_rows = [&](std::uint32_t first_word)
    {
        for (std::uint32_t r = 0; r < BORROWED_ROW_COUNT; ++r)
        {
            constexpr std::uint32_t half           = BORROWED_ROW_COUNT / 2;
            const std::uint32_t row                = physical_dest_row(BORROWED_FIRST_ROW + (r % half)) + ((r < half) ? 0 : 8);
            std::uint32_t row_words[WORDS_PER_ROW] = {0};
            dbg_get_array_row(dbg_array_id::DEST, row, row_words);
            for (std::uint32_t i = 0; i < WORDS_PER_ROW; ++i)
            {
                words[first_word + r * WORDS_PER_ROW + i] = row_words[i];
            }
        }
    };
    if constexpr (SRCA_ROW_CHECK)
    {
        read_borrowed_rows(BORROWED_ROWS_BEFORE_WORD);
    }

    // The call under test.
    std::uint32_t srca_row[WORDS_PER_ROW] = {0};
    dbg_get_array_row(dbg_array_id::SRCA, 0 /*row_addr*/, srca_row);
    for (std::uint32_t i = 0; i < WORDS_PER_ROW; ++i)
    {
        words[i] = srca_row[i];
    }

    if constexpr (SRCA_ROW_CHECK)
    {
        read_borrowed_rows(BORROWED_ROWS_AFTER_WORD);
    }

    if constexpr (!SRCA_ROW_CHECK)
    {
#ifdef ARCH_BLACKHOLE
        dbg_copy_dest_tile<DbgDestTileOp::Read, MathThreadId>(l1_fmt, 0 /*tile_id*/, reinterpret_cast<void*>(params.buffer_Res[RESULT_TILE_DEST]));
#endif
    }

    dbg_thread_unhalt<MathThreadId>();

    if constexpr (SRCA_ROW_CHECK)
    {
        // The offset the dump left behind (the unhalt drained the dump); it must be what the datacopy set.
        words[DEST_OFFSET_AFTER_WORD] = dbg_read_dest_offset();
        words[SRCA_FMT_AFTER_WORD]    = cfg_read(ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32) & SRCA_FMT_SELECT_MASK;
        if constexpr (PRESET_SRCA_OVERRIDE)
        {
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, SRCA_FMT_SELECT_MASK>(0);
        }
        // Release the packer: it packs tile 0 of the lower half, which only the tile_0 case wrote.
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#ifdef ARCH_BLACKHOLE
        if constexpr (DUMP_REMAP)
        {
            // Leave linear addressing for the next test; waits for the pack above to finish.
            _llk_math_reconfig_remap_(false);
        }
#endif
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    if constexpr (SRCA_ROW_CHECK)
    {
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /*tile_size*/, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

        // Tile 0 of the lower half, once the math thread is done with the dump. The packer never flipped halves, so
        // this is the same tile in every variant: the datacopy's output in the tile_0 case, zeros in every other.
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(0 /*tile_index*/, L1_ADDRESS(params.buffer_Res[RESULT_TILE_DEST]));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
