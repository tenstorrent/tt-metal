// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// dbg_get_array_row(dbg_array_id::SRCA, ...) cannot dump SrcA directly, so it borrows dest
// row 0: it saves that row into SFPU registers, stages the SrcA row there via MOVDBGA2D,
// reads it out over the debug bus, then restores the row it borrowed. The row is 32 datums
// wide and the SFPU moves it in two halves, so the save needs two registers -- holding both
// halves in one loses the first half, and the restore writes the surviving half over both.
//
// Every in-tree caller of dbg_get_array_row passes dbg_array_id::DEST, which takes none of
// this path, so nothing else exercises the save/restore. This driver does:
//
//   1. write a known tile into DEST through the RISC-V debug window
//   2. call dbg_get_array_row on SRCA, which borrows and must restore dest row 0
//   3. read DEST back and compare against what went in
//
// SrcA is never unpacked into, so the row staged in step 2 is whatever SrcA happens to hold.
// That is deliberate: the test asserts only that dest survives the borrow, which is the
// contract the helper breaks when the two halves share a register.

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_debug.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif

#ifdef LLK_TRISC_MATH

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    const DataFormat l1_fmt          = static_cast<DataFormat>(formats.unpack_A_src);
    constexpr std::uint32_t TILE_IDX = 0;

    dbg_copy_dest_tile<DbgDestTileOp::Write, MathThreadId>(l1_fmt, TILE_IDX, reinterpret_cast<void*>(params.buffer_A[0]));

    // The call under test. Its return data is not what is being checked -- the dest tile is.
    std::uint32_t srca_row[8] = {0};
    dbg_get_array_row(dbg_array_id::SRCA, 0 /* row_addr */, srca_row);

    dbg_copy_dest_tile<DbgDestTileOp::Read, MathThreadId>(l1_fmt, TILE_IDX, reinterpret_cast<void*>(params.buffer_Res[0]));
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif
