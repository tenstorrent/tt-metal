// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// lane FS  FP-3 architectural model, experiment 1 ("kernel B").
//
// Does NOT record anything.  Launches replay slots 0..1 (TT_REPLAY Load=0 =>
// emit ReplayBuffer[0..1]) and then copies DEST tile 0 back to L1 through the
// RISC-V debug DEST window.
//
// If the Replay-Expander per-thread buffer was cleared between invocations,
// slots 0..1 hold zero words and the launch writes nothing meaningful to DEST
// (readback == 0).  If the buffer PERSISTED from kernel A, the launch replays
// A's SFPLOADI+SFPSTORE and the readback contains the sentinel 0x0000ABCD.

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_debug.h"
#include "ckernel_ops.h"

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
    const DataFormat l1_fmt = static_cast<DataFormat>(formats.unpack_A_src);

    // Launch replay slots 0..1 (Load=0 => replay, emit ReplayBuffer[0..1]).
    // No preceding record in this kernel.
    TTI_REPLAY(0, 2, 0, 0);

    // Drain the SFPU pipe so the replayed store retires before we read DEST.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SFPU1);
    TTI_NOP;

    // Copy DEST tile 0 -> L1 through the RISC-V debug DEST window.
    dbg_copy_dest_tile<DbgDestTileOp::Read, MathThreadId>(l1_fmt, 0, reinterpret_cast<void*>(params.buffer_Res[0]));
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif
