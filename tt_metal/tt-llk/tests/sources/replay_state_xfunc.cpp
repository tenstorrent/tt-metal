// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// lane FS  FP-3 architectural model, experiment 2.
//
// Within ONE kernel launch: record a distinctive payload into replay slots
// 0..1 with a NO-EXEC record in one basic block, then LAUNCH slot 0 in a
// LATER basic block separated by opaque runtime control flow.  Models the
// pfj1 "sibling-arm" shape (FP-3): the compiler's intra-function reachability
// walk cannot connect the record to the launch, but the hardware instruction
// stream delivers them in program order to the one per-thread Replay Expander.
//
// If the DEST readback contains the sentinel, the buffer persists across
// basic-block / control-flow boundaries within a launch, i.e. the reassembly
// the FP-3 reach rule cannot see is real.

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

// Kept in separate noinline functions so the record and the launch land in
// distinct basic blocks / call frames (the compiler-analysis boundary), while
// the hardware stream still issues record-then-launch in order.

static __attribute__((noinline)) void record_arm()
{
    // NO-EXEC record of slots 0..1 (execute_while_loading=0, Load=1).
    TTI_REPLAY(0, 2, 0, 1);
    TTI_SFPLOADI(p_sfpu::LREG0, 2, 0xABCD); // LREG0 = ZeroExtend(0xABCD)
    TTI_SFPSTORE(p_sfpu::LREG0, 4, 0, 0);   // MOD0_FMT_INT32 -> DEST addr 0
}

static __attribute__((noinline)) void launch_arm()
{
    TTI_REPLAY(0, 2, 0, 0); // launch slots 0..1
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SFPU1);
    TTI_NOP;
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const DataFormat l1_fmt = static_cast<DataFormat>(formats.unpack_A_src);

    // Opaque runtime value the compiler cannot fold: buffer_Res base address.
    volatile std::uint32_t gate = params.buffer_Res[0];

    if (gate != 0xFFFFFFFFu)
    {
        record_arm();
    }
    // A second opaque branch so record_arm() and launch_arm() are provably in
    // sibling arms, not a straight-line block.
    if (gate + 1u != 0u)
    {
        launch_arm();
    }

    dbg_copy_dest_tile<DbgDestTileOp::Read, MathThreadId>(l1_fmt, 0, reinterpret_cast<void*>(params.buffer_Res[0]));
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif
