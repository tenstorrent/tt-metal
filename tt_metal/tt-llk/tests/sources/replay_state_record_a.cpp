// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// lane FS  FP-3 architectural model, experiment 1 ("kernel A").
//
// Records a distinctive 2-instruction payload into replay slots 0..1 with a
// NO-EXEC record (TT_REPLAY Load=1, Exec=0), then EXITS without ever launching
// the slots.  Because Exec=0, neither recorded instruction executes here, so
// this kernel never writes DEST (DEST stays whatever ZEROACC left it: zero).
//
// The recorded payload, if later replayed (by kernel B), loads the sentinel
// 0x0000ABCD into LREG0 across all lanes and SFPSTOREs it (INT32 mode) to
// DEST rows 0..3.  Kernel B never records; it only launches slot 0.  If B's
// DEST readback contains the sentinel, the per-thread Replay-Expander buffer
// PERSISTED across the TRISC soft-reset + ELF reload between the two kernel
// invocations.

#include <cstdint>

#include "build.h"
#include "ckernel.h"
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
    // Record slots 0..1: NO-EXEC (execute_while_loading=0), Load=1.
    TTI_REPLAY(0, 2, 0, 1);
    // Payload word 0: LREG0 = ZeroExtend(0xABCD)  (SFPLOADI_MOD0_USHORT = 2)
    TTI_SFPLOADI(p_sfpu::LREG0, 2, 0xABCD);
    // Payload word 1: SFPSTORE LREG0 -> DEST, MOD0_FMT_INT32 = 4, addr 0.
    TTI_SFPSTORE(p_sfpu::LREG0, 4, 0, 0);
    // No launch.  Exit with slots 0..1 armed but un-executed.
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif
