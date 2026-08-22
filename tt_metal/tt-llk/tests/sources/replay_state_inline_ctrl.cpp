// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// lane FS  FP-3 model, vehicle control.
//
// Executes the sentinel SFPLOADI+SFPSTORE payload INLINE (no replay at all),
// then reads DEST tile 0 back through the debug window.  This validates the
// store-address / readback alignment of the experiment vehicle independently
// of the Replay Expander.  If this shows the sentinel but the replay variants
// do not, the store/read path is sound and the replay result is meaningful.

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

    // Inline (executed) payload, no replay.
    TTI_SFPLOADI(p_sfpu::LREG0, 2, 0xABCD); // LREG0 = ZeroExtend(0xABCD)
    TTI_SFPSTORE(p_sfpu::LREG0, 4, 0, 0);   // MOD0_FMT_INT32 -> DEST addr 0
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SFPU1);
    TTI_NOP;

    dbg_copy_dest_tile<DbgDestTileOp::Read, MathThreadId>(l1_fmt, 0, reinterpret_cast<void*>(params.buffer_Res[0]));
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    // idle
}

#endif
