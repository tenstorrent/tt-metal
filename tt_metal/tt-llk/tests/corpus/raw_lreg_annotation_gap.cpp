// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Allocation-only reproducer; the raw consumer is an assembly comment.
// Compile -O2 -mcpu=tt-bh-tensix -S, with/without -DUSE_EFFECT.
// Existing read/write annotations at the consumer do not protect the gap:
// the temporary load/store can overwrite L0 before the annotation executes.
// USE_EFFECT must keep that temporary off L0. No silicon result is claimed.
#if defined(USE_EFFECT)
#define TT_LLK_REQUIRE_RAW_LREG_METADATA
#include "../../common/ckernel_lreg_metadata.h"
#endif
#if defined(USE_MACRO)
namespace ckernel { inline volatile unsigned long instrn_buffer[1]; }
#include "../../tt_llk_blackhole/common/inc/ckernel_ops.h"
#endif
void raw_input_gap()
{
    auto seed = __builtin_rvtt_sfpload(nullptr, 0, 0, 0, 0, 0);
    __builtin_rvtt_sfpwritelreg(seed, 0);
    auto temporary = __builtin_rvtt_sfpload(nullptr, 1, 0, 0, 0, 0);
    __builtin_rvtt_sfpstore(nullptr, temporary, 0, 0, 0, 0, 0);
#if defined(USE_MACRO)
    // Disable lanes, partially overwrite L0, then expose its retained lanes.
    // A compiler that owns lane merging must infer that old L0 is still needed.
    TTI_SFPSETCC(0, 1, 0, 0);
    TTI_SFPLOADI(0, 0, 123);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPMOV(0, 0, 2, 0);
#else
#if !defined(USE_EFFECT)
    __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpreadlreg(0), 0);
#endif
    asm volatile ("# RAW INPUT L0");
#if defined(USE_EFFECT)
    TT_LLK_SFPRAWLREG_EFFECT(1, 0);
#endif
#endif
}
