// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Allocation-only reproducer; the raw consumer is an assembly comment.
// Compile -O2 -mcpu=tt-bh-tensix -S, with/without -DUSE_EFFECT.
// Existing read/write annotations at the consumer do not protect the gap:
// the temporary load/store can overwrite L0 before the annotation executes.
// USE_EFFECT must keep that temporary off L0. No silicon result is claimed.
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
    // SFPSETCC may disable lanes, which then retain the old L0, and SFPMOV
    // reads L0 again once SFPENCC has re-enabled them.
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
    __builtin_rvtt_sfprawlreg_effect(1, 0);
#endif
#endif
}
