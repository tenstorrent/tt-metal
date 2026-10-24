// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compile-only allocation check, not a device test. The raw L0 value written
// by the SFPLOADI word must stay in L0 until the SFPSTORE word reads it, even
// though a typed load/store temporary is allocated in between. SCHEME=2 and
// SCHEME=3 must keep that temporary off L0; SCHEME=0 and 1 are reported only.
//   SCHEME=0: no annotation.
//   SCHEME=1: LREG read/write builtins after the producer and before the consumer.
//   SCHEME=2: raw-LREG effect metadata.
//   SCHEME=3: the value carried through C++ (USE_SFPI_API: via sfpi::l_reg).
#ifndef SCHEME
#error "select SCHEME=0, 1, 2, or 3"
#endif
#if SCHEME == 2
#define TT_LLK_REQUIRE_RAW_LREG_METADATA
#include "../../common/ckernel_lreg_metadata.h"
#endif
#ifdef USE_SFPI_API
namespace ckernel { extern volatile unsigned long instrn_buffer[]; }
#include <sfpi.h>
static_assert(SCHEME == 3, "public API comparison requires threaded state");
#endif
#define MARK(R) __builtin_rvtt_sfpwritelreg(__builtin_rvtt_sfpreadlreg(R), R)
#if defined(USE_MMIO)
#define ISSUE(word) (*issue = (word))
#else
#define ISSUE(word) asm volatile (".ttinsn %0" : : "n"(word))
#endif

void raw_producer_consumer(volatile unsigned *issue)
{
    // SFPLOADI L0, 0, 123. Unknown lane state conservatively reads old L0.
    ISSUE(0x7100007bu);
#if SCHEME == 1
    MARK(0); // Output annotation immediately after the producer.
#elif SCHEME == 2
    TT_LLK_SFPRAWLREG_EFFECT(1, 1);
#elif SCHEME == 3
#ifdef USE_SFPI_API
    sfpi::vFloat saved = sfpi::l_reg[sfpi::LRegs::LReg0];
#else
    auto saved = __builtin_rvtt_sfpreadlreg(0);
#endif
#endif
    // Parentheses retain the builtin ABI when sfpi.h defines arity macros.
    auto temporary = (__builtin_rvtt_sfpload)(nullptr, 1, 0, 0, 0, 0);
    (__builtin_rvtt_sfpstore)(nullptr, temporary, 2, 0, 0, 0, 0);
#if SCHEME == 1
    MARK(0); // Input annotation immediately before the consumer.
#elif SCHEME == 3
#ifdef USE_SFPI_API
    sfpi::l_reg[sfpi::LRegs::LReg0] = saved;
#else
    __builtin_rvtt_sfpwritelreg(saved, 0);
#endif
#endif
    ISSUE(0x72000000u); // SFPSTORE L0, destination 0.
#if SCHEME == 2
    TT_LLK_SFPRAWLREG_EFFECT(1, 0);
#endif
}
