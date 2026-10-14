// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compile-only check that the raw-LREG helper calls a real target builtin.
#include "../../common/ckernel_lreg_metadata.h"

#ifdef TEST_REQUIRE_ACCESS_ONLY_COMPILER
#if !defined(__has_builtin) || !__has_builtin(__builtin_rvtt_sfprawlreg_access) || \
    __has_builtin(__builtin_rvtt_sfprawlreg_effect)
#error "this negative test requires an access-only compiler"
#endif
#endif

void raw_lreg_metadata_abi()
{
    TT_LLK_SFPRAWLREG_EFFECT(1u, 2u);
}

// The raw L0 value must survive the intervening typed temporary. On an
// access-only compiler, the helper is intentionally a no-op compatibility
// path; only the effect compiler promises to reserve L0 across this gap.
void raw_lreg_metadata_lifetime()
{
    asm volatile (".ttinsn %0" : : "n"(0x7100007bu));
    TT_LLK_SFPRAWLREG_EFFECT(1u, 1u);
    auto temporary = __builtin_rvtt_sfpload(nullptr, 1, 0, 0, 0, 0);
    __builtin_rvtt_sfpstore(nullptr, temporary, 2, 0, 0, 0, 0);
    asm volatile (".ttinsn %0" : : "n"(0x72000000u));
    TT_LLK_SFPRAWLREG_EFFECT(1u, 0u);
}
