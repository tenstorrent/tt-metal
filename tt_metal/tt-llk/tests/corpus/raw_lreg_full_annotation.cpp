// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Allocation experiment, not a device test. SCHEME=0: unannotated;
// SCHEME=1: Nathan's before-input/after-output read/write pairs;
// SCHEME=2: effect metadata; SCHEME=3: explicit C++ value threading.
// Both raw producer and consumer are real words.
// This does not implement or substitute for the proposed sfpvalue builtin.
#ifndef SCHEME
#error "select SCHEME=0, 1, 2, or 3"
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
    __builtin_rvtt_sfprawlreg_effect(1, 1);
#elif SCHEME == 3
    auto saved = __builtin_rvtt_sfpreadlreg(0);
#endif
    auto temporary = __builtin_rvtt_sfpload(nullptr, 1, 0, 0, 0, 0);
    __builtin_rvtt_sfpstore(nullptr, temporary, 2, 0, 0, 0, 0);
#if SCHEME == 1
    MARK(0); // Input annotation immediately before the consumer.
#elif SCHEME == 3
    __builtin_rvtt_sfpwritelreg(saved, 0);
#endif
    ISSUE(0x72000000u); // SFPSTORE L0, destination 0.
#if SCHEME == 2
    __builtin_rvtt_sfprawlreg_effect(1, 0);
#endif
}
