// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compile-only allocation experiment, not a correctness or performance test.
// Build with -DLIVE_REGS={2,4,7,8} -DSCHEME={2,3} and optional -DUSE_MMIO.
// Scheme 2 describes raw effects; scheme 3 carries explicit C++ values.
// Eight preserved values plus a temporary may require unsupported spilling:
// compiler rejection is a capacity result, never a correctness pass.
// Example (a Tensix-enabled target is required):
// riscv-tt-elf-g++ -mcpu=tt-bh-tensix -std=c++17 -O2 -S -DSCHEME=3
//   -DLIVE_REGS=7 -fdisable-rtl-rvtt_lreg_livein
//   raw_lreg_threading_pressure.cpp -o pressure.s (join these three lines)
// For scheme 2 omit the pass-disable flag. Optional scheduling stress:
// -fschedule-insns -fschedule-insns2. Inspect return code before assembly.
#ifndef LIVE_REGS
#error "select LIVE_REGS=2, 4, 7, or 8"
#endif
#ifndef SCHEME
#error "select SCHEME=2 or 3"
#endif
static_assert(LIVE_REGS == 2 || LIVE_REGS == 4 || LIVE_REGS == 7 || LIVE_REGS == 8);
static_assert(SCHEME == 2 || SCHEME == 3);

template <unsigned Word>
__attribute__((always_inline)) inline void issue_word(volatile unsigned *issue)
{
#ifdef USE_MMIO
    *issue = Word;
#else
    asm volatile (".ttinsn %0" : : "n"(Word));
#endif
}

template <unsigned Register>
__attribute__((always_inline)) inline auto produce(volatile unsigned *issue)
{
    // SFPLOADI Register, FLOATB, a distinct finite immediate per register.
    issue_word<0x71000000u | (Register << 20) | (0x3f80u + Register * 0x80u)>(issue);
    if constexpr (SCHEME == 2) {
        __builtin_rvtt_sfprawlreg_effect(1u << Register, 1u << Register);
        return 0; // No additional fixed-register reads in the effects arm.
    } else {
        return __builtin_rvtt_sfpreadlreg(Register);
    }
}

template <unsigned Register, typename Value>
__attribute__((always_inline)) inline void consume(volatile unsigned *issue, Value value)
{
    if constexpr (SCHEME == 3)
        __builtin_rvtt_sfpwritelreg(value, Register);
    // SFPSTORE Register, SRCB, address-mod 7, distinct destination row.
    issue_word<0x72000000u | (Register << 20) | (7u << 13) | (64u + 2u * Register)>(issue);
    if constexpr (SCHEME == 2)
        __builtin_rvtt_sfprawlreg_effect(1u << Register, 0);
}

void raw_lreg_threading_pressure(volatile unsigned *issue)
{
    auto v0 = produce<0>(issue);
    auto v1 = produce<1>(issue);
    // Conditional declarations remain in this scope so all selected values
    // overlap the typed temporary's lifetime. Unselected values are unused.
    decltype(v0) v2, v3, v4, v5, v6, v7;
    if constexpr (LIVE_REGS >= 4) {
        v2 = produce<2>(issue);
        v3 = produce<3>(issue);
    }
    if constexpr (LIVE_REGS >= 7) {
        v4 = produce<4>(issue);
        v5 = produce<5>(issue);
        v6 = produce<6>(issue);
    }
    if constexpr (LIVE_REGS == 8) v7 = produce<7>(issue);

    auto temporary = __builtin_rvtt_sfpload(nullptr, 1, 0, 0, 0, 0);
    __builtin_rvtt_sfpstore(nullptr, temporary, 2, 0, 0, 0, 0);

    consume<0>(issue, v0);
    consume<1>(issue, v1);
    if constexpr (LIVE_REGS >= 4) {
        consume<2>(issue, v2);
        consume<3>(issue, v3);
    }
    if constexpr (LIVE_REGS >= 7) {
        consume<4>(issue, v4);
        consume<5>(issue, v5);
        consume<6>(issue, v6);
    }
    if constexpr (LIVE_REGS == 8) consume<7>(issue, v7);
}
