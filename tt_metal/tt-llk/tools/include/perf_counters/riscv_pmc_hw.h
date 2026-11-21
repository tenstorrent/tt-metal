// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(ARCH_WORMHOLE) || defined(ARCH_WORMHOLE_B0)
#error "Wormhole RISC-V cores implement no performance counter CSRs"
#endif

namespace llk::perf::riscv
{

// Machine-mode CSR addresses, the same on every core that has the monitor.
inline constexpr std::uint32_t CSR_MCYCLE       = 0xB00;
inline constexpr std::uint32_t CSR_MINSTRET     = 0xB02;
inline constexpr std::uint32_t CSR_MHPMCOUNTER3 = 0xB03;
inline constexpr std::uint32_t CSR_MHPMCOUNTER4 = 0xB04;
inline constexpr std::uint32_t CSR_MHPMCOUNTER5 = 0xB05;
inline constexpr std::uint32_t CSR_MHPMCOUNTER6 = 0xB06;
inline constexpr std::uint32_t CSR_MHPMEVENT3   = 0x323;
inline constexpr std::uint32_t CSR_MHPMEVENT4   = 0x324;
inline constexpr std::uint32_t CSR_MHPMEVENT5   = 0x325;
inline constexpr std::uint32_t CSR_MHPMEVENT6   = 0x326;

// mcountinhibit is not written: its reset value runs the counters.
inline void select(std::uint32_t event3, std::uint32_t event4)
{
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT3), "r"(event3));
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT4), "r"(event4));
}

// Counters 5 and 6 exist on the Quasar data-movement cores only; elsewhere they read zero and ignore writes.
inline void select_upper(std::uint32_t event5, std::uint32_t event6)
{
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT5), "r"(event5));
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT6), "r"(event6));
}

inline void zero_counters()
{
    asm volatile("csrw %0, zero" ::"i"(CSR_MHPMCOUNTER3));
    asm volatile("csrw %0, zero" ::"i"(CSR_MHPMCOUNTER4));
}

inline void zero_counters_upper()
{
    asm volatile("csrw %0, zero" ::"i"(CSR_MHPMCOUNTER5));
    asm volatile("csrw %0, zero" ::"i"(CSR_MHPMCOUNTER6));
}

// The memory clobber keeps the reads from moving across the code being measured.
inline std::uint32_t read_mcycle()
{
    std::uint32_t value;
    asm volatile("csrr %0, %1" : "=r"(value) : "i"(CSR_MCYCLE) : "memory");
    return value;
}

inline std::uint32_t read_minstret()
{
    std::uint32_t value;
    asm volatile("csrr %0, %1" : "=r"(value) : "i"(CSR_MINSTRET) : "memory");
    return value;
}

inline std::uint32_t read_counter3()
{
    std::uint32_t value;
    asm volatile("csrr %0, %1" : "=r"(value) : "i"(CSR_MHPMCOUNTER3) : "memory");
    return value;
}

inline std::uint32_t read_counter4()
{
    std::uint32_t value;
    asm volatile("csrr %0, %1" : "=r"(value) : "i"(CSR_MHPMCOUNTER4) : "memory");
    return value;
}

inline std::uint32_t read_counter5()
{
    std::uint32_t value;
    asm volatile("csrr %0, %1" : "=r"(value) : "i"(CSR_MHPMCOUNTER5) : "memory");
    return value;
}

inline std::uint32_t read_counter6()
{
    std::uint32_t value;
    asm volatile("csrr %0, %1" : "=r"(value) : "i"(CSR_MHPMCOUNTER6) : "memory");
    return value;
}

// The mcycle and minstret a window starts from. Both free-run and are never written, so a window is a difference.
struct Reference
{
    std::uint32_t cycles;
    std::uint32_t instret;
};

// Selects the events and zeroes the counters they count into.
__attribute__((always_inline)) inline void arm(std::uint32_t event3, std::uint32_t event4)
{
    select(event3, event4);
    zero_counters();
}

__attribute__((always_inline)) inline void arm_upper(std::uint32_t event5, std::uint32_t event6)
{
    select_upper(event5, event6);
    zero_counters_upper();
}

__attribute__((always_inline)) inline Reference reference()
{
    return Reference {read_mcycle(), read_minstret()};
}

// What a window counted: the two programmable counters and the mcycle and minstret deltas, modulo 2^32. The counters are
// read first and mcycle and minstret last, so the window ends as late as the reads allow. Results come back through
// references, not a struct, because a struct result is kept in memory by the firmware build.
__attribute__((always_inline)) inline void sample(
    const Reference& start, std::uint32_t& pmc3, std::uint32_t& pmc4, std::uint32_t& cycles, std::uint32_t& instret)
{
    pmc3    = read_counter3();
    pmc4    = read_counter4();
    cycles  = read_mcycle() - start.cycles;
    instret = read_minstret() - start.instret;
}

// Counters 5 and 6 of a window, Quasar data-movement cores.
__attribute__((always_inline)) inline void sample_upper(std::uint32_t& pmc5, std::uint32_t& pmc6)
{
    pmc5 = read_counter5();
    pmc6 = read_counter6();
}

} // namespace llk::perf::riscv
