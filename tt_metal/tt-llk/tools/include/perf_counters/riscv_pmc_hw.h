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
inline constexpr std::uint32_t CSR_MHPMEVENT3   = 0x323;
inline constexpr std::uint32_t CSR_MHPMEVENT4   = 0x324;

// mcountinhibit is not written: its reset value runs both counters.
inline void select(std::uint32_t event3, std::uint32_t event4)
{
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT3), "r"(event3));
    asm volatile("csrw %0, %1" ::"i"(CSR_MHPMEVENT4), "r"(event4));
}

inline void zero_counters()
{
    asm volatile("csrw %0, zero" ::"i"(CSR_MHPMCOUNTER3));
    asm volatile("csrw %0, zero" ::"i"(CSR_MHPMCOUNTER4));
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

} // namespace llk::perf::riscv
