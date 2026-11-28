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

// The memory clobber keeps the reads from moving across the code being measured.
template <std::uint32_t csr>
inline std::uint32_t read_csr()
{
    std::uint32_t value;
    asm volatile("csrr %0, %1" : "=r"(value) : "i"(csr) : "memory");
    return value;
}

template <std::uint32_t csr>
inline void write_csr(std::uint32_t value)
{
    asm volatile("csrw %0, %z1" ::"i"(csr), "rJ"(value));
}

// mcountinhibit is not written: its reset value runs the counters.
inline void select(std::uint32_t event3, std::uint32_t event4)
{
    write_csr<CSR_MHPMEVENT3>(event3);
    write_csr<CSR_MHPMEVENT4>(event4);
}

// Counters 5 and 6 exist on the Quasar data-movement cores only; elsewhere they read zero and ignore writes.
inline void select_upper(std::uint32_t event5, std::uint32_t event6)
{
    write_csr<CSR_MHPMEVENT5>(event5);
    write_csr<CSR_MHPMEVENT6>(event6);
}

inline void zero_counters()
{
    write_csr<CSR_MHPMCOUNTER3>(0);
    write_csr<CSR_MHPMCOUNTER4>(0);
}

inline void zero_counters_upper()
{
    write_csr<CSR_MHPMCOUNTER5>(0);
    write_csr<CSR_MHPMCOUNTER6>(0);
}

} // namespace llk::perf::riscv
