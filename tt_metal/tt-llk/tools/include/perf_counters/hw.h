// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "perf_counters/inventory.h"
#include "perf_counters/registers.h"

namespace llk::perf
{

inline void write(std::uint32_t addr, std::uint32_t value)
{
    *reinterpret_cast<volatile std::uint32_t*>(addr) = value;
}

inline std::uint32_t read(std::uint32_t addr)
{
    return *reinterpret_cast<volatile std::uint32_t*>(addr);
}

inline void compiler_fence()
{
    asm volatile("" ::: "memory");
}

// Read-modify-write: the low bits of MUX_CTRL hold the INSTRN_THREAD debug-bus select.
inline void set_l1_mux(std::uint8_t position)
{
    const std::uint32_t field = (static_cast<std::uint32_t>(position) << L1_MUX_SHIFT) & L1_MUX_MASK;
    write(PERF_CNT_MUX_CTRL, (read(PERF_CNT_MUX_CTRL) & ~L1_MUX_MASK) | field);
}

inline void clear_debug_feature_disable()
{
    write(DBG_FEATURE_DISABLE, 0);
}

// Free-running count with the reference period at its maximum.
inline void configure(const BankRegs& regs)
{
    write(regs.ref_period, REF_PERIOD_MAX);
    write(regs.mode, MODE_CONTINUOUS);
}

// The hardware acts on the rising edge, so the bit is cleared first. Start also zeroes the counts.
inline void start(const BankRegs& regs)
{
    const std::uint32_t control = regs.control;
    write(control, 0);
    write(control, START);
}

inline void stop(const BankRegs& regs)
{
    const std::uint32_t control = regs.control;
    write(control, 0);
    write(control, STOP);
}

inline void start_all()
{
    write(PERF_CNT_ALL, START);
}

inline void stop_all()
{
    write(PERF_CNT_ALL, STOP);
}

// Route one select to the bank's readout, then poll the mode register back so the next read sees the
// new selection. PollLimit 0 polls without a bound and always returns true (the BRISC firmware is a few
// bytes from its size limit); otherwise returns false when PollLimit reads never matched.
template <std::uint32_t PollLimit = DEFAULT_POLL_LIMIT>
inline bool select(const BankRegs& regs, std::uint16_t sel)
{
    const std::uint32_t mode = (static_cast<std::uint32_t>(sel) << SELECT_SHIFT) | MODE_CONTINUOUS;
    write(regs.mode, mode);
    if constexpr (PollLimit == 0)
    {
        while (read(regs.mode) != mode)
        {
        }
        return true;
    }
    else
    {
        for (std::uint32_t spin = 0; spin < PollLimit; ++spin)
        {
            if (read(regs.mode) == mode)
            {
                return true;
            }
        }
        return false;
    }
}

inline std::uint32_t read_ref(const BankRegs& regs)
{
    return read(regs.out_l);
}

inline std::uint32_t read_count(const BankRegs& regs)
{
    return read(regs.out_h);
}

// Read every entry of a table: emit(PerfCounterType, ref, count) once per select.
template <std::uint32_t PollLimit = DEFAULT_POLL_LIMIT, class Emit>
inline void read_table(const BankRegs& regs, Table table, Emit&& emit)
{
    for (std::size_t i = 0; i < table.size; ++i)
    {
        select<PollLimit>(regs, table.data[i].second);
        const std::uint32_t ref   = read_ref(regs);
        const std::uint32_t count = read_count(regs);
        emit(table.data[i].first, ref, count);
    }
}

} // namespace llk::perf
