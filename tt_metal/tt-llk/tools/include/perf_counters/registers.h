// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/types.h"

#if defined(ARCH_QUASAR)
#error "Quasar counter registers are not in tt-llk yet"
#elif !defined(ARCH_BLACKHOLE) && !defined(ARCH_WORMHOLE) && !defined(ARCH_WORMHOLE_B0)
#error "perf_counters/registers.h needs ARCH_BLACKHOLE, ARCH_WORMHOLE or ARCH_WORMHOLE_B0"
#endif

namespace llk::perf
{

// The three control registers of a bank and its two readout registers. control takes START / STOP,
// out_l is the reference (cycle) count, out_h the count of the selected event.
struct BankRegs
{
    std::uint32_t ref_period;
    std::uint32_t mode;
    std::uint32_t control;
    std::uint32_t out_l;
    std::uint32_t out_h;
};

// tt-1xx debug register block; Wormhole and Blackhole use the same offsets.
inline constexpr std::uint32_t DEBUG_REGS_BASE     = 0xFFB12000;
inline constexpr std::uint32_t PERF_CNT_ALL        = DEBUG_REGS_BASE + 0x3C;
inline constexpr std::uint32_t DBG_FEATURE_DISABLE = DEBUG_REGS_BASE + 0x68;
inline constexpr std::uint32_t PERF_CNT_MUX_CTRL   = DEBUG_REGS_BASE + 0x218;

namespace detail
{

// Indexed by Bank.
inline constexpr BankRegs BANK_REGS[NUM_BANKS] = {
    {DEBUG_REGS_BASE + 0x0, DEBUG_REGS_BASE + 0x4, DEBUG_REGS_BASE + 0x8, DEBUG_REGS_BASE + 0x100, DEBUG_REGS_BASE + 0x104},    // INSTRN_THREAD
    {DEBUG_REGS_BASE + 0x18, DEBUG_REGS_BASE + 0x1C, DEBUG_REGS_BASE + 0x20, DEBUG_REGS_BASE + 0x120, DEBUG_REGS_BASE + 0x124}, // FPU
    {DEBUG_REGS_BASE + 0xC, DEBUG_REGS_BASE + 0x10, DEBUG_REGS_BASE + 0x14, DEBUG_REGS_BASE + 0x108, DEBUG_REGS_BASE + 0x10C},  // TDMA_UNPACK
    {DEBUG_REGS_BASE + 0x30, DEBUG_REGS_BASE + 0x34, DEBUG_REGS_BASE + 0x38, DEBUG_REGS_BASE + 0x118, DEBUG_REGS_BASE + 0x11C}, // L1
    {DEBUG_REGS_BASE + 0xF0, DEBUG_REGS_BASE + 0xF4, DEBUG_REGS_BASE + 0xF8, DEBUG_REGS_BASE + 0x110, DEBUG_REGS_BASE + 0x114}, // TDMA_PACK
};

} // namespace detail

// A reference into the table, so a runtime bank index costs one load per field that is used.
constexpr const BankRegs& bank_regs(Bank bank)
{
    return detail::BANK_REGS[static_cast<std::uint8_t>(bank)];
}

// control register bits
inline constexpr std::uint32_t START = 1;
inline constexpr std::uint32_t STOP  = 2;
// mode register: low byte is the count mode, the select sits above it
inline constexpr std::uint32_t MODE_CONTINUOUS = 0;
inline constexpr std::uint32_t SELECT_SHIFT    = 8;
// PERF_CNT_MUX_CTRL: the L1 mux field starts at bit 4 (its width is the arch L1_MUX_MASK)
inline constexpr std::uint32_t L1_MUX_SHIFT       = 4;
inline constexpr std::uint32_t REF_PERIOD_MAX     = 0xFFFFFFFF;
inline constexpr std::uint32_t DEFAULT_POLL_LIMIT = 1024;

// PERF_CNT_ALL reaches only the INSTRN_THREAD and FPU blocks (RTL confirmed); the others start and stop
// from their own control register.
constexpr bool follows_all(Bank bank)
{
    return bank == Bank::INSTRN_THREAD || bank == Bank::FPU;
}

// When the arch tensix.h is visible, check every address against it.
#ifdef RISCV_DEBUG_REG_PERF_CNT_FPU0
static_assert(bank_regs(Bank::INSTRN_THREAD).ref_period == RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0);
static_assert(bank_regs(Bank::INSTRN_THREAD).mode == RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD1);
static_assert(bank_regs(Bank::INSTRN_THREAD).control == RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD2);
static_assert(bank_regs(Bank::INSTRN_THREAD).out_l == RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD);
static_assert(bank_regs(Bank::INSTRN_THREAD).out_h == RISCV_DEBUG_REG_PERF_CNT_OUT_H_INSTRN_THREAD);
static_assert(bank_regs(Bank::FPU).ref_period == RISCV_DEBUG_REG_PERF_CNT_FPU0);
static_assert(bank_regs(Bank::FPU).mode == RISCV_DEBUG_REG_PERF_CNT_FPU1);
static_assert(bank_regs(Bank::FPU).control == RISCV_DEBUG_REG_PERF_CNT_FPU2);
static_assert(bank_regs(Bank::FPU).out_l == RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU);
static_assert(bank_regs(Bank::FPU).out_h == RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU);
static_assert(bank_regs(Bank::TDMA_UNPACK).ref_period == RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0);
static_assert(bank_regs(Bank::TDMA_UNPACK).mode == RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK1);
static_assert(bank_regs(Bank::TDMA_UNPACK).control == RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2);
static_assert(bank_regs(Bank::TDMA_UNPACK).out_l == RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK);
static_assert(bank_regs(Bank::TDMA_UNPACK).out_h == RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_UNPACK);
static_assert(bank_regs(Bank::L1).ref_period == RISCV_DEBUG_REG_PERF_CNT_L1_0);
static_assert(bank_regs(Bank::L1).mode == RISCV_DEBUG_REG_PERF_CNT_L1_1);
static_assert(bank_regs(Bank::L1).control == RISCV_DEBUG_REG_PERF_CNT_L1_2);
static_assert(bank_regs(Bank::L1).out_l == RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1);
static_assert(bank_regs(Bank::L1).out_h == RISCV_DEBUG_REG_PERF_CNT_OUT_H_DBG_L1);
static_assert(bank_regs(Bank::TDMA_PACK).ref_period == RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0);
static_assert(bank_regs(Bank::TDMA_PACK).mode == RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK1);
static_assert(bank_regs(Bank::TDMA_PACK).control == RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK2);
static_assert(bank_regs(Bank::TDMA_PACK).out_l == RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK);
static_assert(bank_regs(Bank::TDMA_PACK).out_h == RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_PACK);
static_assert(PERF_CNT_ALL == RISCV_DEBUG_REG_PERF_CNT_ALL);
static_assert(PERF_CNT_MUX_CTRL == RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
static_assert(DBG_FEATURE_DISABLE == RISCV_DEBUG_REG_DBG_FEATURE_DISABLE);
#endif

} // namespace llk::perf
