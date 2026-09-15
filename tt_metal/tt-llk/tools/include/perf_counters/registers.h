// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/types.h"

#if !defined(ARCH_BLACKHOLE) && !defined(ARCH_WORMHOLE) && !defined(ARCH_WORMHOLE_B0) && !defined(ARCH_QUASAR)
#error "perf_counters/registers.h needs ARCH_BLACKHOLE, ARCH_WORMHOLE, ARCH_WORMHOLE_B0 or ARCH_QUASAR"
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

// PERF_CNT_ALL reaches only the INSTRN_THREAD and FPU blocks (RTL confirmed on every arch); the others start
// and stop from their own control register.
constexpr bool follows_all(Bank bank)
{
    return bank == Bank::INSTRN_THREAD || bank == Bank::FPU;
}

#if defined(ARCH_QUASAR)

// Quasar: every NEO has its own debug block. A TRISC reaches its NEO's block through the local window; the DM
// cores reach every NEO through the NoC window, NEO n at neo_window(n). The window is the base of the block.
inline constexpr std::uint32_t LOCAL_REGS_WINDOW  = 0x00800000;
inline constexpr std::uint32_t NEO_WINDOW_BASE    = 0x01800000;
inline constexpr std::uint32_t NEO_WINDOW_STRIDE  = 0x10000;
inline constexpr std::uint32_t NUM_NEOS           = 4;
inline constexpr std::uint32_t DEFAULT_WINDOW     = LOCAL_REGS_WINDOW;

constexpr std::uint32_t neo_window(unsigned neo)
{
    return NEO_WINDOW_BASE + neo * NEO_WINDOW_STRIDE;
}

// Same placement rule as the tables in quasar.h (this header may be included first).
#ifndef LLK_PERF_TABLE_SECTION
#if defined(LLK_PERF_TABLES_IN_TEXT)
#define LLK_PERF_TABLE_SECTION __attribute__((section(".text.perf_counter_tables")))
#else
#define LLK_PERF_TABLE_SECTION
#endif
#endif

namespace detail
{

// Offsets inside a window, indexed by Bank. Quasar has no L1 counter bank; that row is never read.
// Not `inline`: LTO drops the section attribute of COMDAT variables.
constexpr BankRegs BANK_OFFSETS[NUM_BANKS] LLK_PERF_TABLE_SECTION = {
    {0x0, 0x4, 0x8, 0x98, 0x9C},    // INSTRN_THREAD
    {0x18, 0x1C, 0x20, 0xB0, 0xB4}, // FPU
    {0xC, 0x10, 0x14, 0xA0, 0xA4},  // TDMA_UNPACK
    {0, 0, 0, 0, 0},                // L1 (absent)
    {0x8C, 0x90, 0x94, 0xA8, 0xAC}, // TDMA_PACK
};

inline constexpr std::uint32_t PERF_CNT_ALL_OFFSET        = 0x24;
inline constexpr std::uint32_t DBG_FEATURE_DISABLE_OFFSET = 0x40;
inline constexpr std::uint32_t PERF_CNT_MUX_CTRL_OFFSET   = 0xF4;
// l1_client CSR pair in the same window (the l1_client block sits at +0xA000).
inline constexpr std::uint32_t L1_CLIENT_CTRL_OFFSET = 0xA0AC;
inline constexpr std::uint32_t L1_CLIENT_CNT_OFFSET  = 0xA0B0;

} // namespace detail

// The L1 bank does not exist: all-zero registers, so callers can skip it by testing control.
constexpr BankRegs bank_regs(Bank bank, std::uint32_t window = DEFAULT_WINDOW)
{
    if (bank == Bank::L1)
    {
        return BankRegs {};
    }
    const BankRegs& o = detail::BANK_OFFSETS[static_cast<std::uint8_t>(bank)];
    return BankRegs {window + o.ref_period, window + o.mode, window + o.control, window + o.out_l, window + o.out_h};
}

constexpr std::uint32_t perf_cnt_all(std::uint32_t window = DEFAULT_WINDOW)
{
    return window + detail::PERF_CNT_ALL_OFFSET;
}

constexpr std::uint32_t dbg_feature_disable(std::uint32_t window = DEFAULT_WINDOW)
{
    return window + detail::DBG_FEATURE_DISABLE_OFFSET;
}

constexpr std::uint32_t perf_cnt_mux_ctrl(std::uint32_t window = DEFAULT_WINDOW)
{
    return window + detail::PERF_CNT_MUX_CTRL_OFFSET;
}

// Local-window addresses, for code that runs on a TRISC.
inline constexpr std::uint32_t PERF_CNT_ALL        = perf_cnt_all();
inline constexpr std::uint32_t DBG_FEATURE_DISABLE = dbg_feature_disable();
inline constexpr std::uint32_t PERF_CNT_MUX_CTRL   = perf_cnt_mux_ctrl();

// The l1_client event counter: one clear-on-read CSR behind a subport/event mux set in ctrl.
struct L1ClientRegs
{
    std::uint32_t ctrl;
    std::uint32_t cnt;
};

constexpr L1ClientRegs l1_client_regs(std::uint32_t window = DEFAULT_WINDOW)
{
    return L1ClientRegs {window + detail::L1_CLIENT_CTRL_OFFSET, window + detail::L1_CLIENT_CNT_OFFSET};
}

// ctrl fields: bit 0 enable, [9:4] subport, [14:12] event
inline constexpr std::uint32_t L1_CLIENT_ENABLE        = 1;
inline constexpr std::uint32_t L1_CLIENT_SUBPORT_SHIFT = 4;
inline constexpr std::uint32_t L1_CLIENT_EVENT_SHIFT   = 12;

// When the generated tensix_neo_reg.h is visible, check every NEO0 NoC-window address against it.
#ifdef NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD0_REG_ADDR
static_assert(bank_regs(Bank::INSTRN_THREAD, neo_window(0)).ref_period == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD0_REG_ADDR);
static_assert(bank_regs(Bank::INSTRN_THREAD, neo_window(0)).mode == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD1_REG_ADDR);
static_assert(bank_regs(Bank::INSTRN_THREAD, neo_window(0)).control == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD2_REG_ADDR);
static_assert(bank_regs(Bank::INSTRN_THREAD, neo_window(0)).out_l == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_INSTRN_THREAD_REG_ADDR);
static_assert(bank_regs(Bank::INSTRN_THREAD, neo_window(0)).out_h == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_H_INSTRN_THREAD_REG_ADDR);
static_assert(bank_regs(Bank::FPU, neo_window(0)).ref_period == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_FPU0_REG_ADDR);
static_assert(bank_regs(Bank::FPU, neo_window(0)).mode == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_FPU1_REG_ADDR);
static_assert(bank_regs(Bank::FPU, neo_window(0)).control == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_FPU2_REG_ADDR);
static_assert(bank_regs(Bank::FPU, neo_window(0)).out_l == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_FPU_REG_ADDR);
static_assert(bank_regs(Bank::FPU, neo_window(0)).out_h == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_H_FPU_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_UNPACK, neo_window(0)).ref_period == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_UNPACK0_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_UNPACK, neo_window(0)).mode == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_UNPACK1_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_UNPACK, neo_window(0)).control == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_UNPACK2_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_UNPACK, neo_window(0)).out_l == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_TDMA_UNPACK_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_UNPACK, neo_window(0)).out_h == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_H_TDMA_UNPACK_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_PACK, neo_window(0)).ref_period == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_PACK0_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_PACK, neo_window(0)).mode == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_PACK1_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_PACK, neo_window(0)).control == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_PACK2_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_PACK, neo_window(0)).out_l == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_TDMA_PACK_REG_ADDR);
static_assert(bank_regs(Bank::TDMA_PACK, neo_window(0)).out_h == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_H_TDMA_PACK_REG_ADDR);
static_assert(perf_cnt_all(neo_window(0)) == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_ALL_REG_ADDR);
static_assert(dbg_feature_disable(neo_window(0)) == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_DBG_FEATURE_DISABLE_REG_ADDR);
static_assert(perf_cnt_mux_ctrl(neo_window(0)) == NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_MUX_CTRL_REG_ADDR);
static_assert(l1_client_regs(neo_window(0)).ctrl == NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CTRL_REG_ADDR);
static_assert(l1_client_regs(neo_window(0)).cnt == NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CNT_REG_ADDR);
static_assert(neo_window(0) == NEO_REGS_0__LOCAL_REGS_REG_MAP_BASE_ADDR);
static_assert(neo_window(1) == NEO_REGS_1__LOCAL_REGS_REG_MAP_BASE_ADDR);
static_assert(neo_window(2) == NEO_REGS_2__LOCAL_REGS_REG_MAP_BASE_ADDR);
static_assert(neo_window(3) == NEO_REGS_3__LOCAL_REGS_REG_MAP_BASE_ADDR);
static_assert(l1_client_regs(neo_window(1)).ctrl == NEO_REGS_1__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CTRL_REG_ADDR);
#endif
// tt_t6_trisc_map.h names the TRISC-local window LOCAL_REGS_BASE.
#ifdef LOCAL_REGS_BASE
static_assert(LOCAL_REGS_WINDOW == LOCAL_REGS_BASE);
#endif

#else // tt-1xx

// tt-1xx debug register block; Wormhole and Blackhole use the same offsets.
inline constexpr std::uint32_t DEBUG_REGS_BASE     = 0xFFB12000;
inline constexpr std::uint32_t DEFAULT_WINDOW      = DEBUG_REGS_BASE;
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

// The same block at another base; tt-1xx has one block, so this only exists for arch-neutral callers.
constexpr BankRegs bank_regs(Bank bank, std::uint32_t window)
{
    const BankRegs& r = bank_regs(bank);
    return BankRegs {
        r.ref_period - DEBUG_REGS_BASE + window,
        r.mode - DEBUG_REGS_BASE + window,
        r.control - DEBUG_REGS_BASE + window,
        r.out_l - DEBUG_REGS_BASE + window,
        r.out_h - DEBUG_REGS_BASE + window};
}

constexpr std::uint32_t perf_cnt_all(std::uint32_t window = DEFAULT_WINDOW)
{
    return PERF_CNT_ALL - DEBUG_REGS_BASE + window;
}

constexpr std::uint32_t dbg_feature_disable(std::uint32_t window = DEFAULT_WINDOW)
{
    return DBG_FEATURE_DISABLE - DEBUG_REGS_BASE + window;
}

constexpr std::uint32_t perf_cnt_mux_ctrl(std::uint32_t window = DEFAULT_WINDOW)
{
    return PERF_CNT_MUX_CTRL - DEBUG_REGS_BASE + window;
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

#endif // ARCH_QUASAR

} // namespace llk::perf
