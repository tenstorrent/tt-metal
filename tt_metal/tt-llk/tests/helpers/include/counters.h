// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "barrier.h"
#include "perf.h" // the PERF_COUNTERS_* L1 region constants

#ifdef PERF_COUNTERS_COMPILED

#include "ckernel.h"
#include "profiler.h" // the zone/timestamp layer (TRISC only)

// BRISC builds the config only; the per-zone measurement layer below also needs LLK_PROFILER.
// Quasar has no BRISC in this harness: the unpack TRISC does the one-time setup before it releases the others.

// Include order matters: hw_counters.h uses PerfCounterType, which perf_counters.hpp defines.
#include <array>
// clang-format off
#include "perf_counters.hpp"
// Metal's kernel build gives the Quasar tables a named section; here they are folded at compile time.
#ifndef PERF_COUNTER_TABLE
#define PERF_COUNTER_TABLE
#endif
#include "hw_counters.h"
// clang-format on
#if defined(ARCH_QUASAR)
#include "tensix_neo_reg.h"
#endif

namespace llk_perf
{

// Register map. tt-1xx names come from tensix.h. Quasar's hw_counters.h names NEO0's NoC window; a TRISC reaches its
// own NEO's registers through the local window at the same offsets, so they are rebased onto LOCAL_REGS_BASE here.
namespace reg
{
#if defined(ARCH_QUASAR)
constexpr std::uint32_t local(std::uint32_t neo0_addr)
{
    return neo0_addr - NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD0_REG_ADDR + LOCAL_REGS_BASE;
}

constexpr std::uint32_t INSTRN_THREAD0      = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD0_REG_ADDR);
constexpr std::uint32_t INSTRN_THREAD1      = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_INSTRN_THREAD1_REG_ADDR);
constexpr std::uint32_t FPU0                = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_FPU0_REG_ADDR);
constexpr std::uint32_t FPU1                = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_FPU1_REG_ADDR);
constexpr std::uint32_t TDMA_UNPACK0        = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_UNPACK0_REG_ADDR);
constexpr std::uint32_t TDMA_UNPACK1        = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_UNPACK1_REG_ADDR);
constexpr std::uint32_t TDMA_UNPACK2        = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_UNPACK2_REG_ADDR);
constexpr std::uint32_t TDMA_PACK0          = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_PACK0_REG_ADDR);
constexpr std::uint32_t TDMA_PACK1          = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_PACK1_REG_ADDR);
constexpr std::uint32_t TDMA_PACK2          = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_TDMA_PACK2_REG_ADDR);
constexpr std::uint32_t PERF_CNT_ALL        = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_ALL_REG_ADDR);
constexpr std::uint32_t DBG_FEATURE_DISABLE = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_DBG_FEATURE_DISABLE_REG_ADDR);
constexpr std::uint32_t OUT_L_INSTRN_THREAD = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_INSTRN_THREAD_REG_ADDR);
constexpr std::uint32_t OUT_L_FPU           = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_FPU_REG_ADDR);
constexpr std::uint32_t OUT_L_TDMA_UNPACK   = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_TDMA_UNPACK_REG_ADDR);
constexpr std::uint32_t OUT_L_TDMA_PACK     = local(NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_PERF_CNT_OUT_L_TDMA_PACK_REG_ADDR);
// Quasar has no L1 counter bank; bank slot 3 carries the l1_client CSR instead (see counter_bank).
constexpr std::uint32_t L1_0           = 0;
constexpr std::uint32_t L1_1           = 0;
constexpr std::uint32_t L1_2           = 0;
constexpr std::uint32_t OUT_L_DBG_L1   = 0;
constexpr std::uint32_t MUX_CTRL       = 0;
constexpr std::uint32_t L1_CLIENT_CTRL = local(NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CTRL_REG_ADDR);
constexpr std::uint32_t L1_CLIENT_CNT  = local(NEO_REGS_0__LOCAL_REGS_L1_CLIENT_GROUP_PERF_CNT_REG_ADDR);
#else
constexpr std::uint32_t INSTRN_THREAD0      = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0;
constexpr std::uint32_t INSTRN_THREAD1      = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD1;
constexpr std::uint32_t FPU0                = RISCV_DEBUG_REG_PERF_CNT_FPU0;
constexpr std::uint32_t FPU1                = RISCV_DEBUG_REG_PERF_CNT_FPU1;
constexpr std::uint32_t TDMA_UNPACK0        = RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0;
constexpr std::uint32_t TDMA_UNPACK1        = RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK1;
constexpr std::uint32_t TDMA_UNPACK2        = RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2;
constexpr std::uint32_t TDMA_PACK0          = RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0;
constexpr std::uint32_t TDMA_PACK1          = RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK1;
constexpr std::uint32_t TDMA_PACK2          = RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK2;
constexpr std::uint32_t PERF_CNT_ALL        = RISCV_DEBUG_REG_PERF_CNT_ALL;
constexpr std::uint32_t DBG_FEATURE_DISABLE = RISCV_DEBUG_REG_DBG_FEATURE_DISABLE;
constexpr std::uint32_t OUT_L_INSTRN_THREAD = RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD;
constexpr std::uint32_t OUT_L_FPU           = RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU;
constexpr std::uint32_t OUT_L_TDMA_UNPACK   = RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK;
constexpr std::uint32_t OUT_L_TDMA_PACK     = RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK;
constexpr std::uint32_t L1_0                = RISCV_DEBUG_REG_PERF_CNT_L1_0;
constexpr std::uint32_t L1_1                = RISCV_DEBUG_REG_PERF_CNT_L1_1;
constexpr std::uint32_t L1_2                = RISCV_DEBUG_REG_PERF_CNT_L1_2;
constexpr std::uint32_t OUT_L_DBG_L1        = RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1;
constexpr std::uint32_t MUX_CTRL            = RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL;
#endif
} // namespace reg

constexpr std::uint32_t PERF_COUNTERS_MAX_ZONES = 8;
constexpr std::uint32_t SYNC_ZONE_COMPLETE      = 0xFFu; // written after readout; host polls for it

constexpr std::uint32_t PERF_COUNTERS_ZONE_DATA_BYTES = (PERF_COUNTERS_BANK_CYCLES_WORDS + PERF_COUNTERS_DATA_WORDS) * 4;
constexpr std::uint32_t PERF_COUNTERS_ZONE_SIZE       = PERF_COUNTERS_ZONE_DATA_BYTES + 40;

constexpr std::uint32_t PERF_COUNTERS_SHARED_CONFIG_ADDR = PERF_COUNTERS_BASE_ADDR;
constexpr std::uint32_t PERF_COUNTERS_ZONES_BASE         = PERF_COUNTERS_BASE_ADDR + PERF_COUNTERS_CONFIG_WORDS * 4;

constexpr std::uint32_t perf_counters_zone_data_addr(std::uint32_t zone)
{
    return PERF_COUNTERS_ZONES_BASE + zone * PERF_COUNTERS_ZONE_SIZE;
}

// +0 holds the SYNC_ZONE_COMPLETE flag the host polls.
constexpr std::uint32_t perf_counters_sync_ctrl_addr(std::uint32_t zone)
{
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
}

constexpr std::uint32_t PERF_COUNTERS_ENABLED_FLAG_ADDR = PERF_COUNTERS_ZONES_BASE + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE;
constexpr std::uint32_t PERF_COUNTERS_BANK_MASK_ADDR    = PERF_COUNTERS_ENABLED_FLAG_ADDR + 4;
constexpr std::uint32_t PERF_COUNTERS_VALID_COUNT_ADDR  = PERF_COUNTERS_BANK_MASK_ADDR + 4;
constexpr std::uint32_t PERF_COUNTERS_LAYOUT_END        = PERF_COUNTERS_VALID_COUNT_ADDR + PERF_COUNTERS_MAX_ZONES * 4;

// A literal because BRISC has no llk_profiler namespace; the LLK_PROFILER section asserts it symbolically.
static_assert(PERF_COUNTERS_LAYOUT_END <= 0x16AFF0u, "Perf counter L1 layout overflows into the profiler region");

// On-wire bank IDs; the order is a contract with base_addrs[], banks[] and the host. On Quasar, which has no L1
// counter bank, slot 3 carries the l1_client event CSR: its counter_sel field holds the subport*8+event selection.
enum class counter_bank : std::uint8_t
{
    instrn_thread = 0,
    fpu           = 1,
    tdma_unpack   = 2,
    l1            = 3,
    tdma_pack     = 4,
};

#if defined(ARCH_QUASAR)
// -1 leaves the l1_client CSR out; otherwise subport*8 + event, as TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL in metal.
#ifndef LLK_PERF_L1_CLIENT_SEL
#define LLK_PERF_L1_CLIENT_SEL (-1)
#endif
constexpr int L1_CLIENT_SEL                = LLK_PERF_L1_CLIENT_SEL;
constexpr bool L1_CLIENT_ENABLED           = L1_CLIENT_SEL >= 0;
constexpr std::uint32_t L1_CLIENT_SUBPORTS = QUASAR_L1_CLIENT_NUM_SUBPORTS;
constexpr std::uint32_t L1_CLIENT_EVENTS   = QUASAR_L1_CLIENT_NUM_EVENTS;
static_assert(
    !L1_CLIENT_ENABLED || L1_CLIENT_SEL < static_cast<int>(L1_CLIENT_SUBPORTS * L1_CLIENT_EVENTS), "LLK_PERF_L1_CLIENT_SEL is subport*8 + event, below 296");
static_assert(!L1_CLIENT_ENABLED || L1_CLIENT_SEL % static_cast<int>(L1_CLIENT_EVENTS) != 0, "l1_client event 0 is unused in the RTL");
static_assert(
    !L1_CLIENT_ENABLED || !(L1_CLIENT_SEL / static_cast<int>(L1_CLIENT_EVENTS) == 4 && L1_CLIENT_SEL % static_cast<int>(L1_CLIENT_EVENTS) <= 3),
    "THCON events 1-3 read the TRISC port's SBank 0 counters; select them through sub-port 0");

constexpr std::uint32_t l1_client_ctrl_word()
{
    return (static_cast<std::uint32_t>(L1_CLIENT_SEL / static_cast<int>(L1_CLIENT_EVENTS)) << QUASAR_L1_CLIENT_PERF_SUBPORT_SHIFT) |
           (static_cast<std::uint32_t>(L1_CLIENT_SEL % static_cast<int>(L1_CLIENT_EVENTS)) << QUASAR_L1_CLIENT_PERF_EVENT_SHIFT) |
           QUASAR_L1_CLIENT_PERF_CTRL_ENABLE;
}
#endif

constexpr std::uint32_t COUNTER_BANK_COUNT = 5;

// Unbounded, a corrupt config word would hang every thread and surface only as TENSIX TIMED OUT.
constexpr std::uint32_t MODE_REG_POLL_LIMIT = 1024;
constexpr std::uint32_t COUNTER_SLOT_COUNT  = PERF_COUNTERS_CONFIG_WORDS;

constexpr std::uint32_t PERF_CFG_VALID_BIT     = 1u << 31; // bit 31: slot active
constexpr std::uint32_t PERF_CFG_L1_MUX_SHIFT  = 17;       // bits 19:17
constexpr std::uint32_t PERF_CFG_L1_MUX_MASK   = 0x7u;
constexpr std::uint32_t PERF_CFG_COUNTER_SHIFT = 8; // bits 16:8 (9-bit counter_sel)
constexpr std::uint32_t PERF_CFG_COUNTER_MASK  = 0x1FFu;
constexpr std::uint32_t PERF_CFG_BANK_MASK     = 0xFFu; // bits 7:0

// hw_counters.h is the authority and L1_MUX_MASK arrives already shifted. Quasar has no L1 mux.
constexpr std::uint32_t PERF_CNT_MUX_CTRL_SHIFT = 4;
#if defined(ARCH_QUASAR)
constexpr std::uint32_t PERF_CNT_MUX_CTRL_MASK = 0;
#else
constexpr std::uint32_t PERF_CNT_MUX_CTRL_MASK = L1_MUX_MASK;
#endif
constexpr std::uint32_t PERF_L1_MUX_MAX = PERF_CNT_MUX_CTRL_MASK >> PERF_CNT_MUX_CTRL_SHIFT;

constexpr std::uint32_t _perf_cfg(std::uint8_t bank, std::uint16_t cid, std::uint8_t mux = 0)
{
    return PERF_CFG_VALID_BIT | (static_cast<std::uint32_t>(mux & PERF_CFG_L1_MUX_MASK) << PERF_CFG_L1_MUX_SHIFT) |
           (static_cast<std::uint32_t>(cid & PERF_CFG_COUNTER_MASK) << PERF_CFG_COUNTER_SHIFT) | static_cast<std::uint32_t>(bank);
}

// The volatile index stops GCC emitting a CSWTCH table, which shifts GP offsets and breaks NC/WC .text equality.
inline std::uint32_t get_counter_base_addr(counter_bank bank)
{
    static constexpr std::uint32_t base_addrs[COUNTER_BANK_COUNT] = {
        reg::INSTRN_THREAD0,
        reg::FPU0,
        reg::TDMA_UNPACK0,
        reg::L1_0,
        reg::TDMA_PACK0,
    };
    static_assert(
        static_cast<std::uint32_t>(counter_bank::tdma_pack) == COUNTER_BANK_COUNT - 1, "counter_bank enumerators must be contiguous 0..COUNTER_BANK_COUNT-1");
    volatile auto b = static_cast<std::uint32_t>(bank);
    return b < COUNTER_BANK_COUNT ? base_addrs[b] : 0u;
}

// Only 8 physical L1 counters exist, and the mux selects which group feeds them while they count,
// not at read time, so a run sees one group and the groups have to be swept across runs.
#ifndef LLK_PERF_L1_MUX_GROUP
#define LLK_PERF_L1_MUX_GROUP 0
#endif

constexpr std::uint8_t L1_MUX_GROUP = LLK_PERF_L1_MUX_GROUP;

#if defined(ARCH_QUASAR)
// Slot 3 holds the one l1_client selection when it is enabled.
constexpr std::uint32_t l1_group_size(std::uint8_t)
{
    return L1_CLIENT_ENABLED ? 1u : 0u;
}
#else
constexpr std::uint32_t l1_group_size(std::uint8_t mux)
{
    return mux == 0   ? l1_0_counters.size()
           : mux == 1 ? l1_1_counters.size()
           : mux == 2 ? l1_2_counters.size()
           : mux == 3 ? l1_3_counters.size()
           : mux == 4 ? l1_4_counters.size()
           : mux == 5 ? l1_5_counters.size()
                      : 0u;
}

static_assert(L1_MUX_GROUP <= PERF_L1_MUX_MAX, "LLK_PERF_L1_MUX_GROUP does not fit this architecture's PERF_CNT_MUX_CTRL mux field");
static_assert(l1_group_size(L1_MUX_GROUP) > 0, "LLK_PERF_L1_MUX_GROUP selects an L1 mux group this architecture does not expose");
#endif

constexpr std::uint32_t builtin_counter_count()
{
    return instrn_counters.size() + fpu_counters.size() + unpack_counters.size() + pack_counters.size() + l1_group_size(L1_MUX_GROUP);
}

// Fixed order, matched by the readout: INSTRN, FPU, TDMA_UNPACK, TDMA_PACK, selected L1 group.
constexpr std::array<std::uint32_t, builtin_counter_count()> build_builtin_config()
{
    std::array<std::uint32_t, builtin_counter_count()> cfg {};
    std::uint32_t k = 0;
    const auto emit = [&](const auto& arr, counter_bank bank, std::uint8_t mux)
    {
        for (const auto& entry : arr)
        {
            cfg[k++] = _perf_cfg(static_cast<std::uint8_t>(bank), entry.second, mux);
        }
    };
    emit(instrn_counters, counter_bank::instrn_thread, 0);
    emit(fpu_counters, counter_bank::fpu, 0);
    emit(unpack_counters, counter_bank::tdma_unpack, 0);
    emit(pack_counters, counter_bank::tdma_pack, 0);
#if defined(ARCH_QUASAR)
    if constexpr (L1_CLIENT_ENABLED)
    {
        cfg[k++] = _perf_cfg(static_cast<std::uint8_t>(counter_bank::l1), static_cast<std::uint16_t>(L1_CLIENT_SEL), 0);
    }
#else
    if constexpr (L1_MUX_GROUP == 0)
    {
        emit(l1_0_counters, counter_bank::l1, 0);
    }
    else if constexpr (L1_MUX_GROUP == 1)
    {
        emit(l1_1_counters, counter_bank::l1, 1);
    }
    else if constexpr (L1_MUX_GROUP == 2)
    {
        emit(l1_2_counters, counter_bank::l1, 2);
    }
    else if constexpr (L1_MUX_GROUP == 3)
    {
        emit(l1_3_counters, counter_bank::l1, 3);
    }
    else if constexpr (L1_MUX_GROUP == 4)
    {
        emit(l1_4_counters, counter_bank::l1, 4);
    }
    else if constexpr (L1_MUX_GROUP == 5)
    {
        emit(l1_5_counters, counter_bank::l1, 5);
    }
#endif
    return cfg;
}

static_assert(L1_MUX_GROUP <= 5, "LLK_PERF_L1_MUX_GROUP has no emitter in build_builtin_config()");

constexpr auto BUILTIN_COUNTER_CONFIG         = build_builtin_config();
constexpr std::uint32_t BUILTIN_COUNTER_COUNT = BUILTIN_COUNTER_CONFIG.size();

static_assert(BUILTIN_COUNTER_COUNT <= COUNTER_SLOT_COUNT, "Counter inventory overflows the shared config region into zone 0 data");

inline std::uint32_t get_active_bank_mask()
{
    return *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_BANK_MASK_ADDR);
}

inline void configure_hardware()
{
    const volatile std::uint32_t* config_mem = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
    std::uint32_t configured_mask            = 0;

    for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; i++)
    {
        const std::uint32_t metadata = config_mem[i];
        if ((metadata & PERF_CFG_VALID_BIT) == 0)
        {
            continue;
        }
        const std::uint8_t bank_id   = static_cast<std::uint8_t>(metadata & PERF_CFG_BANK_MASK);
        const std::uint32_t bank_bit = 1u << bank_id;
        if (configured_mask & bank_bit)
        {
            continue;
        }
        const counter_bank bank = static_cast<counter_bank>(bank_id);
#if defined(ARCH_QUASAR)
        if (bank == counter_bank::l1)
        {
            // The l1_client CSR: route the selection, then read once so the clear-on-read count starts at zero.
            if constexpr (L1_CLIENT_ENABLED)
            {
                ckernel::reg_write(reg::L1_CLIENT_CTRL, l1_client_ctrl_word());
                (void)ckernel::reg_read(reg::L1_CLIENT_CNT);
            }
            configured_mask |= bank_bit;
            continue;
        }
#else
        if (bank == counter_bank::l1)
        {
            const std::uint8_t l1_mux = (metadata >> PERF_CFG_L1_MUX_SHIFT) & PERF_CFG_L1_MUX_MASK;
            std::uint32_t cur         = ckernel::reg_read(reg::MUX_CTRL);
            ckernel::reg_write(
                reg::MUX_CTRL, (cur & ~PERF_CNT_MUX_CTRL_MASK) | ((static_cast<std::uint32_t>(l1_mux) << PERF_CNT_MUX_CTRL_SHIFT) & PERF_CNT_MUX_CTRL_MASK));
        }
#endif
        std::uint32_t counter_base = get_counter_base_addr(bank);
        ckernel::reg_write(counter_base, 0xFFFFFFFF);
        ckernel::reg_write(counter_base + 4, 0);
        configured_mask |= bank_bit;
    }
}

inline void arm_hardware()
{
    for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
    {
        if (!(get_active_bank_mask() & (1u << b)))
        {
            continue;
        }
        std::uint32_t counter_base = get_counter_base_addr(static_cast<counter_bank>(b));
        if (counter_base == 0)
        {
            continue; // Quasar slot 3: the l1_client CSR has no start/stop register
        }
        ckernel::reg_write(counter_base + 8, 1);
        ckernel::reg_write(counter_base + 8, 0);
    }
    ckernel::reg_write(reg::PERF_CNT_ALL, 1);
    ckernel::reg_write(reg::PERF_CNT_ALL, 0);
}

inline void configure_all_zones()
{
    // One config covers every zone, so scan once: the per-zone scan re-read it 8 times on BRISC.
    bool found_valid        = false;
    std::uint32_t bank_mask = 0;
    std::uint32_t count     = 0;

    const volatile std::uint32_t* config_mem = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
    for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; i++)
    {
        const std::uint32_t metadata = config_mem[i];
        if (metadata & PERF_CFG_VALID_BIT)
        {
            found_valid = true;
            count++;
            bank_mask |= (1u << (metadata & PERF_CFG_BANK_MASK));
        }
    }

    *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_ENABLED_FLAG_ADDR) = found_valid ? 1u : 0u;
    *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_BANK_MASK_ADDR)    = bank_mask;
    volatile std::uint32_t* valid_count_ptr                                     = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_VALID_COUNT_ADDR);
    for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
    {
        valid_count_ptr[zone] = count;
    }

    if (found_valid)
    {
        ckernel::reg_write(reg::DBG_FEATURE_DISABLE, 0);
        configure_hardware();
        arm_hardware();
    }
}

// Write shared config to L1, clear per-zone data, then configure + arm hw. BRISC runs it on tt-1xx before it
// releases the TRISCs; on Quasar the unpack TRISC runs it before it releases the other three (trisc.cpp).
inline void configure_and_arm()
{
    volatile std::uint32_t* shared_config = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
    for (std::uint32_t i = 0; i < BUILTIN_COUNTER_COUNT; i++)
    {
        shared_config[i] = BUILTIN_COUNTER_CONFIG[i];
    }
    for (std::uint32_t i = BUILTIN_COUNTER_COUNT; i < COUNTER_SLOT_COUNT; i++)
    {
        shared_config[i] = 0;
    }

    for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
    {
        volatile std::uint32_t* data_mem = reinterpret_cast<volatile std::uint32_t*>(perf_counters_zone_data_addr(zone));
        for (std::uint32_t i = 0; i < PERF_COUNTERS_BANK_CYCLES_WORDS + PERF_COUNTERS_DATA_WORDS; i++)
        {
            data_mem[i] = 0;
        }
        volatile std::uint32_t* sync_mem = reinterpret_cast<volatile std::uint32_t*>(perf_counters_sync_ctrl_addr(zone));
        for (std::uint32_t i = 0; i < 10; i++)
        {
            sync_mem[i] = 0;
        }
    }

    configure_all_zones();
}

inline void configure_and_arm_from_brisc()
{
    configure_and_arm();
}

namespace detail
{
static std::uint32_t zone_hashes[PERF_COUNTERS_MAX_ZONES];
static std::uint32_t next_zone_id;

#ifndef _LLK_PERF_ZONE_ALLOCATOR_DEFINED_
#define _LLK_PERF_ZONE_ALLOCATOR_DEFINED_

constexpr std::uint32_t zone_name_hash(const char* s)
{
    std::uint32_t h = 5381u;
    while (*s)
    {
        h = h * 33u + static_cast<std::uint32_t>(*s++);
    }
    return h ? h : 1u;
}
#endif
} // namespace detail

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    std::uint32_t n = detail::next_zone_id;
    for (std::uint32_t i = 0; i < n; ++i)
    {
        if (detail::zone_hashes[i] == hash_val)
        {
            return i;
        }
    }
    if (n < PERF_COUNTERS_MAX_ZONES)
    {
        detail::zone_hashes[n] = hash_val;
        detail::next_zone_id   = n + 1;
        return n;
    }
    return 0;
}

#if defined(LLK_PROFILER)

static_assert(PERF_COUNTERS_LAYOUT_END <= llk_profiler::EPOCH_ADDR, "Perf counter L1 layout overflows into the profiler region");

inline __attribute__((always_inline)) void arm_all_counters()
{
    ckernel::fence_compiler();
    ckernel::reg_write(reg::PERF_CNT_ALL, 1u);
    ckernel::reg_write(reg::TDMA_UNPACK2, 1u);
#if defined(ARCH_QUASAR)
    if constexpr (L1_CLIENT_ENABLED)
    {
        (void)ckernel::reg_read(reg::L1_CLIENT_CNT); // clear-on-read: the window starts here
    }
#else
    ckernel::reg_write(reg::L1_2, 1u);
#endif
    ckernel::reg_write(reg::TDMA_PACK2, 1u);
    ckernel::fence_compiler();
}

inline __attribute__((always_inline)) void freeze_and_read_all_counters(std::uint32_t zone_id)
{
    ckernel::fence_compiler();
    ckernel::reg_write(reg::PERF_CNT_ALL, 2u);
    ckernel::reg_write(reg::TDMA_UNPACK2, 2u);
#if !defined(ARCH_QUASAR)
    ckernel::reg_write(reg::L1_2, 2u);
#endif
    ckernel::reg_write(reg::TDMA_PACK2, 2u);

    struct bank_regs
    {
        std::uint32_t mode_reg;
        std::uint32_t out_l;
    };

    // Per-bank readout pair: mode_reg drives counter_sel; out_l is the bank's reference count, OUT_H (at out_l + 4)
    // the selected counter.
    static constexpr bank_regs banks[5] = {
        {reg::INSTRN_THREAD1, reg::OUT_L_INSTRN_THREAD},
        {reg::FPU1, reg::OUT_L_FPU},
        {reg::TDMA_UNPACK1, reg::OUT_L_TDMA_UNPACK},
        {reg::L1_1, reg::OUT_L_DBG_L1},
        {reg::TDMA_PACK1, reg::OUT_L_TDMA_PACK},
    };

    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    volatile std::uint32_t* bank_cycles    = reinterpret_cast<volatile std::uint32_t*>(cycles_base);
    volatile std::uint32_t* counter_counts = bank_cycles + PERF_COUNTERS_BANK_CYCLES_WORDS;
    for (std::uint32_t b = 0; b < 5; ++b)
    {
        // Quasar slot 3 has no reference counter; every bank is armed within a few cycles of INSTRN, so its count is
        // reported over the INSTRN reference.
        bank_cycles[b] = banks[b].out_l ? ckernel::reg_read(banks[b].out_l) : bank_cycles[0];
    }

    const volatile std::uint32_t* cfg = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
    std::uint32_t out_idx             = 0;
#pragma GCC unroll 0
    for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; ++i)
    {
        std::uint32_t cw = cfg[i];
        if (!(cw & PERF_CFG_VALID_BIT))
        {
            continue;
        }
        std::uint32_t bank_id    = cw & PERF_CFG_BANK_MASK;
        std::uint32_t counter_id = (cw >> PERF_CFG_COUNTER_SHIFT) & PERF_CFG_COUNTER_MASK;
        if (bank_id >= COUNTER_BANK_COUNT)
        {
            continue; // corrupt config word: do not index banks[] out of range
        }
        const bank_regs& br = banks[bank_id];
#if defined(ARCH_QUASAR)
        if (bank_id == static_cast<std::uint32_t>(counter_bank::l1))
        {
            counter_counts[out_idx] = ckernel::reg_read(reg::L1_CLIENT_CNT); // clear-on-read
            ++out_idx;
            continue;
        }
#endif
        // No mux write: it is fixed once by configure_hardware and cannot be re-aimed afterwards.
        const std::uint32_t expected_mode = counter_id << PERF_CFG_COUNTER_SHIFT;
        ckernel::reg_write(br.mode_reg, expected_mode);
        // reg_write is only a volatile store, so without this fence the read samples the previous counter.
        for (std::uint32_t spin = 0; spin < MODE_REG_POLL_LIMIT && ckernel::reg_read(br.mode_reg) != expected_mode; ++spin)
        {
        }
        counter_counts[out_idx] = ckernel::reg_read(br.out_l + 4u);
        ++out_idx;
    }

    std::uint32_t sync_addr                               = perf_counters_sync_ctrl_addr(zone_id);
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
}

constexpr bool is_single_thread_runtype(PerfRunType run_type)
{
    return run_type == PerfRunType::UNPACK_ISOLATE || run_type == PerfRunType::MATH_ISOLATE || run_type == PerfRunType::PACK_ISOLATE ||
           run_type == PerfRunType::SFPU_ISOLATE;
}

// MATH and PACK_ISOLATE freeze on the measured thread; the rest need every thread stopped first.
constexpr bool exit_barrier_for(PerfRunType run_type)
{
    return !is_single_thread_runtype(run_type) || run_type == PerfRunType::UNPACK_ISOLATE;
}

constexpr bool is_measured_thread(PerfRunType run_type)
{
#if defined(LLK_TRISC_UNPACK)
    return run_type == PerfRunType::UNPACK_ISOLATE;
#elif defined(LLK_TRISC_MATH)
    return run_type == PerfRunType::MATH_ISOLATE;
#elif defined(LLK_TRISC_PACK)
    return run_type == PerfRunType::PACK_ISOLATE;
#elif defined(LLK_TRISC_ISOLATE_SFPU)
    return run_type == PerfRunType::SFPU_ISOLATE;
#else
    return false;
#endif
}

template <PerfRunType RUN_TYPE>
struct perf_counter_scoped
{
    std::uint32_t zone_id;

    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    {
        ckernel::fence_compiler();
        llk_barrier::rendezvous(llk_barrier::is_action_thread(), [] { arm_all_counters(); });
        ckernel::fence_compiler();
    }

    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        ckernel::fence_compiler();
        const std::uint32_t zid = zone_id;
        static_assert(
            exit_barrier_for(RUN_TYPE) || RUN_TYPE == PerfRunType::MATH_ISOLATE || RUN_TYPE == PerfRunType::PACK_ISOLATE ||
                RUN_TYPE == PerfRunType::SFPU_ISOLATE,
            "a run type that skips the exit barrier needs a measured thread in is_measured_thread() to freeze the counters");
        if constexpr (!exit_barrier_for(RUN_TYPE))
        {
            if constexpr (is_measured_thread(RUN_TYPE))
            {
                freeze_and_read_all_counters(zid);
            }
        }
        else
        {
            llk_barrier::rendezvous(llk_barrier::is_action_thread(), [zid] { freeze_and_read_all_counters(zid); });
        }
        ckernel::fence_compiler();
    }
};
#endif // LLK_PROFILER

} // namespace llk_perf

#if defined(LLK_PROFILER)
#define PERF_COUNTER_VAR_CONCAT_(a, b) a##b
#define PERF_COUNTER_VAR_(line)        PERF_COUNTER_VAR_CONCAT_(_perf_ctr_, line)
#define MEASURE_PERF_COUNTERS(zone_name) \
    const llk_perf::perf_counter_scoped<PERF_RUN_TYPE> PERF_COUNTER_VAR_(__LINE__)(llk_perf::get_zone_id(llk_perf::detail::zone_name_hash(zone_name)));
#else
#define MEASURE_PERF_COUNTERS(zone_name)
#endif

#else // !PERF_COUNTERS_COMPILED

#if defined(LLK_PROFILER)
#define MEASURE_PERF_COUNTERS(zone_name) llk_barrier::rendezvous(llk_barrier::is_action_thread());
#else
#define MEASURE_PERF_COUNTERS(zone_name)
#endif

namespace llk_perf
{
inline void configure_and_arm()
{
}

inline void configure_and_arm_from_brisc()
{
}
} // namespace llk_perf

#endif // PERF_COUNTERS_COMPILED

// One measured scope: NC activates timing only, WC both.
#define START_PERF_MEASURE(zone_name) \
    MEASURE_PERF_COUNTERS(zone_name)  \
    ZONE_SCOPED(zone_name)
