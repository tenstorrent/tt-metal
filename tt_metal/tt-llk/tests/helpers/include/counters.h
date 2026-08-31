// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "perf.h" // the PERF_COUNTERS_* L1 region constants

#ifdef PERF_COUNTERS_COMPILED

#include "ckernel.h"
#include "profiler.h" // the zone/timestamp layer (TRISC only)

// BRISC builds the config only; the per-zone measurement layer below also needs LLK_PROFILER.

#ifdef ARCH_QUASAR
#error "Perf counters are not supported on Quasar yet (no Quasar hw_counters.h; untested register set)."
#endif

// Include order matters: hw_counters.h uses PerfCounterType, which perf_counters.hpp defines.
#include <array>
// clang-format off
#include "perf_counters.hpp"
#include "hw_counters.h"
// clang-format on

namespace llk_perf
{

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

// On-wire bank IDs; the order is a contract with base_addrs[], banks[] and the host.
enum class counter_bank : std::uint8_t
{
    instrn_thread = 0,
    fpu           = 1,
    tdma_unpack   = 2,
    l1            = 3,
    tdma_pack     = 4,
};

constexpr std::uint32_t COUNTER_BANK_COUNT = 5;

// Unbounded, a corrupt config word would hang every thread and surface only as TENSIX TIMED OUT.
constexpr std::uint32_t MODE_REG_POLL_LIMIT = 1024;
constexpr std::uint32_t COUNTER_SLOT_COUNT = PERF_COUNTERS_CONFIG_WORDS;

constexpr std::uint32_t PERF_CFG_VALID_BIT     = 1u << 31; // bit 31: slot active
constexpr std::uint32_t PERF_CFG_L1_MUX_SHIFT  = 17;       // bits 19:17
constexpr std::uint32_t PERF_CFG_L1_MUX_MASK   = 0x7u;
constexpr std::uint32_t PERF_CFG_COUNTER_SHIFT = 8; // bits 16:8 (9-bit counter_sel)
constexpr std::uint32_t PERF_CFG_COUNTER_MASK  = 0x1FFu;
constexpr std::uint32_t PERF_CFG_BANK_MASK     = 0xFFu; // bits 7:0

// hw_counters.h is the authority and L1_MUX_MASK arrives already shifted.
constexpr std::uint32_t PERF_CNT_MUX_CTRL_SHIFT = 4;
constexpr std::uint32_t PERF_CNT_MUX_CTRL_MASK  = L1_MUX_MASK;
constexpr std::uint32_t PERF_L1_MUX_MAX         = PERF_CNT_MUX_CTRL_MASK >> PERF_CNT_MUX_CTRL_SHIFT;

constexpr std::uint32_t _perf_cfg(std::uint8_t bank, std::uint16_t cid, std::uint8_t mux = 0)
{
    return PERF_CFG_VALID_BIT | (static_cast<std::uint32_t>(mux & PERF_CFG_L1_MUX_MASK) << PERF_CFG_L1_MUX_SHIFT) |
           (static_cast<std::uint32_t>(cid & PERF_CFG_COUNTER_MASK) << PERF_CFG_COUNTER_SHIFT) | static_cast<std::uint32_t>(bank);
}

// The volatile index stops GCC emitting a CSWTCH table, which shifts GP offsets and breaks NC/WC .text equality.
inline std::uint32_t get_counter_base_addr(counter_bank bank)
{
    static constexpr std::uint32_t base_addrs[COUNTER_BANK_COUNT] = {
        RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0,
        RISCV_DEBUG_REG_PERF_CNT_FPU0,
        RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0,
        RISCV_DEBUG_REG_PERF_CNT_L1_0,
        RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0,
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

constexpr std::uint32_t l1_group_size(std::uint8_t mux)
{
    return mux == 0   ? l1_0_counters.size()
           : mux == 1 ? l1_1_counters.size()
           : mux == 2 ? l1_2_counters.size()
           : mux == 3 ? l1_3_counters.size()
           : mux == 4 ? l1_4_counters.size()
                      : 0u;
}

static_assert(L1_MUX_GROUP <= PERF_L1_MUX_MAX, "LLK_PERF_L1_MUX_GROUP does not fit this architecture's PERF_CNT_MUX_CTRL mux field");
static_assert(l1_group_size(L1_MUX_GROUP) > 0, "LLK_PERF_L1_MUX_GROUP selects an L1 mux group this architecture does not expose");

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
    return cfg;
}

static_assert(L1_MUX_GROUP <= 4, "LLK_PERF_L1_MUX_GROUP has no emitter in build_builtin_config()");

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
        if (bank == counter_bank::l1)
        {
            const std::uint8_t l1_mux = (metadata >> PERF_CFG_L1_MUX_SHIFT) & PERF_CFG_L1_MUX_MASK;
            std::uint32_t cur         = ckernel::reg_read(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
            ckernel::reg_write(
                RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL,
                (cur & ~PERF_CNT_MUX_CTRL_MASK) | ((static_cast<std::uint32_t>(l1_mux) << PERF_CNT_MUX_CTRL_SHIFT) & PERF_CNT_MUX_CTRL_MASK));
        }
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
        ckernel::reg_write(counter_base + 8, 1);
        ckernel::reg_write(counter_base + 8, 0);
    }
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_ALL, 1);
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_ALL, 0);
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
        ckernel::reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 0);
        configure_hardware();
        arm_hardware();
    }
}

// Write shared config to L1, clear per-zone data, then configure + arm hw.
inline void configure_and_arm_from_brisc()
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

namespace detail
{
// Placed last via its own section (see sections.ld) so adding it cannot move other globals.
__attribute__((section(".perf_counters_bss"))) static std::uint32_t zone_hashes[PERF_COUNTERS_MAX_ZONES];
__attribute__((section(".perf_counters_bss"))) static std::uint32_t next_zone_id;

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
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_ALL, 1u);
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2, 1u);
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_L1_2, 1u);
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK2, 1u);
    ckernel::fence_compiler();
}

inline __attribute__((always_inline)) void freeze_and_read_all_counters(std::uint32_t zone_id)
{
    ckernel::fence_compiler();
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_ALL, 2u);
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2, 2u);
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_L1_2, 2u);
    ckernel::reg_write(RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK2, 2u);

    struct bank_regs
    {
        std::uint32_t mode_reg;
        std::uint32_t out_l;
    };

    // Per-bank readout pair: mode_reg drives counter_sel; out_l is the bank's
    // OUT_L (shared cycles); OUT_H sits at out_l + 4 and is sampled per slot.
    static constexpr bank_regs banks[5] = {
        {RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD},
        {RISCV_DEBUG_REG_PERF_CNT_FPU1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU},
        {RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK},
        {RISCV_DEBUG_REG_PERF_CNT_L1_1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1},
        {RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK1, RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK},
    };

    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    volatile std::uint32_t* bank_cycles    = reinterpret_cast<volatile std::uint32_t*>(cycles_base);
    volatile std::uint32_t* counter_counts = bank_cycles + PERF_COUNTERS_BANK_CYCLES_WORDS;
    std::uint32_t shared_cycles            = ckernel::reg_read(banks[0].out_l);
    bank_cycles[0]                         = shared_cycles;
    bank_cycles[1]                         = shared_cycles;
    bank_cycles[2]                         = shared_cycles;
    bank_cycles[3]                         = shared_cycles;
    bank_cycles[4]                         = shared_cycles;

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

// L1_TO_L1/L1_CONGESTION: unpack arms (pipeline source), pack freezes (sink). ISOLATE: same thread arms and freezes.
template <PerfRunType run_type>
constexpr bool is_arm_thread()
{
#if defined(LLK_TRISC_UNPACK)
    return run_type == PerfRunType::L1_TO_L1 || run_type == PerfRunType::L1_CONGESTION || run_type == PerfRunType::UNPACK_ISOLATE;
#elif defined(LLK_TRISC_MATH)
    return run_type == PerfRunType::MATH_ISOLATE;
#elif defined(LLK_TRISC_PACK)
    return run_type == PerfRunType::PACK_ISOLATE;
#else
    return false;
#endif
}

template <PerfRunType run_type>
constexpr bool is_freeze_thread()
{
#if defined(LLK_TRISC_UNPACK)
    return run_type == PerfRunType::UNPACK_ISOLATE;
#elif defined(LLK_TRISC_MATH)
    return run_type == PerfRunType::MATH_ISOLATE;
#elif defined(LLK_TRISC_PACK)
    return run_type == PerfRunType::L1_TO_L1 || run_type == PerfRunType::L1_CONGESTION || run_type == PerfRunType::PACK_ISOLATE;
#else
    return false;
#endif
}

// pc_buf-semaphore barriers: arm/freeze thread sempost ×N, non-active threads spinwait+semget.
constexpr std::uint8_t PERF_ENTRY_SEM        = ckernel::semaphore::FPU_SFPU;
constexpr std::uint8_t PERF_EXIT_SEM         = ckernel::semaphore::UNPACK_TO_DEST;
constexpr std::uint32_t PERF_NUM_SPINWAITERS = 2;

template <PerfRunType run_type>
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
        if constexpr (is_arm_thread<run_type>())
        {
            arm_all_counters();
            for (std::uint32_t i = 0; i < PERF_NUM_SPINWAITERS; ++i)
            {
                ckernel::semaphore_post(PERF_ENTRY_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_ENTRY_SEM) == 0)
            {
                asm volatile("nop");
            }
            ckernel::semaphore_get(PERF_ENTRY_SEM);
        }
        ckernel::fence_compiler();
    }

    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        ckernel::fence_compiler();
        if constexpr (is_freeze_thread<run_type>())
        {
            freeze_and_read_all_counters(zone_id);
            for (std::uint32_t i = 0; i < PERF_NUM_SPINWAITERS; ++i)
            {
                ckernel::semaphore_post(PERF_EXIT_SEM);
            }
        }
        else
        {
            while (ckernel::semaphore_read(PERF_EXIT_SEM) == 0)
            {
                asm volatile("nop");
            }
            ckernel::semaphore_get(PERF_EXIT_SEM);
        }
        ckernel::fence_compiler();
    }
};
#endif // LLK_PROFILER

} // namespace llk_perf

#define PERF_COUNTER_VAR_CONCAT_(a, b) a##b
#define PERF_COUNTER_VAR_(line)        PERF_COUNTER_VAR_CONCAT_(_perf_ctr_, line)
#if defined(LLK_PROFILER)
#define MEASURE_PERF_COUNTERS(zone_name) \
    const llk_perf::perf_counter_scoped<PERF_RUN_TYPE> PERF_COUNTER_VAR_(__LINE__)(llk_perf::get_zone_id(llk_perf::detail::zone_name_hash(zone_name)));
#else
#define MEASURE_PERF_COUNTERS(zone_name)
#endif

#else // !PERF_COUNTERS_COMPILED

#define MEASURE_PERF_COUNTERS(zone_name)

// NC build stubs — let callers invoke unconditionally; compiler folds these to nothing.
namespace llk_perf
{
inline void configure_and_arm_from_brisc()
{
}
} // namespace llk_perf

#endif // PERF_COUNTERS_COMPILED

// One measured scope: NC activates timing only, WC both.
#define START_PERF_MEASURE(zone_name) \
    MEASURE_PERF_COUNTERS(zone_name)  \
    ZONE_SCOPED(zone_name)
