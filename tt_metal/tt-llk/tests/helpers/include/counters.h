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

// Counter inventory, register map and register primitives are shared with the metal profiler. On Quasar
// bank_regs() defaults to the TRISC-local window, so every thread reaches its own NEO's block.
#include <array>

#include "perf_counters/hw.h"
#include "perf_counters/inventory.h"
#include "perf_counters/registers.h"

namespace llk_perf
{

using llk::perf::Bank;
using llk::perf::PerfCounterType;

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

// On-wire bank IDs are llk::perf::Bank; the order is a contract with the host.
constexpr std::uint32_t COUNTER_BANK_COUNT = llk::perf::NUM_BANKS;

// Unbounded, a corrupt config word would hang every thread and surface only as TENSIX TIMED OUT.
constexpr std::uint32_t MODE_REG_POLL_LIMIT = 1024;
constexpr std::uint32_t COUNTER_SLOT_COUNT  = PERF_COUNTERS_CONFIG_WORDS;

constexpr std::uint32_t PERF_CFG_VALID_BIT     = 1u << 31; // bit 31: slot active
constexpr std::uint32_t PERF_CFG_L1_MUX_SHIFT  = 17;       // bits 19:17
constexpr std::uint32_t PERF_CFG_L1_MUX_MASK   = 0x7u;
constexpr std::uint32_t PERF_CFG_COUNTER_SHIFT = 8; // bits 16:8 (9-bit counter_sel)
constexpr std::uint32_t PERF_CFG_COUNTER_MASK  = 0x1FFu;
constexpr std::uint32_t PERF_CFG_BANK_MASK     = 0xFFu; // bits 7:0

// The arch header is the authority and L1_MUX_MASK arrives already shifted (0 on Quasar, which has no L1 mux).
constexpr std::uint32_t PERF_L1_MUX_MAX = llk::perf::L1_MUX_MASK >> llk::perf::L1_MUX_SHIFT;

#if defined(ARCH_QUASAR)
// No L1 counter bank: bank slot 3 carries the l1_client CSR, its counter_sel field holding the subport*8+event
// selection. -1 leaves it out; otherwise the same encoding as TT_METAL_PROFILE_PERF_COUNTERS_L1_SEL in metal.
#ifndef LLK_PERF_L1_CLIENT_SEL
#define LLK_PERF_L1_CLIENT_SEL (-1)
#endif
constexpr int L1_CLIENT_SEL      = LLK_PERF_L1_CLIENT_SEL;
constexpr bool L1_CLIENT_ENABLED = L1_CLIENT_SEL >= 0;
static_assert(
    !L1_CLIENT_ENABLED || llk::perf::l1_client_selection_is_valid(static_cast<std::uint32_t>(L1_CLIENT_SEL)),
    "LLK_PERF_L1_CLIENT_SEL is subport*8 + event below 296 with event not 0; THCON events 1-3 alias selections 1-3");
#endif

constexpr std::uint32_t _perf_cfg(std::uint8_t bank, std::uint16_t cid, std::uint8_t mux = 0)
{
    return PERF_CFG_VALID_BIT | (static_cast<std::uint32_t>(mux & PERF_CFG_L1_MUX_MASK) << PERF_CFG_L1_MUX_SHIFT) |
           (static_cast<std::uint32_t>(cid & PERF_CFG_COUNTER_MASK) << PERF_CFG_COUNTER_SHIFT) | static_cast<std::uint32_t>(bank);
}

// bank_regs() hands out a reference into a table on tt-1xx and a value (offset plus window) on Quasar.
#if defined(ARCH_QUASAR)
using BankRegsRef = llk::perf::BankRegs;
#else
using BankRegsRef = const llk::perf::BankRegs&;
#endif

// The volatile index stops GCC emitting a CSWTCH table, which shifts GP offsets and breaks NC/WC .text equality.
inline BankRegsRef get_bank_regs(Bank bank)
{
    static_assert(static_cast<std::uint32_t>(Bank::TDMA_PACK) == COUNTER_BANK_COUNT - 1, "Bank enumerators must be contiguous 0..COUNTER_BANK_COUNT-1");
    static constexpr llk::perf::BankRegs none {};
    volatile auto b = static_cast<std::uint32_t>(bank);
    return b < COUNTER_BANK_COUNT ? llk::perf::bank_regs(static_cast<Bank>(b)) : none;
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
    return llk::perf::table_for(Bank::L1, mux).size;
}

static_assert(L1_MUX_GROUP <= PERF_L1_MUX_MAX, "LLK_PERF_L1_MUX_GROUP does not fit this architecture's PERF_CNT_MUX_CTRL mux field");
static_assert(L1_MUX_GROUP < llk::perf::L1_MUX_POSITIONS, "LLK_PERF_L1_MUX_GROUP is past the L1 mux positions this architecture decodes");
static_assert(l1_group_size(L1_MUX_GROUP) > 0, "LLK_PERF_L1_MUX_GROUP selects an L1 mux group this architecture does not expose");
#endif

constexpr std::uint32_t builtin_counter_count()
{
    return llk::perf::table_for(Bank::INSTRN_THREAD).size + llk::perf::table_for(Bank::FPU).size + llk::perf::table_for(Bank::TDMA_UNPACK).size +
           llk::perf::table_for(Bank::TDMA_PACK).size + l1_group_size(L1_MUX_GROUP);
}

// Fixed order, matched by the readout: INSTRN, FPU, TDMA_UNPACK, TDMA_PACK, selected L1 group.
constexpr std::array<std::uint32_t, builtin_counter_count()> build_builtin_config()
{
    std::array<std::uint32_t, builtin_counter_count()> cfg {};
    std::uint32_t k = 0;
    const auto emit = [&](Bank bank, std::uint8_t mux)
    {
        const llk::perf::Table table = llk::perf::table_for(bank, mux);
        for (std::size_t i = 0; i < table.size; ++i)
        {
            cfg[k++] = _perf_cfg(static_cast<std::uint8_t>(bank), table.data[i].second, mux);
        }
    };
    emit(Bank::INSTRN_THREAD, 0);
    emit(Bank::FPU, 0);
    emit(Bank::TDMA_UNPACK, 0);
    emit(Bank::TDMA_PACK, 0);
#if defined(ARCH_QUASAR)
    if constexpr (L1_CLIENT_ENABLED)
    {
        cfg[k++] = _perf_cfg(static_cast<std::uint8_t>(Bank::L1), static_cast<std::uint16_t>(L1_CLIENT_SEL), 0);
    }
#else
    emit(Bank::L1, L1_MUX_GROUP);
#endif
    return cfg;
}

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
        const Bank bank = static_cast<Bank>(bank_id);
#if defined(ARCH_QUASAR)
        if (bank == Bank::L1)
        {
            // The l1_client CSR: route the selection and clear it; there is no bank to configure.
            if constexpr (L1_CLIENT_ENABLED)
            {
                llk::perf::l1_client_start(llk::perf::l1_client_regs(), static_cast<std::uint32_t>(L1_CLIENT_SEL));
            }
            configured_mask |= bank_bit;
            continue;
        }
#else
        if (bank == Bank::L1)
        {
            const std::uint8_t l1_mux = (metadata >> PERF_CFG_L1_MUX_SHIFT) & PERF_CFG_L1_MUX_MASK;
            llk::perf::set_l1_mux(l1_mux);
        }
#endif
        llk::perf::configure(get_bank_regs(bank));
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
        const BankRegsRef regs = get_bank_regs(static_cast<Bank>(b));
#if defined(ARCH_QUASAR)
        if (regs.control == 0)
        {
            continue; // slot 3: the l1_client CSR has no start/stop register
        }
#endif
        llk::perf::write(regs.control, llk::perf::START);
        llk::perf::write(regs.control, 0);
    }
    llk::perf::start_all();
    llk::perf::write(llk::perf::PERF_CNT_ALL, 0);
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
        llk::perf::clear_debug_feature_disable();
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

// PERF_CNT_ALL reaches only INSTRN_THREAD and FPU; the other banks take the pulse on their own control register.
// Quasar has no L1 bank: the l1_client CSR is clear-on-read, so its window starts with a read instead.
inline __attribute__((always_inline)) void arm_all_counters()
{
    ckernel::fence_compiler();
    llk::perf::start_all();
    llk::perf::write(llk::perf::bank_regs(Bank::TDMA_UNPACK).control, llk::perf::START);
#if defined(ARCH_QUASAR)
    if constexpr (L1_CLIENT_ENABLED)
    {
        (void)llk::perf::l1_client_read(llk::perf::l1_client_regs());
    }
#else
    llk::perf::write(llk::perf::bank_regs(Bank::L1).control, llk::perf::START);
#endif
    llk::perf::write(llk::perf::bank_regs(Bank::TDMA_PACK).control, llk::perf::START);
    ckernel::fence_compiler();
}

inline __attribute__((always_inline)) void freeze_and_read_all_counters(std::uint32_t zone_id)
{
    ckernel::fence_compiler();
    llk::perf::stop_all();
    llk::perf::write(llk::perf::bank_regs(Bank::TDMA_UNPACK).control, llk::perf::STOP);
#if !defined(ARCH_QUASAR)
    llk::perf::write(llk::perf::bank_regs(Bank::L1).control, llk::perf::STOP);
#endif
    llk::perf::write(llk::perf::bank_regs(Bank::TDMA_PACK).control, llk::perf::STOP);

    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    volatile std::uint32_t* bank_cycles    = reinterpret_cast<volatile std::uint32_t*>(cycles_base);
    volatile std::uint32_t* counter_counts = bank_cycles + PERF_COUNTERS_BANK_CYCLES_WORDS;
    // One reference count per bank, in Bank order.
    for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
    {
#if defined(ARCH_QUASAR)
        // Slot 3 has no reference counter; every bank is armed within a few cycles of INSTRN, so its count is
        // reported over the INSTRN reference.
        const llk::perf::BankRegs regs = llk::perf::bank_regs(static_cast<Bank>(b));
        bank_cycles[b]                  = regs.out_l ? llk::perf::read_ref(regs) : bank_cycles[0];
#else
        bank_cycles[b] = llk::perf::read_ref(llk::perf::bank_regs(static_cast<Bank>(b)));
#endif
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
            continue; // corrupt config word: do not index the register table out of range
        }
#if defined(ARCH_QUASAR)
        if (bank_id == static_cast<std::uint32_t>(Bank::L1))
        {
            counter_counts[out_idx] = llk::perf::l1_client_read(llk::perf::l1_client_regs()); // clear-on-read
            ++out_idx;
            continue;
        }
#endif
        const BankRegsRef regs = llk::perf::bank_regs(static_cast<Bank>(bank_id));
        // No mux write: it is fixed once by configure_hardware and cannot be re-aimed afterwards.
        // select() polls the mode register back; without that the read samples the previous counter.
        llk::perf::select<MODE_REG_POLL_LIMIT>(regs, static_cast<std::uint16_t>(counter_id));
        counter_counts[out_idx] = llk::perf::read_count(regs);
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

// The NC build keeps the zone rendezvous so both builds measure the same window.
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
