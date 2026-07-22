// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "perf.h" // the PERF_COUNTERS_* L1 region constants

#ifdef PERF_COUNTERS_COMPILED

#include "ckernel.h"
#include "profiler.h" // llk_profiler::sync_point — the unified cross-thread rendezvous (TRISC only)

// BRISC gets PERF_COUNTERS_COMPILED (config only, sections 1-6); TRISCs also get LLK_PROFILER
// and the per-zone measurement layer (sections 7-8, guarded by #if defined(LLK_PROFILER)).

// Quasar has no counter config here and is unsupported; the build keeps the flag off (this guards it).
#ifdef ARCH_QUASAR
#error "Perf counters are not supported on Quasar yet (no Quasar hw_counters.h; untested register set)."
#endif

// Canonical counter inventory (single source of truth, shared with metal + the host decode):
// hw_counters.h resolves per-arch via -I and lists {PerfCounterType, id} per bank;
#include <array>

// Order matters: hw_counters.h uses PerfCounterType, which perf_counters.hpp defines.
// clang-format off
#include "perf_counters.hpp" // PerfCounterType enum (canonical counter names)
#include "hw_counters.h"      // per-arch {PerfCounterType, id} inventory
// clang-format on

namespace llk_perf
{

// === 1. L1 layout: shared config, per-zone result blocks, trailing host metadata (see docs) ===

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

// sync_ctrl block sits after each zone's data; +0 holds the SYNC_ZONE_COMPLETE flag the host polls.
constexpr std::uint32_t perf_counters_sync_ctrl_addr(std::uint32_t zone)
{
    return perf_counters_zone_data_addr(zone) + PERF_COUNTERS_ZONE_DATA_BYTES;
}

// Trailing metadata written by configure_all_zones() for the host.
constexpr std::uint32_t PERF_COUNTERS_ENABLED_FLAG_ADDR = PERF_COUNTERS_ZONES_BASE + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE;
constexpr std::uint32_t PERF_COUNTERS_BANK_MASK_ADDR    = PERF_COUNTERS_ENABLED_FLAG_ADDR + 4;
constexpr std::uint32_t PERF_COUNTERS_VALID_COUNT_ADDR  = PERF_COUNTERS_BANK_MASK_ADDR + 4;
// Host-written L1 mux group (0..4) measured this run; the mux is a count-time selector (one group/window).
constexpr std::uint32_t PERF_COUNTERS_L1_MUX_SEL_ADDR = PERF_COUNTERS_VALID_COUNT_ADDR + PERF_COUNTERS_MAX_ZONES * 4;
constexpr std::uint32_t PERF_COUNTERS_LAYOUT_END      = PERF_COUNTERS_L1_MUX_SEL_ADDR + 4;

// Ceiling = profiler's lowest L1 addr (llk_profiler::EPOCH_ADDR, 0x16AFF0). Literal here (BRISC has
// no llk_profiler ns); the LLK_PROFILER section adds a symbolic assert so it can't drift.
static_assert(PERF_COUNTERS_LAYOUT_END <= 0x16AFF0u, "Perf counter L1 layout overflows into the profiler region");

// === 2. Config-word encoding + banks: [valid|l1_mux|counter_sel|bank], built by _perf_cfg() ===

// Values are the on-wire bank IDs; their order is a contract with base_addrs[], banks[], and the host.
enum class counter_bank : std::uint8_t
{
    instrn_thread = 0,
    fpu           = 1,
    tdma_unpack   = 2,
    l1            = 3,
    tdma_pack     = 4,
};

constexpr std::uint32_t COUNTER_BANK_COUNT = 5;
// Number of config slots (length of the shared config array); every config-scan loop uses this.
constexpr std::uint32_t COUNTER_SLOT_COUNT = PERF_COUNTERS_CONFIG_WORDS;

constexpr std::uint32_t PERF_CFG_VALID_BIT     = 1u << 31; // bit 31: slot active
constexpr std::uint32_t PERF_CFG_L1_MUX_SHIFT  = 17;       // bits 19:17
constexpr std::uint32_t PERF_CFG_L1_MUX_MASK   = 0x7u;
constexpr std::uint32_t PERF_CFG_COUNTER_SHIFT = 8; // bits 16:8 (9-bit counter_sel)
constexpr std::uint32_t PERF_CFG_COUNTER_MASK  = 0x1FFu;
constexpr std::uint32_t PERF_CFG_BANK_MASK     = 0xFFu; // bits 7:0

// L1 mux position inside PERF_CNT_MUX_CTRL (bits 6:4).
constexpr std::uint32_t PERF_CNT_MUX_CTRL_SHIFT = 4;

constexpr std::uint32_t _perf_cfg(std::uint8_t bank, std::uint16_t cid, std::uint8_t mux = 0)
{
    return PERF_CFG_VALID_BIT | (static_cast<std::uint32_t>(mux & PERF_CFG_L1_MUX_MASK) << PERF_CFG_L1_MUX_SHIFT) |
           (static_cast<std::uint32_t>(cid & PERF_CFG_COUNTER_MASK) << PERF_CFG_COUNTER_SHIFT) | static_cast<std::uint32_t>(bank);
}

// === 3. Counter-bank base-address lookup (raw access is via ckernel::reg_read/reg_write) ===

// Volatile index cast prevents GCC CSWTCH (shifts GP-offsets, breaking counter/no-counter bit-identity).
inline std::uint32_t get_counter_base_addr(counter_bank bank)
{
    static constexpr std::uint32_t base_addrs[COUNTER_BANK_COUNT] = {
        RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0,
        RISCV_DEBUG_REG_PERF_CNT_FPU0,
        RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0,
        RISCV_DEBUG_REG_PERF_CNT_L1_0,
        RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0,
    };
    // static_assert keeps the enum a safe index into base_addrs[]; the runtime bound is defensive.
    static_assert(
        static_cast<std::uint32_t>(counter_bank::tdma_pack) == COUNTER_BANK_COUNT - 1, "counter_bank enumerators must be contiguous 0..COUNTER_BANK_COUNT-1");
    volatile auto b = static_cast<std::uint32_t>(bank);
    return b < COUNTER_BANK_COUNT ? base_addrs[b] : 0u;
}

// === 4. Built-in counter inventory: built from the canonical hw_counters.h arrays (see docs) ===

// Total counters across all bank arrays (per arch), evaluated at compile time.
constexpr std::uint32_t builtin_counter_count()
{
    return instrn_counters.size() + fpu_counters.size() + unpack_counters.size() + pack_counters.size() + l1_0_counters.size() + l1_1_counters.size() +
           l1_2_counters.size() + l1_3_counters.size() + l1_4_counters.size();
}

// Concatenate the per-bank arrays into config words, in the fixed order the readout expects:
// INSTRN, FPU, TDMA_UNPACK, TDMA_PACK, then L1 by ascending mux.
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
    emit(l1_0_counters, counter_bank::l1, 0);
    emit(l1_1_counters, counter_bank::l1, 1);
    emit(l1_2_counters, counter_bank::l1, 2);
    emit(l1_3_counters, counter_bank::l1, 3);
    emit(l1_4_counters, counter_bank::l1, 4);
    return cfg;
}

constexpr auto BUILTIN_COUNTER_CONFIG         = build_builtin_config();
constexpr std::uint32_t BUILTIN_COUNTER_COUNT = BUILTIN_COUNTER_CONFIG.size();

// === 5. One-time HW setup (BRISC): write config to L1, clear scratch, configure + arm (see docs) ===
// Stateless — all state lives in L1 at fixed addresses, so these are plain free functions.

inline std::uint32_t get_active_bank_mask()
{
    return *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_BANK_MASK_ADDR);
}

// Configure each populated bank's reference-period + mode register once (configured_mask dedups to banks).
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
            // Count-time mux group for this run (host-selected; see PERF_COUNTERS_L1_MUX_SEL_ADDR).
            const std::uint8_t l1_mux = *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_L1_MUX_SEL_ADDR) & PERF_CFG_L1_MUX_MASK;
            std::uint32_t cur         = ckernel::reg_read(RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL);
            ckernel::reg_write(
                RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL, (cur & ~(PERF_CFG_L1_MUX_MASK << PERF_CNT_MUX_CTRL_SHIFT)) | (l1_mux << PERF_CNT_MUX_CTRL_SHIFT));
        }
        std::uint32_t counter_base = get_counter_base_addr(bank);
        ckernel::reg_write(counter_base, 0xFFFFFFFF);
        ckernel::reg_write(counter_base + 4, 0);
        configured_mask |= bank_bit;
    }
}

// Per-bank arm + global PERF_CNT_ALL broadcast (rising edge 0→1 clears + starts).
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

// Scan config, publish enabled flag + bank mask + per-zone valid_count to L1, then configure + arm.
inline void configure_all_zones()
{
    bool found_valid                                    = false;
    std::uint32_t bank_mask                             = 0;
    std::uint32_t valid_counts[PERF_COUNTERS_MAX_ZONES] = {};

    for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
    {
        const volatile std::uint32_t* config_mem = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
        std::uint32_t count                      = 0;
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
        valid_counts[zone] = count;
    }

    *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_ENABLED_FLAG_ADDR) = found_valid ? 1u : 0u;
    *reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_BANK_MASK_ADDR)    = bank_mask;
    volatile std::uint32_t* valid_count_ptr                                     = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_VALID_COUNT_ADDR);
    for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
    {
        valid_count_ptr[zone] = valid_counts[zone];
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

// === 6. Zone-id allocator: zone name -> DJB2 hash -> sequential id (0..MAX_ZONES-1) ===

namespace detail
{
// `.bss.perf_counters` is NOBITS, zero-initialized by do_crt0() on every cold start (grouped for objdump).
__attribute__((section(".bss.perf_counters"))) static std::uint32_t zone_hashes[PERF_COUNTERS_MAX_ZONES];
__attribute__((section(".bss.perf_counters"))) static std::uint32_t next_zone_id;

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

// === Per-zone measurement layer (TRISC only — needs the profiler sync_point rendezvous) ===
#if defined(LLK_PROFILER)

// Drift-proof version of the layout ceiling above (tracks the profiler region symbolically).
static_assert(PERF_COUNTERS_LAYOUT_END <= llk_profiler::EPOCH_ADDR, "Perf counter L1 layout overflows into the profiler region");

// === 7. Arm / freeze+read: per-zone counter window. Write 1 = rising-edge start, 2 = stop (see docs) ===

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
        const bank_regs& br      = banks[bank_id];
        // L1 mux is set once per run in configure_hardware (count-time); the readout ignores it.
        const std::uint32_t mode_val = counter_id << PERF_CFG_COUNTER_SHIFT;
        ckernel::reg_write(br.mode_reg, mode_val);
        // Readback poll: the readout mux is clocked, so OUT lags the mode write by a cycle (else off-by-one reads).
        while (ckernel::reg_read(br.mode_reg) != mode_val)
        {
        }
        counter_counts[out_idx] = ckernel::reg_read(br.out_l + 4u);
        ++out_idx;
    }

    std::uint32_t sync_addr                               = perf_counters_sync_ctrl_addr(zone_id);
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
}

// === 8. Per-thread role + RAII: the pack thread arms+freezes; sync_point keeps it correct (see docs) ===

// The pack thread is the designated actor that arms and freezes+reads counters for every zone.
constexpr bool is_perf_actor_thread()
{
#if defined(LLK_TRISC_PACK)
    return true;
#else
    return false;
#endif
}

// RAII arm/freeze. Constructed first, destructed last relative to the profiler zone_scoped:
// perf_ctor(arm) → zone_ctor(ZONE_START) → body → zone_dtor(ZONE_END) → perf_dtor(freeze).
#ifndef _LLK_PERF_COUNTER_SCOPED_DEFINED_
#define _LLK_PERF_COUNTER_SCOPED_DEFINED_

struct perf_counter_scoped
{
    std::uint32_t zone_id;

    perf_counter_scoped(const perf_counter_scoped&)            = delete;
    perf_counter_scoped(perf_counter_scoped&&)                 = delete;
    perf_counter_scoped& operator=(const perf_counter_scoped&) = delete;
    perf_counter_scoped& operator=(perf_counter_scoped&&)      = delete;

    // Entry: actor thread waits for all, arms, releases (spin is before the arm, outside the window).
    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    {
        ckernel::fence_compiler();
        llk_profiler::sync_point(is_perf_actor_thread(), [] { arm_all_counters(); });
        ckernel::fence_compiler();
    }

    // Exit: actor thread waits for all to finish, then freezes+reads all counters and releases.
    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        ckernel::fence_compiler();
        const std::uint32_t zid = zone_id;
        llk_profiler::sync_point(is_perf_actor_thread(), [zid] { freeze_and_read_all_counters(zid); });
        ckernel::fence_compiler();
    }
};
#endif // _LLK_PERF_COUNTER_SCOPED_DEFINED_

#endif // LLK_PROFILER (per-zone measurement layer)

} // namespace llk_perf

#if defined(LLK_PROFILER)
// TRISC: instantiate a perf_counter_scoped keyed by the zone name.
#define PERF_COUNTER_VAR_CONCAT_(a, b) a##b
#define PERF_COUNTER_VAR_(line)        PERF_COUNTER_VAR_CONCAT_(_perf_ctr_, line)
#define MEASURE_PERF_COUNTERS(zone_name) \
    const llk_perf::perf_counter_scoped PERF_COUNTER_VAR_(__LINE__)(llk_perf::get_zone_id(llk_perf::detail::zone_name_hash(zone_name)));
#else
// BRISC (counters configured but no per-zone profiler rendezvous available): no-op.
#define MEASURE_PERF_COUNTERS(zone_name)
#endif

#else // !PERF_COUNTERS_COMPILED

#if defined(LLK_PROFILER)
#define MEASURE_PERF_COUNTERS(zone_name) llk_profiler::sync_point(llk_profiler::TRISC_ID == 0, [] {});
#else
#define MEASURE_PERF_COUNTERS(zone_name)
#endif

// No-counter build stub — brisc calls this unconditionally; compiler folds it to nothing.
namespace llk_perf
{
inline void configure_and_arm_from_brisc()
{
}
} // namespace llk_perf

#endif // PERF_COUNTERS_COMPILED

// One measured scope = a perf-counter window + a profiler timing zone (one half is active per build), keyed by `zone_name`.
#define START_PERF_MEASURE(zone_name) \
    MEASURE_PERF_COUNTERS(zone_name)  \
    ZONE_SCOPED(zone_name)
