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
constexpr std::uint32_t PERF_COUNTERS_LAYOUT_END        = PERF_COUNTERS_VALID_COUNT_ADDR + PERF_COUNTERS_MAX_ZONES * 4;

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

// The L1 bank has only 8 physical counters, and PERF_CNT_MUX_CTRL[6:4] selects which group of 8 client
// interfaces feeds them *while they count* - it is not a read-time selector. One run therefore observes
// exactly one group. Emitting every group made all of them read back the counting group's 8 values under
// several different client names, so 4 of every 5 L1 columns were duplicates at an indeterminate mux.
// Capture one group per run and select it here; sweep the groups across runs to cover all interfaces.
#ifndef LLK_PERF_L1_MUX_GROUP
#define LLK_PERF_L1_MUX_GROUP 0
#endif

constexpr std::uint8_t L1_MUX_GROUP = LLK_PERF_L1_MUX_GROUP;

// Groups an arch does not expose are declared empty, so a zero size means "not available here".
constexpr std::uint32_t l1_group_size(std::uint8_t mux)
{
    return mux == 0   ? l1_0_counters.size()
           : mux == 1 ? l1_1_counters.size()
           : mux == 2 ? l1_2_counters.size()
           : mux == 3 ? l1_3_counters.size()
           : mux == 4 ? l1_4_counters.size()
                      : 0u;
}

static_assert(L1_MUX_GROUP <= PERF_CFG_L1_MUX_MASK, "LLK_PERF_L1_MUX_GROUP must fit in PERF_CNT_MUX_CTRL[6:4]");
static_assert(l1_group_size(L1_MUX_GROUP) > 0, "LLK_PERF_L1_MUX_GROUP selects an L1 mux group this architecture does not expose");

// Total counters across all bank arrays (per arch), evaluated at compile time.
constexpr std::uint32_t builtin_counter_count()
{
    return instrn_counters.size() + fpu_counters.size() + unpack_counters.size() + pack_counters.size() + l1_group_size(L1_MUX_GROUP);
}

// Concatenate the per-bank arrays into config words, in the fixed order the readout expects:
// INSTRN, FPU, TDMA_UNPACK, TDMA_PACK, then the single selected L1 mux group.
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
    else
    {
        emit(l1_4_counters, counter_bank::l1, 4);
    }
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
            const std::uint8_t l1_mux = (metadata >> PERF_CFG_L1_MUX_SHIFT) & PERF_CFG_L1_MUX_MASK;
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
    // Every zone is described by the one shared config, so the scan answer is identical for all of
    // them. Scan once and replicate: scanning per zone read the same COUNTER_SLOT_COUNT words 8 times
    // over, and those are volatile L1 reads issued on BRISC immediately before it releases the TRISCs.
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

// === 6. Zone-id allocator: zone name -> DJB2 hash -> sequential id (0..MAX_ZONES-1) ===

namespace detail
{
// `.perf_counters_bss` is NOBITS, zero-initialized by do_crt0() on every cold start (which clears
// the whole [__ldm_bss_start, __ldm_bss_end) range by address). Its dedicated section name keeps it
// out of the generic `.bss.*` catch-all so the linker can deterministically place it last — see
// ld/sections.ld. (Grouped in one section so the two symbols stay together for objdump.)
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
        // No L1 mux write here: the mux routes interfaces into the counters as they count, so it is fixed
        // once by configure_hardware and cannot be re-aimed after the fact.
        const std::uint32_t expected_mode = counter_id << PERF_CFG_COUNTER_SHIFT;
        ckernel::reg_write(br.mode_reg, expected_mode);
        // Readback poll: MMIO fence so the counter-select store lands before the OUT_H read.
        // reg_write is only a volatile store, so without this the read can race the in-flight
        // select and sample the previous counter (matches tools/profiler/perf_counters.hpp).
        while (ckernel::reg_read(br.mode_reg) != expected_mode)
        {
        }
        counter_counts[out_idx] = ckernel::reg_read(br.out_l + 4u);
        ++out_idx;
    }

    std::uint32_t sync_addr                               = perf_counters_sync_ctrl_addr(zone_id);
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE;
}

// === 8. Per-thread role + RAII: the pack thread arms+freezes; sync_point keeps it correct (see docs) ===

// The pack thread is the designated actor that freezes+reads counters for every zone, and arms them for
// the run types whose entry goes through sem_rendezvous. It does NOT arm the entry_via_sync_point run
// types (L1_CONGESTION, MATH_ISOLATE): those elect TRISC0 so the barrier matches the no-counter build's
// stub exactly, so for them the arming thread and the freezing thread are different threads.
constexpr bool is_perf_actor_thread()
{
#if defined(LLK_TRISC_PACK)
    return true;
#else
    return false;
#endif
}

// === Rendezvous implementation choice ===
//   0 = L1 epoch/barrier words, with per-run-type poll backoff and a split arm/release actor
//   1 = Tensix hardware semaphore (no L1 traffic at all, no release-detection latency)
//
// Mode 1 exists because every poll of the L1 protocol is a fence plus an uncached L1 load, so a
// thread waiting at a zone's exit is a continuous L1 requester competing with the thread still being
// measured. semaphore_read() instead reads pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] — the semaphore
// register file, not L1 — so waiting costs the measured thread nothing and needs no backoff.
// Both implementations are always compiled; the choice is made per run type below, because each run
// type is built into its own ELFs and they want different things.

// Blackhole has exactly 8 semaphores and none can be added: SEM_COUNT = 8 (tt_tensix.sv:3375) and
// tt_sync_exu.sv:390 instantiates that many physical tt_semaphore_reg blocks. semaphore::PACK_DONE is
// used because it is documented for precisely this purpose ("For recording perf events and inserting
// delay", ckernel_structs.h:19) and nothing reachable from a perf kernel touches it — no *perf*.cpp
// source uses it, and its only LLK users are the tilize_polluter_* tests, which never share a kernel
// launch with these. (semaphore::UNPACK_MATH_DONE carries the same comment but is NOT safe: it is used
// by llk_unpack_AB_reduce_custom_runtime.h, which perf_eltwise_bcast_col_custom reaches.)
//
// Firmware initialises this semaphore with max=1 (c_tensix_core.h:421), which does NOT matter here:
// per tt_semaphore_reg.sv, SEMPOST increments up to 4'hf regardless of semaphore_max, and the max only
// drives o_stall_cond for SEMWAIT — which this code never issues. So no SEMINIT is needed, which also
// avoids having to order a backend instruction against RISC-side accesses.
constexpr std::uint8_t PERF_RENDEZVOUS_SEM = ckernel::semaphore::PACK_DONE;

// Cross-thread rendezvous on a hardware semaphore. Every thread announces by incrementing; the action
// thread waits for all announcements, runs action(), then drains the count back to zero — and that
// return to zero IS the release signal, so no separate release flag or actor is needed.
//
// Waiters watch for zero rather than for the full count, which removes the obvious race in a
// count-to-N barrier (a thread that decremented first would otherwise stop its peers from ever
// observing N). A waiter has already incremented before it starts polling, so the value is >= 1 on
// entry and can only return to zero after the action thread has completed action() and drained.
//
// The drain loop is `while (read() != 0) get()` rather than a fixed N decrements so that the protocol
// self-normalises: whatever value the semaphore happens to hold when the first zone starts, the first
// rendezvous leaves it at zero and every later one begins from zero.
template <std::uint32_t BACKOFF_NOPS_UNUSED = 0, typename Action>
inline __attribute__((always_inline)) void sem_rendezvous(bool is_action_thread, Action action)
{
    ckernel::fence_compiler();

    ckernel::semaphore_post(PERF_RENDEZVOUS_SEM);

    if (is_action_thread)
    {
        while (ckernel::semaphore_read(PERF_RENDEZVOUS_SEM) < llk_profiler::NUM_CORES)
        {
        }
        action();
        while (ckernel::semaphore_read(PERF_RENDEZVOUS_SEM) != 0)
        {
            ckernel::semaphore_get(PERF_RENDEZVOUS_SEM);
        }
    }
    else
    {
        while (ckernel::semaphore_read(PERF_RENDEZVOUS_SEM) != 0)
        {
        }
    }

    ckernel::fence_compiler();
}

// Which entry rendezvous a run type uses. Measured per run type, not chosen a priori: the two barriers
// differ in medium — llk_profiler::sync_point polls an L1 barrier array and invalidates dcache, while
// sem_rendezvous polls a hardware semaphore through pc_buf — so they release the threads with different
// stagger, and the measured window inherits that phase. Measured on perf_unpack_tilize over all 772
// variants, counting variants past 50 cycles:
//   UNPACK_ISOLATE  49 -> 0    worst -2563 -> -14, mean -106 -> +4              (semaphore wins)
//   L1_CONGESTION   sync_point kept: moving it to the semaphore keeps the worst magnitude and only flips
//                   its sign (-1277 -> +1277) while the pack row goes 17 -> 23 variants and +349 ->
//                   +1270. That row is a bistable L1 arbiter orbit, so changing the barrier re-phases
//                   which variants latch the slow orbit instead of removing it.
// Choosing per run type is legitimate because PERF_RUN_TYPE is compile-time: each run type is its own
// ELF, so this selects the measured-best barrier for each rather than trading one row against another.
// sync_point is also what the no-counter build itself calls (counters.h NC stub), so where it is chosen
// the two builds reach ZONE_START through the identical barrier.
//   L1_TO_L1        sync_point required, and for a different reason than the other two: it is the only
//                   run type whose window is a cross-thread span (pack's ZONE_END minus unpack's
//                   ZONE_START, profiler.py _stats_l1_to_l1), so its measurement includes the thread
//                   spacing at zone entry. Entering through the semaphore while the no-counter build
//                   entered through sync_point therefore put the two barriers' different alignment
//                   straight into every INIT row: -21 cycles on average, negative in 462 of 462
//                   variants, reproducing to the cycle across independent build pairs. The *_ISOLATE
//                   run types time their own thread start-to-end, so they are blind to entry spacing by
//                   construction and keep the cheaper semaphore.
constexpr bool entry_via_sync_point(PerfRunType rt)
{
    return rt == PerfRunType::L1_CONGESTION || rt == PerfRunType::MATH_ISOLATE || rt == PerfRunType::L1_TO_L1;
}

// TRISC0 — the thread the no-counter path elects as its rendezvous actor (its MEASURE_PERF_COUNTERS
// stub passes TRISC_ID == 0). Only used by the LLK_PERF_ARM_ON == 0 setting, which arms on the same
// thread the no-counter build releases from. Measurement preferred arming on pack (ARM_ON 2): it
// halves the L1_CONGESTION error and costs nothing elsewhere.
constexpr bool is_perf_release_thread()
{
#if defined(LLK_TRISC_UNPACK)
    return true;
#else
    return false;
#endif
}

// RAII arm/freeze. Constructed first, destructed last relative to the profiler zone_scoped:
// perf_ctor(arm) → zone_ctor(ZONE_START) → body → zone_dtor(ZONE_END) → perf_dtor(freeze).
#ifndef _LLK_PERF_COUNTER_SCOPED_DEFINED_
#define _LLK_PERF_COUNTER_SCOPED_DEFINED_

// Run types whose reported metric is ONE thread's own ZONE_START..ZONE_END window. For these the
// counters only need to be frozen at that thread's own zone end, so no cross-thread rendezvous is
// required on exit at all.
constexpr bool is_single_thread_runtype(PerfRunType run_type)
{
    return run_type == PerfRunType::UNPACK_ISOLATE || run_type == PerfRunType::MATH_ISOLATE || run_type == PerfRunType::PACK_ISOLATE;
}

// === Per-run-type rendezvous shape ===
//
// Every run type is built into its OWN set of ELFs (PERF_RUN_TYPE is a compile-time constant, so one
// test variant produces 5 run types x 3 threads = 15 ELFs). Nothing about the shape is shared at run
// time, so these choices are INDEPENDENT: tuning one run type cannot perturb another. That matters
// because the residual WC-vs-NC difference is a thread-phase effect, and each run type has a different
// thread doing the work, so each one wants a different phase.
//
// Measured on perf_eltwise_binary_fpu (Float16->Float16_b, LoFi, Elwadd, tile_cnt 16, dest_acc No),
// sweeping exit-barrier x arming-thread. Delta vs the no-counter build, in cycles:
//
//   shape                 L1_TO_L1  UNPACK_ISO  MATH_ISO  PACK_ISO  CONG[UNP]  CONG[PACK]
//   no barrier, TRISC0 arms     -6        -18        +2        +6       -12         +2
//   no barrier, pack arms       -6        -13        +1         0        -6         +3
//   barrier,    TRISC0 arms     -6        -13       +12        +5       -12         +2
//   barrier,    pack arms       -6         -6       +12        +1        -6         +3
//
// (The L1_CONGESTION columns are against NC 691/471. An earlier revision of this table used 685/474,
// which are WC numbers, and so wrongly reported these two as reaching exactly zero.)
//
// The arming-thread rows are superseded for MATH_ISO and both CONG columns: entry_via_sync_point now
// elects TRISC0 for those run types whatever LLK_PERF_ARM_ON says, so those three columns can no longer
// differ between the two arming rows. Only the L1_TO_L1 and *_ISOLATE columns still respond to it.
//
// Two conclusions. Arming on the pack thread halves the total L1_CONGESTION error (|-6|+|+3| = 9 versus
// |-12|+|+2| = 14 for TRISC0 arming), so it is still the better choice there. And the
// exit barrier is wanted by UNPACK_ISOLATE (-6 with it, -13 without) but not by MATH/PACK_ISOLATE
// (+12/+1 with it, +1/0 without) - so it has to be chosen per run type, which is safe because each run
// type is built into its own ELFs and shares nothing at run time.
//
// Not a knob: arming on the *measured* thread deadlocks for the multi-thread run types, where no thread
// is the measured one, so no thread becomes the actor and nobody releases the rendezvous.
constexpr bool wants_exit_barrier(PerfRunType run_type)
{
    // Multi-thread metrics must keep it - their window spans more than one thread.
    if (!is_single_thread_runtype(run_type))
    {
        return true;
    }
    // UNPACK_ISOLATE measures TRISC0, which is also first out of the entry rendezvous; keeping the exit
    // barrier holds it closer to the no-counter build's phase.
    return run_type == PerfRunType::UNPACK_ISOLATE;
}

// Override for experiments: -1 = per-run-type table above, 0/1 = force off/on for every run type.
#ifndef LLK_PERF_EXIT_BARRIER
#define LLK_PERF_EXIT_BARRIER (-1)
#endif
// 0 = TRISC0 arms, 2 = pack arms. (1 = measured thread: deadlocks, see above.)
#ifndef LLK_PERF_ARM_ON
#define LLK_PERF_ARM_ON 2
#endif

constexpr bool exit_barrier_for(PerfRunType run_type)
{
    return (LLK_PERF_EXIT_BARRIER < 0) ? wants_exit_barrier(run_type) : (LLK_PERF_EXIT_BARRIER != 0);
}

// Whether THIS thread is the one whose window a single-thread run type reports.
constexpr bool is_measured_thread(PerfRunType run_type)
{
#if defined(LLK_TRISC_UNPACK)
    return run_type == PerfRunType::UNPACK_ISOLATE;
#elif defined(LLK_TRISC_MATH)
    return run_type == PerfRunType::MATH_ISOLATE;
#elif defined(LLK_TRISC_PACK)
    return run_type == PerfRunType::PACK_ISOLATE;
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

    // Entry: arm once every thread has finished the previous zone, then release. The rendezvous is
    // required here — without it the previous zone's trailing activity would be counted into this one.
    //
    // The arming thread is TRISC0 for every run type, which is exactly what the no-counter path elects
    // (its MEASURE_PERF_COUNTERS stub passes TRISC_ID == 0). Matching it matters: whichever thread runs
    // the action also walks on first, so electing a different one re-phases the threads relative to the
    // no-counter build. Electing the measured thread instead was measured to swing UNPACK_ISOLATE from
    // -6 to -18 cycles. Counters are global debug registers, so arming here and freezing on the
    // measured thread (see the destructor) is fine.
    inline __attribute__((always_inline)) explicit perf_counter_scoped(std::uint32_t zid) : zone_id(zid)
    {
        ckernel::fence_compiler();
#if LLK_PERF_ARM_ON == 1
        // Total by construction: for a multi-thread run type no thread is the measured one, and electing
        // nobody hangs the rendezvous (every thread SEMPOSTs then waits for a drain that never comes).
        [[maybe_unused]] constexpr bool arms_here = is_single_thread_runtype(RUN_TYPE) ? is_measured_thread(RUN_TYPE) : is_perf_actor_thread();
#elif LLK_PERF_ARM_ON == 2
        [[maybe_unused]] constexpr bool arms_here = is_perf_actor_thread();
#else
        [[maybe_unused]] constexpr bool arms_here = is_perf_release_thread(); // TRISC0, as the no-counter path elects
#endif
        if constexpr (entry_via_sync_point(RUN_TYPE))
        {
            // Same call the no-counter build makes, with arming as its action.
            llk_profiler::sync_point(llk_profiler::TRISC_ID == 0, [] { arm_all_counters(); });
        }
        else
        {
            sem_rendezvous(arms_here, [] { arm_all_counters(); });
        }
        ckernel::fence_compiler();
    }

    // Exit.
    //
    // For a single-thread run type there is NO rendezvous: the measured thread freezes and reads its
    // own window at its own ZONE_END, and the other threads simply leave. Ablation showed the exit
    // rendezvous — merely existing — was worth +11 of the +12 cycles that this build differed from the
    // no-counter build on eltwise_binary_fpu/MATH_ISOLATE, while the arming writes cost 1 and the
    // BRISC-side configuration cost 0. Ordering stays safe because the next zone's entry rendezvous
    // still waits for every thread, and the measured thread only reaches it after freezing.
    //
    // L1_TO_L1 and L1_CONGESTION keep the barrier: their metrics span more than one thread, so the
    // freeze genuinely has to wait for all of them.
    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        ckernel::fence_compiler();
        const std::uint32_t zid = zone_id;
        if constexpr (!exit_barrier_for(RUN_TYPE))
        {
            if constexpr (is_measured_thread(RUN_TYPE))
            {
                freeze_and_read_all_counters(zid);
            }
        }
        else
        {
            sem_rendezvous(is_perf_actor_thread(), [zid] { freeze_and_read_all_counters(zid); });
        }
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
// PERF_RUN_TYPE is declared by the generated params.h, which lands after this header is parsed, so
// the run type is read here at macro-expansion time (inside the kernel) rather than at header scope.
#define MEASURE_PERF_COUNTERS(zone_name) \
    const llk_perf::perf_counter_scoped<PERF_RUN_TYPE> PERF_COUNTER_VAR_(__LINE__)(llk_perf::get_zone_id(llk_perf::detail::zone_name_hash(zone_name)));
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

// One measured scope = a perf-counter window + a profiler timing zone; NC activates timing only, WC activates both, keyed by `zone_name`.
#define START_PERF_MEASURE(zone_name) \
    MEASURE_PERF_COUNTERS(zone_name)  \
    ZONE_SCOPED(zone_name)
