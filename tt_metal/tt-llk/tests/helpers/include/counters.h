// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "barrier.h"
#include "ckernel.h"
#include "perf.h" // the PERF_COUNTERS_* L1 region constants
// START_PERF_MEASURE expands to ZONE_SCOPED, so the zone layer must be in scope here; self-guarded on LLK_PROFILER.
#include "profiler.h"

// Counter inventory, register map and register primitives are shared with the metal profiler. On Quasar
// bank_regs() defaults to the TRISC-local window, so every thread reaches the block of its own NEO.
#include <array>
#include <type_traits>

#include "perf_counters/hw.h"
#include "perf_counters/inventory.h"
#include "perf_counters/registers.h"

namespace llk_perf
{

// Both builds, before the TRISCs run, as metal's firmware does before each kernel: zero the debug feature chicken bits
// (L1 arbitration among them) an earlier program may have left set. Not counter work.
inline void clear_debug_features()
{
#if !defined(TT_METAL_TTSIM)
    llk::perf::clear_debug_feature_disable();
#endif
}

// Counters on (PERF_COUNTERS_COMPILED, set only by --enable-perf-counters builds) counts the zones the host names for a
// launch. Counters off compiles no counter code: a zone keeps only its rendezvous, the same barriers in both builds.
#if defined(PERF_COUNTERS_COMPILED)

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

// +4 of the zone 0 sync block: the frozen zone still waiting for its readout, stored as zone + 1 so 0 means none.
constexpr std::uint32_t PERF_COUNTERS_PENDING_ZONE_ADDR = perf_counters_sync_ctrl_addr(0) + 4;

// Quasar's launch word (it has no stream registers): the word after the zone blocks.
constexpr std::uint32_t PERF_COUNTERS_LAUNCH_WORD_ADDR = PERF_COUNTERS_ZONES_BASE + PERF_COUNTERS_MAX_ZONES * PERF_COUNTERS_ZONE_SIZE;
constexpr std::uint32_t PERF_COUNTERS_LAYOUT_END       = PERF_COUNTERS_LAUNCH_WORD_ADDR + 4;

// A literal because BRISC has no llk_profiler namespace; the LLK_PROFILER section asserts it symbolically.
static_assert(PERF_COUNTERS_LAYOUT_END <= 0x16AFF0u, "Perf counter L1 layout overflows into the profiler region");

// On-wire bank IDs are llk::perf::Bank; the order is a contract with the host.
constexpr std::uint32_t COUNTER_BANK_COUNT = llk::perf::NUM_BANKS;

// Unbounded, a corrupt config word would hang every thread and surface only as TENSIX TIMED OUT.
constexpr std::uint32_t MODE_REG_POLL_LIMIT = 1024;
// Stored when the mode register never confirms the selection: a stale count would look plausible, 2^32-1 does not.
constexpr std::uint32_t COUNTER_SELECT_MISSED = 0xFFFFFFFFu;
constexpr std::uint32_t COUNTER_SLOT_COUNT    = PERF_COUNTERS_CONFIG_WORDS;

constexpr std::uint32_t PERF_CFG_VALID_BIT     = 1u << 31; // bit 31: slot active
constexpr std::uint32_t PERF_CFG_L1_MUX_SHIFT  = 17;       // bits 19:17
constexpr std::uint32_t PERF_CFG_L1_MUX_MASK   = 0x7u;
constexpr std::uint32_t PERF_CFG_COUNTER_SHIFT = 8; // bits 16:8 (9-bit counter_sel)
constexpr std::uint32_t PERF_CFG_COUNTER_MASK  = 0x1FFu;
constexpr std::uint32_t PERF_CFG_BANK_MASK     = 0xFFu; // bits 7:0

// The arch header is the authority and L1_MUX_MASK arrives already shifted (0 on Quasar, which has no L1 mux).
constexpr std::uint32_t PERF_L1_MUX_MAX = llk::perf::L1_MUX_MASK >> llk::perf::L1_MUX_SHIFT;

#if defined(ARCH_QUASAR)
// No L1 counter bank: slot 3 carries the l1_client CSR instead. -1 leaves it out; otherwise subport*8 + event, the
// selection metal takes in bits 16-24 of TT_METAL_PROFILE_PERF_COUNTERS.
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

// bank_regs() hands out a reference into a table on tt-1xx and a value (offset plus window) on Quasar; use this
// alias for every local so each arch keeps its own form.
#if defined(ARCH_QUASAR)
using BankRegsRef = llk::perf::BankRegs;
#else
using BankRegsRef = const llk::perf::BankRegs&;
#endif

// Only 8 physical L1 counters exist, and the mux selects which group feeds them while they count,
// not at read time, so a run sees one group and the groups have to be swept across runs.
#ifndef LLK_PERF_L1_MUX_GROUP
#define LLK_PERF_L1_MUX_GROUP 0
#endif

constexpr std::uint8_t L1_MUX_GROUP = LLK_PERF_L1_MUX_GROUP;

#if defined(ARCH_QUASAR)
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

// The host's word for a launch, written before it (test_config.py _write_launch_word): the zones it counts (a bit per
// zone id, 0 for none), and above them a token the readouts store with the results (at the same shift in the sync word).
constexpr std::uint32_t LAUNCH_TOKEN_SHIFT = 8;
constexpr std::uint32_t LAUNCH_ZONES_MASK  = (1u << LAUNCH_TOKEN_SHIFT) - 1;
constexpr std::uint32_t LAUNCH_WORD_MASK   = 0xFFFFFFu; // the stream registers are 24 bits wide
static_assert(LAUNCH_TOKEN_SHIFT == PERF_COUNTERS_MAX_ZONES, "one launch word bit per zone id, the token above them");
#if defined(ARCH_QUASAR)
constexpr std::uint32_t COUNTED_ZONES_REG = PERF_COUNTERS_LAUNCH_WORD_ADDR;
#else
// STREAM_SCRATCH_1 of overlay stream 0 (metal's noc_overlay_parameters.h, not included: boot.h declares names it defines)
constexpr std::uint32_t LAUNCH_WORD_OVERLAY_BASE = 0xFFB40000;
#if defined(ARCH_WORMHOLE)
constexpr std::uint32_t LAUNCH_WORD_SCRATCH_INDEX = 249;
#elif defined(ARCH_BLACKHOLE)
constexpr std::uint32_t LAUNCH_WORD_SCRATCH_INDEX = 37;
#endif
constexpr std::uint32_t COUNTED_ZONES_REG = LAUNCH_WORD_OVERLAY_BASE + LAUNCH_WORD_SCRATCH_INDEX * 4;
#endif

inline __attribute__((always_inline)) std::uint32_t launch_word()
{
#if defined(TT_METAL_TTSIM) // ttsim models no counter registers
    return 0;
#else
    return *reinterpret_cast<volatile std::uint32_t*>(COUNTED_ZONES_REG) & LAUNCH_WORD_MASK;
#endif
}

inline __attribute__((always_inline)) std::uint32_t counted_zones()
{
    return launch_word() & LAUNCH_ZONES_MASK;
}

namespace detail
{
// The launch's counted zones and each zone's arm command (START if counted, 0 = no command if not), copied once per
// kernel by load_counted_zones: every launch runs the same arm and freeze instructions.
inline std::uint32_t counted                                    = 0;
inline std::uint32_t token                                      = 0;
inline volatile std::uint32_t arm_cmds[PERF_COUNTERS_MAX_ZONES] = {};
#if defined(ARCH_QUASAR)
// The l1_client selection per zone, with the enable bit only when the zone is counted.
inline volatile std::uint32_t l1_client_cmds[PERF_COUNTERS_MAX_ZONES] = {};
#endif

// value if bit, else 0, without a branch: a branch on launch data would leave launches with different predictor state.
__attribute__((always_inline)) inline std::uint32_t when(bool bit, std::uint32_t value)
{
    return value & (0u - static_cast<std::uint32_t>(bit));
}
} // namespace detail

inline __attribute__((always_inline)) bool counts(std::uint32_t zone_id)
{
    return (detail::counted >> zone_id) & 1u;
}

// Before the kernel (trisc.cpp), in every launch.
inline void load_counted_zones()
{
    const std::uint32_t word = launch_word();
    detail::counted          = word & LAUNCH_ZONES_MASK;
    detail::token            = word >> LAUNCH_TOKEN_SHIFT;
    for (std::uint32_t zone = 0; zone < PERF_COUNTERS_MAX_ZONES; ++zone)
    {
        detail::arm_cmds[zone] = detail::when(counts(zone), llk::perf::START);
#if defined(ARCH_QUASAR)
        if constexpr (L1_CLIENT_ENABLED)
        {
            const std::uint32_t ctrl     = llk::perf::l1_client_ctrl_word(static_cast<std::uint32_t>(L1_CLIENT_SEL));
            detail::l1_client_cmds[zone] = (ctrl & ~llk::perf::L1_CLIENT_ENABLE) | detail::when(counts(zone), llk::perf::L1_CLIENT_ENABLE);
        }
#endif
    }
}

#if defined(ARCH_QUASAR)
// Route the selection and clear the counter: the l1_client CSR is clear-on-read.
inline __attribute__((always_inline)) void route_l1_client(std::uint32_t cmd)
{
    llk::perf::write(llk::perf::l1_client_regs().ctrl, cmd);
    (void)llk::perf::l1_client_read(llk::perf::l1_client_regs());
}
#endif

// The bank registers the readout needs: reference period, continuous mode and the L1 mux group. Registers only: BRISC's
// reads of an L1 config before the release alone moved the TRISCs' timing.
inline void configure_hardware()
{
#if defined(TT_METAL_TTSIM) // ttsim models no counter registers
    return;
#endif
    llk::perf::configure(llk::perf::bank_regs(Bank::INSTRN_THREAD));
    llk::perf::configure(llk::perf::bank_regs(Bank::FPU));
    llk::perf::configure(llk::perf::bank_regs(Bank::TDMA_UNPACK));
    llk::perf::configure(llk::perf::bank_regs(Bank::TDMA_PACK));
#if !defined(ARCH_QUASAR) // no L1 counter bank on Quasar
    llk::perf::set_l1_mux(L1_MUX_GROUP);
    llk::perf::configure(llk::perf::bank_regs(Bank::L1));
#endif
}

// Runs on BRISC before it releases the TRISCs. Quasar has no BRISC: the host configures its counting launches
// (counters.py). The zone arms start the counters and the readouts write the results and the config they used.
inline void configure_from_brisc()
{
    if (counted_zones() != 0)
    {
        configure_hardware();
    }
}

#else
// Counters off: the launch hooks brisc.cpp and trisc.cpp call compile to nothing.
__attribute__((always_inline)) inline void configure_from_brisc()
{
}

__attribute__((always_inline)) inline void load_counted_zones()
{
}

__attribute__((always_inline)) inline void read_last_zone()
{
}
#endif // PERF_COUNTERS_COMPILED

namespace detail
{
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

#if defined(PERF_COUNTERS_COMPILED)
namespace detail
{
static std::uint32_t zone_hashes[PERF_COUNTERS_MAX_ZONES];
static std::uint32_t next_zone_id;
} // namespace detail

// INIT and TILE_LOOP get fixed ids (the host maps ZONE_0 and ZONE_1 to them by index), resolved at compile time:
// an id kept live across the measured loop cost math_matmul a reload per iteration. Other names are allocated after.
constexpr std::uint32_t ZONE_ID_INIT       = 0;
constexpr std::uint32_t ZONE_ID_TILE_LOOP  = 1;
constexpr std::uint32_t FIRST_DYNAMIC_ZONE = 2;

constexpr std::uint32_t fixed_zone_id(std::uint32_t hash_val)
{
    return hash_val == detail::zone_name_hash("INIT")        ? ZONE_ID_INIT
           : hash_val == detail::zone_name_hash("TILE_LOOP") ? ZONE_ID_TILE_LOOP
                                                             : PERF_COUNTERS_MAX_ZONES;
}

__attribute__((always_inline)) inline std::uint32_t get_zone_id(std::uint32_t hash_val)
{
    if (fixed_zone_id(hash_val) < PERF_COUNTERS_MAX_ZONES) // a constant for the two fixed names
    {
        return fixed_zone_id(hash_val);
    }
    std::uint32_t n = detail::next_zone_id < FIRST_DYNAMIC_ZONE ? FIRST_DYNAMIC_ZONE : detail::next_zone_id;
    for (std::uint32_t i = FIRST_DYNAMIC_ZONE; i < n; ++i)
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

#endif // PERF_COUNTERS_COMPILED

// The per-zone measurement layer also needs LLK_PROFILER; BRISC builds only the config above.
#if defined(LLK_PROFILER)

#if defined(PERF_COUNTERS_COMPILED)
static_assert(PERF_COUNTERS_LAYOUT_END <= llk_profiler::EPOCH_ADDR, "Perf counter L1 layout overflows into the profiler region");

// PERF_CNT_ALL reaches only INSTRN_THREAD and FPU; the other banks take the pulse on their own control register.
inline __attribute__((always_inline)) void arm_all_counters(std::uint32_t zone_id)
{
#if defined(TT_METAL_TTSIM)
    return;
#endif
    ckernel::fence_compiler();
    const std::uint32_t arm_cmd = detail::arm_cmds[zone_id];
    llk::perf::write(llk::perf::PERF_CNT_ALL, arm_cmd);
    llk::perf::write(llk::perf::bank_regs(Bank::TDMA_UNPACK).control, arm_cmd);
#if defined(ARCH_QUASAR)
    if constexpr (L1_CLIENT_ENABLED)
    {
        route_l1_client(detail::l1_client_cmds[zone_id]);
    }
#else
    llk::perf::write(llk::perf::bank_regs(Bank::L1).control, arm_cmd);
#endif
    llk::perf::write(llk::perf::bank_regs(Bank::TDMA_PACK).control, arm_cmd);
    ckernel::fence_compiler();
}

inline __attribute__((always_inline)) void freeze_all_counters()
{
#if defined(TT_METAL_TTSIM)
    return;
#endif
    ckernel::fence_compiler();
    llk::perf::stop_all();
    llk::perf::write(llk::perf::bank_regs(Bank::TDMA_UNPACK).control, llk::perf::STOP);
#if defined(ARCH_QUASAR)
    // Stopped before the readout touches L1, or a TRISC-port selection counts the readout into a frozen window.
    if constexpr (L1_CLIENT_ENABLED)
    {
        llk::perf::l1_client_stop(llk::perf::l1_client_regs());
    }
#else
    llk::perf::write(llk::perf::bank_regs(Bank::L1).control, llk::perf::STOP);
#endif
    llk::perf::write(llk::perf::bank_regs(Bank::TDMA_PACK).control, llk::perf::STOP);
    ckernel::fence_compiler();
}

// Register hungry: inlined into a measured window it would reshuffle the registers of the loop (one reload per loop level
// in math_matmul), so it runs only after the windows: in a rendezvous, or on the measured thread in read_last_zone.
inline __attribute__((always_inline)) void read_all_counters([[maybe_unused]] std::uint32_t zone_id)
{
#if defined(TT_METAL_TTSIM)
    return;
#endif
    ckernel::fence_compiler();
    std::uint32_t cycles_base              = PERF_COUNTERS_ZONES_BASE + zone_id * PERF_COUNTERS_ZONE_SIZE;
    volatile std::uint32_t* bank_cycles    = reinterpret_cast<volatile std::uint32_t*>(cycles_base);
    volatile std::uint32_t* counter_counts = bank_cycles + PERF_COUNTERS_BANK_CYCLES_WORDS;
    for (std::uint32_t b = 0; b < COUNTER_BANK_COUNT; ++b)
    {
#if defined(ARCH_QUASAR)
        // Slot 3 has no reference counter; it is armed within cycles of INSTRN, so it borrows the INSTRN reference.
        const BankRegsRef regs = llk::perf::bank_regs(static_cast<Bank>(b));
        bank_cycles[b]         = regs.out_l ? llk::perf::read_ref(regs) : bank_cycles[0];
#else
        bank_cycles[b] = llk::perf::read_ref(llk::perf::bank_regs(static_cast<Bank>(b)));
#endif
    }

    volatile std::uint32_t* cfg = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_SHARED_CONFIG_ADDR);
    std::uint32_t out_idx       = 0;
#pragma GCC unroll 0
    for (std::uint32_t i = 0; i < COUNTER_SLOT_COUNT; ++i)
    {
        // The built-in config, stored for the host's decode next to the counts it describes.
        const std::uint32_t cw = i < BUILTIN_COUNTER_COUNT ? BUILTIN_COUNTER_CONFIG[i] : 0;
        cfg[i]                 = cw;
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
        // Without the readback the read samples the previous counter; a poll that never converges stores the sentinel.
        const bool selected     = llk::perf::select<MODE_REG_POLL_LIMIT>(regs, static_cast<std::uint16_t>(counter_id));
        counter_counts[out_idx] = selected ? llk::perf::read_count(regs) : COUNTER_SELECT_MISSED;
        ++out_idx;
    }

    std::uint32_t sync_addr                               = perf_counters_sync_ctrl_addr(zone_id);
    *reinterpret_cast<volatile std::uint32_t*>(sync_addr) = SYNC_ZONE_COMPLETE | (detail::token << LAUNCH_TOKEN_SHIFT); // this launch's
    ckernel::fence_compiler();
}

inline __attribute__((always_inline)) void freeze_zone(std::uint32_t zone_id)
{
    freeze_all_counters();
    volatile std::uint32_t* pending = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_PENDING_ZONE_ADDR);
    *pending                        = zone_id + 1;
    (void)*pending; // landed in L1 before this thread arrives at the rendezvous the reader waits in
}

// Runs only once every thread is past the frozen zone: inside a rendezvous or in read_last_zone after run_kernel.
inline __attribute__((always_inline)) void read_pending_zone()
{
    volatile std::uint32_t* pending = reinterpret_cast<volatile std::uint32_t*>(PERF_COUNTERS_PENDING_ZONE_ADDR);
    const std::uint32_t zone        = *pending;
    if (zone != 0)
    {
        *pending = 0;
        read_all_counters(zone - 1);
    }
}
#endif // PERF_COUNTERS_COMPILED

constexpr bool is_single_thread_runtype(PerfRunType run_type)
{
    return run_type == PerfRunType::UNPACK_ISOLATE || run_type == PerfRunType::MATH_ISOLATE || run_type == PerfRunType::PACK_ISOLATE ||
           run_type == PerfRunType::SFPU_ISOLATE;
}

// A single thread run type freezes on its measured thread; peers held in an exit barrier would spin through its
// whole measured window. A span needs every thread stopped before the read, so those keep the barrier.
constexpr bool exit_barrier_for(PerfRunType run_type)
{
    return !is_single_thread_runtype(run_type);
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

// A peer of the pack thread in PACK_ISOLATE, at its TILE_LOOP zone: holds its epilogue until pack is done (profiler.h).
constexpr bool holds_quiet(PerfRunType run_type, bool tile_loop)
{
#if defined(LLK_DBG_BARRIER) && defined(LLK_BARRIER_ON_TRISC) && !defined(LLK_PERF_INIT_ONLY)
    // With the release schedule every other thread starts before the measured one, so none of their start-up is in the
    // window; held at their TILE_LOOP end, none of their epilogue is either. L1_TO_L1 ends at pack's end: unpack and math
    // hold until pack is done.
    if (run_type == PerfRunType::L1_TO_L1)
    {
        return tile_loop && llk_barrier::THREAD_ID != 2;
    }
    return tile_loop && is_single_thread_runtype(run_type) && !is_measured_thread(run_type);
#else
    return tile_loop && run_type == PerfRunType::PACK_ISOLATE && !is_measured_thread(run_type);
#endif
}

// The thread that ends the held peers' wait after its TILE_LOOP: the measured thread, or pack in L1_TO_L1.
constexpr bool releases_holds(PerfRunType run_type)
{
#if defined(LLK_DBG_BARRIER) && defined(LLK_BARRIER_ON_TRISC) && !defined(LLK_PERF_INIT_ONLY)
    return run_type == PerfRunType::L1_TO_L1 ? llk_barrier::THREAD_ID == 2 : is_measured_thread(run_type);
#else
    return run_type == PerfRunType::PACK_ISOLATE && is_measured_thread(run_type);
#endif
}

// The thread that reads the last zone of a single thread run type after run_kernel: the measured thread itself, whose
// window has ended, so no peer polls for it. Span run types read in their exit rendezvous.
constexpr bool is_reader_thread(PerfRunType run_type)
{
    return is_single_thread_runtype(run_type) && is_measured_thread(run_type);
}

#if defined(LLK_DBG_BARRIER) && defined(LLK_BARRIER_ON_TRISC) && !defined(LLK_PERF_INIT_ONLY)
// The order in which BRISC releases the threads from the TILE_LOOP park, each by its own flush (barrier.h schedule):
// threads that only wait or idle first, the measured thread last, so every one starts from the same state whatever code
// ran before the park, and the measured window holds no other thread's start-up. 0: all three released together.
constexpr std::uint32_t release_schedule(PerfRunType run_type)
{
    switch (run_type)
    {
        case PerfRunType::UNPACK_ISOLATE:
            return llk_barrier::schedule(2, 1, 0); // idle pack, the math mock that consumes, unpack
        case PerfRunType::MATH_ISOLATE:
            return llk_barrier::schedule(2, 0, 1); // idle pack, the unpack mock that produces, math
        case PerfRunType::PACK_ISOLATE:
            return llk_barrier::schedule(0, 1, 2); // idle unpack and math, pack
        case PerfRunType::L1_TO_L1:
            return llk_barrier::schedule(2, 1, 0); // pack and math wait for data, unpack starts the chain
        default:
            return 0;
    }
}

// The last thread of a schedule names it to BRISC and arms the counters after its release, so they count its window only.
// The INIT measurement build keeps the plain release (its TILE_LOOP is not reported): its code stays as it was, so the
// INIT helpers placed after run_kernel do not move.
constexpr bool releases_last(PerfRunType run_type)
{
    return release_schedule(run_type) != 0 && ((release_schedule(run_type) >> 4) & 3) == llk_barrier::THREAD_ID;
}
#else
constexpr std::uint32_t release_schedule(PerfRunType)
{
    return 0;
}

constexpr bool releases_last(PerfRunType)
{
    return false;
}
#endif

#if defined(PERF_COUNTERS_COMPILED)
namespace detail
{
// A run time flag because trisc.cpp has no PERF_RUN_TYPE of its own; the first zone object of the kernel sets it.
static bool reader_here;
} // namespace detail

// After run_kernel on the measured thread: read back the zone it froze last, if the launch counts it.
inline void read_last_zone()
{
    if (detail::reader_here && counts(ZONE_ID_TILE_LOOP))
    {
        read_pending_zone();
    }
}
#endif // PERF_COUNTERS_COMPILED

// The zone object of both builds: the same rendezvous, with the counter work only in counters on.
template <PerfRunType RUN_TYPE, bool LOOP_PAD = false>
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
#if defined(PERF_COUNTERS_COMPILED)
        if constexpr (is_reader_thread(RUN_TYPE))
        {
            detail::reader_here = true;
        }
#endif
#if defined(LLK_DBG_BARRIER)
        if constexpr (LOOP_PAD && releases_last(RUN_TYPE))
        {
            llk_barrier::announce_schedule(release_schedule(RUN_TYPE));
        }
#endif
#if defined(LLK_PERF_INIT_ONLY)
        llk_barrier::rendezvous<LOOP_PAD ? llk_barrier::PARK_LOOP : llk_barrier::PARK_INIT_TS>(
#else
        llk_barrier::rendezvous<LOOP_PAD>(
#endif
            llk_barrier::is_action_thread(),
            [zid]
            {
#if defined(PERF_COUNTERS_COMPILED)
                // Span run types read at their exit rendezvous, so this read would be dead for them, and not
                // harmless: an L1 read here hung every Wormhole L1_CONGESTION kernel in the math thread.
                if constexpr (is_single_thread_runtype(RUN_TYPE))
                {
                    // An earlier zone's readout: only a launch that counts one (INIT's own launch) takes it.
                    if (detail::counted & ((1u << zid) - 1u))
                    {
                        read_pending_zone();
                    }
                }
                if constexpr (!(LOOP_PAD && release_schedule(RUN_TYPE) != 0))
                {
                    arm_all_counters(zid); // START, or 0 (no command) when the zone is not counted
                }
#endif
            });
#if defined(PERF_COUNTERS_COMPILED)
        if constexpr (LOOP_PAD && releases_last(RUN_TYPE)) // released last, by its own flush: the window starts here
        {
            arm_all_counters(zone_id);
        }
#endif
#if defined(LLK_DBG_BARRIER)
        if constexpr (holds_quiet(RUN_TYPE, LOOP_PAD)) // after the entry release: the level the pack thread flips when done
        {
#if !defined(LLK_PERF_INIT_ONLY) // opaque, as quiet_wait_here (profiler.h): the kernel compiles as if it were not there
            asm volatile(
                "addi  sp, sp, -16\n\tsw    t0, 0(sp)\n\tsw    t1, 4(sp)\n\t"
                "lui   t0, %%hi(%[sem])\n\tlw    t1, %%lo(%[sem])(t0)\n\tandi  t1, t1, 0xff\n\tsw    t1, %[seen]\n\t"
                "lw    t1, 4(sp)\n\tlw    t0, 0(sp)\n\taddi  sp, sp, 16"
                : [seen] "=m"(llk_profiler::quiet_seen)
                : [sem] "i"(PC_BUF_BASE + 4 * (ckernel::PC_BUF_SEMAPHORE_BASE + llk_barrier::RELEASE_SEM))
                : "memory");
#else
            llk_profiler::quiet_seen = llk_barrier::detail::settled(ckernel::semaphore_read(llk_barrier::RELEASE_SEM));
#endif
        }
#endif
        ckernel::fence_compiler();
    }

    inline __attribute__((always_inline)) ~perf_counter_scoped()
    {
        ckernel::fence_compiler();
        [[maybe_unused]] const std::uint32_t zid = zone_id;
        static_assert(
            exit_barrier_for(RUN_TYPE) || is_single_thread_runtype(RUN_TYPE),
            "a run type that skips the exit barrier needs a measured thread in is_measured_thread() to freeze the counters");
        if constexpr (!exit_barrier_for(RUN_TYPE))
        {
#if defined(PERF_COUNTERS_COMPILED)
            if constexpr (is_measured_thread(RUN_TYPE))
            {
                // Every launch: STOP to counters that never started changes nothing.
                freeze_zone(zid);
            }
#endif
        }
        else
        {
#if defined(LLK_DBG_BARRIER)
            if constexpr (LOOP_PAD && releases_holds(RUN_TYPE)) // pack's TILE_LOOP has ended: let the held peers go on
            {
                llk_barrier::detail::flip(llk_barrier::RELEASE_SEM);
            }
#endif
            llk_barrier::rendezvous(
                llk_barrier::is_action_thread(),
                [zid]
                {
#if defined(PERF_COUNTERS_COMPILED)
                    freeze_all_counters();
                    if (counts(zid))
                    {
                        read_all_counters(zid);
                    }
#endif
                });
        }
        ckernel::fence_compiler();
    }
};
#endif // LLK_PROFILER

#if !defined(LLK_PROFILER) && defined(PERF_COUNTERS_COMPILED)
// Without the profiler there are no zones, so there is nothing to read after run_kernel.
inline void read_last_zone()
{
}
#endif

} // namespace llk_perf

#if defined(LLK_PROFILER)
#define PERF_COUNTER_VAR_CONCAT_(a, b) a##b
#define PERF_COUNTER_VAR_(line)        PERF_COUNTER_VAR_CONCAT_(_perf_ctr_, line)
#define LLK_IS_TILE_LOOP_(zone_name)   (llk_perf::detail::zone_name_hash(zone_name) == llk_perf::detail::zone_name_hash("TILE_LOOP"))
#if defined(PERF_COUNTERS_COMPILED)
#define LLK_PERF_ZONE_ID_(zone_name) llk_perf::get_zone_id(llk_perf::detail::zone_name_hash(zone_name))
#else
#define LLK_PERF_ZONE_ID_(zone_name) 0u // counters off: no zone needs an id
#endif
#define MEASURE_PERF_COUNTERS(zone_name) \
    const llk_perf::perf_counter_scoped<PERF_RUN_TYPE, LLK_IS_TILE_LOOP_(zone_name)> PERF_COUNTER_VAR_(__LINE__)(LLK_PERF_ZONE_ID_(zone_name));
#else
#define MEASURE_PERF_COUNTERS(zone_name)
#endif

// INIT measurement build (LLK_PERF_INIT_ONLY, launched by perf/core.py): times INIT's body from fixed sections ahead of
// main; the spin after the end read and the closing rendezvous keep every thread's epilogue out of the other INITs.
#if defined(LLK_PROFILER) && defined(LLK_PERF_INIT_ONLY) && defined(LLK_DBG_BARRIER)
namespace llk_perf
{
// INIT ends inside its body, read at the end of the body (init_measurement_no_zone), so the trampoline's return is not
// timed: the prefetch stream's run past the body cannot meet the return line in the icache
inline std::uint32_t init_end_ts[2] = {0, 0};

// no .balign as at the trampoline's read: inside the body its padding would change with the code before it
__attribute__((always_inline)) inline void read_init_end(std::uint32_t (&ts)[2])
{
    std::uint32_t lo, hi;
    asm volatile("lui %[h], %%hi(%[c])\n\tlw %[l], %%lo(%[c])(%[h])\n\tlw %[h], %%lo(%[c] + 8)(%[h])"
                 : [l] "=&r"(lo), [h] "=&r"(hi)
                 : [c] "i"(RISCV_DEBUG_REG_WALL_CLOCK_L)
                 : "memory");
    ts[0] = lo;
    ts[1] = hi;
}

// LLK_INIT_ZONE_END (perf.h): INIT's end in a block that goes on to prepare the loop's data
inline std::uint32_t init_zone_end_ts[2] = {0, 0};

__attribute__((always_inline)) inline void init_zone_end()
{
    ckernel::tensix_sync(); // the thread's own init done, as PROFILER_SYNC ends every other INIT
    read_init_end(init_zone_end_ts);
}

// ZONE_END: the body ended INIT's zone at LLK_INIT_ZONE_END (LLK_INIT_PREP_END)
template <PerfRunType RUN_TYPE, typename F, bool ZONE_END = false>
__attribute__((noipa, section(".llk_init_tramp"))) void init_zone(const F& body)
{
    const bool opened = !llk_profiler::is_buffer_full();
    if (opened)
    {
        llk_profiler::zone_reserve();
    }
    std::uint32_t end_lo, end_hi;
    // the id and its profiler_meta entry on one line: MARKER_ID hashes the file name and line
    PROFILER_META(MARKER_FULL("INIT")) constexpr std::uint16_t init_id = MARKER_ID("INIT");
    {
        const perf_counter_scoped<RUN_TYPE, false> counters(LLK_PERF_ZONE_ID_("INIT"));
        body();
        asm volatile( // INIT has ended in the body: this read and spin only hold every thread's epilogue
            ".balign 16\n\tlui %[h], %%hi(%[c])\n\tlw %[l], %%lo(%[c])(%[h])\n\tlw %[h], %%lo(%[c] + 8)(%[h])\n\t"
            "li t0, 2048\n1:\n\taddi t0, t0, -1\n\tbnez t0, 1b"
            : [l] "=&r"(end_lo), [h] "=&r"(end_hi)
            : [c] "i"(RISCV_DEBUG_REG_WALL_CLOCK_L)
            : "t0", "memory");
        end_lo = ZONE_END ? init_zone_end_ts[0] : init_end_ts[0];
        end_hi = ZONE_END ? init_zone_end_ts[1] : init_end_ts[1];
        if (opened)
        {
            std::uint32_t id; // fixed size lui + addi, after the end read, whatever the id's value
            asm volatile("lui %0, %%hi(%1)\n\taddi %0, %0, %%lo(%1)" : "=r"(id) : "i"(init_id));
            llk_profiler::zone_record(
                static_cast<std::uint16_t>(id),
                (static_cast<std::uint64_t>(llk_barrier::init_start_ts[1]) << 32) | llk_barrier::init_start_ts[0],
                (static_cast<std::uint64_t>(end_hi) << 32) | end_lo);
        }
    }
    if constexpr (!exit_barrier_for(RUN_TYPE)) // no INIT exit rendezvous: hold here until every INIT has ended
    {
        llk_barrier::rendezvous<llk_barrier::PARK_PLAIN>(llk_barrier::is_action_thread(), [] {});
    }
    ckernel::icache_prefetch_init_end(); // trisc.cpp turned it on; the loop runs without it, as after tt-metal's first wait
}
} // namespace llk_perf

namespace llk_perf
{
struct init_measurement_second_object // START_PERF_MEASURE("INIT")'s second object: math's head start runs once
{
    init_measurement_second_object()
    {
    }
};

struct init_measurement_no_zone // START_PERF_MEASURE("INIT") in the INIT measurement build: the trampoline opens the zone
{
    explicit init_measurement_no_zone(std::uint32_t) // the top of INIT's body, where math's init starts as in trisck.cc
    {
#if defined(LLK_TRISC_MATH)
        ckernel::icache_prefetch_head_start();
#else
        // tt-metal reaches unpack / pack init code mid-stream, so INIT starts after a 2-line lead past the cold jal
        // entry; the restores and jal ahead of it leave the window too
        asm volatile(".rept 8\n\tnop\n\t.endr" ::: "memory");
        read_init_end(llk_barrier::init_start_ts);
#endif
    }

    init_measurement_no_zone()
    {
    }

    ~init_measurement_no_zone() // the end of INIT's body: INIT's end
    {
        read_init_end(init_end_ts);
    }
};
} // namespace llk_perf

// The rest of the kernel runs as in the TILE_LOOP build (its TILE_LOOP zone is recorded but not reported), so every
// peer's work that INIT starts or waits for (data INIT unpacks or computes, semaphores a loop posts) completes.
#define START_PERF_MEASURE(zone_name)                                                                                                              \
    const std::conditional_t<LLK_IS_TILE_LOOP_(zone_name), llk_perf::perf_counter_scoped<PERF_RUN_TYPE, true>, llk_perf::init_measurement_no_zone> \
        PERF_COUNTER_VAR_(__LINE__)(LLK_PERF_ZONE_ID_(zone_name));                                                                                 \
    PROFILER_META(MARKER_FULL(zone_name))                                                                                                          \
    const std::conditional_t<                                                                                                                      \
        LLK_IS_TILE_LOOP_(zone_name),                                                                                                              \
        llk_profiler::zone_scoped<MARKER_ID(zone_name), true, llk_perf::holds_quiet(PERF_RUN_TYPE, LLK_IS_TILE_LOOP_(zone_name))>,                 \
        llk_perf::init_measurement_second_object>                                                                                                  \
        _zone_scoped_;
#elif defined(LLK_PROFILER)
#define START_PERF_MEASURE(zone_name)                  \
    LLK_ZONE_PRERESERVE_(LLK_IS_TILE_LOOP_(zone_name)) \
    MEASURE_PERF_COUNTERS(zone_name)                   \
    ZONE_SCOPED_Q(zone_name, llk_perf::holds_quiet(PERF_RUN_TYPE, LLK_IS_TILE_LOOP_(zone_name)))
#else
#define START_PERF_MEASURE(zone_name) MEASURE_PERF_COUNTERS(zone_name)
#endif
