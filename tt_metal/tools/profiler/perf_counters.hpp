// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/types.h"

// Global so the host profiler can keep reflecting the enumerator names unqualified.
using llk::perf::PerfCounterType;

constexpr std::uint16_t PERF_COUNTER_PROFILER_ID = 9090;

// Profiler groups: the index of every *_for_group table and the bit order of PROFILE_PERF_COUNTERS_*.
enum PerfCounterGroup : std::uint8_t { FPU, PACK, UNPACK, L1_0, L1_1, INSTRN, L1_2, L1_3, L1_4, L1_5 };

union PerfCounter {
    struct {
        std::uint32_t counter_value;
        std::uint32_t ref_cnt;
        std::uint32_t counter_type : 16;
        // NEO the record came from: Quasar's DM0 reads all four; always 0 on tt-1xx.
        std::uint32_t neo : 4;
        std::uint32_t unused : 12;
    } __attribute__((packed));
    struct {
        std::uint64_t raw_data_1;
        std::uint64_t raw_data_2;
    } __attribute__((packed));
    // The device constructors fill this view: one plain store per word instead of bit-field inserts.
    struct {
        std::uint32_t value_word;
        std::uint32_t ref_word;
        std::uint32_t type_word;  // counter_type | neo << NEO_SHIFT, unused bits 0
        std::uint32_t pad_word;
    } __attribute__((packed));

    static constexpr std::uint32_t NEO_SHIFT = 16;

    PerfCounter() = delete;
    PerfCounter(
        std::uint32_t counter_value, std::uint32_t ref_cnt, PerfCounterType counter_type, std::uint32_t neo = 0) :
        value_word(counter_value),
        ref_word(ref_cnt),
        type_word(static_cast<std::uint32_t>(counter_type) | (neo << NEO_SHIFT)) {}
    // Raw counter_type, for the l1_client encoding above the enum.
    PerfCounter(std::uint32_t counter_value, std::uint32_t ref_cnt, std::uint32_t counter_type_raw, std::uint32_t neo) :
        value_word(counter_value), ref_word(ref_cnt), type_word(counter_type_raw | (neo << NEO_SHIFT)) {}

    PerfCounter(std::uint64_t raw_data_1, std::uint64_t raw_data_2) : raw_data_1(raw_data_1), raw_data_2(raw_data_2) {}
};
static_assert(sizeof(PerfCounter) == sizeof(std::uint64_t) * 2, "PerfCounter must be 128-bit");

// The RISC that orchestrates the kernel owns the counters. tt-1xx: TRISC1 arms and stops, BRISC reads
// once the TRISCs are done. Quasar: DM0 arms all four NEOs before GO and stops and reads them after DONE.
#if defined(ARCH_QUASAR)
#if defined(COMPILE_FOR_DM)
#define PERF_COUNTER_WRAP_RISC 1
#define PERF_COUNTER_READ_RISC 1
#endif
#else
#if COMPILE_FOR_TRISC == 1
#define PERF_COUNTER_WRAP_RISC 1
#endif
#if defined(COMPILE_FOR_BRISC)
#define PERF_COUNTER_READ_RISC 1
#endif
#endif

#if defined(PROFILE_PERF_COUNTERS) && (defined(PERF_COUNTER_WRAP_RISC) || defined(PERF_COUNTER_READ_RISC))

#include <array>
#include <utility>

#include "kernel_profiler.hpp"

// Quasar's DM firmware has a 2 KB RW data region and its linker script folds .rodata into it; the constant
// tables go to the 12 KB text region instead (the shared headers honour LLK_PERF_TABLES_IN_TEXT).
#if defined(ARCH_QUASAR)
#define LLK_PERF_TABLES_IN_TEXT
// Own section name: the shared tables are COMDAT and cannot share a section with these local ones.
#define PERF_COUNTER_TABLE __attribute__((section(".text.perf_counter_groups")))
#else
#define PERF_COUNTER_TABLE
#endif

#include "perf_counters/inventory.h"
#include "perf_counters/registers.h"
#include "perf_counters/hw.h"

namespace kernel_profiler {

// bit masks for the different counter groups
#define PROFILE_PERF_COUNTERS_FPU (1 << 0)
#define PROFILE_PERF_COUNTERS_PACK (1 << 1)
#define PROFILE_PERF_COUNTERS_UNPACK (1 << 2)
#define PROFILE_PERF_COUNTERS_L1_0 (1 << 3)
#define PROFILE_PERF_COUNTERS_L1_1 (1 << 4)
#define PROFILE_PERF_COUNTERS_INSTRN (1 << 5)
#define PROFILE_PERF_COUNTERS_L1_2 (1 << 6)
#define PROFILE_PERF_COUNTERS_L1_3 (1 << 7)
#define PROFILE_PERF_COUNTERS_L1_4 (1 << 8)
#define PROFILE_PERF_COUNTERS_L1_5 (1 << 9)

#if defined(ARCH_QUASAR) &&                                                                                            \
    ((PROFILE_PERF_COUNTERS) & (PROFILE_PERF_COUNTERS_L1_0 | PROFILE_PERF_COUNTERS_L1_1 | PROFILE_PERF_COUNTERS_L1_2 | \
                                PROFILE_PERF_COUNTERS_L1_3 | PROFILE_PERF_COUNTERS_L1_4 | PROFILE_PERF_COUNTERS_L1_5))
#error "Quasar has no L1 perf counter groups; valid bits are FPU(1)|PACK(2)|UNPACK(4)|INSTRN(32) = 39"
#endif

// Counter groups and their corresponding enable bitmask bits. Shared; used on both
// the wrap thread (start/stop loop) and the read thread (read loop).
constexpr std::pair<PerfCounterGroup, std::uint32_t> counter_group_flags[] PERF_COUNTER_TABLE = {
    {PerfCounterGroup::FPU, PROFILE_PERF_COUNTERS_FPU},
    {PerfCounterGroup::PACK, PROFILE_PERF_COUNTERS_PACK},
    {PerfCounterGroup::UNPACK, PROFILE_PERF_COUNTERS_UNPACK},
#if !defined(ARCH_QUASAR)
    {PerfCounterGroup::L1_0, PROFILE_PERF_COUNTERS_L1_0},
    {PerfCounterGroup::L1_1, PROFILE_PERF_COUNTERS_L1_1},
#endif
    {PerfCounterGroup::INSTRN, PROFILE_PERF_COUNTERS_INSTRN},
#if !defined(ARCH_QUASAR)
    {PerfCounterGroup::L1_2, PROFILE_PERF_COUNTERS_L1_2},
    {PerfCounterGroup::L1_3, PROFILE_PERF_COUNTERS_L1_3},
    {PerfCounterGroup::L1_4, PROFILE_PERF_COUNTERS_L1_4},
    {PerfCounterGroup::L1_5, PROFILE_PERF_COUNTERS_L1_5},
#endif
};
constexpr std::uint32_t NUM_COUNTER_GROUPS = sizeof(counter_group_flags) / sizeof(counter_group_flags[0]);

// Indexed by PerfCounterGroup; keep the enum order. The six L1 groups are the one L1 bank at different
// mux positions.
constexpr llk::perf::Bank bank_for_group[10] PERF_COUNTER_TABLE = {
    llk::perf::Bank::FPU,            // FPU
    llk::perf::Bank::TDMA_PACK,      // PACK
    llk::perf::Bank::TDMA_UNPACK,    // UNPACK
    llk::perf::Bank::L1,             // L1_0
    llk::perf::Bank::L1,             // L1_1
    llk::perf::Bank::INSTRN_THREAD,  // INSTRN
    llk::perf::Bank::L1,             // L1_2
    llk::perf::Bank::L1,             // L1_3
    llk::perf::Bank::L1,             // L1_4
    llk::perf::Bank::L1,             // L1_5
};

constexpr std::uint8_t l1_mux_for_group[10] PERF_COUNTER_TABLE = {
    0,  // FPU (unused)
    0,  // PACK (unused)
    0,  // UNPACK (unused)
    0,  // L1_0
    1,  // L1_1
    0,  // INSTRN (unused)
    2,  // L1_2
    3,  // L1_3
    4,  // L1_4
    5,  // L1_5
};

// The per-group functions take the NEO index only on Quasar (DM0 serves four NEOs through the NoC window).
// tt-1xx has one Tensix, and keeping the parameter out leaves its call sites and code as they were.
#if defined(ARCH_QUASAR)
constexpr std::uint32_t NUM_NEOS = llk::perf::NUM_NEOS;
#define PERF_COUNTER_NEO_PARAM , std::uint32_t neo
#define PERF_COUNTER_NEO_ARG(n) , n

inline llk::perf::BankRegs regs_for(PerfCounterGroup counter_group PERF_COUNTER_NEO_PARAM) {
    return llk::perf::bank_regs(bank_for_group[counter_group], llk::perf::neo_window(neo));
}
#else
constexpr std::uint32_t NUM_NEOS = 1;
constexpr std::uint32_t neo = 0;
#define PERF_COUNTER_NEO_PARAM
#define PERF_COUNTER_NEO_ARG(n)

// Register block per group, resolved at compile time so the runtime path is one indexed load.
constexpr std::array<const llk::perf::BankRegs*, 10> regs_for_group = [] {
    std::array<const llk::perf::BankRegs*, 10> regs{};
    for (std::uint32_t g = 0; g < 10; g++) {
        regs[g] = &llk::perf::bank_regs(bank_for_group[g]);
    }
    return regs;
}();

inline const llk::perf::BankRegs& regs_for(PerfCounterGroup counter_group) { return *regs_for_group[counter_group]; }
#endif

#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
constexpr std::uint32_t QUASAR_L1_CLIENT_SEL = PROFILE_PERF_COUNTERS_L1_SEL;
static_assert(
    llk::perf::l1_client_selection_is_valid(QUASAR_L1_CLIENT_SEL),
    "PROFILE_PERF_COUNTERS_L1_SEL must be subport*8 + event below 296 with event not 0; THCON events 1-3 alias "
    "selections 1-3");
static_assert(
    llk::perf::QUASAR_L1_CLIENT_EVENT_BASE + llk::perf::QUASAR_L1_CLIENT_NUM_SELECTIONS < (1u << 16),
    "l1_client encoding must fit the 16-bit counter_type");
// The CSR has no reference counter; the wall-clock span from arming to freezing is the denominator.
static std::uint32_t l1_client_start_cycles;
static std::uint32_t l1_client_elapsed_cycles;

inline void start_l1_client_event_counter(std::uint32_t neo) {
    llk::perf::l1_client_start(llk::perf::l1_client_regs(llk::perf::neo_window(neo)), QUASAR_L1_CLIENT_SEL);
}

inline void stop_l1_client_event_counter(std::uint32_t neo) {
    llk::perf::l1_client_stop(llk::perf::l1_client_regs(llk::perf::neo_window(neo)));
}
#endif  // ARCH_QUASAR && PROFILE_PERF_COUNTERS_L1_SEL

#if defined(PERF_COUNTER_WRAP_RISC)
// --- Wrap thread only: start/stop counters around the compute kernel -------

__attribute__((noinline)) void start_single_group(PerfCounterGroup counter_group PERF_COUNTER_NEO_PARAM) {
#if !defined(ARCH_QUASAR)
    if (bank_for_group[counter_group] == llk::perf::Bank::L1) {
        llk::perf::set_l1_mux(l1_mux_for_group[counter_group]);
    }
#endif
    const auto& regs = regs_for(counter_group PERF_COUNTER_NEO_ARG(neo));
    llk::perf::configure(regs);
    llk::perf::start(regs);
}

__attribute__((noinline)) void stop_single_group(PerfCounterGroup counter_group PERF_COUNTER_NEO_PARAM) {
    llk::perf::stop(regs_for(counter_group PERF_COUNTER_NEO_ARG(neo)));
}

void start_perf_counter() {
    for (std::uint32_t n = 0; n < NUM_NEOS; n++) {
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
        start_l1_client_event_counter(n);
#endif
        for (std::uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
            if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
                start_single_group(counter_group_flags[i].first PERF_COUNTER_NEO_ARG(n));
            }
        }
    }
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
    l1_client_start_cycles = static_cast<std::uint32_t>(quasar_read_wall_clock_64());
#endif
}

void stop_perf_counter() {
    for (std::uint32_t n = 0; n < NUM_NEOS; n++) {
        for (std::uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
            if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
                stop_single_group(counter_group_flags[i].first PERF_COUNTER_NEO_ARG(n));
            }
        }
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
        stop_l1_client_event_counter(n);
#endif
    }
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
    l1_client_elapsed_cycles = static_cast<std::uint32_t>(quasar_read_wall_clock_64()) - l1_client_start_cycles;
#endif
}

struct PerfCounterWrapper {
    PerfCounterWrapper() { kernel_profiler::start_perf_counter(); }
    ~PerfCounterWrapper() { kernel_profiler::stop_perf_counter(); }
};

#endif  // PERF_COUNTER_WRAP_RISC

#if defined(PERF_COUNTER_READ_RISC)
// --- Read thread only: counter readout (BRISC on tt-1xx, DM0 on Quasar) ---

// Select table per group (same ordering as bank_for_group).
constexpr std::array<llk::perf::Table, 10> table_for_group PERF_COUNTER_TABLE = [] {
    std::array<llk::perf::Table, 10> tables{};
    for (std::uint32_t g = 0; g < 10; g++) {
        tables[g] = llk::perf::table_for(bank_for_group[g], l1_mux_for_group[g]);
    }
    return tables;
}();

#if defined(ARCH_QUASAR)
// Quasar's profiler is L1-only (about 80 records per risc buffer), so DM0 files each NEO's readout into that
// NEO's own TRISC buffers (complete after wait_subordinates()); the record's neo field says which NEO.
constexpr std::uint32_t TRISCS_PER_NEO = 4;
constexpr std::uint32_t PERF_RECORD_WORDS = PROFILER_L1_MARKER_UINT32_SIZE * 3;  // marker + two 64-bit words

FORCE_INLINE std::uint32_t first_trisc_of(std::uint32_t neo) {
    return static_cast<std::uint32_t>(TensixProcessorTypes::E0_MATH0) + neo * TRISCS_PER_NEO;
}

FORCE_INLINE bool neo_enabled(std::uint32_t enables, std::uint32_t neo) {
    return (enables >> first_trisc_of(neo)) & 1u;
}

struct SpillCursor {
    std::uint32_t risc;
    std::uint32_t last_risc;
    std::uint32_t index;
};
static SpillCursor spill;

FORCE_INLINE std::uint32_t spill_end_index(std::uint32_t risc) {
    std::uint32_t index = profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + risc];
    // A TRISC that published nothing has no run sentinel for the host to anchor on; leave it alone.
    return index < CUSTOM_MARKERS ? PROFILER_L1_VECTOR_SIZE : index;
}

inline void spill_open(std::uint32_t neo) {
    spill.risc = first_trisc_of(neo);
    spill.last_risc = spill.risc + TRISCS_PER_NEO - 1;
    spill.index = spill_end_index(spill.risc);
}

inline void spill_close() { profiler_control_buffer[DEVICE_BUFFER_END_INDEX_BR_ER + spill.risc] = spill.index; }

inline void spill_record(const PerfCounter& counter) {
    while (spill.index + PERF_RECORD_WORDS > PROFILER_L1_VECTOR_SIZE) {
        if (spill.risc == spill.last_risc) {
            mark_dropped_timestamps(spill.risc);  // all four buffers full: let the host's dropped-markers warning fire
            return;
        }
        spill_close();
        spill.risc++;
        spill.index = spill_end_index(spill.risc);
    }
    volatile tt_l1_ptr std::uint32_t* data = profiler_data_buffer[spill.risc].data;
    std::uint64_t wall_clock = quasar_read_wall_clock_64();
    data[spill.index] =
        PROFILER_MARKER_VALID |
        ((get_const_id(PERF_COUNTER_PROFILER_ID, PacketTypes::TS_DATA_16B) & PROFILER_MARKER_TIMER_ID_MASK)
         << PROFILER_MARKER_TIMER_ID_SHIFT) |
        (static_cast<std::uint32_t>(wall_clock >> 32) & PROFILER_MARKER_TS_HIGH_MASK);
    data[spill.index + 1] = static_cast<std::uint32_t>(wall_clock);
    data[spill.index + 2] = counter.raw_data_1 >> 32;
    data[spill.index + 3] = static_cast<std::uint32_t>(counter.raw_data_1);
    data[spill.index + 4] = counter.raw_data_2 >> 32;
    data[spill.index + 5] = static_cast<std::uint32_t>(counter.raw_data_2);
    spill.index += PERF_RECORD_WORDS;
}
#endif  // ARCH_QUASAR

inline void emit_record(const PerfCounter& counter) {
#if defined(ARCH_QUASAR)
    spill_record(counter);
#else
    kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>(
        kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE * 2);
    kernel_profiler::timeStampedData<
        PERF_COUNTER_PROFILER_ID,
        kernel_profiler::DoingDispatch::DISPATCH,
        kernel_profiler::PacketTypes::TS_DATA_16B>(counter.raw_data_1, counter.raw_data_2);
#endif
}

#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
inline void read_l1_client_event_counter(std::uint32_t neo) {
    const std::uint32_t count = llk::perf::l1_client_read(llk::perf::l1_client_regs(llk::perf::neo_window(neo)));
    PerfCounter counter(
        count, l1_client_elapsed_cycles, llk::perf::QUASAR_L1_CLIENT_EVENT_BASE + QUASAR_L1_CLIENT_SEL, neo);
    emit_record(counter);
}
#endif

// The mode readback poll is unbounded here (PollLimit 0): BRISC firmware is within a few bytes of its
// size limit, and the poll never fails on hardware.
__attribute__((noinline)) void read_single_group(PerfCounterGroup counter_group PERF_COUNTER_NEO_PARAM) {
    const auto& regs = regs_for(counter_group PERF_COUNTER_NEO_ARG(neo));
    llk::perf::read_table<0>(
        regs, table_for_group[counter_group], [&](PerfCounterType type, std::uint32_t ref_cnt, std::uint32_t value) {
            PerfCounter counter(value, ref_cnt, type, neo);
            emit_record(counter);
        });
    // Toggle start bit to clear the counters for this group
    llk::perf::start(regs);
}

// One L1 group per pass at most; its mux position is routed here so passes without L1 carry no mux code.
// trisc_enables is the launch message's processor enable mask; Quasar skips NEOs that ran nothing.
void read_perf_counters([[maybe_unused]] std::uint32_t trisc_enables) {
    if (kernel_profiler::get_profiler_zone_invalid()) {
        return;
    }
    for (std::uint32_t n = 0; n < NUM_NEOS; n++) {
#if defined(ARCH_QUASAR)
        if (!neo_enabled(trisc_enables, n)) {
            continue;
        }
        spill_open(n);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_FPU
        read_single_group(PerfCounterGroup::FPU PERF_COUNTER_NEO_ARG(n));
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_PACK
        read_single_group(PerfCounterGroup::PACK PERF_COUNTER_NEO_ARG(n));
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_UNPACK
        read_single_group(PerfCounterGroup::UNPACK PERF_COUNTER_NEO_ARG(n));
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_0
        llk::perf::set_l1_mux(0);
        read_single_group(PerfCounterGroup::L1_0 PERF_COUNTER_NEO_ARG(n));
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_1
        llk::perf::set_l1_mux(1);
        read_single_group(PerfCounterGroup::L1_1 PERF_COUNTER_NEO_ARG(n));
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_INSTRN
        read_single_group(PerfCounterGroup::INSTRN PERF_COUNTER_NEO_ARG(n));
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_2
        llk::perf::set_l1_mux(2);
        read_single_group(PerfCounterGroup::L1_2 PERF_COUNTER_NEO_ARG(n));
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_3
        llk::perf::set_l1_mux(3);
        read_single_group(PerfCounterGroup::L1_3 PERF_COUNTER_NEO_ARG(n));
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_4
        llk::perf::set_l1_mux(4);
        read_single_group(PerfCounterGroup::L1_4 PERF_COUNTER_NEO_ARG(n));
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_5
        llk::perf::set_l1_mux(5);
        read_single_group(PerfCounterGroup::L1_5 PERF_COUNTER_NEO_ARG(n));
#endif
#if defined(ARCH_QUASAR) && defined(PROFILE_PERF_COUNTERS_L1_SEL)
        read_l1_client_event_counter(n);
#endif
#if defined(ARCH_QUASAR)
        spill_close();
#endif
    }
}

#endif  // PERF_COUNTER_READ_RISC

}  // namespace kernel_profiler

#if defined(PERF_COUNTER_WRAP_RISC)
#define StartPerfCounters() kernel_profiler::start_perf_counter();
#define StopPerfCounters() kernel_profiler::stop_perf_counter();
#define RecordPerfCounters() kernel_profiler::PerfCounterWrapper _perf_counter_wrapper_;
#else
#define StartPerfCounters()
#define StopPerfCounters()
#define RecordPerfCounters()
#endif

#if defined(PERF_COUNTER_READ_RISC)
#define ReadPerfCounters(trisc_enables) kernel_profiler::read_perf_counters(trisc_enables);
#else
#define ReadPerfCounters(trisc_enables)
#endif

#else

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define ReadPerfCounters(trisc_enables)
#define RecordPerfCounters()

#endif
