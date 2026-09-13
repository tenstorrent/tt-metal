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
        std::uint32_t counter_type : 8;
        std::uint32_t unused : 24;
    } __attribute__((packed));
    struct {
        std::uint64_t raw_data_1;
        std::uint64_t raw_data_2;
    } __attribute__((packed));

    PerfCounter() = delete;
    PerfCounter(std::uint32_t counter_value, std::uint32_t ref_cnt, PerfCounterType counter_type) :
        counter_value(counter_value), ref_cnt(ref_cnt), counter_type(static_cast<std::uint32_t>(counter_type)) {}

    PerfCounter(std::uint64_t raw_data_1, std::uint64_t raw_data_2) : raw_data_1(raw_data_1), raw_data_2(raw_data_2) {}
};
static_assert(sizeof(PerfCounter) == sizeof(std::uint64_t) * 2, "PerfCounter must be 128-bit");

// Perf counter start/stop runs on TRISC1 (wraps the compute kernel).
// Counter readout and DRAM push runs on BRISC (has NOC access for DRAM writes).
#if defined(PROFILE_PERF_COUNTERS) && (COMPILE_FOR_TRISC == 1 || defined(COMPILE_FOR_BRISC))

#include <array>
#include <utility>

#include "kernel_profiler.hpp"
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

// Counter groups and their corresponding enable bitmask bits. Shared; used on both
// TRISC1 (start/stop loop) and BRISC (read loop).
constexpr std::pair<PerfCounterGroup, std::uint32_t> counter_group_flags[] = {
    {PerfCounterGroup::FPU, PROFILE_PERF_COUNTERS_FPU},
    {PerfCounterGroup::PACK, PROFILE_PERF_COUNTERS_PACK},
    {PerfCounterGroup::UNPACK, PROFILE_PERF_COUNTERS_UNPACK},
    {PerfCounterGroup::L1_0, PROFILE_PERF_COUNTERS_L1_0},
    {PerfCounterGroup::L1_1, PROFILE_PERF_COUNTERS_L1_1},
    {PerfCounterGroup::INSTRN, PROFILE_PERF_COUNTERS_INSTRN},
    {PerfCounterGroup::L1_2, PROFILE_PERF_COUNTERS_L1_2},
    {PerfCounterGroup::L1_3, PROFILE_PERF_COUNTERS_L1_3},
    {PerfCounterGroup::L1_4, PROFILE_PERF_COUNTERS_L1_4},
    {PerfCounterGroup::L1_5, PROFILE_PERF_COUNTERS_L1_5},
};
constexpr std::uint32_t NUM_COUNTER_GROUPS = sizeof(counter_group_flags) / sizeof(counter_group_flags[0]);

// Indexed by PerfCounterGroup; keep the enum order. The six L1 groups are the one L1 bank at different
// mux positions.
constexpr llk::perf::Bank bank_for_group[10] = {
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

constexpr std::uint8_t l1_mux_for_group[10] = {
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

// Register block per group, resolved at compile time so the runtime path is one indexed load.
constexpr std::array<const llk::perf::BankRegs*, 10> regs_for_group = [] {
    std::array<const llk::perf::BankRegs*, 10> regs{};
    for (std::uint32_t g = 0; g < 10; g++) {
        regs[g] = &llk::perf::bank_regs(bank_for_group[g]);
    }
    return regs;
}();

#if COMPILE_FOR_TRISC == 1
// --- TRISC1-only: start/stop counters around the compute kernel ------------

__attribute__((noinline)) void start_single_group(PerfCounterGroup counter_group) {
    if (bank_for_group[counter_group] == llk::perf::Bank::L1) {
        llk::perf::set_l1_mux(l1_mux_for_group[counter_group]);
    }
    const llk::perf::BankRegs& regs = *regs_for_group[counter_group];
    llk::perf::configure(regs);
    llk::perf::start(regs);
}

__attribute__((noinline)) void stop_single_group(PerfCounterGroup counter_group) {
    llk::perf::stop(*regs_for_group[counter_group]);
}

void start_perf_counter() {
    for (std::uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
        if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
            start_single_group(counter_group_flags[i].first);
        }
    }
}

void stop_perf_counter() {
    for (std::uint32_t i = 0; i < NUM_COUNTER_GROUPS; i++) {
        if (PROFILE_PERF_COUNTERS & counter_group_flags[i].second) {
            stop_single_group(counter_group_flags[i].first);
        }
    }
}

struct PerfCounterWrapper {
    PerfCounterWrapper() { kernel_profiler::start_perf_counter(); }
    ~PerfCounterWrapper() { kernel_profiler::stop_perf_counter(); }
};

#endif  // COMPILE_FOR_TRISC == 1

#if defined(COMPILE_FOR_BRISC)
// --- BRISC-only: counter readout and DRAM push -----------------------------

// Select table per group (same ordering as bank_for_group).
constexpr std::array<llk::perf::Table, 10> table_for_group = [] {
    std::array<llk::perf::Table, 10> tables{};
    for (std::uint32_t g = 0; g < 10; g++) {
        tables[g] = llk::perf::table_for(bank_for_group[g], l1_mux_for_group[g]);
    }
    return tables;
}();

// The mode readback poll is unbounded here (PollLimit 0): BRISC firmware is within a few bytes of its
// size limit, and the poll never fails on hardware.
__attribute__((noinline)) void read_single_group(PerfCounterGroup counter_group) {
    const llk::perf::BankRegs& regs = *regs_for_group[counter_group];
    llk::perf::read_table<0>(
        regs, table_for_group[counter_group], [](PerfCounterType type, std::uint32_t ref_cnt, std::uint32_t value) {
            PerfCounter counter(value, ref_cnt, type);
            kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>(
                kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE * 2);
            kernel_profiler::timeStampedData<
                PERF_COUNTER_PROFILER_ID,
                kernel_profiler::DoingDispatch::DISPATCH,
                kernel_profiler::PacketTypes::TS_DATA_16B>(counter.raw_data_1, counter.raw_data_2);
        });
    // Toggle start bit to clear the counters for this group
    llk::perf::start(regs);
}

// One L1 group per pass at most; its mux position is routed here so passes without L1 carry no mux code.
void read_perf_counters() {
    if (kernel_profiler::get_profiler_zone_invalid()) {
        return;
    }
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_FPU
    read_single_group(PerfCounterGroup::FPU);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_PACK
    read_single_group(PerfCounterGroup::PACK);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_UNPACK
    read_single_group(PerfCounterGroup::UNPACK);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_0
    llk::perf::set_l1_mux(0);
    read_single_group(PerfCounterGroup::L1_0);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_1
    llk::perf::set_l1_mux(1);
    read_single_group(PerfCounterGroup::L1_1);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_INSTRN
    read_single_group(PerfCounterGroup::INSTRN);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_2
    llk::perf::set_l1_mux(2);
    read_single_group(PerfCounterGroup::L1_2);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_3
    llk::perf::set_l1_mux(3);
    read_single_group(PerfCounterGroup::L1_3);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_4
    llk::perf::set_l1_mux(4);
    read_single_group(PerfCounterGroup::L1_4);
#endif
#if (PROFILE_PERF_COUNTERS) & PROFILE_PERF_COUNTERS_L1_5
    llk::perf::set_l1_mux(5);
    read_single_group(PerfCounterGroup::L1_5);
#endif
}

#endif  // COMPILE_FOR_BRISC

}  // namespace kernel_profiler

#if COMPILE_FOR_TRISC == 1
#define StartPerfCounters() kernel_profiler::start_perf_counter();
#define StopPerfCounters() kernel_profiler::stop_perf_counter();
#define RecordPerfCounters() kernel_profiler::PerfCounterWrapper _perf_counter_wrapper_;
#else
#define StartPerfCounters()
#define StopPerfCounters()
#define RecordPerfCounters()
#endif

#if defined(COMPILE_FOR_BRISC)
#define ReadPerfCounters() kernel_profiler::read_perf_counters();
#else
#define ReadPerfCounters()
#endif

#else

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define ReadPerfCounters()
#define RecordPerfCounters()

#endif
