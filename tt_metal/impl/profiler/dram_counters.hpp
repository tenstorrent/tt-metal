// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/device_types.hpp>

#include "perf_counters/dram_counters_inventory.h"

namespace tt::tt_metal {
class IDevice;
class Program;
class MetalContext;
}  // namespace tt::tt_metal

// Measured DRAM traffic per program: a host snapshot of the free running counters of every DRAM channel tile before
// the program is launched and after it is done, written as deltas to dram_traffic.csv in the profiler log directory.
// Reference: tt_metal/tt-llk/docs/performance_counters/dram_counters.md.
namespace tt::tt_metal::dram_counters {

inline constexpr std::string_view TRAFFIC_LOG = "dram_traffic.csv";
inline constexpr std::string_view IDLE_LOG = "dram_idle.csv";
inline constexpr std::string_view TRAFFIC_HEADER =
    "device_id,program_id,launch_seq,flags,elapsed_ns,dram_x,dram_y,noc,counter,delta_words,self_words";
inline constexpr std::string_view IDLE_HEADER =
    "device_id,dram_x,dram_y,noc,counter,words_per_snapshot_pair,words_per_second";

// Bit set in the flags column of a row.
inline constexpr uint32_t FLAG_OVERLAPPED = 1u << 0;  // another program was in flight on the device
inline constexpr uint32_t FLAG_UNSETTLED = 1u << 1;   // write counters were still moving when the settle loop gave up

struct Reading {
    std::chrono::steady_clock::time_point time;
    // [tile][noc][captured counter], layout fixed by Plan.
    std::vector<uint32_t> values;
    // Register reads issued to the tile of each value, counted up to and including the read of that value.
    std::vector<uint32_t> tile_reads;
};

// Which registers are read; independent of any device access so it can be tested without hardware.
struct Plan {
    llk::perf::dram::Arch arch = llk::perf::dram::Arch::BLACKHOLE;
    uint32_t num_tiles = 0;
    uint32_t num_nocs = 0;
    std::vector<llk::perf::dram::DramCounter> counters;  // captured subset of the NIU table, in read order
    std::vector<uint32_t> offsets;                       // register offset of each captured counter

    size_t reading_size() const { return size_t(num_tiles) * num_nocs * counters.size(); }
};

// Words that the capture's own register reads add to a read counter, per NoC: 1 when a host read of a register of the
// tile counts as one read word on that NoC, else 0. Measured on the device, see measure_coupling.
using Coupling = std::vector<uint32_t>;

// nullopt when the architecture has no known register map.
std::optional<Plan> make_plan(llk::perf::dram::Arch arch, uint32_t num_tiles);

bool is_read_counter(llk::perf::dram::DramCounter counter);
bool is_write_counter(llk::perf::dram::DramCounter counter);

struct Delta {
    uint32_t tile;
    int noc;
    llk::perf::dram::DramCounter counter;
    uint32_t words;       // wrapped difference minus self_words, never negative
    uint32_t self_words;  // words the capture's own reads added between the two readings
};

// Self words of value i: the reads issued to its tile between its two readings, on a NoC whose read counters
// the reads count on. Exact whether a read sees itself or not, since it is the difference of two read indices.
uint32_t self_words(const Plan& plan, const Coupling& coupling, const Reading& before, const Reading& after, size_t i);

// Nonzero corrected differences only.
std::vector<Delta> diff(const Plan& plan, const Coupling& coupling, const Reading& before, const Reading& after);

// Drift of one register while nothing of ours runs: words = per_pair + per_second * window seconds.
struct IdleFit {
    double per_pair = 0.0;
    double per_second = 0.0;
};

// Line through a back-to-back snapshot pair and a spaced one, clamped to no negative term.
IdleFit fit_idle(double tight_words, double tight_seconds, double spaced_words, double spaced_seconds);

std::string counter_name(llk::perf::dram::DramCounter counter);

// Slow dispatch hooks, no-ops unless TT_METAL_PROFILE_DRAM_COUNTERS is set. begin_program snapshots right before the
// go signal; end_pending closes the program in flight on the chip once its cores are idle.
void begin_program(MetalContext& ctx, IDevice& device, const Program& program);
void end_pending(MetalContext& ctx, ChipId chip);
// Drops the state of every chip of the context, which closes its devices.
void release(MetalContext& ctx);

}  // namespace tt::tt_metal::dram_counters
