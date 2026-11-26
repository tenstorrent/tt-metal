// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

// Measured DRAM traffic per slow-dispatch program. Reference: tt-llk docs/performance_counters/dram_counters.md.
namespace tt::tt_metal::dram_counters {

inline constexpr std::string_view TRAFFIC_LOG = llk::perf::dram::TRAFFIC_LOG;
inline constexpr std::string_view IDLE_LOG = llk::perf::dram::IDLE_LOG;
inline constexpr std::string_view CONTROLLER_LOG = llk::perf::dram::CONTROLLER_LOG;
using llk::perf::dram::FLAG_INTERVAL_WRAPPED;
using llk::perf::dram::FLAG_NOT_FINISHED;
using llk::perf::dram::FLAG_OVERLAPPED;
using llk::perf::dram::FLAG_SERVICE_ACTIVE;
using llk::perf::dram::FLAG_UNSETTLED;

uint32_t launch_flags(bool overlapped, bool settled, uint32_t extra_flags);

// The longest controller interval, and the shortest time it can take (at a 2 GHz memory clock).
inline constexpr uint32_t CONTROLLER_INTERVAL = 0xFFFFFFFFu;
inline constexpr uint64_t CONTROLLER_INTERVAL_MIN_NS = CONTROLLER_INTERVAL / 2;

// Index of the first tile whose controller registers answer, judged by the AXI control word read through each tile.
std::optional<size_t> answering_port(
    const std::vector<uint32_t>& axi_control, const llk::perf::dram::ControllerLayout& l);

struct Reading {
    std::chrono::steady_clock::time_point time;
    std::vector<uint32_t> values;      // [tile][noc][captured counter], layout fixed by Plan
    std::vector<uint32_t> tile_reads;  // register reads issued to the tile of each value, up to and including it
};

// Which registers are read; no device access, so it can be tested without hardware.
struct Plan {
    llk::perf::dram::Arch arch = llk::perf::dram::Arch::BLACKHOLE;
    uint32_t num_tiles = 0;
    uint32_t num_nocs = 0;
    std::vector<llk::perf::noc::NocCounter> counters;  // captured NIU counters, in read order
    std::vector<uint32_t> regs;                        // their status register indices

    size_t reading_size() const { return size_t(num_tiles) * num_nocs * counters.size(); }
};

// Per tile and NoC ([tile * num_nocs + noc]): 1 when a host read of a register of the tile counts as one read word.
using Coupling = std::vector<uint32_t>;

// nullopt when the architecture has no known register map.
std::optional<Plan> make_plan(llk::perf::dram::Arch arch, uint32_t num_tiles);

bool is_read_counter(llk::perf::noc::NocCounter counter);
bool is_write_counter(llk::perf::noc::NocCounter counter);
std::string counter_name(llk::perf::noc::NocCounter counter);

struct Delta {
    uint32_t tile;
    int noc;
    llk::perf::noc::NocCounter counter;
    uint32_t words;       // wrapped difference minus self_words, never negative
    uint32_t self_words;  // words the capture's own reads added between the two readings
};

// Words the capture's own reads added to value i between the two readings.
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

// Slow dispatch hooks, no-ops unless TT_METAL_PROFILE_DRAM_COUNTERS is set. begin_program snapshots right before the
// go signal; end_pending closes the program in flight once its cores are idle, only that program when program_id is
// given.
void begin_program(MetalContext& ctx, IDevice& device, const Program& program);
void end_pending(MetalContext& ctx, ChipId chip, std::optional<uint64_t> program_id = std::nullopt);
// Closes what is still open and drops the state of every chip of the context.
void release(MetalContext& ctx);

}  // namespace tt::tt_metal::dram_counters
