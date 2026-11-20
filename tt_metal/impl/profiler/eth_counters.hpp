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

#include "perf_counters/eth_counters_inventory.h"

namespace tt::tt_metal {
class IDevice;
class Program;
class MetalContext;
}  // namespace tt::tt_metal

// Measured ethernet link traffic: a host snapshot of the queue counters and the firmware link counters of every active
// ethernet core, taken around each slow-dispatch program and around the whole device lifetime, written as deltas to
// eth_traffic.csv and eth_run.csv in the profiler log directory.
// Reference: tt_metal/tt-llk/docs/performance_counters/eth_counters.md.
namespace tt::tt_metal::eth_counters {

inline constexpr std::string_view TRAFFIC_LOG = "eth_traffic.csv";
inline constexpr std::string_view RUN_LOG = "eth_run.csv";
inline constexpr std::string_view IDLE_LOG = "eth_idle.csv";
inline constexpr std::string_view TRAFFIC_HEADER =
    "device_id,program_id,launch_seq,flags,elapsed_ns,eth_x,eth_y,channel,counter,queue,delta";
inline constexpr std::string_view RUN_HEADER =
    "device_id,run_id,flags,elapsed_ns,eth_x,eth_y,channel,counter,queue,delta";
inline constexpr std::string_view IDLE_HEADER = "device_id,eth_x,eth_y,channel,counter,queue,counts_per_second";

// Bit set in the flags column of a row.
inline constexpr uint32_t FLAG_OVERLAPPED = 1u << 0;  // another program was in flight on the device
inline constexpr uint32_t FLAG_RESET = 1u << 1;       // a 64-bit counter went down in the window, its row is dropped
inline constexpr uint32_t FLAG_STALE =
    1u << 2;  // the firmware link counters did not refresh in the window, their rows are dropped

inline constexpr int8_t NO_QUEUE = -1;

enum class Kind : uint8_t { TX_QUEUE, RX_QUEUE, L1 };

// One value of a reading.
struct Item {
    llk::perf::eth::EthCounter counter = llk::perf::eth::EthCounter::UNDEF;
    Kind kind = Kind::L1;
    int8_t queue = NO_QUEUE;
    uint8_t words = 1;        // 32-bit words of the value
    bool high_first = false;  // L1 values only: the high word comes first in memory
    uint32_t address = 0;     // register address, or L1 byte address
};

struct Reading {
    std::chrono::steady_clock::time_point time;
    // [core][item], layout fixed by Plan.
    std::vector<uint64_t> values;
};

// Which values are read from every ethernet core; independent of any device access so it can be tested without
// hardware.
struct Plan {
    llk::perf::eth::Arch arch = llk::perf::eth::Arch::BLACKHOLE;
    uint32_t num_cores = 0;
    std::vector<Item> items;
    uint32_t l1_base = 0;  // one block read per core covers every L1 item
    uint32_t l1_bytes = 0;

    size_t reading_size() const { return size_t(num_cores) * items.size(); }
};

// nullopt when the architecture has no known register map. include_l1 is false when the ethernet firmware is too old
// to keep the link counters in L1.
std::optional<Plan> make_plan(llk::perf::eth::Arch arch, uint32_t num_cores, bool include_l1);

bool is_register(const Item& item);

struct Delta {
    uint32_t core;
    size_t item;
    uint64_t count;
};

// Nonzero differences only. A 32-bit value wraps; a 64-bit value that went down is dropped and sets FLAG_RESET in
// *flags. When the firmware poll counter of a core did not move, the values read from its L1 are stale: those rows
// are dropped and FLAG_STALE is set.
std::vector<Delta> diff(const Plan& plan, const Reading& before, const Reading& after, uint32_t* flags);

// Adds the wrapped difference of every register value to totals, which is reading_size() long. Summed snapshot by
// snapshot, a run keeps the counter periods that one open to close difference of a 32-bit register would lose.
void add_register_deltas(const Plan& plan, const Reading& before, const Reading& after, std::vector<uint64_t>& totals);

std::string counter_name(llk::perf::eth::EthCounter counter);

// Hooks, no-ops unless TT_METAL_PROFILE_ETH_COUNTERS is set. begin_run is the first snapshot of a device and end_run
// the last one, written as one set of rows. begin_program snapshots right before the go signal of a slow-dispatch
// program; end_pending closes the program in flight on the chip once its cores are idle.
void begin_run(MetalContext& ctx, IDevice& device);
void end_run(MetalContext& ctx, IDevice& device);
void begin_program(MetalContext& ctx, IDevice& device, const Program& program);
void end_pending(MetalContext& ctx, ChipId chip);

}  // namespace tt::tt_metal::eth_counters
