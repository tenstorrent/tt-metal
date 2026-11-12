// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Max-utilization workload test.
// Architecture:
//   1. Pre-fill phase: BRISC runs on each core to read 2 DRAM buffers into L1
//      - Buffer 0: 8 tiles of bfloat16 from DRAM
//      - Buffer 1: 8 tiles of bfloat16 from DRAM
//   2. Main phase: these generators run concurrently, each with its own knobs
//      (see tests/didt/README.md, "MaxUtil Power Virus"):
//      - TRISC compute (FPU)
//      - NCRISC Tensix→DRAM streaming on every worker (NOC0/NOC1 checkerboard)
//      - active-ETH DRAM streaming (ERISC0, or ERISC1 when link traffic runs)
//      - Ethernet link traffic on ERISC0 of every link-up core whose peer chip
//        is dispatched together (AllDevices test only)
//      An optional BRISC kernel synchronizes TRISC launch across worker cores.
//   No CB dependencies between DRAM streaming and compute.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_metal/test_utils/stimulus.hpp>
#include <distributed/mesh_device_impl.hpp>
#include "multi_device_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"
#include "impl/buffers/buffer_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include <random>
#include <set>
#include <vector>

namespace tt::tt_metal {

using namespace std;
using namespace tt;

namespace unit_tests::didt::max_util_workload {

static constexpr uint32_t kOutputSentinel = 0xFFFFFFFFu;
static constexpr uint32_t kEthL1StagingHeaderBytes = 64;

// ---------------------------------------------------------------------------
// Test configuration
// ---------------------------------------------------------------------------

struct MaxUtilConfig {
    // Core grid used for the workload (logical coordinates).
    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {0, 0};

    // Number of tiles for pre-filled buffers.
    uint32_t num_tiles = 8;  // 8 tiles as requested

    // Number of times the main program is enqueued to the device.
    uint32_t num_iterations = 1;

    // Number of loops within each kernel of a single program dispatch.
    uint32_t num_wl_loops = 100;

    // Number of loops for the slow (cos) workload kernel per dispatch.
    // Controlled via the duty-cycle map; 0 means no slow workload is run.
    uint32_t num_slow_wl_loops = 0;

    // L1 buffer addresses (filled by pre-fill phase, passed to main phase).
    uint32_t l1_buffer0_addr = 0;     // input 0 bfloat16 data
    uint32_t l1_buffer1_addr = 0;     // input 1 bfloat16 data
    uint32_t l1_buffer2_addr = 0;     // output bfloat16 data
    uint32_t l1_super_sync_addr = 0;  // super sync semaphore
    uint32_t l1_fpu_timing_addr = 0;  // compute-pipeline start/end timestamps

    // FPU utilization target percentage [1, 92].
    // Passed to the compute kernel as compile-time arg 5.
    uint32_t fpu_utilization_pct = 92;

    // ETH DRAM streaming fields (filled by setup_eth_stream_config before build_program).
    uint32_t eth_dram_buffer_addr = 0;  // DRAM src base address for ETH streaming
    uint32_t eth_pages_per_bank = 0;    // pages per bank read per iteration
    uint32_t eth_l1_staging_addr = 0;   // ETH L1 unreserved base (64-byte header, first 16 bytes = timing scratch)
    // DRAM read transaction size. Larger values generally amortize NOC command
    // overhead better; Blackhole supports bursts up to 16 KiB.
    uint32_t eth_page_size = 1024;
    // ETH DRAM run duration in wall-clock ticks (matches the compute window).
    uint64_t eth_duration_ticks = 0;  // set by setup_eth_stream_config

    // DRAM utilization target percentage [1, 100].
    // 100 = maximum utilization (no wait between NOC page issues).
    // Lower values insert busy-wait cycles between individual page-read commands
    // to throttle the effective DRAM bandwidth.
    uint32_t eth_dram_util_pct = 100;
    // Cycles to busy-wait between consecutive NOC page issues; derived from
    // eth_dram_util_pct by setup_eth_stream_config via kDramUtilToCyclesMap.
    uint32_t eth_noc_wait_cycles = 0;

    // Host-measured seconds per main dispatch (0 = not measured). Device wall
    // clocks tick at AICLK, which DVFS lowers under power limits, so long runs
    // convert ticks to time with this instead of the nominal clock.
    double dispatch_seconds = 0.0;

    // When true, compute kernels perform a super-sync barrier at program start.
    bool super_sync = false;

    // -- Tensix DRAM streaming (NCRISC on every worker) ----------------------
    // Throttle knobs follow the syseng ttx NOC power virus: transfers per burst
    // (0 = continuous) and idle time after each burst in 512 wall-clock ticks.
    // tdram_xfer_bytes == 0 disables the Tensix DRAM readers.
    uint32_t tdram_xfer_bytes = 8192;
    uint32_t tdram_burst = 0;
    uint32_t tdram_idle_units = 0;
    uint32_t tdram_period_units = 0;  // phase-locked mode (from _UTILIZATION_PCT); 0 = off
    uint32_t tdram_active_units = 0;
    uint32_t tdram_pages_per_bank = 16;
    uint32_t tdram_num_banks = 0;        // set by setup_tensix_dram_config
    uint32_t tdram_buffer_addr = 0;      // set by setup_tensix_dram_config
    uint64_t tdram_duration_ticks = 0;   // set by setup_tensix_dram_config
    uint32_t l1_tdram_timing_addr = 0;   // set by build_prefill_program
    uint32_t l1_tdram_staging_addr = 0;  // set by build_prefill_program

    // -- Ethernet link traffic (active ERISC0, both ends of each link) --------
    // Only enabled when both ends of a link are opened and dispatched together
    // (the AllDevices test). eth_link_xfer_bytes == 0 disables link traffic.
    bool eth_link_allowed = false;
    uint32_t eth_link_xfer_bytes = 8192;
    uint32_t eth_link_burst = 0;
    uint32_t eth_link_idle_units = 0;
    uint32_t eth_link_period_units = 0;  // phase-locked mode (from _UTILIZATION_PCT); 0 = off
    uint32_t eth_link_active_units = 0;
    uint32_t eth_link_num_slots = 8;
    uint64_t eth_link_duration_ticks = 0;  // send window; set by setup_eth_link_config
    uint32_t eth_link_startup_delay_ticks = 0;
    struct EthLink {
        CoreCoord core;  // logical ETH core on this device
        ChipId peer_chip;
        CoreCoord peer_core;  // logical ETH core on the peer
        uint32_t role;        // 0 or 1; opposite on the peer
    };
    std::vector<EthLink> eth_links;            // set by setup_eth_link_config
    uint32_t eth_link_timing_buffer_addr = 0;  // set by setup_eth_link_config
    uint32_t eth_link_l1_size = 0;             // bytes of ETH L1 given to link kernel
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint32_t tile_size_bytes(DataFormat fmt, uint32_t h = 32, uint32_t w = 32) {
    if (fmt == DataFormat::Float16_b) {
        return uint32_t(w * h * 2);  // two bytes per float16_b element
    }
    throw std::invalid_argument("Invalid data format");
}

static std::optional<int> parse_env_int(const char* text) {
    const std::string_view value(text);
    int parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return std::nullopt;
    }
    return parsed;
}

static std::vector<uint32_t> rng_bfp16(
    uint32_t num_bytes, uint32_t tile_rows, uint32_t tile_cols, float mean, float stdev, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(mean, stdev);
    std::vector<bfloat16> results(
        num_bytes / tile_size_bytes(DataFormat::Float16_b, tile_rows, tile_cols) * tile_rows * tile_cols);
    std::generate(results.begin(), results.end(), [&]() { return bfloat16(dis(gen)); });
    std::vector<uint32_t> packed_results = tt::test_utils::pack_vector<uint32_t, bfloat16>(results);
    return packed_results;
}

/// Reads MAX_UTIL_FPU_UTILIZATION_PCT from the environment; defaults to 92.
/// Valid range: [0, 92]. 0 disables the compute kernel (FPU off).
static uint32_t get_fpu_utilization_pct() {
    const char* env = std::getenv("MAX_UTIL_FPU_UTILIZATION_PCT");
    if (env != nullptr) {
        const auto val = parse_env_int(env);
        if (val.has_value() && *val >= 0 && *val <= 92) {
            return static_cast<uint32_t>(*val);
        }
        log_warning(
            LogTest, "MAX_UTIL_FPU_UTILIZATION_PCT='{}' is not an integer in [0, 92] – using default of 92", env);
    }
    return 92;
}

/// Reads MAX_UTIL_DRAM_UTILIZATION_PCT (ETH-core DRAM readers) from the
/// environment; defaults to 100. Valid range: [0, 100]. 100 = maximum DRAM
/// utilization (no throttle); 0 disables the ETH-core DRAM readers.
static uint32_t get_dram_utilization_pct() {
    const char* env = std::getenv("MAX_UTIL_DRAM_UTILIZATION_PCT");
    if (env != nullptr) {
        const auto val = parse_env_int(env);
        if (val.has_value() && *val >= 0 && *val <= 100) {
            return static_cast<uint32_t>(*val);
        }
        log_warning(
            LogTest, "MAX_UTIL_DRAM_UTILIZATION_PCT='{}' is not an integer in [0, 100] – using default of 100", env);
    }
    return 100;
}

/// Reads an unsigned integer environment variable in [lo, hi]. Returns
/// std::nullopt (with a warning when set but invalid) so callers can keep
/// their defaults.
static std::optional<uint32_t> get_env_u32(const char* name, uint32_t lo, uint32_t hi) {
    const char* env = std::getenv(name);
    if (env == nullptr) {
        return std::nullopt;
    }
    const auto val = parse_env_int(env);
    if (val.has_value() && *val >= 0 && static_cast<uint32_t>(*val) >= lo && static_cast<uint32_t>(*val) <= hi) {
        return static_cast<uint32_t>(*val);
    }
    log_warning(LogTest, "{}='{}' is not an integer in [{}, {}] – ignoring", name, env, lo, hi);
    return std::nullopt;
}

static bool is_power_of_two(uint32_t v) { return v != 0 && (v & (v - 1)) == 0; }

// Throttle knobs shared by the Tensix-DRAM and ETH-link generators.
//
// Raw knobs follow the syseng ttx NOC power virus (alewycky/math-noc-power-virus):
//   <prefix>_XFER_BYTES  bytes per transfer; 0 = generator off
//   <prefix>_BURST       transfers per burst; 0 = continuous (no idle)
//   <prefix>_IDLE_512    idle time after each burst, units of 512 wall-clock ticks
//
// Convenience knob:
//   <prefix>_UTILIZATION_PCT  phase-locked duty cycle. Every instance is active
//                        while (wall_clock / 512) % PERIOD < PERIOD * pct / 100,
//                        so all generators toggle together and the average
//                        throughput scales ~linearly with pct. 0 = off.
//   <prefix>_PERIOD_512  phase period, units of 512 ticks, power of two
//                        (default 256 ≈ 97 µs at 1.35 GHz).
// Explicit BURST / IDLE_512 take precedence over UTILIZATION_PCT.
static constexpr uint32_t kThrottleDefaultPeriodUnits = 256;

struct ThrottleKnobs {
    uint32_t xfer_bytes;
    uint32_t burst;
    uint32_t idle_units;
    uint32_t period_units;  // 0 = phase-locked mode off
    uint32_t active_units;
};

/// Resolves the throttle knobs for one generator from the environment.
static ThrottleKnobs resolve_throttle_knobs(
    const std::string& prefix, uint32_t default_xfer_bytes, uint32_t min_xfer_bytes, uint32_t max_xfer_bytes) {
    ThrottleKnobs k{default_xfer_bytes, 0, 0, 0, 0};
    if (auto v = get_env_u32((prefix + "_XFER_BYTES").c_str(), 0, max_xfer_bytes)) {
        if (*v == 0 || (is_power_of_two(*v) && *v >= min_xfer_bytes)) {
            k.xfer_bytes = *v;
        } else {
            log_warning(
                LogTest,
                "{}_XFER_BYTES={} must be 0 or a power of two in [{}, {}] – using {}",
                prefix,
                *v,
                min_xfer_bytes,
                max_xfer_bytes,
                default_xfer_bytes);
        }
    }
    const auto burst = get_env_u32((prefix + "_BURST").c_str(), 0, 1u << 24);
    const auto idle = get_env_u32((prefix + "_IDLE_512").c_str(), 0, 1u << 24);
    if (burst.has_value() || idle.has_value()) {
        k.burst = burst.value_or(0);
        k.idle_units = k.burst == 0 ? 0 : idle.value_or(0);  // continuous mode has no idle period
    } else if (auto pct = get_env_u32((prefix + "_UTILIZATION_PCT").c_str(), 0, 100)) {
        if (*pct == 0) {
            k.xfer_bytes = 0;
        } else if (*pct < 100) {
            uint32_t period = kThrottleDefaultPeriodUnits;
            if (auto p = get_env_u32((prefix + "_PERIOD_512").c_str(), 2, 1u << 20)) {
                if (is_power_of_two(*p)) {
                    period = *p;
                } else {
                    log_warning(LogTest, "{}_PERIOD_512={} is not a power of two – using {}", prefix, *p, period);
                }
            }
            k.period_units = period;
            k.active_units = std::clamp<uint32_t>(
                static_cast<uint32_t>(std::lround(static_cast<double>(period) * *pct / 100.0)), 1, period - 1);
        }
    }
    return k;
}

/// Applies the Tensix-DRAM and ETH-link power-virus knobs to @p cfg.
static void apply_power_virus_knobs(MaxUtilConfig& cfg) {
    const auto tdram = resolve_throttle_knobs("MAX_UTIL_TDRAM", 8192, 64, 16384);
    cfg.tdram_xfer_bytes = tdram.xfer_bytes;
    cfg.tdram_burst = tdram.burst;
    cfg.tdram_idle_units = tdram.idle_units;
    cfg.tdram_period_units = tdram.period_units;
    cfg.tdram_active_units = tdram.active_units;

    const auto link = resolve_throttle_knobs("MAX_UTIL_ETH_LINK", 8192, 64, 8192);
    cfg.eth_link_xfer_bytes = link.xfer_bytes;
    cfg.eth_link_burst = link.burst;
    cfg.eth_link_idle_units = link.idle_units;
    cfg.eth_link_period_units = link.period_units;
    cfg.eth_link_active_units = link.active_units;
    cfg.eth_link_startup_delay_ticks = get_env_u32("MAX_UTIL_ETH_LINK_STARTUP_DELAY_TICKS", 0, UINT32_MAX).value_or(0);
}

/// Reads MAX_UTIL_DRAM_PAGE_SIZE_BYTES from the environment; defaults to 1024.
/// Valid values are power-of-two NOC burst sizes in [512, 16384].
static uint32_t get_dram_page_size_bytes() {
    constexpr uint32_t default_page_size = 1024;
    const char* env = std::getenv("MAX_UTIL_DRAM_PAGE_SIZE_BYTES");
    if (env != nullptr) {
        const auto val = parse_env_int(env);
        if (val.has_value() && *val >= 512 && *val <= 16384 && (*val & (*val - 1)) == 0) {
            return static_cast<uint32_t>(*val);
        }
        log_warning(
            LogTest,
            "MAX_UTIL_DRAM_PAGE_SIZE_BYTES='{}' is not a power of two in [512, 16384] – using default of {}",
            env,
            default_page_size);
    }
    return default_page_size;
}

/// Reads MAX_UTIL_SUPER_SYNC from the environment.
/// Any non-empty value other than "0" or "false" enables super-sync.
/// Defaults to false.
static bool get_super_sync() {
    const char* env = std::getenv("MAX_UTIL_SUPER_SYNC");
    if (env != nullptr) {
        std::string val(env);
        return !val.empty() && val != "0" && val != "false";
    }
    return false;
}

/// Returns a MaxUtilConfig that covers every compute core on @p device.
static MaxUtilConfig full_grid_config(
    IDevice* device,
    uint32_t num_tiles,
    uint32_t num_iterations,
    uint32_t num_wl_loops,
    uint32_t num_slow_wl_loops = 0,
    bool super_sync = false) {
    auto grid = device->compute_with_storage_grid_size();
    MaxUtilConfig cfg;
    cfg.grid_start = {0, 0};
    cfg.grid_end = {grid.x - 1, grid.y - 1};
    cfg.num_tiles = num_tiles;
    cfg.num_iterations = num_iterations;
    cfg.num_wl_loops = num_wl_loops;
    cfg.num_slow_wl_loops = num_slow_wl_loops;
    cfg.fpu_utilization_pct = get_fpu_utilization_pct();
    cfg.eth_dram_util_pct = get_dram_utilization_pct();
    cfg.eth_page_size = get_dram_page_size_bytes();
    cfg.super_sync = super_sync;
    apply_power_virus_knobs(cfg);
    return cfg;
}

// ---------------------------------------------------------------------------
// DRAM utilization throttle map
//
// Maps DRAM utilization percentage → noc_wait_cycles inserted between
// consecutive NOC page-read issues in the ETH DRAM streaming kernel.
// noc_wait_cycles == 0 means no throttle (maximum DRAM utilization).
//
// Values are calibrated on Blackhole p300c against the measured four-bank
// bandwidth at 100%. Integer NOP granularity limits exact matching near 100%.
// ---------------------------------------------------------------------------

// clang-format off
static const std::map<uint32_t, uint32_t> kDramUtilToCyclesMap = {
    {10,  215},
    {20,   95},
    {30,   55},
    {40,   35},
    {50,   23},
    {60,   15},
    {70,    9},
    {80,    7},
    {90,    3},
    {100,   0},
};
// clang-format on

/// Returns the key in a percentage calibration table nearest to @p pct.
static uint32_t nearest_calibration_pct(const std::map<uint32_t, uint32_t>& table, uint32_t pct) {
    TT_ASSERT(!table.empty(), "Calibration table must not be empty");
    auto it = table.lower_bound(pct);
    if (it == table.end()) {
        return std::prev(it)->first;
    }
    if (it == table.begin() || it->first == pct) {
        return it->first;
    }
    auto prev = std::prev(it);
    return (pct - prev->first <= it->first - pct) ? prev->first : it->first;
}

/// Converts a requested DRAM utilization percentage to a noc_wait_cycles value
/// by snapping to the nearest entry in kDramUtilToCyclesMap.
static uint32_t dram_pct_to_noc_wait_cycles(uint32_t pct) {
    return kDramUtilToCyclesMap.at(nearest_calibration_pct(kDramUtilToCyclesMap, pct));
}

// ---------------------------------------------------------------------------
// eth_noc0_coord – translates a logical ETH core to its NOC0 physical
//   coordinate via the SoC descriptor.
// dram_noc0_coord – returns the NOC0 physical coordinate of a DRAM bank
//   (channel) via the SoC descriptor (subchannel 0).
// ---------------------------------------------------------------------------

static CoreCoord eth_noc0_coord(IDevice* device, const CoreCoord& logical_eth) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    tt::umd::CoreCoord noc0 = soc_desc.translate_coord_to(
        {logical_eth.x, logical_eth.y, tt::CoreType::ETH, tt::CoordSystem::LOGICAL}, tt::CoordSystem::NOC0);
    return {noc0.x, noc0.y};
}

static CoreCoord dram_noc0_coord(IDevice* device, uint32_t bank_id) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    tt::umd::CoreCoord noc0 =
        soc_desc.get_dram_core_for_channel(static_cast<int>(bank_id), /*subchannel=*/0, tt::CoordSystem::NOC0);
    return {noc0.x, noc0.y};
}

// ---------------------------------------------------------------------------
// assign_eth_cores_to_banks – partitions active ETH cores by NOC0 x-coordinate
// and assigns each to one DRAM bank. NOC0 x < 8 maps to banks 0-3 and NOC0
// x >= 8 maps to banks 4-7. At most four cores are selected per side.
// ---------------------------------------------------------------------------

static std::vector<std::pair<CoreCoord, uint32_t>> assign_eth_cores_to_banks(IDevice* device) {
    auto active_eth = device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true);
    std::vector<CoreCoord> left_cores;
    std::vector<CoreCoord> right_cores;
    for (const auto& core : active_eth) {
        if (eth_noc0_coord(device, core).x < 8) {
            left_cores.push_back(core);
        } else {
            right_cores.push_back(core);
        }
    }

    auto cmp = [](const CoreCoord& a, const CoreCoord& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); };
    std::sort(left_cores.begin(), left_cores.end(), cmp);
    std::sort(right_cores.begin(), right_cores.end(), cmp);
    left_cores.resize(std::min<size_t>(left_cores.size(), 4));
    right_cores.resize(std::min<size_t>(right_cores.size(), 4));

    std::vector<std::pair<CoreCoord, uint32_t>> assignments;
    assignments.reserve(left_cores.size() + right_cores.size());
    for (size_t i = 0; i < left_cores.size(); ++i) {
        assignments.emplace_back(left_cores[i], static_cast<uint32_t>(i));
    }
    for (size_t i = 0; i < right_cores.size(); ++i) {
        assignments.emplace_back(right_cores[i], static_cast<uint32_t>(4 + i));
    }
    return assignments;
}

// ---------------------------------------------------------------------------
// setup_eth_stream_config – configures DRAM buffer and ETH L1 addresses for
//   the ETH DRAM streaming kernel.
//
// Each selected active ETH core reads from exactly one DRAM bank so that
// summing per-core bandwidths gives the aggregate ETH-to-DRAM bandwidth.
//
// Returns a shared_ptr<Buffer> holding the DRAM staging buffer; the caller
// must keep this alive until the program finishes.  Returns nullptr when no
// active Ethernet cores are assignable.
// ---------------------------------------------------------------------------

static uint64_t estimate_compute_ticks(const MaxUtilConfig& cfg);  // defined after the FPU calibration map

static shared_ptr<Buffer> setup_eth_stream_config(IDevice* device, MaxUtilConfig& cfg) {
    if (cfg.eth_dram_util_pct == 0) {
        log_info(LogTest, "Device {}: ETH DRAM streaming disabled (MAX_UTIL_DRAM_UTILIZATION_PCT=0)", device->id());
        return nullptr;
    }
    auto active_eth = device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true);
    auto inactive_eth = device->get_inactive_ethernet_cores();

    log_info(
        LogTest,
        "Device {}: ETH cores available: {} active (connected), {} inactive (idle); using active only",
        device->id(),
        active_eth.size(),
        inactive_eth.size());

    auto assignments = assign_eth_cores_to_banks(device);
    if (assignments.empty()) {
        log_warning(
            LogTest, "Device {}: no active ETH cores available – skipping ETH DRAM streaming kernel", device->id());
        return nullptr;
    }

    const auto& hal = MetalContext::instance().hal();
    cfg.eth_l1_staging_addr = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    uint32_t eth_l1_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    // When the link-traffic kernel owns ERISC0 and the start of unreserved L1,
    // the DRAM reader moves to ERISC1 and uses the remainder.
    if (!cfg.eth_links.empty()) {
        if (hal.get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH) < 2) {
            log_warning(
                LogTest,
                "Device {}: only one user ERISC on active ETH cores – ETH DRAM streaming disabled in favour of "
                "link traffic",
                device->id());
            return nullptr;
        }
        cfg.eth_l1_staging_addr += cfg.eth_link_l1_size;
        eth_l1_size -= cfg.eth_link_l1_size;
    }

    uint32_t num_banks = static_cast<uint32_t>(device->num_dram_channels());
    uint32_t page_size_bytes = cfg.eth_page_size;

    cfg.eth_pages_per_bank = (eth_l1_size - kEthL1StagingHeaderBytes) / page_size_bytes;

    // Run for the estimated compute window; the utilization knob throttles the
    // issue rate, not the duration.
    cfg.eth_duration_ticks = estimate_compute_ticks(cfg);

    cfg.eth_noc_wait_cycles = dram_pct_to_noc_wait_cycles(cfg.eth_dram_util_pct);
    const uint32_t matched_dram_pct = nearest_calibration_pct(kDramUtilToCyclesMap, cfg.eth_dram_util_pct);
    log_info(
        LogTest,
        "DRAM utilization: requested={}%  matched={}%  noc_wait_cycles={}",
        cfg.eth_dram_util_pct,
        matched_dram_pct,
        cfg.eth_noc_wait_cycles);

    log_info(
        LogTest,
        "Device {}: ETH DRAM streaming – {} active cores (1 bank each), pages_per_bank={}, "
        "page_size={} B ({}KB), eth_l1_staging_addr=0x{:x}, eth_l1_size={} B",
        device->id(),
        assignments.size(),
        cfg.eth_pages_per_bank,
        page_size_bytes,
        page_size_bytes / 1024,
        cfg.eth_l1_staging_addr,
        eth_l1_size);

    // Buffer spans all 8 banks so every assigned bank_id has pages_per_bank pages.
    uint32_t total_size = num_banks * cfg.eth_pages_per_bank * page_size_bytes;
    auto dram_buffer = BufferImpl::create(device, total_size, page_size_bytes, BufferType::DRAM);
    cfg.eth_dram_buffer_addr = dram_buffer->address();

    // Populate with a recognisable pattern so DRAM contains live data.
    std::vector<uint32_t> pattern(total_size / sizeof(uint32_t), 0xDEADBEEFu);
    detail::WriteToBuffer(dram_buffer, pattern);

    return dram_buffer;
}

// ---------------------------------------------------------------------------
// build_prefill_program – constructs a pre-fill Program and owns its inputs
//
// Creates DRAM buffers, fills them with random data, and builds a program
// that reads from DRAM into L1 on all cores.
//   - Buffer 0: 8 tiles of bfloat16 from DRAM
//   - Buffer 1: 8 tiles of bfloat16 from DRAM
// ---------------------------------------------------------------------------

struct PrefillProgram {
    Program program;
    shared_ptr<Buffer> dram_buffer0;
    shared_ptr<Buffer> dram_buffer1;
};

static PrefillProgram build_prefill_program(IDevice* device, MaxUtilConfig& cfg) {
    const CoreRange core_range(cfg.grid_start, cfg.grid_end);
    const CoreRangeSet core_range_set({core_range});

    // Create DRAM buffers (using 2 bytes per element for bfloat16, 32x32 tiles)
    const uint32_t tile_rows = 32, tile_cols = 32;
    uint32_t tile_bytes_bfloat16 = tile_size_bytes(DataFormat::Float16_b, tile_rows, tile_cols);
    uint32_t buffer_size_bfloat16 = cfg.num_tiles * tile_bytes_bfloat16;
    auto dram_buffer0 = BufferImpl::create(device, buffer_size_bfloat16, tile_bytes_bfloat16, BufferType::DRAM);
    auto dram_buffer1 = BufferImpl::create(device, buffer_size_bfloat16, tile_bytes_bfloat16, BufferType::DRAM);

    uint32_t dram_buffer0_addr = dram_buffer0->address();
    uint32_t dram_buffer1_addr = dram_buffer1->address();

    // Fill buffer 0 with random data
    std::vector<uint32_t> data0 =
        rng_bfp16(buffer_size_bfloat16, tile_rows, tile_cols, /*mean=*/0.0f, /*stdev=*/1.0f, /*seed=*/42);
    detail::WriteToBuffer(dram_buffer0, data0);

    // Fill buffer 1 with random data
    std::vector<uint32_t> data1 =
        rng_bfp16(buffer_size_bfloat16, tile_rows, tile_cols, /*mean=*/0.0f, /*stdev=*/1.0f, /*seed=*/43);
    detail::WriteToBuffer(dram_buffer1, data1);

    // Get L1 base address and pack all buffers densely
    uint64_t l1_base_addr = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    uint32_t addr = static_cast<uint32_t>(l1_base_addr);
    cfg.l1_buffer0_addr = addr;
    addr += buffer_size_bfloat16;
    cfg.l1_buffer1_addr = addr;
    addr += buffer_size_bfloat16;
    cfg.l1_buffer2_addr = addr;  // output, 8 float16_b tiles, no init
    addr += buffer_size_bfloat16;
    cfg.l1_super_sync_addr = addr;  // super sync semaphore
    addr += 16;
    cfg.l1_fpu_timing_addr = addr;  // four uint32 words: t0 low/high, t1 low/high
    addr += 16;
    cfg.l1_tdram_timing_addr = addr;  // four uint32 words: elapsed lo/hi, transfers lo/hi
    addr += 16;
    cfg.l1_tdram_staging_addr = (addr + 63) & ~63u;  // NCRISC DRAM read destination (tdram_xfer_bytes)

    log_info(
        LogTest,
        "Pre-fill: L1 inputs=0x{:x},0x{:x}, output=0x{:x}, super_sync=0x{:x}, fpu_timing=0x{:x}, "
        "tdram_timing=0x{:x}, tdram_staging=0x{:x}",
        cfg.l1_buffer0_addr,
        cfg.l1_buffer1_addr,
        cfg.l1_buffer2_addr,
        cfg.l1_super_sync_addr,
        cfg.l1_fpu_timing_addr,
        cfg.l1_tdram_timing_addr,
        cfg.l1_tdram_staging_addr);

    Program program = CreateProgram();

    std::vector<uint32_t> prefill_compile_args = {
        dram_buffer0_addr,       // 0: dram_buffer0_addr
        dram_buffer1_addr,       // 1: dram_buffer1_addr
        cfg.l1_buffer0_addr,     // 2: l1_buffer0_addr
        cfg.l1_buffer1_addr,     // 3: l1_buffer1_addr
        tile_bytes_bfloat16,     // 4: tile_size_bytes (bfloat16, 2048)
        cfg.num_tiles,           // 5: num_tiles (8)
        cfg.l1_super_sync_addr,  // 6: l1_super_sync_addr
        cfg.l1_fpu_timing_addr,  // 7: l1_fpu_timing_addr
        cfg.l1_buffer2_addr,     // 8: l1_buffer2_addr
        kOutputSentinel,         // 9: output sentinel
    };
    TensorAccessorArgs(*dram_buffer0).append_to(prefill_compile_args);
    TensorAccessorArgs(*dram_buffer1).append_to(prefill_compile_args);

    // Pre-fill kernel on BRISC - reads from DRAM to L1
    CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/prefill_l1.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = std::move(prefill_compile_args),
        });

    return PrefillProgram{
        .program = std::move(program),
        .dram_buffer0 = std::move(dram_buffer0),
        .dram_buffer1 = std::move(dram_buffer1),
    };
}

// ---------------------------------------------------------------------------
// FPU utilization throttle map
//
// Maps FPU utilization percentage → cycles_to_wait between compute operations.
// The kernel uses cycles_to_wait to insert idle cycles and throttle the FPU.
// Values are calibrated on Blackhole p300c; keys cover the valid [1, 92]
// range at representative intervals. Device timestamps below verify that the
// resulting sustained utilization remains close to the selected table entry.
// ---------------------------------------------------------------------------

// clang-format off
static const std::map<uint32_t, uint32_t> kFpuUtilToCyclesMap = {
    {10,  1100},
    {20,  500},
    {30,  280},
    {40,  180},
    {50,  120},
    {60,  75},
    {70,  47},
    {80,  23},
    {90,  6},
    {92,  0},
};
// clang-format on

/// Converts a requested FPU utilization percentage to a cycles_to_wait value
/// by snapping to the nearest entry in kFpuUtilToCyclesMap.
static uint32_t fpu_pct_to_cycles_to_wait(uint32_t pct) {
    return kFpuUtilToCyclesMap.at(nearest_calibration_pct(kFpuUtilToCyclesMap, pct));
}

/// Estimated duration of one main-program compute pass in wall-clock ticks
/// (AICLK). Used to size the Tensix-DRAM and ETH-link generators so they cover
/// the compute window. With FPU off, the 92% duration is used.
static uint64_t estimate_compute_ticks(const MaxUtilConfig& cfg) {
    constexpr uint64_t useful_fpu_cycles_per_tile = 8 * 16;
    const uint32_t pct =
        cfg.fpu_utilization_pct == 0 ? 92 : nearest_calibration_pct(kFpuUtilToCyclesMap, cfg.fpu_utilization_pct);
    return useful_fpu_cycles_per_tile * cfg.num_tiles * cfg.num_wl_loops * 100 / pct;
}

/// Fills a DRAM buffer with an alternating bit pattern so reads toggle data.
static void fill_toggle_pattern(const shared_ptr<Buffer>& buffer) {
    std::vector<uint32_t> pattern(buffer->size() / sizeof(uint32_t));
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = (i & 1) ? 0xA5A5A5A5u : 0x5A5A5A5Au;
    }
    detail::WriteToBuffer(buffer, pattern);
}

// ---------------------------------------------------------------------------
// setup_tensix_dram_config – allocates the interleaved DRAM source buffer for
//   the Tensix (NCRISC) DRAM readers and sizes their run duration.
//   Returns nullptr when the generator is disabled.
// ---------------------------------------------------------------------------

static shared_ptr<Buffer> setup_tensix_dram_config(IDevice* device, MaxUtilConfig& cfg) {
    if (cfg.tdram_xfer_bytes == 0) {
        log_info(LogTest, "Device {}: Tensix DRAM streaming disabled", device->id());
        return nullptr;
    }
    cfg.tdram_num_banks = device->allocator()->get_num_banks(BufferType::DRAM);
    const uint32_t total_size = cfg.tdram_num_banks * cfg.tdram_pages_per_bank * cfg.tdram_xfer_bytes;
    auto buffer = BufferImpl::create(device, total_size, cfg.tdram_xfer_bytes, BufferType::DRAM);
    fill_toggle_pattern(buffer);
    cfg.tdram_buffer_addr = buffer->address();
    cfg.tdram_duration_ticks = estimate_compute_ticks(cfg);
    log_info(
        LogTest,
        "Device {}: Tensix DRAM streaming – {} banks, {} pages/bank × {} B, burst={} idle={}×512 ticks, "
        "phase={}/{}×512 ticks, duration={} ticks",
        device->id(),
        cfg.tdram_num_banks,
        cfg.tdram_pages_per_bank,
        cfg.tdram_xfer_bytes,
        cfg.tdram_burst,
        cfg.tdram_idle_units,
        cfg.tdram_active_units,
        cfg.tdram_period_units,
        cfg.tdram_duration_ticks);
    return buffer;
}

// ---------------------------------------------------------------------------
// setup_eth_link_config – finds every active ETH link that is up and whose
//   peer chip is also being dispatched, assigns roles, sizes the per-dispatch
//   message count and allocates a DRAM buffer for per-link timing records.
//   Returns nullptr when link traffic is disabled or no link qualifies.
//
//   Both ends compute identical num_slots (L1 layout) from the shared env
//   knobs, which the kernel's flow control requires.
// ---------------------------------------------------------------------------

static constexpr uint32_t kEthLinkHeaderBytes = 64;
static constexpr uint32_t kEthLinkTimingPageBytes = 64;

static shared_ptr<Buffer> setup_eth_link_config(
    IDevice* device, MaxUtilConfig& cfg, const std::set<ChipId>& dispatched_chips) {
    cfg.eth_links.clear();
    if (cfg.eth_link_xfer_bytes == 0) {
        log_info(LogTest, "Device {}: ETH link traffic disabled", device->id());
        return nullptr;
    }

    auto& cluster = MetalContext::instance().get_cluster();
    const ChipId chip = device->id();
    uint32_t links_up = 0;
    for (const auto& core : device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true)) {
        if (!cluster.is_ethernet_link_up(chip, core)) {
            continue;
        }
        ++links_up;
        // Skip links that leave this cluster (e.g. Galaxy exit links to another
        // host); their peer cannot be dispatched from here.
        const auto chan = cluster.get_soc_desc(chip).logical_eth_core_to_chan_map.at(core);
        const auto& internal = cluster.get_ethernet_connections();
        if (!internal.contains(chip) || !internal.at(chip).contains(chan)) {
            continue;
        }
        const auto [peer_chip, peer_core] = device->get_connected_ethernet_core(core);
        if (!cfg.eth_link_allowed || !dispatched_chips.contains(peer_chip)) {
            continue;
        }
        const bool lower =
            chip < peer_chip ||
            (chip == peer_chip && (core.x < peer_core.x || (core.x == peer_core.x && core.y < peer_core.y)));
        cfg.eth_links.push_back({core, peer_chip, peer_core, lower ? 0u : 1u});
    }

    if (cfg.eth_links.empty()) {
        log_info(
            LogTest,
            "Device {}: ETH link traffic skipped – {} link(s) up{}",
            device->id(),
            links_up,
            (links_up > 0 && !cfg.eth_link_allowed) ? " but this test does not dispatch peer chips (use AllDevices)"
                                                    : "");
        return nullptr;
    }

    // Fit the two slot rings into ETH L1, leaving room for the ETH DRAM reader.
    const auto& hal = MetalContext::instance().hal();
    const uint32_t eth_l1_size = hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    auto link_l1_bytes = [&](uint32_t slots) {
        return kEthLinkHeaderBytes + 2 * slots * (cfg.eth_link_xfer_bytes + 16);
    };
    while (cfg.eth_link_num_slots > 2 && link_l1_bytes(cfg.eth_link_num_slots) > eth_l1_size / 2) {
        cfg.eth_link_num_slots /= 2;
    }
    TT_FATAL(
        link_l1_bytes(cfg.eth_link_num_slots) <= eth_l1_size,
        "ETH link buffers ({} B) do not fit in {} B of ETH L1; reduce MAX_UTIL_ETH_LINK_XFER_BYTES",
        link_l1_bytes(cfg.eth_link_num_slots),
        eth_l1_size);
    cfg.eth_link_l1_size = (link_l1_bytes(cfg.eth_link_num_slots) + 63) & ~63u;

    // Duration-bounded: links run at SerDes speed regardless of AICLK, so a
    // message count would finish early when the chip throttles.
    cfg.eth_link_duration_ticks = estimate_compute_ticks(cfg);

    auto timing_buffer = BufferImpl::create(
        device, cfg.eth_links.size() * kEthLinkTimingPageBytes, kEthLinkTimingPageBytes, BufferType::DRAM);
    std::vector<uint32_t> zeros(timing_buffer->size() / sizeof(uint32_t), 0);
    detail::WriteToBuffer(timing_buffer, zeros);
    cfg.eth_link_timing_buffer_addr = timing_buffer->address();

    log_info(
        LogTest,
        "Device {}: ETH link traffic – {} link(s) ({} up), {} ticks × {} B messages per dispatch, {} slots, "
        "burst={} idle={}×512 ticks, phase={}/{}×512 ticks, L1={} B",
        device->id(),
        cfg.eth_links.size(),
        links_up,
        cfg.eth_link_duration_ticks,
        cfg.eth_link_xfer_bytes,
        cfg.eth_link_num_slots,
        cfg.eth_link_burst,
        cfg.eth_link_idle_units,
        cfg.eth_link_active_units,
        cfg.eth_link_period_units,
        cfg.eth_link_l1_size);
    return timing_buffer;
}

// ---------------------------------------------------------------------------
// Duty-cycle control map
//
// Maps hot-workload duty cycle percentage (in 10% increments) to the number
// of slow cos() loops to interleave between each max-util dispatch.
//
// Values are CALIBRATED FOR num_wl_loops = 1000. At runtime the raw map value
// is scaled linearly for the requested hot-workload loop count. Integer
// rounding limits accuracy at very small loop counts:
//
//   num_slow_wl_loops = map_value_at_1000 × num_wl_loops / 1000
//
// A duty_cycle of 100 means no slow workload is injected (num_slow_wl_loops=0).
// Lower duty cycles inject progressively more slow cos() loops so that:
//   duty_cycle ≈ T_hot / (T_hot + T_slow × num_slow_wl_loops)
//
// Tune the values below on hardware by timing both workloads at 1000 loops.
// ---------------------------------------------------------------------------

static constexpr uint32_t kDutyCycleCalibrationLoops = 1000;

// clang-format off
// Derived experimentally at 1000 loops.
static const std::map<uint32_t, uint32_t> kDutyCycleToSlowLoopsMap = {
    {10,  1260},  // 90/10
    {20,   560},  // 80/20
    {30,   327},  // 70/30
    {40,   210},  // 60/40
    {50,   140},  // 50/50
    {60,    95},  // 40/60
    {70,    61},  // 30/70
    {80,    36},  // 20/80
    {90,    18},  // 10/90
    {100,    0},  // full duty cycle: no slow workload injected
};
// clang-format on

/// Reads MAX_UTIL_DUTY_CYCLE_PCT from the environment.
/// Valid values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.
/// Returns 100 (no slow workload) if the variable is absent or invalid.
static uint32_t get_duty_cycle_pct() {
    const char* env = std::getenv("MAX_UTIL_DUTY_CYCLE_PCT");
    if (env == nullptr) {
        // Preserve compatibility with the original, undocumented spelling.
        env = std::getenv("MAX_UTIL_DUTY_CYCLE");
    }
    if (env != nullptr) {
        const auto val = parse_env_int(env);
        if (val.has_value() && *val >= 0 && kDutyCycleToSlowLoopsMap.contains(static_cast<uint32_t>(*val))) {
            return static_cast<uint32_t>(*val);
        }
        log_warning(
            LogTest,
            "MAX_UTIL_DUTY_CYCLE_PCT='{}' is not one of {{10,20,30,40,50,60,70,80,90,100}} – using default of 100",
            env);
    }
    return 100;
}

/// Returns the num_slow_wl_loops for a given duty cycle, scaled to match the
/// actual hot-workload loop count. Map values are calibrated at
/// kDutyCycleCalibrationLoops; integer rounding limits accuracy for very small
/// loop counts.
static uint32_t duty_cycle_to_slow_loops(uint32_t duty_cycle_pct, uint32_t num_wl_loops) {
    auto it = kDutyCycleToSlowLoopsMap.find(duty_cycle_pct);
    if (it == kDutyCycleToSlowLoopsMap.end() || it->second == 0) {
        return 0;
    }
    // 64-bit multiply to avoid overflow before dividing back down.
    uint64_t scaled = static_cast<uint64_t>(it->second) * num_wl_loops / kDutyCycleCalibrationLoops;
    if (scaled == 0) {
        log_warning(
            LogTest,
            "Duty cycle {}% cannot be represented with only {} hot-workload loops; no slow workload will run",
            duty_cycle_pct,
            num_wl_loops);
    }
    return static_cast<uint32_t>(scaled);
}

// ---------------------------------------------------------------------------
// build_program – constructs the main Program
//
// TRISC uses pre-filled L1 buffers directly without CB waits. An optional
// BRISC kernel synchronizes compute launch across the worker grid.
// ---------------------------------------------------------------------------

static Program build_program(IDevice* device, const MaxUtilConfig& cfg) {
    const CoreRange core_range(cfg.grid_start, cfg.grid_end);
    const CoreRangeSet core_range_set({core_range});

    log_info(
        LogTest,
        "Main program: {} cores, {} enqueues × {} inner iterations",
        core_range.size(),
        cfg.num_iterations,
        cfg.num_wl_loops);

    Program program = CreateProgram();

    log_info(LogTest, "Super sync: {}", cfg.super_sync);

    // -- Compute kernel (TRISC) ---------------------------------------------
    // Uses pre-filled L1 buffers directly, no CB waits. FPU 0% = no compute.
    const bool fpu_on = cfg.fpu_utilization_pct != 0;
    if (!fpu_on) {
        log_info(LogTest, "FPU utilization: 0% – compute kernel disabled");
    }
    if (fpu_on) {
        const uint32_t matched_pct = nearest_calibration_pct(kFpuUtilToCyclesMap, cfg.fpu_utilization_pct);
        const uint32_t cycles_to_wait = fpu_pct_to_cycles_to_wait(cfg.fpu_utilization_pct);
        log_info(
            LogTest,
            "FPU utilization: requested={}%  matched={}%  cycles_to_wait={}",
            cfg.fpu_utilization_pct,
            matched_pct,
            cycles_to_wait);

        CreateKernel(
            program,
            "tests/didt/max_util_workload/kernels/max_util_compute.cpp",
            core_range_set,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .compile_args =
                    {
                        cfg.l1_buffer0_addr,     // 0: l1_buffer0_addr (bfloat16)
                        cfg.l1_buffer1_addr,     // 1: l1_buffer1_addr (bfloat16)
                        cfg.l1_buffer2_addr,     // 2: l1_buffer2_addr (output, 8 float16_b tiles)
                        cfg.num_tiles,           // 3: num_tiles (8)
                        cfg.num_wl_loops,        // 4: num_iterations (inner loops per dispatch)
                        cycles_to_wait,          // 5: cycles_to_wait (derived from fpu_utilization_pct)
                        cfg.super_sync,          // 6: super_sync
                        cfg.l1_super_sync_addr,  // 7: l1_super_sync_addr
                        cfg.l1_fpu_timing_addr   // 8: l1_fpu_timing_addr
                    },
            });
    }

    // Super-sync releases the compute kernels; it is meaningless without them.
    if (cfg.super_sync && fpu_on) {
        auto sender_semaphore_id = tt_metal::CreateSemaphore(program, core_range_set, INVALID);
        auto receiver_semaphore_id = tt_metal::CreateSemaphore(program, core_range_set, INVALID);
        auto barrier_kernel = CreateKernel(
            program,
            "tests/didt/max_util_workload/kernels/super_sync.cpp",
            core_range_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {sender_semaphore_id, receiver_semaphore_id, cfg.l1_super_sync_addr},
            });

        CoreCoord phys_tl = device->worker_core_from_logical_core(cfg.grid_start);
        CoreCoord phys_br = device->worker_core_from_logical_core(cfg.grid_end);
        uint32_t num_dests = core_range.size() - 1;
        for (const auto& core : core_range) {
            CoreCoord worker_core = device->worker_core_from_logical_core(core);
            bool is_sender = core == cfg.grid_start;
            SetRuntimeArgs(
                program,
                barrier_kernel,
                core,
                {
                    is_sender,
                    static_cast<uint32_t>(phys_tl.x),
                    static_cast<uint32_t>(phys_tl.y),
                    static_cast<uint32_t>(phys_br.x),
                    static_cast<uint32_t>(phys_br.y),
                    num_dests,
                    static_cast<uint32_t>(worker_core.x),
                    static_cast<uint32_t>(worker_core.y),
                });
        }
    }

    // -- Tensix DRAM streaming kernel (NCRISC on every worker) --
    // Guard: skip when setup_tensix_dram_config was not called / disabled.
    if (cfg.tdram_buffer_addr != 0) {
        const uint64_t duration = cfg.tdram_duration_ticks;
        const std::vector<uint32_t> tdram_compile_args = {
            cfg.tdram_num_banks,                            // 0: num_banks
            cfg.tdram_pages_per_bank,                       // 1: pages_per_bank
            cfg.tdram_xfer_bytes,                           // 2: xfer_bytes
            cfg.tdram_burst,                                // 3: burst
            cfg.tdram_idle_units,                           // 4: idle_units
            static_cast<uint32_t>(duration & 0xFFFFFFFFu),  // 5: duration lo
            static_cast<uint32_t>(duration >> 32),          // 6: duration hi
            cfg.tdram_period_units,                         // 7: period_units (0 = off)
            cfg.tdram_active_units,                         // 8: active_units
        };
        // Split workers in a checkerboard across NOC0 and NOC1 so a single NOC's
        // links do not cap the aggregate DRAM bandwidth.
        std::set<CoreRange> noc_cores[2];
        for (const auto& core : core_range) {
            noc_cores[(core.x + core.y) & 1].insert(CoreRange(core, core));
        }
        uint32_t core_idx = 0;
        for (uint32_t noc = 0; noc < 2; ++noc) {
            if (noc_cores[noc].empty()) {
                continue;
            }
            auto tdram_kernel = CreateKernel(
                program,
                "tests/didt/max_util_workload/kernels/tensix_dram_reader.cpp",
                CoreRangeSet(noc_cores[noc]),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = noc == 0 ? NOC::NOC_0 : NOC::NOC_1,
                    .compile_args = tdram_compile_args,
                });
            for (const auto& range : noc_cores[noc]) {
                SetRuntimeArgs(
                    program,
                    tdram_kernel,
                    range.start_coord,
                    {cfg.tdram_buffer_addr, cfg.l1_tdram_staging_addr, cfg.l1_tdram_timing_addr, core_idx++});
            }
        }
    }

    // -- ETH link traffic kernel (ERISC0 on every link-up core) --
    if (!cfg.eth_links.empty()) {
        std::set<CoreRange> link_ranges;
        for (const auto& link : cfg.eth_links) {
            link_ranges.insert(CoreRange(link.core, link.core));
        }
        EthernetConfig link_cfg{
            .eth_mode = Eth::SENDER,
            .noc = NOC::NOC_0,
            .processor = DataMovementProcessor::RISCV_0,
            .compile_args =
                {
                    cfg.eth_link_num_slots,            // 0: num_slots
                    cfg.eth_link_xfer_bytes,           // 1: xfer_bytes
                    cfg.eth_link_burst,                // 2: burst
                    cfg.eth_link_idle_units,           // 3: idle_units
                    cfg.eth_link_startup_delay_ticks,  // 4: startup_delay_ticks
                    cfg.eth_link_period_units,         // 5: period_units (0 = off)
                    cfg.eth_link_active_units,         // 6: active_units
                },
        };
        eth_test_common::set_arch_specific_eth_config(link_cfg);
        auto link_kernel = CreateKernel(
            program, "tests/didt/max_util_workload/kernels/eth_link_traffic.cpp", CoreRangeSet(link_ranges), link_cfg);

        const auto& hal = MetalContext::instance().hal();
        const uint32_t l1_base = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
        const uint32_t num_banks = device->allocator()->get_num_banks(BufferType::DRAM);
        for (uint32_t i = 0; i < cfg.eth_links.size(); ++i) {
            const auto& link = cfg.eth_links[i];
            // Timing record i is page i of the interleaved timing buffer.
            const uint32_t timing_bank = i % num_banks;
            const uint32_t timing_addr = cfg.eth_link_timing_buffer_addr + (i / num_banks) * kEthLinkTimingPageBytes;
            SetRuntimeArgs(
                program,
                link_kernel,
                link.core,
                {l1_base,
                 link.role,
                 static_cast<uint32_t>(cfg.eth_link_duration_ticks),
                 static_cast<uint32_t>(cfg.eth_link_duration_ticks >> 32),
                 timing_bank,
                 timing_addr});
        }
    }

    // -- ETH DRAM streaming kernel (one active core per bank) --
    // Guard: skip when setup_eth_stream_config was not called.
    // Runs on ERISC1 when the link-traffic kernel owns ERISC0.
    if (cfg.eth_dram_buffer_addr != 0) {
        auto assignments = assign_eth_cores_to_banks(device);
        if (!assignments.empty()) {
            std::set<CoreRange> eth_ranges;
            for (const auto& [core, bank_id] : assignments) {
                eth_ranges.insert(CoreRange(core, core));
            }

            const bool use_erisc1 = !cfg.eth_links.empty();
            EthernetConfig eth_cfg{
                .eth_mode = Eth::SENDER,
                .noc = use_erisc1 ? NOC::NOC_1 : NOC::NOC_0,
                .processor = use_erisc1 ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0,
                .compile_args =
                    {
                        static_cast<uint32_t>(cfg.eth_duration_ticks),  // 0: duration lo
                        cfg.eth_pages_per_bank,
                        cfg.eth_page_size,
                        cfg.eth_noc_wait_cycles,  // 3: noc_wait_cycles (0 = max DRAM util)
                        static_cast<uint32_t>(cfg.eth_duration_ticks >> 32),  // 4: duration hi
                    },
            };
            eth_test_common::set_arch_specific_eth_config(eth_cfg);
            auto eth_kernel = CreateKernel(
                program, "tests/didt/max_util_workload/kernels/eth_dram_reader.cpp", CoreRangeSet(eth_ranges), eth_cfg);

            for (uint32_t stream = 0; stream < assignments.size(); ++stream) {
                const auto& [core, bank_id] = assignments[stream];
                // Spread concurrent read requests across VCs, matching the
                // DRAM bandwidth microbenchmark's contention-avoidance strategy.
                const uint32_t read_vc = stream & 0x3;
                SetRuntimeArgs(
                    program, eth_kernel, core, {cfg.eth_dram_buffer_addr, cfg.eth_l1_staging_addr, bank_id, read_vc});
            }

            log_info(
                LogTest,
                "ETH DRAM streaming: {} bank streams, {} enqueues × {} ticks, "
                "{} pages/bank × {} B/page ({}KB)",
                assignments.size(),
                cfg.num_iterations,
                cfg.eth_duration_ticks,
                cfg.eth_pages_per_bank,
                cfg.eth_page_size,
                cfg.eth_page_size / 1024);
        }
    }

    return program;
}

// ---------------------------------------------------------------------------
// build_slow_cos_program – constructs the slow (cos) program
//
// The compute kernel runs SFPU cosine instead of MVMUL, making it significantly
// lighter on the FPU. The slow workload loops num_slow_wl_loops times per dispatch.
// ---------------------------------------------------------------------------

static Program build_slow_cos_program(const MaxUtilConfig& cfg) {
    const CoreRange core_range(cfg.grid_start, cfg.grid_end);
    const CoreRangeSet core_range_set({core_range});

    log_info(
        LogTest,
        "Slow-cos program: {} cores, {} enqueues × {} slow inner iterations",
        core_range.size(),
        cfg.num_iterations,
        cfg.num_slow_wl_loops);

    Program program = CreateProgram();

    // -- Compute kernel (TRISC): SFPU cosine on L1 data --
    CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/slow_cos_compute.cpp",
        core_range_set,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .compile_args =
                {
                    cfg.l1_buffer0_addr,    // 0: l1_buffer0_addr (bfloat16 input)
                    cfg.l1_buffer2_addr,    // 1: l1_buffer2_addr (output)
                    cfg.num_tiles,          // 2: num_tiles (8)
                    cfg.num_slow_wl_loops,  // 3: num_loops
                },
        });

    return program;
}

// ---------------------------------------------------------------------------
// validate_compute_output – verifies that PACK produced plausible output on a
// representative worker. This is a workload-liveness check rather than a
// numerical-correctness test: different Blackhole SKUs may write different
// portions of the reserved output region. The host readback occurs after the
// measured device workload completes.
// ---------------------------------------------------------------------------

static bool validate_compute_output(IDevice* device, const MaxUtilConfig& cfg) {
    const uint32_t output_size_bytes = cfg.num_tiles * tile_size_bytes(DataFormat::Float16_b);
    const size_t expected_words = output_size_bytes / sizeof(uint32_t);
    std::vector<uint32_t> output;
    detail::ReadFromDeviceL1(device, cfg.grid_start, cfg.l1_buffer2_addr, output_size_bytes, output, CoreType::WORKER);

    if (output.size() != expected_words) {
        log_warning(
            LogTest,
            "Device {}: compute output readback returned {} words, expected {}",
            device->id(),
            output.size(),
            expected_words);
        return false;
    }

    size_t written_words = 0;
    size_t non_finite_values = 0;
    bool any_nonzero = false;
    for (uint32_t word : output) {
        if (word == kOutputSentinel) {
            continue;
        }
        ++written_words;
        const uint32_t low_bfloat16 = word & 0xFFFFu;
        const uint32_t high_bfloat16 = word >> 16;
        any_nonzero |= (low_bfloat16 & 0x7FFFu) != 0 || (high_bfloat16 & 0x7FFFu) != 0;
        non_finite_values += (low_bfloat16 & 0x7F80u) == 0x7F80u;
        non_finite_values += (high_bfloat16 & 0x7F80u) == 0x7F80u;
    }

    const bool valid = written_words > 0 && non_finite_values == 0 && any_nonzero;
    if (!valid) {
        log_warning(
            LogTest,
            "Device {}: compute output validation failed: written_words={}/{}, non_finite_bfloat16_values={}, "
            "any_nonzero={}",
            device->id(),
            written_words,
            expected_words,
            non_finite_values,
            any_nonzero);
        return false;
    }

    log_info(
        LogTest,
        "Device {}: compute output validation passed ({} of {} words written, {} finite bfloat16 values)",
        device->id(),
        written_words,
        expected_words,
        written_words * 2);
    return true;
}

// ---------------------------------------------------------------------------
// log_fpu_utilization – reads back-pressured unpack-thread timestamps from a
// representative worker and reports useful MVMUL cycles / pipeline cycles.
// ---------------------------------------------------------------------------

static bool log_fpu_utilization(IDevice* device, const MaxUtilConfig& cfg) {
    std::vector<uint32_t> timing;
    detail::ReadFromDeviceL1(
        device, cfg.grid_start, cfg.l1_fpu_timing_addr, 4 * sizeof(uint32_t), timing, CoreType::WORKER);
    if (timing.size() != 4) {
        log_warning(LogTest, "Device {}: FPU timing readback returned {} words", device->id(), timing.size());
        return false;
    }

    uint64_t t0 = (static_cast<uint64_t>(timing[1]) << 32) | timing[0];
    uint64_t t1 = (static_cast<uint64_t>(timing[3]) << 32) | timing[2];
    if (t1 <= t0) {
        log_warning(LogTest, "Device {}: invalid FPU timing (t0={}, t1={})", device->id(), t0, t1);
        return false;
    }

    // Each tile executes eight repetitions of sixteen MVMUL instructions.
    constexpr uint32_t useful_fpu_cycles_per_tile = 8 * 16;
    uint64_t useful_cycles = static_cast<uint64_t>(useful_fpu_cycles_per_tile) * cfg.num_tiles * cfg.num_wl_loops;
    uint64_t elapsed_cycles = t1 - t0;
    double measured_pct = 100.0 * static_cast<double>(useful_cycles) / static_cast<double>(elapsed_cycles);

    // Very short smoke runs do not fill the unpack/math/pack pipeline, so the
    // back-pressure timing window is not a meaningful utilization estimate.
    if (cfg.num_wl_loops < 100) {
        log_info(
            LogTest,
            "Device {} FPU utilization measurement skipped: {} loop(s) do not fill the compute pipeline",
            device->id(),
            cfg.num_wl_loops);
        return true;
    }

    const uint32_t matched_fpu_pct = nearest_calibration_pct(kFpuUtilToCyclesMap, cfg.fpu_utilization_pct);
    const double matched_pct = static_cast<double>(matched_fpu_pct);
    constexpr double tolerance_pct_points = 2.0;
    bool within_tolerance = std::abs(measured_pct - matched_pct) <= tolerance_pct_points;
    log_info(
        LogTest,
        "Device {} FPU utilization: requested={}% matched={}% measured={:.1f}% "
        "({} useful MVMUL cycles / {} elapsed cycles on worker ({},{}))",
        device->id(),
        cfg.fpu_utilization_pct,
        matched_fpu_pct,
        measured_pct,
        useful_cycles,
        elapsed_cycles,
        cfg.grid_start.x,
        cfg.grid_start.y);
    if (!within_tolerance) {
        log_warning(
            LogTest,
            "Device {} FPU utilization is outside the ±{:.1f} percentage-point tolerance",
            device->id(),
            tolerance_pct_points);
    }
    return within_tolerance;
}

// ---------------------------------------------------------------------------
// log_eth_bw – reads per-core timing persisted in DRAM and logs bandwidth
// ---------------------------------------------------------------------------

/// Minimum host-timed dispatch length for converting ticks with host time;
/// shorter runs are dominated by launch latency and use the nominal clock.
static constexpr double kMinHostTimedSeconds = 2.0;

struct TickClock {
    double ghz;
    std::string source;
};

/// Clock rate for converting a generator's elapsed ticks to time. Device wall
/// clocks tick at AICLK, which DVFS drops (to 800 MHz on Blackhole) when the
/// chip hits its power limit. For long runs the average rate is the busiest
/// generator's ticks over the host-measured dispatch time; generators are
/// sized to span the same window, so this is valid for all of them.
static TickClock tick_clock(IDevice* device, const MaxUtilConfig& cfg, uint64_t max_elapsed_ticks) {
    const double nominal = static_cast<double>(device->get_clock_rate_mhz()) / 1000.0;
    if (cfg.dispatch_seconds < kMinHostTimedSeconds || max_elapsed_ticks == 0) {
        return {nominal, "nominal AICLK"};
    }
    const double measured = static_cast<double>(max_elapsed_ticks) / (cfg.dispatch_seconds * 1e9);
    if (measured >= nominal) {
        return {nominal, "nominal AICLK"};
    }
    return {measured, measured < 0.95 * nominal ? "host-timed avg AICLK, THROTTLED" : "host-timed avg AICLK"};
}

static bool log_eth_bw(IDevice* device, const MaxUtilConfig& cfg, const shared_ptr<Buffer>& dram_buffer) {
    if (cfg.eth_dram_buffer_addr == 0 || dram_buffer == nullptr) {
        log_info(LogTest, "Device {}: ETH DRAM bandwidth validation skipped (no active ETH streams)", device->id());
        return true;
    }

    auto assignments = assign_eth_cores_to_banks(device);
    if (assignments.empty()) {
        return true;
    }

    std::vector<uint32_t> readback;
    detail::ReadFromBuffer(dram_buffer, readback);

    // Each stream reads cfg.eth_pages_per_bank pages from 1 bank per iteration.
    const uint64_t bytes_per_iter = static_cast<uint64_t>(cfg.eth_pages_per_bank) * cfg.eth_page_size;

    struct Stream {
        CoreCoord core;
        uint32_t bank;
        uint64_t elapsed;
        uint64_t bytes;
    };
    std::vector<Stream> streams;
    uint64_t max_elapsed = 0;
    for (const auto& [eth_core, bank_id] : assignments) {
        const size_t timing_offset = static_cast<size_t>(bank_id) * cfg.eth_page_size / sizeof(uint32_t);
        if (readback.size() < timing_offset + 4) {
            log_warning(
                LogTest,
                "Device {} ETH core ({},{}) bank {} – timing readback too short ({}), skipping",
                device->id(),
                eth_core.x,
                eth_core.y,
                bank_id,
                readback.size());
            continue;
        }
        const uint64_t elapsed =
            (static_cast<uint64_t>(readback[timing_offset + 1]) << 32) | readback[timing_offset];
        const uint64_t iters =
            (static_cast<uint64_t>(readback[timing_offset + 3]) << 32) | readback[timing_offset + 2];
        if (elapsed == 0 || iters == 0) {
            log_warning(
                LogTest,
                "Device {} ETH core ({},{}) bank {} – invalid timing (elapsed={}, iterations={}), skipping",
                device->id(),
                eth_core.x,
                eth_core.y,
                bank_id,
                elapsed,
                iters);
            continue;
        }
        streams.push_back({eth_core, bank_id, elapsed, iters * bytes_per_iter});
        max_elapsed = std::max(max_elapsed, elapsed);
    }

    const auto clk = tick_clock(device, cfg, max_elapsed);
    double total_bw_bpc = 0.0;  // bytes/cycle
    for (const auto& st : streams) {
        const double bw_bpc = static_cast<double>(st.bytes) / static_cast<double>(st.elapsed);
        total_bw_bpc += bw_bpc;
        auto eth_noc0 = eth_noc0_coord(device, st.core);
        auto dram_noc0 = dram_noc0_coord(device, st.bank);
        log_info(
            LogTest,
            "Device {} ETH ({},{}) [NOC0 ({},{})] → bank {} [NOC0 ({},{})] "
            "BW: {:.3f} bytes/cycle  {:.2f} GB/s  ({} bytes, {} cycles)",
            device->id(),
            st.core.x,
            st.core.y,
            eth_noc0.x,
            eth_noc0.y,
            st.bank,
            dram_noc0.x,
            dram_noc0.y,
            bw_bpc,
            bw_bpc * clk.ghz,
            st.bytes,
            st.elapsed);
    }
    const size_t reported = streams.size();

    if (reported > 0) {
        log_info(
            LogTest,
            "Device {}: aggregate ETH-to-DRAM BW ({} banks): {:.3f} bytes/cycle  {:.2f} GB/s  "
            "(@ {:.2f} GHz, {})",
            device->id(),
            reported,
            total_bw_bpc,
            total_bw_bpc * clk.ghz,
            clk.ghz,
            clk.source);
    }
    return reported == assignments.size();
}

// ---------------------------------------------------------------------------
// log_tensix_dram_bw – reads every worker's NCRISC timing record and reports
//   the aggregate Tensix-to-DRAM bandwidth of the last dispatch.
// ---------------------------------------------------------------------------

static bool log_tensix_dram_bw(IDevice* device, const MaxUtilConfig& cfg) {
    if (cfg.tdram_buffer_addr == 0) {
        return true;
    }
    uint64_t total_bytes = 0;
    uint64_t max_elapsed = 0;
    uint64_t min_elapsed = UINT64_MAX;
    uint32_t bad_cores = 0;
    const CoreRange core_range(cfg.grid_start, cfg.grid_end);
    for (const auto& core : core_range) {
        std::vector<uint32_t> rec;
        detail::ReadFromDeviceL1(device, core, cfg.l1_tdram_timing_addr, 16, rec, CoreType::WORKER);
        if (rec.size() != 4) {
            ++bad_cores;
            continue;
        }
        const uint64_t elapsed = (static_cast<uint64_t>(rec[1]) << 32) | rec[0];
        const uint64_t xfers = (static_cast<uint64_t>(rec[3]) << 32) | rec[2];
        if (elapsed == 0 || xfers == 0) {
            ++bad_cores;
            continue;
        }
        total_bytes += xfers * cfg.tdram_xfer_bytes;
        max_elapsed = std::max(max_elapsed, elapsed);
        min_elapsed = std::min(min_elapsed, elapsed);
    }
    if (max_elapsed == 0) {
        log_warning(LogTest, "Device {}: no valid Tensix DRAM timing records", device->id());
        return false;
    }
    const auto clk = tick_clock(device, cfg, max_elapsed);
    const double ghz = clk.ghz;
    const double bpc = static_cast<double>(total_bytes) / static_cast<double>(max_elapsed);
    log_info(
        LogTest,
        "Device {}: aggregate Tensix-to-DRAM BW ({} cores, {} banks): {:.2f} bytes/cycle  {:.2f} GB/s  "
        "({} bytes, window {}..{} cycles @ {:.2f} GHz, {})",
        device->id(),
        core_range.size() - bad_cores,
        cfg.tdram_num_banks,
        bpc,
        bpc * ghz,
        total_bytes,
        min_elapsed,
        max_elapsed,
        ghz,
        clk.source);
    if (bad_cores > 0) {
        log_warning(LogTest, "Device {}: {} worker(s) reported no Tensix DRAM traffic", device->id(), bad_cores);
    }
    return bad_cores == 0;
}

// ---------------------------------------------------------------------------
// log_eth_link_bw – reads per-link timing records persisted in DRAM and
//   reports per-direction Ethernet link bandwidth of the last dispatch.
// ---------------------------------------------------------------------------

static bool log_eth_link_bw(IDevice* device, const MaxUtilConfig& cfg, const shared_ptr<Buffer>& timing_buffer) {
    if (cfg.eth_links.empty() || timing_buffer == nullptr) {
        return true;
    }
    std::vector<uint32_t> readback;
    detail::ReadFromBuffer(timing_buffer, readback);
    constexpr size_t words_per_page = kEthLinkTimingPageBytes / sizeof(uint32_t);
    uint64_t max_link_elapsed = 0;
    for (size_t i = 0; i < cfg.eth_links.size(); ++i) {
        const size_t off = i * words_per_page;
        if (readback.size() >= off + 2) {
            max_link_elapsed =
                std::max(max_link_elapsed, (static_cast<uint64_t>(readback[off + 1]) << 32) | readback[off]);
        }
    }
    const auto clk = tick_clock(device, cfg, max_link_elapsed);
    const double ghz = clk.ghz;
    bool ok = true;
    double total_tx_bpc = 0.0;
    for (size_t i = 0; i < cfg.eth_links.size(); ++i) {
        const auto& link = cfg.eth_links[i];
        const size_t off = i * words_per_page;
        if (readback.size() < off + 4) {
            log_warning(LogTest, "Device {}: ETH link timing readback too short", device->id());
            return false;
        }
        const uint64_t elapsed = (static_cast<uint64_t>(readback[off + 1]) << 32) | readback[off];
        const uint32_t sent = readback[off + 2];
        const uint32_t received = readback[off + 3];
        // Each side drains the peer's traffic before exiting, so a record with
        // traffic both ways means the link ran and shut down cleanly.
        const bool complete = elapsed != 0 && sent != 0 && received != 0;
        ok &= complete;
        const double tx_bpc =
            elapsed == 0 ? 0.0 : static_cast<double>(sent) * cfg.eth_link_xfer_bytes / static_cast<double>(elapsed);
        total_tx_bpc += tx_bpc;
        log_info(
            LogTest,
            "Device {} ETH ({},{}) ↔ chip {} ({},{}) role {}: TX {:.2f} GB/s  ({} sent, {} received, {} cycles){}",
            device->id(),
            link.core.x,
            link.core.y,
            link.peer_chip,
            link.peer_core.x,
            link.peer_core.y,
            link.role,
            tx_bpc * ghz,
            sent,
            received,
            elapsed,
            complete ? "" : "  INCOMPLETE");
    }
    log_info(
        LogTest,
        "Device {}: aggregate ETH link TX BW ({} links): {:.2f} GB/s (RX is symmetric) (@ {:.2f} GHz, {})",
        device->id(),
        cfg.eth_links.size(),
        total_tx_bpc * ghz,
        ghz,
        clk.source);
    return ok;
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Runs the pre-fill program to populate L1 buffers, then runs the main program.
static bool run_single_device(const shared_ptr<distributed::MeshDevice>& mesh_device, MaxUtilConfig& cfg) {
    IDevice* device = mesh_device->impl().get_device(0);

    auto& cq = mesh_device->mesh_command_queue();
    auto target =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate({0, 0}), distributed::MeshCoordinate({0, 0}));

    // Phase 1: Pre-fill L1 buffers from DRAM
    auto prefill_program = build_prefill_program(device, cfg);

    // Run pre-fill program
    auto prefill_workload = distributed::MeshWorkload();
    prefill_workload.add_program(target, std::move(prefill_program.program));
    distributed::EnqueueMeshWorkload(cq, prefill_workload, /*blocking=*/true);
    distributed::Finish(cq);

    log_info(LogTest, "Pre-fill phase complete");

    // Link traffic needs both ends dispatched; single-device runs only log the skip.
    auto eth_link_buf = setup_eth_link_config(device, cfg, {device->id()});
    auto tdram_buf = setup_tensix_dram_config(device, cfg);
    // Set up ETH DRAM streaming (active ETH cores only); keep buffer alive until Finish.
    auto eth_dram_buf = setup_eth_stream_config(device, cfg);

    // Phase 2: Build and run main program, optionally interleaved with slow cos.
    auto mesh_workload = distributed::MeshWorkload();
    mesh_workload.add_program(target, build_program(device, cfg));

    const bool has_slow_wl = cfg.num_slow_wl_loops > 0;
    auto slow_cos_workload = distributed::MeshWorkload();
    if (has_slow_wl) {
        slow_cos_workload.add_program(target, build_slow_cos_program(cfg));
        log_info(
            LogTest, "Duty-cycle interleave: slow cos loops={} between each max-util dispatch", cfg.num_slow_wl_loops);
    }

    const auto t_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < cfg.num_iterations; ++i) {
        bool is_last = (i == cfg.num_iterations - 1);
        if (has_slow_wl) {
            // Pattern: max_util (non-blocking) → slow_cos (blocking on last)
            distributed::EnqueueMeshWorkload(cq, mesh_workload, /*blocking=*/false);
            distributed::EnqueueMeshWorkload(cq, slow_cos_workload, /*blocking=*/is_last);
        } else {
            distributed::EnqueueMeshWorkload(cq, mesh_workload, /*blocking=*/is_last);
        }
    }
    distributed::Finish(cq);
    if (!has_slow_wl) {
        cfg.dispatch_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() / cfg.num_iterations;
    }

    const bool fpu_on = cfg.fpu_utilization_pct != 0;
    bool valid_compute_output = !fpu_on || validate_compute_output(device, cfg);
    bool valid_fpu_timing = !fpu_on || log_fpu_utilization(device, cfg);

    // Report per-ETH-core DRAM bandwidth after the program completes.
    bool valid_eth_timing = log_eth_bw(device, cfg, eth_dram_buf);
    valid_eth_timing &= log_eth_link_bw(device, cfg, eth_link_buf);
    bool valid_tdram_timing = log_tensix_dram_bw(device, cfg);

    log_info(
        LogTest,
        "MaxUtilWorkload [single-device] done: device={}, grid=[{},{}]->[{},{}], tiles={}, "
        "enqueues={}, inner_iters={}, slow_cos_loops={}",
        device->id(),
        cfg.grid_start.x,
        cfg.grid_start.y,
        cfg.grid_end.x,
        cfg.grid_end.y,
        cfg.num_tiles,
        cfg.num_iterations,
        cfg.num_wl_loops,
        cfg.num_slow_wl_loops);

    return valid_compute_output && valid_fpu_timing && valid_eth_timing && valid_tdram_timing;
}

/// Runs the workload on every unit-mesh device simultaneously. Active Ethernet
/// link cores are reserved by a full-system mesh, so each device must use the
/// same isolated unit-mesh setup as the active-ETH API tests.
static bool run_all_devices(
    const std::vector<shared_ptr<distributed::MeshDevice>>& mesh_devices,
    uint32_t num_tiles,
    uint32_t num_iterations,
    uint32_t num_wl_loops,
    uint32_t num_slow_wl_loops = 0,
    bool super_sync = false) {
    struct DeviceRun {
        shared_ptr<distributed::MeshDevice> mesh_device;
        IDevice* device;
        MaxUtilConfig cfg;
        shared_ptr<Buffer> eth_dram_buffer;
        shared_ptr<Buffer> eth_link_buffer;
        shared_ptr<Buffer> tdram_buffer;
        std::unique_ptr<distributed::MeshWorkload> main_workload;
        std::unique_ptr<distributed::MeshWorkload> slow_workload;
    };

    const auto zero = distributed::MeshCoordinate({0, 0});
    const auto target = distributed::MeshCoordinateRange(zero, zero);
    std::vector<DeviceRun> runs;
    runs.reserve(mesh_devices.size());

    // Every opened device is dispatched together, so links between any two of
    // them can carry traffic (both ends run the link kernel concurrently).
    std::set<ChipId> dispatched_chips;
    for (const auto& mesh_device : mesh_devices) {
        dispatched_chips.insert(mesh_device->impl().get_devices().at(0)->id());
    }

    // Pre-fill each device before the simultaneous stress phase.
    for (const auto& mesh_device : mesh_devices) {
        auto* device = mesh_device->impl().get_devices().at(0);
        MaxUtilConfig cfg =
            full_grid_config(device, num_tiles, num_iterations, num_wl_loops, num_slow_wl_loops, super_sync);
        auto prefill_program = build_prefill_program(device, cfg);
        auto prefill_workload = distributed::MeshWorkload();
        prefill_workload.add_program(target, std::move(prefill_program.program));
        auto& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueMeshWorkload(cq, prefill_workload, /*blocking=*/true);
        distributed::Finish(cq);
        log_info(LogTest, "Pre-fill complete: device={}", device->id());

        cfg.eth_link_allowed = true;
        auto eth_link_buffer = setup_eth_link_config(device, cfg, dispatched_chips);
        auto tdram_buffer = setup_tensix_dram_config(device, cfg);
        auto eth_dram_buffer = setup_eth_stream_config(device, cfg);
        auto main_workload = std::make_unique<distributed::MeshWorkload>();
        main_workload->add_program(target, build_program(device, cfg));
        std::unique_ptr<distributed::MeshWorkload> slow_workload;
        if (cfg.num_slow_wl_loops > 0) {
            slow_workload = std::make_unique<distributed::MeshWorkload>();
            slow_workload->add_program(target, build_slow_cos_program(cfg));
        }
        runs.push_back(DeviceRun{
            mesh_device,
            device,
            cfg,
            std::move(eth_dram_buffer),
            std::move(eth_link_buffer),
            std::move(tdram_buffer),
            std::move(main_workload),
            std::move(slow_workload)});
        log_info(LogTest, "Main program ready: device={}", device->id());
    }

    if (num_slow_wl_loops > 0) {
        log_info(LogTest, "Duty-cycle interleave active: slow cos between each max-util dispatch");
    }

    const auto t_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < num_iterations; ++i) {
        bool is_last = (i == num_iterations - 1);
        bool is_log_iter = ((i + 1) % 100 == 0) || is_last;
        for (auto& run : runs) {
            auto& cq = run.mesh_device->mesh_command_queue();
            distributed::EnqueueMeshWorkload(cq, *run.main_workload, /*blocking=*/false);
            if (run.slow_workload != nullptr) {
                distributed::EnqueueMeshWorkload(cq, *run.slow_workload, /*blocking=*/false);
            }
        }
        if (is_log_iter) {
            fmt::print("\rIteration {}/{}   ", i + 1, num_iterations);
            std::fflush(stdout);
            if (is_last) {
                fmt::print("\n");
            }
        }
    }
    for (auto& run : runs) {
        distributed::Finish(run.mesh_device->mesh_command_queue());
    }
    const double dispatch_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() / num_iterations;
    for (auto& run : runs) {
        if (run.cfg.num_slow_wl_loops == 0) {
            run.cfg.dispatch_seconds = dispatch_seconds;
        }
    }

    // Report FPU utilization and per-ETH-core DRAM bandwidth for every device.
    bool valid_compute_output = true;
    bool valid_fpu_timing = true;
    bool valid_eth_timing = true;
    bool valid_tdram_timing = true;
    for (const auto& run : runs) {
        if (run.cfg.fpu_utilization_pct != 0) {
            valid_compute_output &= validate_compute_output(run.device, run.cfg);
            valid_fpu_timing &= log_fpu_utilization(run.device, run.cfg);
        }
        valid_eth_timing &= log_eth_bw(run.device, run.cfg, run.eth_dram_buffer);
        valid_eth_timing &= log_eth_link_bw(run.device, run.cfg, run.eth_link_buffer);
        valid_tdram_timing &= log_tensix_dram_bw(run.device, run.cfg);
    }

    return valid_compute_output && valid_fpu_timing && valid_eth_timing && valid_tdram_timing;
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------

/// Returns the number of iterations to use for stress / all-devices tests.
/// Reads MAX_UTIL_NUM_ITERATIONS from the environment; defaults to 1.
static uint32_t get_num_iterations() {
    const char* env = std::getenv("MAX_UTIL_NUM_ITERATIONS");
    if (env != nullptr) {
        const auto val = parse_env_int(env);
        if (val.has_value() && *val > 0) {
            return static_cast<uint32_t>(*val);
        }
        log_warning(LogTest, "MAX_UTIL_NUM_ITERATIONS='{}' is not a positive integer – using default of 1", env);
    }
    return 1;
}

/// Reads MAX_UTIL_NUM_WL_LOOPS from the environment; defaults to 1000.
static uint32_t get_num_wl_loops() {
    const char* env = std::getenv("MAX_UTIL_NUM_WL_LOOPS");
    if (env != nullptr) {
        const auto val = parse_env_int(env);
        if (val.has_value() && *val > 0) {
            return static_cast<uint32_t>(*val);
        }
        log_warning(LogTest, "MAX_UTIL_NUM_WL_LOOPS='{}' is not a positive integer – using default of 1000", env);
    }
    return 1000;
}

/// Quick smoke run: 2x2 grid on device 0, one iteration.
void max_util_smoke(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    MaxUtilConfig cfg;
    cfg.grid_start = {0, 0};
    cfg.grid_end = {1, 1};
    cfg.num_tiles = 8;
    cfg.num_iterations = 1;
    cfg.num_wl_loops = 1;
    cfg.fpu_utilization_pct = get_fpu_utilization_pct();
    cfg.eth_dram_util_pct = get_dram_utilization_pct();
    cfg.eth_page_size = get_dram_page_size_bytes();
    cfg.super_sync = get_super_sync();
    apply_power_virus_knobs(cfg);
    EXPECT_TRUE(run_single_device(mesh_device, cfg));
}

/// Sustained stress run on device 0: all available cores, many iterations.
void max_util_stress(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    IDevice* device = mesh_device->impl().get_device(0);
    uint32_t num_iterations = get_num_iterations();
    uint32_t num_wl_loops = get_num_wl_loops();
    uint32_t duty_cycle_pct = get_duty_cycle_pct();
    uint32_t num_slow_wl_loops = duty_cycle_to_slow_loops(duty_cycle_pct, num_wl_loops);
    bool super_sync = get_super_sync();
    log_info(
        LogTest,
        "MaxUtilWorkload_Stress: num_iterations={}, num_wl_loops={}, duty_cycle={}%, num_slow_wl_loops={}, "
        "super_sync={}",
        num_iterations,
        num_wl_loops,
        duty_cycle_pct,
        num_slow_wl_loops,
        super_sync);
    MaxUtilConfig cfg = full_grid_config(
        device,
        /*num_tiles=*/8,
        /*num_iterations=*/num_iterations,
        /*num_wl_loops=*/num_wl_loops,
        /*num_slow_wl_loops=*/num_slow_wl_loops,
        /*super_sync=*/super_sync);
    EXPECT_TRUE(run_single_device(mesh_device, cfg));
}

/// All-devices stress run: every device in the mesh, all cores, many iterations.
void max_util_all_devices(const std::vector<shared_ptr<distributed::MeshDevice>>& mesh_devices) {
    uint32_t num_iterations = get_num_iterations();
    uint32_t num_wl_loops = get_num_wl_loops();
    uint32_t duty_cycle_pct = get_duty_cycle_pct();
    uint32_t num_slow_wl_loops = duty_cycle_to_slow_loops(duty_cycle_pct, num_wl_loops);
    bool super_sync = get_super_sync();
    log_info(
        LogTest,
        "MaxUtilWorkload_AllDevices: num_iterations={}, num_wl_loops={}, duty_cycle={}%, num_slow_wl_loops={}, "
        "super_sync={}",
        num_iterations,
        num_wl_loops,
        duty_cycle_pct,
        num_slow_wl_loops,
        super_sync);
    EXPECT_TRUE(run_all_devices(
        mesh_devices,
        /*num_tiles=*/8,
        /*num_iterations=*/num_iterations,
        /*num_wl_loops=*/num_wl_loops,
        /*num_slow_wl_loops=*/num_slow_wl_loops,
        /*super_sync=*/super_sync));
}

}  // namespace unit_tests::didt::max_util_workload

// ---------------------------------------------------------------------------
// GTest fixtures
// ---------------------------------------------------------------------------

TEST_F(MeshDispatchFixture, MaxUtilWorkload_Smoke) {
    if (MetalContext::instance().hal().get_arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Max-util workload is calibrated for Blackhole hardware";
    }
    unit_tests::didt::max_util_workload::max_util_smoke(devices_.at(0));
}

TEST_F(MeshDispatchFixture, MaxUtilWorkload_Stress) {
    if (MetalContext::instance().hal().get_arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Max-util workload is calibrated for Blackhole hardware";
    }
    unit_tests::didt::max_util_workload::max_util_stress(devices_.at(0));
}

TEST_F(MeshDispatchFixture, MaxUtilWorkload_AllDevices) {
    if (MetalContext::instance().hal().get_arch() != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Max-util workload is calibrated for Blackhole hardware";
    }
    unit_tests::didt::max_util_workload::max_util_all_devices(devices_);
}

}  // namespace tt::tt_metal
