// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Max-utilization workload test.
// Architecture:
//   1. Pre-fill phase: BRISC runs on each core to read 2 DRAM buffers into L1
//      - Buffer 0: 8 tiles of bfloat16 from DRAM
//      - Buffer 1: 8 tiles of bfloat16 from DRAM
//   2. Main phase: TRISC compute and active-ETH DRAM streaming run concurrently.
//      An optional BRISC kernel synchronizes TRISC launch across worker cores.
//   No CB dependencies between DRAM streaming and compute.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string_view>

#include <fmt/core.h>
#include <gtest/gtest.h>

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
    // ETH loop count: 8x fewer loops than compute to match kernel duration.
    uint32_t eth_num_wl_loops = 0;  // set by setup_eth_stream_config

    // DRAM utilization target percentage [1, 100].
    // 100 = maximum utilization (no wait between NOC page issues).
    // Lower values insert busy-wait cycles between individual page-read commands
    // to throttle the effective DRAM bandwidth.
    uint32_t eth_dram_util_pct = 100;
    // Cycles to busy-wait between consecutive NOC page issues; derived from
    // eth_dram_util_pct by setup_eth_stream_config via kDramUtilToCyclesMap.
    uint32_t eth_noc_wait_cycles = 0;

    // When true, compute kernels perform a super-sync barrier at program start.
    bool super_sync = false;
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
/// Valid range: [1, 92].
static uint32_t get_fpu_utilization_pct() {
    const char* env = std::getenv("MAX_UTIL_FPU_UTILIZATION_PCT");
    if (env != nullptr) {
        const auto val = parse_env_int(env);
        if (val.has_value() && *val >= 1 && *val <= 92) {
            return static_cast<uint32_t>(*val);
        }
        log_warning(
            LogTest, "MAX_UTIL_FPU_UTILIZATION_PCT='{}' is not an integer in [1, 92] – using default of 92", env);
    }
    return 92;
}

/// Reads MAX_UTIL_DRAM_UTILIZATION_PCT from the environment; defaults to 100.
/// Valid range: [1, 100].  100 = maximum DRAM utilization (no throttle).
static uint32_t get_dram_utilization_pct() {
    const char* env = std::getenv("MAX_UTIL_DRAM_UTILIZATION_PCT");
    if (env != nullptr) {
        const auto val = parse_env_int(env);
        if (val.has_value() && *val >= 1 && *val <= 100) {
            return static_cast<uint32_t>(*val);
        }
        log_warning(
            LogTest, "MAX_UTIL_DRAM_UTILIZATION_PCT='{}' is not an integer in [1, 100] – using default of 100", env);
    }
    return 100;
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

static shared_ptr<Buffer> setup_eth_stream_config(IDevice* device, MaxUtilConfig& cfg) {
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

    uint32_t num_banks = static_cast<uint32_t>(device->num_dram_channels());
    uint32_t page_size_bytes = cfg.eth_page_size;

    cfg.eth_pages_per_bank = (eth_l1_size - kEthL1StagingHeaderBytes) / page_size_bytes;

    // ETH kernel runs 8x fewer loops than the compute kernel to match duration.
    // Also scale by utilization percentage to match the compute kernel duration.
    const uint32_t eth_max_util_loops = std::max(1u, cfg.num_wl_loops / 8);
    cfg.eth_num_wl_loops = std::max(
        1u,
        static_cast<uint32_t>(
            static_cast<uint64_t>(eth_max_util_loops) * static_cast<uint64_t>(cfg.eth_dram_util_pct) / 100));

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
    auto dram_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device,
        .size = total_size,
        .page_size = page_size_bytes,
        .buffer_type = BufferType::DRAM,
    });
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
    auto dram_cfg_bfloat16 = InterleavedBufferConfig{
        .device = device,
        .size = buffer_size_bfloat16,
        .page_size = tile_bytes_bfloat16,
        .buffer_type = BufferType::DRAM,
    };

    auto dram_buffer0 = CreateBuffer(dram_cfg_bfloat16);
    auto dram_buffer1 = CreateBuffer(dram_cfg_bfloat16);

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

    log_info(
        LogTest,
        "Pre-fill: L1 inputs=0x{:x},0x{:x}, output=0x{:x}, super_sync=0x{:x}, fpu_timing=0x{:x}",
        cfg.l1_buffer0_addr,
        cfg.l1_buffer1_addr,
        cfg.l1_buffer2_addr,
        cfg.l1_super_sync_addr,
        cfg.l1_fpu_timing_addr);

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
    // Uses pre-filled L1 buffers directly, no CB waits
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

    if (cfg.super_sync) {
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

    // -- ETH DRAM streaming kernel (one active core per bank) --
    // Guard: skip when setup_eth_stream_config was not called.
    if (cfg.eth_dram_buffer_addr != 0) {
        auto assignments = assign_eth_cores_to_banks(device);
        if (!assignments.empty()) {
            std::set<CoreRange> eth_ranges;
            for (const auto& [core, bank_id] : assignments) {
                eth_ranges.insert(CoreRange(core, core));
            }

            EthernetConfig eth_cfg{
                .eth_mode = Eth::SENDER,
                .noc = NOC::NOC_0,
                .processor = DataMovementProcessor::RISCV_0,
                .compile_args =
                    {
                        cfg.eth_num_wl_loops,
                        cfg.eth_pages_per_bank,
                        cfg.eth_page_size,
                        cfg.eth_noc_wait_cycles,  // 3: noc_wait_cycles (0 = max DRAM util)
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
                "ETH DRAM streaming: {} bank streams, {} enqueues × {} inner iterations "
                "(compute loops={}, base ratio=1/8, scaled by DRAM utilization), "
                "{} pages/bank × {} B/page ({}KB)",
                assignments.size(),
                cfg.num_iterations,
                cfg.eth_num_wl_loops,
                cfg.num_wl_loops,
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
    uint64_t bytes_per_stream =
        static_cast<uint64_t>(cfg.eth_num_wl_loops) * cfg.eth_pages_per_bank * cfg.eth_page_size;

    // Blackhole's active-ETH wall-clock counter runs in the device AICLK domain.
    // Use the live clock rate because AICLK is DVFS-controlled (800 MHz idle,
    // nominally 1.35 GHz while busy).
    const double eth_wall_clock_ghz = static_cast<double>(device->get_clock_rate_mhz()) / 1000.0;

    double total_bw_bpc = 0.0;  // bytes/cycle
    uint32_t reported = 0;

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

        uint64_t t0 = (static_cast<uint64_t>(readback[timing_offset + 1]) << 32) | readback[timing_offset];
        uint64_t t1 = (static_cast<uint64_t>(readback[timing_offset + 3]) << 32) | readback[timing_offset + 2];
        if (t1 <= t0) {
            log_warning(
                LogTest,
                "Device {} ETH core ({},{}) bank {} – invalid timing (t0={}, t1={}), skipping",
                device->id(),
                eth_core.x,
                eth_core.y,
                bank_id,
                t0,
                t1);
            continue;
        }
        uint64_t cycles = t1 - t0;
        double bw_bpc = static_cast<double>(bytes_per_stream) / static_cast<double>(cycles);
        double bw_gbps = bw_bpc * eth_wall_clock_ghz;
        total_bw_bpc += bw_bpc;
        ++reported;

        auto eth_noc0 = eth_noc0_coord(device, eth_core);
        auto dram_noc0 = dram_noc0_coord(device, bank_id);
        log_info(
            LogTest,
            "Device {} ETH ({},{}) [NOC0 ({},{})] → bank {} [NOC0 ({},{})] "
            "BW: {:.3f} bytes/cycle  {:.2f} GB/s  ({} bytes, {} cycles)",
            device->id(),
            eth_core.x,
            eth_core.y,
            eth_noc0.x,
            eth_noc0.y,
            bank_id,
            dram_noc0.x,
            dram_noc0.y,
            bw_bpc,
            bw_gbps,
            bytes_per_stream,
            cycles);
    }

    if (reported > 0) {
        log_info(
            LogTest,
            "Device {}: aggregate ETH-to-DRAM BW ({} banks): {:.3f} bytes/cycle  {:.2f} GB/s  "
            "(@ {:.2f} GHz ETH wall clock)",
            device->id(),
            reported,
            total_bw_bpc,
            total_bw_bpc * eth_wall_clock_ghz,
            eth_wall_clock_ghz);
    }
    return reported == assignments.size();
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

    bool valid_compute_output = validate_compute_output(device, cfg);
    bool valid_fpu_timing = log_fpu_utilization(device, cfg);

    // Report per-ETH-core DRAM bandwidth after the program completes.
    bool valid_eth_timing = log_eth_bw(device, cfg, eth_dram_buf);

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

    return valid_compute_output && valid_fpu_timing && valid_eth_timing;
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
        std::unique_ptr<distributed::MeshWorkload> main_workload;
        std::unique_ptr<distributed::MeshWorkload> slow_workload;
    };

    const auto zero = distributed::MeshCoordinate({0, 0});
    const auto target = distributed::MeshCoordinateRange(zero, zero);
    std::vector<DeviceRun> runs;
    runs.reserve(mesh_devices.size());

    // Pre-fill each device before the simultaneous stress phase.
    for (const auto& mesh_device : mesh_devices) {
        IDevice* device = mesh_device->get_devices().at(0);
        MaxUtilConfig cfg =
            full_grid_config(device, num_tiles, num_iterations, num_wl_loops, num_slow_wl_loops, super_sync);
        auto prefill_program = build_prefill_program(device, cfg);
        auto prefill_workload = distributed::MeshWorkload();
        prefill_workload.add_program(target, std::move(prefill_program.program));
        auto& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueMeshWorkload(cq, prefill_workload, /*blocking=*/true);
        distributed::Finish(cq);
        log_info(LogTest, "Pre-fill complete: device={}", device->id());

        auto eth_dram_buffer = setup_eth_stream_config(device, cfg);
        auto main_workload = std::make_unique<distributed::MeshWorkload>();
        main_workload->add_program(target, build_program(device, cfg));
        std::unique_ptr<distributed::MeshWorkload> slow_workload;
        if (cfg.num_slow_wl_loops > 0) {
            slow_workload = std::make_unique<distributed::MeshWorkload>();
            slow_workload->add_program(target, build_slow_cos_program(cfg));
        }
        runs.push_back(DeviceRun{
            mesh_device, device, cfg, std::move(eth_dram_buffer), std::move(main_workload), std::move(slow_workload)});
        log_info(LogTest, "Main program ready: device={}", device->id());
    }

    if (num_slow_wl_loops > 0) {
        log_info(LogTest, "Duty-cycle interleave active: slow cos between each max-util dispatch");
    }

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

    // Report FPU utilization and per-ETH-core DRAM bandwidth for every device.
    bool valid_compute_output = true;
    bool valid_fpu_timing = true;
    bool valid_eth_timing = true;
    for (const auto& run : runs) {
        valid_compute_output &= validate_compute_output(run.device, run.cfg);
        valid_fpu_timing &= log_fpu_utilization(run.device, run.cfg);
        valid_eth_timing &= log_eth_bw(run.device, run.cfg, run.eth_dram_buffer);
    }

    return valid_compute_output && valid_fpu_timing && valid_eth_timing;
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
