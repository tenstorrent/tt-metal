// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Max-utilization workload test.
// Architecture:
//   1. Pre-fill phase: BRISC runs on each core to read 2 DRAM buffers into L1
//      - Buffer 0: 8 tiles of bfloat16 from DRAM
//      - Buffer 1: 8 tiles of bfloat16 from DRAM
//   2. Main phase: Three decoupled kernels run simultaneously:
//      - BRISC (NOC0): Sends 8KB to right/down neighbors only
//      - NCRISC (NOC1): Sends 8KB to left/up neighbors only
//      - TRISC: Uses pre-filled L1 buffers directly, runs compute at full speed
//   No CB dependencies between data movement and compute.

#include <cstdio>
#include <memory>

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

    // Data transfer size in bytes.
    uint32_t data_transfer_size = 2048;  // 2KB

    // L1 buffer addresses (filled by pre-fill phase, passed to main phase).
    uint32_t l1_buffer0_addr = 0;     // input 0 bfloat16 data
    uint32_t l1_buffer1_addr = 0;     // input 1 bfloat16 data
    uint32_t l1_buffer2_addr = 0;     // output bfloat16 data
    uint32_t l1_buffer3_addr = 0;     // NOC0 send data pattern A
    uint32_t l1_buffer4_addr = 0;     // NOC0 send data pattern B
    uint32_t l1_buffer5_addr = 0;     // NOC1 send data pattern A
    uint32_t l1_buffer6_addr = 0;     // NOC1 send data pattern B
    uint32_t l1_buffer7_addr = 0;     // rx buffer from left neighbor
    uint32_t l1_buffer8_addr = 0;     // rx buffer from up neighbor
    uint32_t l1_buffer9_addr = 0;     // rx buffer from right neighbor
    uint32_t l1_buffer10_addr = 0;    // rx buffer from down neighbor
    uint32_t l1_super_sync_addr = 0;  // super sync semaphore

    // FPU utilization target percentage [1, 92].
    // Passed to the compute kernel as compile-time arg 5.
    uint32_t fpu_utilization_pct = 92;

    // ETH DRAM streaming fields (filled by setup_eth_stream_config before build_program).
    uint32_t eth_dram_buffer_addr = 0;   // DRAM src base address for ETH streaming
    uint32_t eth_pages_per_bank = 0;     // pages per bank read per iteration
    uint32_t eth_l1_staging_addr = 0;    // ETH L1 unreserved base (first 16 bytes = timing scratch)
    uint32_t eth_l1_staging_stride = 0;  // Per-processor slice of the shared ETH L1 staging region
    // DRAM read transaction size.  Larger values saturate bandwidth better.
    uint32_t eth_page_size = 1024;  // 1KB found to be optimal for BH
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

    // Use one idle Ethernet core per DRAM bank. Idle Ethernet kernels require
    // TT_METAL_SLOW_DISPATCH_MODE=1; the default fast-dispatch path uses both
    // processors on each connected Ethernet core.
    bool use_idle_eth = false;

    // When true, all kernels perform a super-sync barrier at program start
    // before entering their main loop.
    bool super_sync = false;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint32_t tile_size_bytes(DataFormat fmt, uint32_t h = 32, uint32_t w = 32) {
    if (fmt == DataFormat::Float16_b) {
        return uint32_t(w * h * 2);  // two bytes per float16_b element
    } else {
        throw std::invalid_argument("Invalid data format");
    }
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
        try {
            int val = std::stoi(env);
            if (val >= 1 && val <= 92) {
                return static_cast<uint32_t>(val);
            }
        } catch (...) {
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
        try {
            int val = std::stoi(env);
            if (val >= 1 && val <= 100) {
                return static_cast<uint32_t>(val);
            }
        } catch (...) {
        }
        log_warning(
            LogTest, "MAX_UTIL_DRAM_UTILIZATION_PCT='{}' is not an integer in [1, 100] – using default of 100", env);
    }
    return 100;
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

/// Reads MAX_UTIL_USE_IDLE_ETH from the environment.
/// Any non-empty value other than "0" or "false" selects one idle ETH core per bank.
static bool get_use_idle_eth() {
    const char* env = std::getenv("MAX_UTIL_USE_IDLE_ETH");
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
    cfg.use_idle_eth = get_use_idle_eth();
    // A single idle-ETH stream benefits from Blackhole's full 16 KiB NOC burst.
    // The dual-processor active-ETH path retains its empirically optimal 1 KiB pages.
    if (cfg.use_idle_eth) {
        cfg.eth_page_size = 16 * 1024;
    }
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
// Calibrated on Blackhole p300c using 1 KiB reads from active ETH cores. The
// values are empirical because wait instructions overlap NOC issue latency.
// ---------------------------------------------------------------------------

// clang-format off
static const std::map<uint32_t, uint32_t> kDramUtilToCyclesMap = {
    {10,  360},
    {20,  160},
    {30,   93},
    {40,   60},
    {50,   40},
    {60,   27},
    {70,   17},
    {80,   10},
    {90,    4},
    {100,   0},
};
// clang-format on

/// Returns the key in kDramUtilToCyclesMap nearest to @p pct.
static uint32_t nearest_dram_pct(uint32_t pct) {
    TT_ASSERT(!kDramUtilToCyclesMap.empty(), "kDramUtilToCyclesMap must not be empty");
    auto it = kDramUtilToCyclesMap.lower_bound(pct);
    if (it == kDramUtilToCyclesMap.end()) {
        return std::prev(it)->first;
    }
    if (it == kDramUtilToCyclesMap.begin() || it->first == pct) {
        return it->first;
    }
    auto prev = std::prev(it);
    return (pct - prev->first <= it->first - pct) ? prev->first : it->first;
}

/// Converts a requested DRAM utilization percentage to a noc_wait_cycles value
/// by snapping to the nearest entry in kDramUtilToCyclesMap.
static uint32_t dram_pct_to_noc_wait_cycles(uint32_t pct) { return kDramUtilToCyclesMap.at(nearest_dram_pct(pct)); }

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
// assign_eth_streams_to_banks – assigns every DRAM bank to a unique Ethernet
// core/processor stream. The default fast-dispatch path uses two processors on
// each of four active cores. The slow-dispatch idle path uses one physical core
// per bank.
// ---------------------------------------------------------------------------

struct EthStreamAssignment {
    CoreCoord core;
    uint32_t processor;
    uint32_t bank_id;
};

static std::vector<EthStreamAssignment> assign_eth_streams_to_banks(IDevice* device, bool use_idle_eth) {
    auto active_eth = device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true);
    auto inactive_eth = device->get_inactive_ethernet_cores();
    std::vector<CoreCoord> eth_cores = use_idle_eth ? std::vector<CoreCoord>(inactive_eth.begin(), inactive_eth.end())
                                                    : std::vector<CoreCoord>(active_eth.begin(), active_eth.end());
    auto cmp = [](const CoreCoord& a, const CoreCoord& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); };
    std::sort(eth_cores.begin(), eth_cores.end(), cmp);

    const uint32_t num_processors =
        use_idle_eth ? 1 : MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);

    std::vector<EthStreamAssignment> available_streams;
    for (uint32_t processor = 0; processor < num_processors; ++processor) {
        for (const auto& core : eth_cores) {
            available_streams.push_back({core, processor, 0});
        }
    }

    std::vector<EthStreamAssignment> assignments;
    const uint32_t num_banks = static_cast<uint32_t>(device->num_dram_channels());
    for (uint32_t bank_id = 0; bank_id < num_banks && !available_streams.empty(); ++bank_id) {
        const CoreCoord dram_coord = dram_noc0_coord(device, bank_id);
        auto closest =
            std::min_element(available_streams.begin(), available_streams.end(), [&](const auto& lhs, const auto& rhs) {
                const CoreCoord lhs_coord = eth_noc0_coord(device, lhs.core);
                const CoreCoord rhs_coord = eth_noc0_coord(device, rhs.core);
                const uint32_t lhs_distance = std::abs(static_cast<int>(lhs_coord.x) - static_cast<int>(dram_coord.x)) +
                                              std::abs(static_cast<int>(lhs_coord.y) - static_cast<int>(dram_coord.y));
                const uint32_t rhs_distance = std::abs(static_cast<int>(rhs_coord.x) - static_cast<int>(dram_coord.x)) +
                                              std::abs(static_cast<int>(rhs_coord.y) - static_cast<int>(dram_coord.y));
                return lhs_distance < rhs_distance;
            });
        closest->bank_id = bank_id;
        assignments.push_back(*closest);
        available_streams.erase(closest);
    }
    return assignments;
}

// ---------------------------------------------------------------------------
// setup_eth_stream_config – configures DRAM buffer and ETH L1 addresses for
//   the ETH DRAM streaming kernel.
//
// Each selected ETH processor reads from exactly one DRAM bank so that summing
// per-stream bandwidths gives the aggregate ETH-to-DRAM bandwidth.
//
// Returns a shared_ptr<Buffer> holding the DRAM staging buffer; the caller
// must keep this alive until the program finishes.  Returns nullptr when no
// Ethernet cores are assignable.
// ---------------------------------------------------------------------------

static shared_ptr<Buffer> setup_eth_stream_config(IDevice* device, MaxUtilConfig& cfg) {
    auto active_eth = device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true);
    auto inactive_eth = device->get_inactive_ethernet_cores();

    log_info(
        LogTest,
        "Device {}: ETH cores available: {} active (connected), {} inactive (idle); using {}",
        device->id(),
        active_eth.size(),
        inactive_eth.size(),
        cfg.use_idle_eth ? "idle (one physical core per bank)" : "active (two processors per core)");

    auto assignments = assign_eth_streams_to_banks(device, cfg.use_idle_eth);
    if (assignments.empty()) {
        log_warning(
            LogTest, "Device {}: no selected ETH cores available – skipping ETH DRAM streaming kernel", device->id());
        return nullptr;
    }

    auto& hal = MetalContext::instance().hal();
    const auto eth_core_type =
        cfg.use_idle_eth ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH;
    cfg.eth_l1_staging_addr = hal.get_dev_addr(eth_core_type, HalL1MemAddrType::UNRESERVED);
    uint32_t eth_l1_size = hal.get_dev_size(eth_core_type, HalL1MemAddrType::UNRESERVED);

    uint32_t num_banks = static_cast<uint32_t>(device->num_dram_channels());
    uint32_t page_size_bytes = cfg.eth_page_size;

    const uint32_t num_processors =
        cfg.use_idle_eth ? 1 : hal.get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
    cfg.eth_l1_staging_stride = eth_l1_size / num_processors;
    cfg.eth_pages_per_bank = (cfg.eth_l1_staging_stride - 16) / page_size_bytes;

    // ETH kernel runs 8x fewer loops than the compute kernel to match duration.
    // Also scale by utilization percentage to match the compute kernel duration.
    cfg.eth_num_wl_loops = static_cast<uint32_t>(
        std::max<uint64_t>(1, static_cast<uint64_t>(cfg.num_wl_loops) * cfg.eth_dram_util_pct / 800));

    cfg.eth_noc_wait_cycles = dram_pct_to_noc_wait_cycles(cfg.eth_dram_util_pct);
    const uint32_t matched_dram_pct = nearest_dram_pct(cfg.eth_dram_util_pct);
    log_info(
        LogTest,
        "DRAM utilization: requested={}%  matched={}%  noc_wait_cycles={}",
        cfg.eth_dram_util_pct,
        matched_dram_pct,
        cfg.eth_noc_wait_cycles);

    log_info(
        LogTest,
        "Device {}: ETH DRAM streaming – {} bank streams across {} {} cores, pages_per_bank={}, "
        "page_size={} B ({}KB), eth_l1_staging_addr=0x{:x}, eth_l1_size={} B",
        device->id(),
        assignments.size(),
        cfg.use_idle_eth ? inactive_eth.size() : active_eth.size(),
        cfg.use_idle_eth ? "idle" : "active",
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
// build_prefill_program – constructs a pre-fill Program
//
// Creates DRAM buffers, fills them with random data, and builds a program
// that reads from DRAM into L1 on all cores.
//   - Buffer 0: 8 tiles of bfloat16 from DRAM
//   - Buffer 1: 8 tiles of bfloat16 from DRAM
// ---------------------------------------------------------------------------

static Program build_prefill_program(IDevice* device, MaxUtilConfig& cfg) {
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

    // Buffers 3,4,5,6: uint32 pattern data
    const uint32_t buffer_size_uint32 = cfg.data_transfer_size;

    // Buffers 7,8,9,10: rx buffers (no init)
    const uint32_t rx_buffer_size = buffer_size_uint32;

    auto dram_buffer0 = CreateBuffer(dram_cfg_bfloat16);
    auto dram_buffer1 = CreateBuffer(dram_cfg_bfloat16);

    // DRAM buffers for uint32 patterns: one for 0xAAAA, one for 0x5555
    auto dram_cfg_uint32 = InterleavedBufferConfig{
        .device = device,
        .size = buffer_size_uint32,
        .page_size = buffer_size_uint32,
        .buffer_type = BufferType::DRAM,
    };
    auto dram_buffer_0xAAAA = CreateBuffer(dram_cfg_uint32);
    auto dram_buffer_0x5555 = CreateBuffer(dram_cfg_uint32);

    uint32_t dram_buffer0_addr = dram_buffer0->address();
    uint32_t dram_buffer1_addr = dram_buffer1->address();
    uint32_t dram_buffer_0xAAAA_addr = dram_buffer_0xAAAA->address();
    uint32_t dram_buffer_0x5555_addr = dram_buffer_0x5555->address();

    // Fill buffer 0 with random data
    std::vector<uint32_t> data0 =
        rng_bfp16(buffer_size_bfloat16, tile_rows, tile_cols, /*mean=*/0.0f, /*stdev=*/1.0f, /*seed=*/42);
    detail::WriteToBuffer(dram_buffer0, data0);

    // Fill buffer 1 with random data
    std::vector<uint32_t> data1 =
        rng_bfp16(buffer_size_bfloat16, tile_rows, tile_cols, /*mean=*/0.0f, /*stdev=*/1.0f, /*seed=*/43);
    detail::WriteToBuffer(dram_buffer1, data1);

    // Fill pattern buffers: 0xAAAAAAAA and 0x55555555 (32-bit patterns)
    std::vector<uint32_t> data_0xAAAA(buffer_size_uint32 / sizeof(uint32_t), 0xAAAAAAAAu);
    std::vector<uint32_t> data_0x5555(buffer_size_uint32 / sizeof(uint32_t), 0x55555555u);
    detail::WriteToBuffer(dram_buffer_0xAAAA, data_0xAAAA);
    detail::WriteToBuffer(dram_buffer_0x5555, data_0x5555);

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
    cfg.l1_buffer3_addr = addr;  // pattern 0xAAAA
    addr += buffer_size_uint32;
    cfg.l1_buffer4_addr = addr;  // pattern 0x5555
    addr += buffer_size_uint32;
    cfg.l1_buffer5_addr = addr;  // pattern 0xAAAA
    addr += buffer_size_uint32;
    cfg.l1_buffer6_addr = addr;  // pattern 0x5555
    addr += buffer_size_uint32;
    cfg.l1_buffer7_addr = addr;  // rx from left, no init
    addr += rx_buffer_size;
    cfg.l1_buffer8_addr = addr;  // rx from up, no init
    addr += rx_buffer_size;
    cfg.l1_buffer9_addr = addr;  // rx from right, no init
    addr += rx_buffer_size;
    cfg.l1_buffer10_addr = addr;  // rx from down, no init
    addr += rx_buffer_size;
    cfg.l1_super_sync_addr = addr;  // super sync semaphore

    // Pre-fill: L1 buf0..10=0x1b200, 0x1f200, 0x21400, 0x25400, 0x27400, 0x29400, 0x2b400, 0x2d400, 0x2f400, 0x31400,
    // 0x33400
    log_info(
        LogTest,
        "Pre-fill: L1 buf0..10=0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}",
        cfg.l1_buffer0_addr,
        cfg.l1_buffer1_addr,
        cfg.l1_buffer2_addr,
        cfg.l1_buffer3_addr,
        cfg.l1_buffer4_addr,
        cfg.l1_buffer5_addr,
        cfg.l1_buffer6_addr,
        cfg.l1_buffer7_addr,
        cfg.l1_buffer8_addr,
        cfg.l1_buffer9_addr,
        cfg.l1_buffer10_addr);

    Program program = CreateProgram();

    // Pre-fill kernel on BRISC - reads from DRAM to L1
    CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/prefill_l1.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {
                    dram_buffer0_addr,        // 0: dram_buffer0_addr
                    dram_buffer1_addr,        // 1: dram_buffer1_addr
                    dram_buffer_0xAAAA_addr,  // 2: dram_buffer_0xAAAA_addr
                    dram_buffer_0x5555_addr,  // 3: dram_buffer_0x5555_addr
                    cfg.l1_buffer0_addr,      // 4: l1_buffer0_addr
                    cfg.l1_buffer1_addr,      // 5: l1_buffer1_addr
                    cfg.l1_buffer3_addr,      // 6: l1_buffer3_addr
                    cfg.l1_buffer4_addr,      // 7: l1_buffer4_addr
                    cfg.l1_buffer5_addr,      // 8: l1_buffer5_addr
                    cfg.l1_buffer6_addr,      // 9: l1_buffer6_addr
                    tile_bytes_bfloat16,      // 10: tile_size_bytes (bfloat16, 2048)
                    buffer_size_uint32,       // 11: data_transfer_size (8KB)
                    cfg.num_tiles,            // 12: num_tiles (8)
                    cfg.l1_super_sync_addr,   // 13: l1_super_sync_addr
                },
        });

    return program;
}

// ---------------------------------------------------------------------------
// FPU utilization throttle map
//
// Maps FPU utilization percentage → cycles_to_wait between compute operations.
// The kernel uses cycles_to_wait to insert idle cycles and throttle the FPU.
// Fill in the values for your target architecture; keys cover the valid [1, 92]
// range at representative intervals.
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

/// Returns the key in kFpuUtilToCyclesMap nearest to @p pct.
static uint32_t nearest_fpu_pct(uint32_t pct) {
    TT_ASSERT(!kFpuUtilToCyclesMap.empty(), "kFpuUtilToCyclesMap must not be empty");
    auto it = kFpuUtilToCyclesMap.lower_bound(pct);
    if (it == kFpuUtilToCyclesMap.end()) {
        return std::prev(it)->first;
    }
    if (it == kFpuUtilToCyclesMap.begin() || it->first == pct) {
        return it->first;
    }
    auto prev = std::prev(it);
    return (pct - prev->first <= it->first - pct) ? prev->first : it->first;
}

/// Converts a requested FPU utilization percentage to a cycles_to_wait value
/// by snapping to the nearest entry in kFpuUtilToCyclesMap.
static uint32_t fpu_pct_to_cycles_to_wait(uint32_t pct) { return kFpuUtilToCyclesMap.at(nearest_fpu_pct(pct)); }

// ---------------------------------------------------------------------------
// Duty-cycle control map
//
// Maps hot-workload duty cycle percentage (in 10% increments) to the number
// of slow cos() loops to interleave between each max-util dispatch.
//
// Values are CALIBRATED FOR num_wl_loops = 1000.  At runtime the raw map
// value is scaled linearly by (num_wl_loops / 1000) so that the duty cycle
// stays correct regardless of the hot-workload loop count:
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
    const char* env = std::getenv("MAX_UTIL_DUTY_CYCLE");
    if (env != nullptr) {
        try {
            int val = std::stoi(env);
            if (kDutyCycleToSlowLoopsMap.count(static_cast<uint32_t>(val))) {
                return static_cast<uint32_t>(val);
            }
        } catch (...) {
        }
        log_warning(
            LogTest,
            "MAX_UTIL_DUTY_CYCLE_PCT='{}' is not one of {{10,20,30,40,50,60,70,80,90,100}} – using default of 100",
            env);
    }
    return 100;
}

/// Returns the num_slow_wl_loops for a given duty cycle, scaled to match the
/// actual hot-workload loop count.  Map values are calibrated at
/// kDutyCycleCalibrationLoops; scaling is linear so that the ratio of
/// hot-time to slow-time remains constant across different loop counts.
static uint32_t duty_cycle_to_slow_loops(uint32_t duty_cycle_pct, uint32_t num_wl_loops) {
    auto it = kDutyCycleToSlowLoopsMap.find(duty_cycle_pct);
    if (it == kDutyCycleToSlowLoopsMap.end() || it->second == 0) {
        return 0;
    }
    // 64-bit multiply to avoid overflow before dividing back down.
    uint64_t scaled = static_cast<uint64_t>(it->second) * num_wl_loops / kDutyCycleCalibrationLoops;
    return static_cast<uint32_t>(scaled);
}

// ---------------------------------------------------------------------------
// build_program – constructs the main Program
//
// Decoupled kernels: BRISC and NCRISC generate NOC traffic only,
// TRISC uses pre-filled L1 buffers directly without CB waits.
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

    auto super_sync_sender_semaphore_id = 0;
    auto super_sync_receiver_semaphore_id = 0;
    if (cfg.super_sync) {
        super_sync_sender_semaphore_id = tt_metal::CreateSemaphore(program, core_range_set, INVALID);
        super_sync_receiver_semaphore_id = tt_metal::CreateSemaphore(program, core_range_set, INVALID);
    }
    log_info(LogTest, "Super sync: {}", cfg.super_sync);

    // -- Reader kernel (BRISC / RISCV_0 / NOC0) -----------------------------
    // Top-left core multicasts to entire grid; all other cores are no-ops.
    auto reader_kernel = CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/max_util_reader.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {
                    cfg.num_wl_loops,                  // 0: num_loops (inner loops per dispatch)
                    cfg.l1_buffer3_addr,               // 1: l1_tx_A_addr – pattern A (0x5555)
                    cfg.l1_buffer4_addr,               // 2: l1_tx_B_addr – pattern B (0xAAAA)
                    cfg.l1_buffer7_addr,               // 3: l1_rx_addr – destination on receiving cores
                    cfg.l1_buffer8_addr,               // 4: (unused)
                    cfg.l1_buffer9_addr,               // 5: (unused)
                    cfg.l1_buffer10_addr,              // 6: (unused)
                    cfg.data_transfer_size,            // 7: transfer_size
                    cfg.super_sync,                    // 8: super_sync
                    super_sync_sender_semaphore_id,    // 9: super_sync_sender_semaphore_id
                    super_sync_receiver_semaphore_id,  // 10: super_sync_receiver_semaphore_id
                    cfg.l1_super_sync_addr             // 11: l1_super_sync_addr

                },
        });

    // -- Writer kernel (NCRISC / RISCV_1 / NOC1) ----------------------------
    // Bottom-right core multicasts to entire grid; all other cores are no-ops.
    auto writer_kernel = CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/max_util_writer.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args =
                {
                    cfg.num_wl_loops,        // 0: num_loops (inner loops per dispatch)
                    cfg.l1_buffer5_addr,     // 1: l1_tx_A_addr – pattern A (0x5555)
                    cfg.l1_buffer6_addr,     // 2: l1_tx_B_addr – pattern B (0xAAAA)
                    cfg.l1_buffer7_addr,     // 3: (unused)
                    cfg.l1_buffer8_addr,     // 4: (unused)
                    cfg.l1_buffer9_addr,     // 5: l1_rx_addr – destination on receiving cores
                    cfg.l1_buffer10_addr,    // 6: (unused)
                    cfg.data_transfer_size,  // 7: transfer_size
                },
        });

    // -- Compute kernel (TRISC) ---------------------------------------------
    // Uses pre-filled L1 buffers directly, no CB waits
    const uint32_t matched_pct = nearest_fpu_pct(cfg.fpu_utilization_pct);
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
                    cfg.l1_buffer0_addr,    // 0: l1_buffer0_addr (bfloat16)
                    cfg.l1_buffer1_addr,    // 1: l1_buffer1_addr (bfloat16)
                    cfg.l1_buffer2_addr,    // 2: l1_buffer2_addr (output, 8 float16_b tiles)
                    cfg.num_tiles,          // 3: num_tiles (8)
                    cfg.num_wl_loops,       // 4: num_iterations (inner loops per dispatch)
                    cycles_to_wait,         // 5: cycles_to_wait (derived from fpu_utilization_pct)
                    cfg.super_sync,         // 6: super_sync
                    cfg.l1_super_sync_addr  // 7: l1_super_sync_addr
                },
        });

    // -- Set runtime arguments for NOC traffic kernels ----------------------
    // Reader  (NOC0): only the top-left  core multicasts to the whole grid.
    // Writer  (NOC1): only the bottom-right core multicasts to the whole grid.
    // All other cores receive is_sender=0 and exit immediately.

    // Physical NOC0 coordinates of the grid corners (used by both kernels).
    CoreCoord phys_tl = device->worker_core_from_logical_core(cfg.grid_start);
    CoreCoord phys_br = device->worker_core_from_logical_core(cfg.grid_end);

    // Number of destination cores for multicast (all cores minus the sender itself).
    uint32_t grid_size = (cfg.grid_end.x - cfg.grid_start.x + 1) * (cfg.grid_end.y - cfg.grid_start.y + 1);
    uint32_t num_dests = grid_size - 1;

    for (uint32_t y = cfg.grid_start.y; y <= cfg.grid_end.y; ++y) {
        for (uint32_t x = cfg.grid_start.x; x <= cfg.grid_end.x; ++x) {
            CoreCoord core = {x, y};
            CoreCoord worker_core = device->worker_core_from_logical_core(core);

            // Runtime arg vectors for reader and writer kernels
            std::vector<uint32_t> sender_args = {
                1u,
                static_cast<uint32_t>(phys_tl.x),
                static_cast<uint32_t>(phys_tl.y),
                static_cast<uint32_t>(phys_br.x),
                static_cast<uint32_t>(phys_br.y),
                num_dests,
                static_cast<uint32_t>(worker_core.x),
                static_cast<uint32_t>(worker_core.y)};
            std::vector<uint32_t> idle_args = {
                0u,
                static_cast<uint32_t>(phys_tl.x),
                static_cast<uint32_t>(phys_tl.y),
                static_cast<uint32_t>(phys_br.x),
                static_cast<uint32_t>(phys_br.y),
                num_dests,
                static_cast<uint32_t>(worker_core.x),
                static_cast<uint32_t>(worker_core.y)};

            bool is_reader_sender = (x == cfg.grid_start.x && y == cfg.grid_start.y);
            bool is_writer_sender = (x == cfg.grid_end.x && y == cfg.grid_end.y);

            SetRuntimeArgs(program, reader_kernel, core, is_reader_sender ? sender_args : idle_args);
            SetRuntimeArgs(program, writer_kernel, core, is_writer_sender ? sender_args : idle_args);

            // Compute kernel doesn't need runtime args - uses compile-time L1 addresses
        }
    }

    // -- ETH DRAM streaming kernels (one bank per selected ETH processor) --
    // Guard: skip when setup_eth_stream_config was not called.
    if (cfg.eth_dram_buffer_addr != 0) {
        auto assignments = assign_eth_streams_to_banks(device, cfg.use_idle_eth);
        if (!assignments.empty()) {
            const uint32_t num_processors =
                cfg.use_idle_eth
                    ? 1
                    : MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::ACTIVE_ETH);
            for (uint32_t processor = 0; processor < num_processors; ++processor) {
                std::set<CoreRange> eth_ranges;
                for (const auto& assignment : assignments) {
                    if (assignment.processor == processor) {
                        eth_ranges.insert(CoreRange(assignment.core, assignment.core));
                    }
                }
                if (eth_ranges.empty()) {
                    continue;
                }

                EthernetConfig eth_cfg{
                    .eth_mode = cfg.use_idle_eth ? Eth::IDLE : Eth::RECEIVER,
                    .noc = static_cast<NOC>(processor),
                    .processor = static_cast<DataMovementProcessor>(processor),
                    .compile_args =
                        {
                            cfg.eth_num_wl_loops,
                            cfg.eth_pages_per_bank,
                            cfg.eth_page_size,
                            cfg.eth_noc_wait_cycles,  // 3: noc_wait_cycles (0 = max DRAM util)
                        },
                    .defines = cfg.use_idle_eth ? std::map<std::string, std::string>{{"USE_IDLE_ETH", "1"}}
                                                : std::map<std::string, std::string>{},
                };
                if (!cfg.use_idle_eth) {
                    eth_test_common::set_arch_specific_eth_config(eth_cfg);
                }
                auto eth_kernel = CreateKernel(
                    program,
                    "tests/didt/max_util_workload/kernels/eth_dram_reader.cpp",
                    CoreRangeSet(eth_ranges),
                    eth_cfg);

                for (const auto& assignment : assignments) {
                    if (assignment.processor == processor) {
                        SetRuntimeArgs(
                            program,
                            eth_kernel,
                            assignment.core,
                            {
                                cfg.eth_dram_buffer_addr,
                                cfg.eth_l1_staging_addr + processor * cfg.eth_l1_staging_stride,
                                assignment.bank_id,
                            });
                    }
                }
            }

            log_info(
                LogTest,
                "ETH DRAM streaming: {} bank streams, {} enqueues × {} inner iterations "
                "(compute loops={}, ratio=1/8), {} pages/bank × {} B/page ({}KB)",
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
// All three RISCVs run the same noc-address layout as build_program but the
// compute kernel runs SFPU cosine instead of MVMUL, making it significantly
// lighter on the FPU.  Reader/writer kernels are the same noop variants as in
// build_program.  The slow workload loops num_slow_wl_loops times per dispatch.
// ---------------------------------------------------------------------------

static Program build_slow_cos_program(IDevice* device, const MaxUtilConfig& cfg) {
    const CoreRange core_range(cfg.grid_start, cfg.grid_end);
    const CoreRangeSet core_range_set({core_range});

    log_info(
        LogTest,
        "Slow-cos program: {} cores, {} enqueues × {} slow inner iterations",
        core_range.size(),
        cfg.num_iterations,
        cfg.num_slow_wl_loops);

    Program program = CreateProgram();

    // -- Reader kernel (BRISC / NOC0): reuse the same noop reader --
    auto reader_kernel = CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/max_util_reader.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {
                    cfg.num_slow_wl_loops,   // 0: num_loops
                    cfg.l1_buffer3_addr,     // 1: l1_tx_A_addr
                    cfg.l1_buffer4_addr,     // 2: l1_tx_B_addr
                    cfg.l1_buffer7_addr,     // 3: l1_rx_addr
                    cfg.l1_buffer8_addr,     // 4: (unused)
                    cfg.l1_buffer9_addr,     // 5: (unused)
                    cfg.l1_buffer10_addr,    // 6: (unused)
                    cfg.data_transfer_size,  // 7: transfer_size
                    0,                       // 8: super_sync
                    0,                       // 9: super_sync_sender_semaphore_id
                    0,                       // 10: super_sync_receiver_semaphore_id
                    0,                       // 11: l1_super_sync_addr
                },
        });

    // -- Writer kernel (NCRISC / NOC1): reuse the same noop writer --
    auto writer_kernel = CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/max_util_writer.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args =
                {
                    cfg.num_slow_wl_loops,   // 0: num_loops
                    cfg.l1_buffer5_addr,     // 1: l1_tx_A_addr
                    cfg.l1_buffer6_addr,     // 2: l1_tx_B_addr
                    cfg.l1_buffer7_addr,     // 3: (unused)
                    cfg.l1_buffer8_addr,     // 4: (unused)
                    cfg.l1_buffer9_addr,     // 5: l1_rx_addr
                    cfg.l1_buffer10_addr,    // 6: (unused)
                    cfg.data_transfer_size,  // 7: transfer_size
                },
        });

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
                    cfg.l1_buffer1_addr,    // 1: l1_buffer1_addr (bfloat16, kept for unpack HW compat)
                    cfg.l1_buffer2_addr,    // 2: l1_buffer2_addr (output)
                    cfg.num_tiles,          // 3: num_tiles (8)
                    cfg.num_slow_wl_loops,  // 4: num_loops
                },
        });

    // Set runtime args for reader/writer (same noop pattern as build_program).
    CoreCoord phys_tl = device->worker_core_from_logical_core(cfg.grid_start);
    CoreCoord phys_br = device->worker_core_from_logical_core(cfg.grid_end);
    uint32_t grid_size = (cfg.grid_end.x - cfg.grid_start.x + 1) * (cfg.grid_end.y - cfg.grid_start.y + 1);
    uint32_t num_dests = grid_size - 1;

    std::vector<uint32_t> sender_args = {
        1u,
        static_cast<uint32_t>(phys_tl.x),
        static_cast<uint32_t>(phys_tl.y),
        static_cast<uint32_t>(phys_br.x),
        static_cast<uint32_t>(phys_br.y),
        num_dests};
    std::vector<uint32_t> idle_args = {0u, 0u, 0u, 0u, 0u, 0u};

    for (uint32_t y = cfg.grid_start.y; y <= cfg.grid_end.y; ++y) {
        for (uint32_t x = cfg.grid_start.x; x <= cfg.grid_end.x; ++x) {
            CoreCoord core = {x, y};
            bool is_reader_sender = (x == cfg.grid_start.x && y == cfg.grid_start.y);
            bool is_writer_sender = (x == cfg.grid_end.x && y == cfg.grid_end.y);
            SetRuntimeArgs(program, reader_kernel, core, is_reader_sender ? sender_args : idle_args);
            SetRuntimeArgs(program, writer_kernel, core, is_writer_sender ? sender_args : idle_args);
        }
    }

    return program;
}

// ---------------------------------------------------------------------------
// log_eth_bw – reads per-core timing persisted in DRAM and logs bandwidth
// ---------------------------------------------------------------------------

static bool log_eth_bw(IDevice* device, const MaxUtilConfig& cfg, const shared_ptr<Buffer>& dram_buffer) {
    if (cfg.eth_dram_buffer_addr == 0 || dram_buffer == nullptr) {
        return true;
    }

    auto assignments = assign_eth_streams_to_banks(device, cfg.use_idle_eth);
    if (assignments.empty()) {
        return true;
    }

    std::vector<uint32_t> readback;
    detail::ReadFromBuffer(dram_buffer, readback);

    // Each stream reads cfg.eth_pages_per_bank pages from 1 bank per iteration.
    uint64_t bytes_per_stream =
        static_cast<uint64_t>(cfg.eth_num_wl_loops) * cfg.eth_pages_per_bank * cfg.eth_page_size;

    // The active-ETH wall clock runs at 1 GHz (ETH_CLOCK_CYCLE_1MS = 1,000,000),
    // independently of the 1.35 GHz Tensix AI clock.
    constexpr double kEthWallClockGHz = 1.0;

    double total_bw_bpc = 0.0;  // bytes/cycle
    uint32_t reported = 0;

    for (const auto& assignment : assignments) {
        const auto& eth_core = assignment.core;
        const uint32_t processor = assignment.processor;
        const uint32_t bank_id = assignment.bank_id;
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
        double bw_gbps = bw_bpc * kEthWallClockGHz;
        total_bw_bpc += bw_bpc;
        ++reported;

        auto eth_noc0 = eth_noc0_coord(device, eth_core);
        auto dram_noc0 = dram_noc0_coord(device, bank_id);
        log_info(
            LogTest,
            "Device {} ETH ({},{}) DM{} [NOC0 ({},{})] → bank {} [NOC0 ({},{})] "
            "BW: {:.3f} bytes/cycle  {:.2f} GB/s  ({} bytes, {} cycles)",
            device->id(),
            eth_core.x,
            eth_core.y,
            processor,
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
            total_bw_bpc * kEthWallClockGHz,
            kEthWallClockGHz);
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
    prefill_workload.add_program(target, std::move(prefill_program));
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
        slow_cos_workload.add_program(target, build_slow_cos_program(device, cfg));
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

    return valid_eth_timing;
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
        auto prefill_workload = distributed::MeshWorkload();
        prefill_workload.add_program(target, build_prefill_program(device, cfg));
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
            slow_workload->add_program(target, build_slow_cos_program(device, cfg));
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

    // Report per-ETH-core DRAM bandwidth for every device.
    bool valid_eth_timing = true;
    for (const auto& run : runs) {
        valid_eth_timing &= log_eth_bw(run.device, run.cfg, run.eth_dram_buffer);
    }

    return valid_eth_timing;
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------

/// Returns the number of iterations to use for stress / all-devices tests.
/// Reads MAX_UTIL_NUM_ITERATIONS from the environment; defaults to 1.
static uint32_t get_num_iterations() {
    const char* env = std::getenv("MAX_UTIL_NUM_ITERATIONS");
    if (env != nullptr) {
        try {
            int val = std::stoi(env);
            if (val > 0) {
                return static_cast<uint32_t>(val);
            }
        } catch (...) {
        }
        log_warning(LogTest, "MAX_UTIL_NUM_ITERATIONS='{}' is not a positive integer – using default of 1", env);
    }
    return 1;
}

/// Reads MAX_UTIL_NUM_WL_LOOPS from the environment; defaults to 1000.
static uint32_t get_num_wl_loops() {
    const char* env = std::getenv("MAX_UTIL_NUM_WL_LOOPS");
    if (env != nullptr) {
        try {
            int val = std::stoi(env);
            if (val > 0) {
                return static_cast<uint32_t>(val);
            }
        } catch (...) {
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
    unit_tests::didt::max_util_workload::max_util_smoke(devices_.at(0));
}

TEST_F(MeshDispatchFixture, MaxUtilWorkload_Stress) {
    unit_tests::didt::max_util_workload::max_util_stress(devices_.at(0));
}

TEST_F(MeshDispatchFixture, MaxUtilWorkload_AllDevices) {
    unit_tests::didt::max_util_workload::max_util_all_devices(devices_);
}

}  // namespace tt::tt_metal
