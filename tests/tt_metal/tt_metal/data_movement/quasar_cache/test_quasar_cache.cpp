// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for Quasar DM core cache management functions.
// These tests verify L1 D$ and L2 cache flush/invalidate operations.

#include <tt-logger/tt-logger.hpp>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt::test_utils;

namespace unit_tests::dm::quasar_cache {

// Skip test if simulator is not available (arch check is handled by fixture)
bool should_skip_test() {
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    return env_var == nullptr;
}

// =============================================================================
// L2 Cache Flush Tests
// =============================================================================

struct L2FlushTestConfig {
    uint32_t base_addr;
    uint32_t num_words;
    uint32_t value;
    uint32_t test_mode;  // 0=flush_line, 1=flush_range, 2=flush_full, 3=invalidate_line
    bool expect_new_values = true;  // true for flush tests, false for invalidate tests
};

bool run_l2_flush_test(distributed::MeshDevice& mesh_device, const L2FlushTestConfig& config) {
    constexpr CoreCoord core = {0, 0};
    const experimental::NodeCoord node{0, 0};

    // For invalidate tests, pre-populate with known "old" values
    // that should persist after invalidation (since invalidate doesn't write back)
    uint32_t old_value = 0xDEADBEEF;
    std::vector<uint32_t> init_data(config.num_words, config.expect_new_values ? 0 : old_value);
    slow_dispatch::WriteToL1(mesh_device, core, config.base_addr, init_data);

    const experimental::KernelSpecName DM_KERNEL{"l2_flush"};

    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = "tests/tt_metal/tt_metal/data_movement/quasar_cache/kernels/l2_flush_test.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"base_addr", "test_mode"},
                .common_runtime_arg_names = {"value", "num_words"},
            },
        .hw_config = experimental::DataMovementHardwareConfig{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "l2_flush",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node, {{"base_addr", config.base_addr}, {"test_mode", config.test_mode}}),
        .common_runtime_arg_values = {{"value", config.value}, {"num_words", config.num_words}},
    }};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // Read back and verify
    std::vector<uint32_t> output_data;
    slow_dispatch::ReadFromL1(mesh_device, core, config.base_addr, config.num_words * sizeof(uint32_t), output_data);

    bool pass = true;
    for (uint32_t i = 0; i < config.num_words; i++) {
        uint32_t expected = config.expect_new_values ? (config.value + i) : old_value;
        if (output_data[i] != expected) {
            log_error(tt::LogTest, "Mismatch at index {}: expected 0x{:08x}, got 0x{:08x}",
                      i, expected, output_data[i]);
            pass = false;
        }
    }

    return pass;
}

// =============================================================================
// L1 D$ Tests
// =============================================================================

struct L1DCacheTestConfig {
    uint32_t base_addr;
    uint32_t num_words;
    uint32_t value;
    uint32_t test_mode;  // 0=flush_line, 1=flush_full, 2=invalidate_line, 3=invalidate_full
    bool expect_new_values;  // true for flush tests, false for invalidate tests
};

bool run_l1_dcache_test(distributed::MeshDevice& mesh_device, const L1DCacheTestConfig& config) {
    constexpr CoreCoord core = {0, 0};
    const experimental::NodeCoord node{0, 0};

    // For invalidate tests, we need to pre-populate with known "old" values
    // that should persist after invalidation (since invalidate doesn't write back)
    uint32_t old_value = 0xDEADBEEF;
    std::vector<uint32_t> init_data(config.num_words, old_value);
    slow_dispatch::WriteToL1(mesh_device, core, config.base_addr, init_data);

    const experimental::KernelSpecName DM_KERNEL{"l1_dcache"};

    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = "tests/tt_metal/tt_metal/data_movement/quasar_cache/kernels/l1_dcache_test.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"base_addr", "test_mode"},
                .common_runtime_arg_names = {"value", "num_words"},
            },
        .hw_config = experimental::DataMovementHardwareConfig{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "l1_dcache",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node, {{"base_addr", config.base_addr}, {"test_mode", config.test_mode}}),
        .common_runtime_arg_values = {{"value", config.value}, {"num_words", config.num_words}},
    }};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // Read back and verify
    std::vector<uint32_t> output_data;
    slow_dispatch::ReadFromL1(mesh_device, core, config.base_addr, config.num_words * sizeof(uint32_t), output_data);

    bool pass = true;
    for (uint32_t i = 0; i < config.num_words; i++) {
        uint32_t expected = config.expect_new_values ? (config.value + i) : old_value;
        if (output_data[i] != expected) {
            log_error(tt::LogTest, "Mismatch at index {}: expected 0x{:08x}, got 0x{:08x}",
                      i, expected, output_data[i]);
            pass = false;
        }
    }

    return pass;
}

}  // namespace unit_tests::dm::quasar_cache

// =============================================================================
// Test Suite: L2 Cache Operations
// =============================================================================

class QuasarL2CacheOps : public QuasarMeshDeviceSingleCardFixture {};

TEST_F(QuasarL2CacheOps, FlushLine) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    unit_tests::dm::quasar_cache::L2FlushTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 16,
        .value = 0x12340000,
        .test_mode = 0  // line flush
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(this->device(), config));
}

TEST_F(QuasarL2CacheOps, FlushRange) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    unit_tests::dm::quasar_cache::L2FlushTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 64,  // Multiple cache lines (4 lines worth)
        .value = 0xABCD0000,
        .test_mode = 1  // range flush
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(this->device(), config));
}

TEST_F(QuasarL2CacheOps, FlushFull) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    unit_tests::dm::quasar_cache::L2FlushTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 64,
        .value = 0x55550000,
        .test_mode = 2  // full flush
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(this->device(), config));
}

TEST_F(QuasarL2CacheOps, InvalidateLine) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    unit_tests::dm::quasar_cache::L2FlushTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 16,
        .value = 0x99990000,
        .test_mode = 3,  // invalidate line
        .expect_new_values = false  // Invalidate doesn't write back, so old values should remain
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(this->device(), config));
}

TEST_F(QuasarL2CacheOps, InvalidateFreshRead) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    // Test that invalidation causes fresh read from TL1:
    // Kernel reads (caches), writes new value via uncached path, invalidates, host verifies new value
    unit_tests::dm::quasar_cache::L2FlushTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 16,
        .value = 0xFE5A0000,
        .test_mode = 4,  // invalidate fresh read
        .expect_new_values = true  // After invalidation, new values written via uncached path should be visible
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(this->device(), config));
}

// =============================================================================
// Test Suite: L1 Data Cache Operations
// =============================================================================

class QuasarL1DCacheOps : public QuasarMeshDeviceSingleCardFixture {};

TEST_F(QuasarL1DCacheOps, FlushLine) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    unit_tests::dm::quasar_cache::L1DCacheTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 16,
        .value = 0x11110000,
        .test_mode = 0,  // flush line
        .expect_new_values = true
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(this->device(), config));
}

TEST_F(QuasarL1DCacheOps, FlushFull) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    unit_tests::dm::quasar_cache::L1DCacheTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 16,
        .value = 0x22220000,
        .test_mode = 1,  // flush full
        .expect_new_values = true
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(this->device(), config));
}

TEST_F(QuasarL1DCacheOps, InvalidateLine) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    // Invalidate should NOT write back - old values should persist
    unit_tests::dm::quasar_cache::L1DCacheTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 16,
        .value = 0x33330000,
        .test_mode = 2,  // invalidate line
        .expect_new_values = false  // should see old values
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(this->device(), config));
}

TEST_F(QuasarL1DCacheOps, InvalidateFull) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    // Invalidate should NOT write back - old values should persist
    unit_tests::dm::quasar_cache::L1DCacheTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 16,
        .value = 0x44440000,
        .test_mode = 3,  // invalidate full
        .expect_new_values = false  // should see old values
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(this->device(), config));
}

TEST_F(QuasarL1DCacheOps, InvalidateFreshRead) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    // Test that invalidation causes fresh read from TL1:
    // Kernel reads (caches), writes new value via uncached path, invalidates L1+L2, host verifies new value
    unit_tests::dm::quasar_cache::L1DCacheTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 16,
        .value = 0xFE5B0000,
        .test_mode = 4,  // invalidate fresh read
        .expect_new_values = true  // After invalidation, new values written via uncached path should be visible
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(this->device(), config));
}

// =============================================================================
// Snoop coherence and cache read latency
// =============================================================================

namespace unit_tests::dm::quasar_snoop {

constexpr CoreCoord kCore = {0, 0};
constexpr std::uint32_t kEntryBase = 100 * 1024;
constexpr std::uint32_t kFlagAddr = 110 * 1024;
constexpr std::uint32_t kResultAddr = 120 * 1024;
constexpr std::uint32_t kNumEntries = 32;

Program make_single_kernel_program(
    distributed::MeshDevice& mesh_device,
    const std::string& name,
    const std::string& source,
    const std::vector<std::pair<std::string, std::uint32_t>>& args) {
    const experimental::NodeCoord node{0, 0};
    const experimental::KernelSpecName kernel{name};
    std::vector<std::string> arg_names;
    for (const auto& [arg_name, value] : args) {
        arg_names.push_back(arg_name);
    }
    experimental::KernelSpec kernel_spec{
        .unique_id = kernel,
        .source = source,
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = arg_names},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {kernel}, .target_nodes = node};
    experimental::ProgramSpec spec{.name = name, .kernels = {kernel_spec}, .work_units = {main_wu}};
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs::KernelRunArgs::RuntimeArgValues values;
    for (const auto& [arg_name, value] : args) {
        values[arg_name][node] = value;
    }
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = kernel,
        .runtime_arg_values = values,
    }};
    experimental::SetProgramRunArgs(program, params);
    return program;
}

// Opens a host window onto kCore's L1, anchored at 0, the same way the fetch queue window is created.
std::unique_ptr<tt::umd::IoWindow> open_l1_window(
    distributed::MeshDevice& mesh_device, std::uint32_t size, tt::umd::WindowFlags flags) {
    IDevice* device = mesh_device.get_devices()[0];
    auto& cluster = MetalContext::instance().get_cluster();
    const CoreCoord virtual_core = device->worker_core_from_logical_core(kCore);
    const tt::umd::CoreCoord translated =
        cluster.get_soc_desc(device->id())
            .get_coord_at(tt_cxy_pair(device->id(), virtual_core), tt::CoordSystem::TRANSLATED);
    return cluster.get_driver()->create_io_window(
        device->id(), translated, /*addr=*/0, {.size = size}, tt::umd::IoOrdering::Strict, std::nullopt, flags);
}

std::uint32_t old_value(std::uint32_t i) { return 0x0DD00000 + i; }
std::uint32_t new_value(std::uint32_t i) { return 0x0EE00000 + i; }

// Returns false if a result that holds on any coherent memory system is wrong. Whether the cached view
// went stale, and whether an L2 invalidate alone refreshes it, are reported and checked by the caller.
bool run_coherence(
    distributed::MeshDevice& mesh_device,
    bool snoop,
    std::uint32_t stride,
    std::uint32_t& stale_entries,
    std::uint32_t& stale_after_l2_invalidate) {
    std::vector<std::uint32_t> entries(kNumEntries * stride / sizeof(std::uint32_t), 0);
    for (std::uint32_t i = 0; i < kNumEntries; i++) {
        entries[i * stride / sizeof(std::uint32_t)] = old_value(i);
    }
    slow_dispatch::WriteToL1(mesh_device, kCore, kEntryBase, entries);
    std::vector<std::uint32_t> zeros(8, 0);
    slow_dispatch::WriteToL1(mesh_device, kCore, kFlagAddr, zeros);
    std::vector<std::uint32_t> unset(kNumEntries * 4 + 4, 0xFFFFFFFF);
    slow_dispatch::WriteToL1(mesh_device, kCore, kResultAddr, unset);

    Program program = make_single_kernel_program(
        mesh_device,
        "snoop_coherence",
        "tests/tt_metal/tt_metal/data_movement/quasar_cache/kernels/snoop_coherence_test.cpp",
        {{"base_addr", kEntryBase},
         {"num_entries", kNumEntries},
         {"stride", stride},
         {"flag_addr", kFlagAddr},
         {"result_addr", kResultAddr}});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<std::uint32_t> armed;
    do {
        slow_dispatch::ReadFromL1(mesh_device, kCore, kFlagAddr, sizeof(std::uint32_t), armed);
    } while (armed[0] != 1);

    auto window = open_l1_window(
        mesh_device,
        kEntryBase + kNumEntries * stride,
        snoop ? tt::umd::WindowFlags::Snoop : tt::umd::WindowFlags::None);
    for (std::uint32_t i = 0; i < kNumEntries; i++) {
        window->write32(kEntryBase + i * stride, new_value(i));
    }
    std::vector<std::uint32_t> done = {1};
    slow_dispatch::WriteToL1(mesh_device, kCore, kFlagAddr + 16, done);
    distributed::Finish(cq);

    std::vector<std::uint32_t> results;
    slow_dispatch::ReadFromL1(mesh_device, kCore, kResultAddr, kNumEntries * 16, results);

    bool pass = true;
    stale_entries = 0;
    stale_after_l2_invalidate = 0;
    for (std::uint32_t i = 0; i < kNumEntries; i++) {
        const std::uint32_t cached = results[i * 4];
        const std::uint32_t uncached = results[i * 4 + 1];
        const std::uint32_t after_l2 = results[i * 4 + 2];
        const std::uint32_t after_both = results[i * 4 + 3];
        log_info(
            tt::LogTest,
            "snoop={} stride={} entry {:2}: cached=0x{:08x} uncached=0x{:08x} after_l2_inv=0x{:08x} "
            "after_l1d_l2_inv=0x{:08x}",
            snoop,
            stride,
            i,
            cached,
            uncached,
            after_l2,
            after_both);
        if (uncached != new_value(i) || after_both != new_value(i)) {
            log_error(tt::LogTest, "entry {}: TL1 or fully invalidated read does not hold the host value", i);
            pass = false;
        }
        if (cached != new_value(i)) {
            stale_entries++;
        }
        if (after_l2 != new_value(i)) {
            stale_after_l2_invalidate++;
        }
    }
    log_info(
        tt::LogTest,
        "snoop={} stride={}: {} of {} cached reads stale after the host write, {} still stale after an L2-only "
        "invalidate",
        snoop,
        stride,
        stale_entries,
        kNumEntries,
        stale_after_l2_invalidate);
    return pass;
}

void run_read_latency(distributed::MeshDevice& mesh_device, std::uint32_t stride) {
    // 16 KB of 64B lines: four times the L1 D$, an eighth of L2.
    constexpr std::uint32_t kSweepLines = 256;
    static_assert(kEntryBase + kSweepLines * 64 <= kResultAddr, "sweep region overlaps the results");
    std::vector<std::uint32_t> entries(kSweepLines * 64 / sizeof(std::uint32_t), 0x1);
    slow_dispatch::WriteToL1(mesh_device, kCore, kEntryBase, entries);

    Program program = make_single_kernel_program(
        mesh_device,
        "cache_read_latency",
        "tests/tt_metal/tt_metal/data_movement/quasar_cache/kernels/cache_read_latency_test.cpp",
        {{"base_addr", kEntryBase},
         {"num_entries", kNumEntries},
         {"stride", stride},
         {"result_addr", kResultAddr},
         {"sweep_lines", kSweepLines}});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    constexpr std::uint32_t kNumPhases = 10;
    constexpr std::uint32_t kNumCounters = 6;
    std::vector<std::uint32_t> results;
    slow_dispatch::ReadFromL1(mesh_device, kCore, kResultAddr, 16 + kNumPhases * kNumCounters * 8, results);
    const std::uint32_t event3_readback = results[0];
    log_info(
        tt::LogTest,
        "stride={:2} mhpmevent3 read back 0x{:x}: HPM counters {}",
        stride,
        event3_readback,
        event3_readback != 0 ? "implemented" : "look unimplemented, their columns will read 0");

    auto counter = [&](std::uint32_t phase, std::uint32_t index) {
        const std::uint32_t word = 4 + (phase * kNumCounters + index) * 2;
        return (static_cast<std::uint64_t>(results[word + 1]) << 32) | results[word];
    };
    const char* names[kNumPhases] = {
        "measurement overhead",
        "uncached read (TL1)",
        "cached read, L1D+L2 miss",
        "cached read, L2 hit",
        "cached read, L1D hit",
        "uncached zero store",
        "uncached store + L2 invalidate",
        "uncached store + inval reg write only",
        "16KB sweep, cold (L1D+L2 miss)",
        "16KB sweep again (L1D evicted, L2 hit)"};
    const char* counter_names[kNumCounters] = {"cycles", "instret", "d$_blocked", "long_lat", "d$_miss", "load_use"};
    for (std::uint32_t p = 1; p < kNumPhases; p++) {
        std::string row;
        for (std::uint32_t c = 0; c < kNumCounters; c++) {
            // Signed, since a counter can tick slightly more in the empty baseline than in a phase.
            const double per_entry = static_cast<double>(static_cast<std::int64_t>(counter(p, c) - counter(0, c))) /
                                     static_cast<double>(p >= 8 ? kSweepLines : kNumEntries);
            row += fmt::format(" {}={:.1f}", counter_names[c], per_entry);
        }
        log_info(tt::LogTest, "stride={:2} {:38} per entry:{}", stride, names[p], row);
    }
    EXPECT_GT(counter(1, 0), counter(0, 0));
}

// One snooped host write per 64B line. Offsets and lengths are multiples of 4 because the RTL simulator
// write path only carries whole 32-bit words.
struct AlignmentCase {
    std::uint32_t offset;
    std::uint32_t length;
};
const std::vector<AlignmentCase> kAlignmentCases = {
    {0, 4},
    {4, 4},
    {8, 4},
    {12, 4},
    {0, 8},
    {4, 8},
    {8, 8},
    {12, 8},
    {0, 12},
    {4, 12},
    {0, 16},
    {4, 16},
    {0, 32},
    {16, 32},
    {48, 4},
    {0, 64}};

std::uint32_t poison_word(std::uint32_t line, std::uint32_t w) { return 0xA0000000 | (line << 8) | w; }
std::uint32_t written_word(std::uint32_t line, std::uint32_t w) { return 0x5E000000 | (line << 8) | w; }

// Returns false only if TL1 itself ends up wrong. How the cached views fare is reported per case.
bool run_alignment(distributed::MeshDevice& mesh_device) {
    constexpr std::uint32_t kWordsPerLine = 16;
    const std::uint32_t num_lines = kAlignmentCases.size();

    std::vector<std::uint32_t> lines(num_lines * kWordsPerLine);
    for (std::uint32_t line = 0; line < num_lines; line++) {
        for (std::uint32_t w = 0; w < kWordsPerLine; w++) {
            lines[line * kWordsPerLine + w] = poison_word(line, w);
        }
    }
    slow_dispatch::WriteToL1(mesh_device, kCore, kEntryBase, lines);
    std::vector<std::uint32_t> zeros(8, 0);
    slow_dispatch::WriteToL1(mesh_device, kCore, kFlagAddr, zeros);
    std::vector<std::uint32_t> unset(num_lines * kWordsPerLine * 3 + 4, 0xFFFFFFFF);
    slow_dispatch::WriteToL1(mesh_device, kCore, kResultAddr, unset);

    Program program = make_single_kernel_program(
        mesh_device,
        "snoop_alignment",
        "tests/tt_metal/tt_metal/data_movement/quasar_cache/kernels/snoop_alignment_test.cpp",
        {{"base_addr", kEntryBase}, {"num_lines", num_lines}, {"flag_addr", kFlagAddr}, {"result_addr", kResultAddr}});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<std::uint32_t> armed;
    do {
        slow_dispatch::ReadFromL1(mesh_device, kCore, kFlagAddr, sizeof(std::uint32_t), armed);
    } while (armed[0] != 1);

    auto window = open_l1_window(mesh_device, kEntryBase + num_lines * 64, tt::umd::WindowFlags::Snoop);
    for (std::uint32_t line = 0; line < num_lines; line++) {
        const auto& c = kAlignmentCases[line];
        std::vector<std::uint32_t> data(c.length / 4);
        for (std::uint32_t k = 0; k < data.size(); k++) {
            data[k] = written_word(line, c.offset / 4 + k);
        }
        window->write_block(kEntryBase + line * 64 + c.offset, data.data(), c.length);
    }
    std::vector<std::uint32_t> done = {1};
    slow_dispatch::WriteToL1(mesh_device, kCore, kFlagAddr + 16, done);
    distributed::Finish(cq);

    std::vector<std::uint32_t> results;
    slow_dispatch::ReadFromL1(mesh_device, kCore, kResultAddr, num_lines * kWordsPerLine * 3 * 4, results);

    bool tl1_ok = true;
    const char* view_names[3] = {"cached", "tl1", "after_l2_inv"};
    for (std::uint32_t line = 0; line < num_lines; line++) {
        const auto& c = kAlignmentCases[line];
        std::string summary;
        for (std::uint32_t view = 0; view < 3; view++) {
            std::string bad;
            for (std::uint32_t w = 0; w < kWordsPerLine; w++) {
                const bool in_range = w * 4 >= c.offset && w * 4 < c.offset + c.length;
                const std::uint32_t expected = in_range ? written_word(line, w) : poison_word(line, w);
                const std::uint32_t got = results[(line * 3 + view) * kWordsPerLine + w];
                if (got != expected) {
                    bad += fmt::format(" w{}=0x{:08x}(exp 0x{:08x})", w, got, expected);
                }
            }
            summary += fmt::format(" | {}: {}", view_names[view], bad.empty() ? "ok" : "BAD" + bad);
            if (view == 1 && !bad.empty()) {
                tl1_ok = false;
            }
        }
        log_info(tt::LogTest, "snooped write len={:2} offset={:2}{}", c.length, c.offset, summary);
    }
    return tl1_ok;
}

struct PollResult {
    std::uint32_t armed_value;
    std::uint32_t tl1_seen_at;
    std::uint32_t cached_seen_at;
    std::uint32_t last_cached;
    std::uint32_t polls;
};

// The kernel polls one 16B slot through its cache while the host overwrites the slot through a window opened
// with the given flags. The write is a whole aligned 16B beat, since a snooped write that does not start on a
// beat boundary leaves the cached copy wrong on Quasar A0 (QUAS-4226).
PollResult run_poll(distributed::MeshDevice& mesh_device, tt::umd::WindowFlags flags) {
    constexpr std::uint32_t kSlotWords = 4;
    constexpr std::uint32_t kPollsAfterTl1 = 1000;
    constexpr std::uint32_t kMaxPolls = 1U << 24;
    std::vector<std::uint32_t> old_slot = {old_value(0), old_value(1), old_value(2), old_value(3)};
    const std::vector<std::uint32_t> new_slot = {new_value(0), new_value(1), new_value(2), new_value(3)};

    slow_dispatch::WriteToL1(mesh_device, kCore, kEntryBase, old_slot);
    std::vector<std::uint32_t> zeros(8, 0);
    slow_dispatch::WriteToL1(mesh_device, kCore, kFlagAddr, zeros);
    std::vector<std::uint32_t> unset(8, 0xFFFFFFFF);
    slow_dispatch::WriteToL1(mesh_device, kCore, kResultAddr, unset);

    Program program = make_single_kernel_program(
        mesh_device,
        "snoop_poll",
        "tests/tt_metal/tt_metal/data_movement/quasar_cache/kernels/snoop_poll_test.cpp",
        {{"slot_addr", kEntryBase},
         {"new_value", new_value(0)},
         {"polls_after_tl1", kPollsAfterTl1},
         {"max_polls", kMaxPolls},
         {"flag_addr", kFlagAddr},
         {"result_addr", kResultAddr}});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<std::uint32_t> armed;
    do {
        slow_dispatch::ReadFromL1(mesh_device, kCore, kFlagAddr, sizeof(std::uint32_t), armed);
    } while (armed[0] != 1);

    auto window = open_l1_window(mesh_device, kEntryBase + kSlotWords * sizeof(std::uint32_t), flags);
    window->write_block(kEntryBase, new_slot.data(), kSlotWords * sizeof(std::uint32_t));
    distributed::Finish(cq);

    std::vector<std::uint32_t> results;
    slow_dispatch::ReadFromL1(mesh_device, kCore, kResultAddr, 5 * sizeof(std::uint32_t), results);
    PollResult r{results[0], results[1], results[2], results[3], results[4]};
    log_info(
        tt::LogTest,
        "snoop={}: armed=0x{:08x} tl1_seen_at={} cached_seen_at={} last_cached=0x{:08x} polls={}",
        flags == tt::umd::WindowFlags::Snoop,
        r.armed_value,
        r.tl1_seen_at,
        r.cached_seen_at,
        r.last_cached,
        r.polls);
    return r;
}

// What programming and issuing a NoC read through the RoCC command buffer costs, per operation.
void run_rocc_latency(distributed::MeshDevice& mesh_device) {
    constexpr std::uint32_t kIters = 64;
    constexpr std::uint32_t kNumPhases = 12;
    constexpr std::uint32_t kNumCounters = 6;
    constexpr std::uint32_t kDstAddr = kEntryBase + 4096;
    std::vector<std::uint32_t> src(16, 0x5A5A5A5A);
    slow_dispatch::WriteToL1(mesh_device, kCore, kEntryBase, src);

    Program program = make_single_kernel_program(
        mesh_device,
        "rocc_issue_latency",
        "tests/tt_metal/tt_metal/data_movement/quasar_cache/kernels/rocc_issue_latency_test.cpp",
        {{"src_addr", kEntryBase}, {"dst_addr", kDstAddr}, {"num_iters", kIters}, {"result_addr", kResultAddr}});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);

    std::vector<std::uint32_t> results;
    slow_dispatch::ReadFromL1(mesh_device, kCore, kResultAddr, 16 + (kNumPhases + 1) * kNumCounters * 8, results);
    log_info(tt::LogTest, "rocc: mhpmevent3 read back 0x{:x}", results[0]);

    // Slot kNumPhases holds the bare counter read pair, the baseline for the unfenced phases 4 to 6.
    auto counter = [&](std::uint32_t slot, std::uint32_t index) {
        const std::uint32_t word = 4 + (slot * kNumCounters + index) * 2;
        return (static_cast<std::uint64_t>(results[word + 1]) << 32) | results[word];
    };
    const char* names[kNumPhases] = {
        "",
        "1 cmdbuf register write",
        "2 cmdbuf register writes",
        "6 register writes of a read",
        "1 read issue, previous read done",
        "per read, 4 issued back to back",
        "set trid + read issue + reset trid",
        "1 DRAM read issue, previous read done",
        "DRAM read issue after write + atomic",
        "DRAM read issue after uncached store",
        "DRAM read issue after cached store",
        "DRAM read issue after uncached load"};
    const std::uint32_t per_phase_ops[kNumPhases] = {
        1, kIters, kIters, kIters, kIters, 4 * kIters, kIters, kIters, kIters, kIters, kIters, kIters};
    const char* counter_names[kNumCounters] = {"cycles", "instret", "d$_blocked", "long_lat", "d$_miss", "load_use"};
    for (std::uint32_t p = 1; p < kNumPhases; p++) {
        const std::uint32_t baseline = p <= 3 ? 0 : kNumPhases;
        std::string row;
        for (std::uint32_t c = 0; c < kNumCounters; c++) {
            const double per_op = static_cast<double>(static_cast<std::int64_t>(counter(p, c) - counter(baseline, c))) /
                                  static_cast<double>(per_phase_ops[p]);
            row += fmt::format(" {}={:.1f}", counter_names[c], per_op);
        }
        log_info(tt::LogTest, "rocc {:36} per op:{}", names[p], row);
    }
    EXPECT_GT(counter(4, 0), counter(kNumPhases, 0));
}

}  // namespace unit_tests::dm::quasar_snoop

class QuasarSnoopCoherence : public QuasarMeshDeviceSingleCardFixture {};

// Without the snoop bit a host write leaves the cached copy stale; with it the cached view picks the
// write up. Also reports whether an L2-only invalidate refreshes the L1 D$ copy, which the fast
// dispatch fetch queue relies on when it clears consumed entries.
TEST_F(QuasarSnoopCoherence, HostWriteVisibility) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    for (std::uint32_t stride : {16u, 64u}) {
        std::uint32_t stale = 0;
        std::uint32_t stale_after_l2 = 0;
        EXPECT_TRUE(unit_tests::dm::quasar_snoop::run_coherence(this->device(), true, stride, stale, stale_after_l2));
        EXPECT_EQ(stale, 0u) << "snooped host writes left " << stale << " cached entries stale, stride " << stride;
        EXPECT_TRUE(unit_tests::dm::quasar_snoop::run_coherence(this->device(), false, stride, stale, stale_after_l2));
        EXPECT_GT(stale, 0u) << "non-snooped host writes never left the cached view stale, stride " << stride;
    }
}

// End to end check of the snoop path: UMD window flag, NOC header bit, and the DM core's caches. The kernel keeps
// the slot hot in its cache by polling it. A snooped host write must show up in the cached view. Without snoop
// it must not, which proves the poll really reads a cached copy, so a pass on the snooped half means something.
TEST_F(QuasarSnoopCoherence, PolledHostWrite) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    using unit_tests::dm::quasar_snoop::old_value;
    using unit_tests::dm::quasar_snoop::run_poll;

    const auto snooped = run_poll(this->device(), tt::umd::WindowFlags::Snoop);
    EXPECT_EQ(snooped.armed_value, old_value(0)) << "the kernel did not cache the old value before the write";
    EXPECT_NE(snooped.cached_seen_at, 0u) << "a snooped host write never reached the cached view";

    const auto plain = run_poll(this->device(), tt::umd::WindowFlags::None);
    EXPECT_EQ(plain.armed_value, old_value(0)) << "the kernel did not cache the old value before the write";
    EXPECT_NE(plain.tl1_seen_at, 0u) << "the host write never reached TL1";
    EXPECT_EQ(plain.cached_seen_at, 0u) << "the cached view picked up a write without snoop, so this test cannot "
                                           "tell snoop apart on this hardware";
    EXPECT_EQ(plain.last_cached, old_value(0)) << "the cached copy changed without the new value showing up";
}

class QuasarRoccLatency : public QuasarMeshDeviceSingleCardFixture {};

TEST_F(QuasarRoccLatency, Sweep) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    unit_tests::dm::quasar_snoop::run_rocc_latency(this->device());
}

class QuasarSnoopAlignment : public QuasarMeshDeviceSingleCardFixture {};

// QUAS-4226: snooped writes that do not start on a 16B boundary are expected to leave the cached copy
// wrong on Quasar A0. Reports every shape; only TL1 corruption fails the test.
TEST_F(QuasarSnoopAlignment, Sweep) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(unit_tests::dm::quasar_snoop::run_alignment(this->device()));
}

class QuasarCacheReadLatency : public QuasarMeshDeviceSingleCardFixture {};

TEST_F(QuasarCacheReadLatency, Sweep) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    for (std::uint32_t stride : {16u, 64u}) {
        unit_tests::dm::quasar_snoop::run_read_latency(this->device(), stride);
    }
}

}  // namespace tt::tt_metal
