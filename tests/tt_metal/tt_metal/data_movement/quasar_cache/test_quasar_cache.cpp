// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for Quasar DM core cache management functions.
// These tests verify L1 D$ and L2 cache flush/invalidate operations.

#include <tt-logger/tt-logger.hpp>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/host_api.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt::test_utils;

namespace unit_tests::dm::quasar_cache {

// Skip test if not running on Quasar with simulator
bool should_skip_test() {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch != tt::ARCH::QUASAR) {
        return true;
    }
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

bool run_l2_flush_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const L2FlushTestConfig& config) {

    IDevice* device = mesh_device->get_devices()[0];
    constexpr CoreCoord core = {0, 0};

    // For invalidate tests, pre-populate with known "old" values
    // that should persist after invalidation (since invalidate doesn't write back)
    uint32_t old_value = 0xDEADBEEF;
    std::vector<uint32_t> init_data(config.num_words, config.expect_new_values ? 0 : old_value);
    tt_metal::detail::WriteToDeviceL1(device, core, config.base_addr, init_data);

    // Create program with Quasar DM kernel
    Program program = CreateProgram();

    KernelHandle kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/quasar_cache/kernels/l2_flush_test.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1});

    // Set runtime args
    SetRuntimeArgs(program, kernel, core, {config.base_addr, config.test_mode});
    SetCommonRuntimeArgs(program, kernel, {config.value, config.num_words});

    // Execute
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // Read back and verify
    std::vector<uint32_t> output_data;
    tt_metal::detail::ReadFromDeviceL1(
        device, core, config.base_addr, config.num_words * sizeof(uint32_t), output_data);

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

bool run_l1_dcache_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const L1DCacheTestConfig& config) {

    IDevice* device = mesh_device->get_devices()[0];
    constexpr CoreCoord core = {0, 0};

    // For invalidate tests, we need to pre-populate with known "old" values
    // that should persist after invalidation (since invalidate doesn't write back)
    uint32_t old_value = 0xDEADBEEF;
    std::vector<uint32_t> init_data(config.num_words, old_value);
    tt_metal::detail::WriteToDeviceL1(device, core, config.base_addr, init_data);

    // Create program with Quasar DM kernel
    Program program = CreateProgram();

    KernelHandle kernel = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/quasar_cache/kernels/l1_dcache_test.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1});

    // Set runtime args
    SetRuntimeArgs(program, kernel, core, {config.base_addr, config.test_mode});
    SetCommonRuntimeArgs(program, kernel, {config.value, config.num_words});

    // Execute
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // Read back and verify
    std::vector<uint32_t> output_data;
    tt_metal::detail::ReadFromDeviceL1(
        device, core, config.base_addr, config.num_words * sizeof(uint32_t), output_data);

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
// Test Suite: L2 Cache Flush Operations
// =============================================================================

class QuasarL2CacheFlush : public MeshDeviceSingleCardFixture {};

TEST_F(QuasarL2CacheFlush, FlushLine) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    unit_tests::dm::quasar_cache::L2FlushTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 16,
        .value = 0x12340000,
        .test_mode = 0  // line flush
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(devices_[0], config));
}

TEST_F(QuasarL2CacheFlush, FlushRange) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    unit_tests::dm::quasar_cache::L2FlushTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 64,  // Multiple cache lines (4 lines worth)
        .value = 0xABCD0000,
        .test_mode = 1  // range flush
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(devices_[0], config));
}

TEST_F(QuasarL2CacheFlush, FlushFull) {
    if (unit_tests::dm::quasar_cache::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }

    unit_tests::dm::quasar_cache::L2FlushTestConfig config = {
        .base_addr = 100 * 1024,
        .num_words = 64,
        .value = 0x55550000,
        .test_mode = 2  // full flush
    };
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(devices_[0], config));
}

TEST_F(QuasarL2CacheFlush, InvalidateLine) {
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
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(devices_[0], config));
}

TEST_F(QuasarL2CacheFlush, InvalidateFreshRead) {
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
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l2_flush_test(devices_[0], config));
}

// =============================================================================
// Test Suite: L1 Data Cache Operations
// =============================================================================

class QuasarL1DCacheOps : public MeshDeviceSingleCardFixture {};

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
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(devices_[0], config));
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
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(devices_[0], config));
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
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(devices_[0], config));
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
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(devices_[0], config));
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
    EXPECT_TRUE(unit_tests::dm::quasar_cache::run_l1_dcache_test(devices_[0], config));
}

}  // namespace tt::tt_metal
