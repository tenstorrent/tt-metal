// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tests/tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/dispatch/util/size_literals.hpp"
#include "tt_metal/common/env_lib.hpp"

namespace tt::tt_metal::kernel_size_tests {

// Fixture for testing with DEFAULT kernel config buffer size (69KB)
// Sets worker_l1_size = max_worker_l1_size_ - 69KB, resulting in a 69KB kernel config buffer.
// This represents the typical production configuration
class KernelSizeTestDefaultBuffer : public UnitMeshCQFixture {
    static constexpr uint32_t DEFAULT_KERNEL_CONFIG_BUFFER_SIZE = 69 * 1024;

protected:
    uint32_t actual_kernel_config_size_{};
    uint32_t unreserved_base_{};
    uint32_t unreserved_size_{};
    const Hal& hal_{MetalContext::instance().hal()};
    // Total L1 memory available to be partitioned between kernel config buffer and user-allocatable space
    const uint32_t max_worker_l1_size_{hal::get_max_worker_l1_unreserved_size()};
    const uint32_t kernel_config_base_{
        hal_.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG)};

    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        // These tests are only supported on Wormhole and Blackhole (skip Quasar)
        skip_if_unsupported_arch();

        uint32_t worker_l1_size = max_worker_l1_size_ - DEFAULT_KERNEL_CONFIG_BUFFER_SIZE;
        create_devices(DEFAULT_TRACE_REGION_SIZE, worker_l1_size);
        compute_memory_layout();
        log_l1_memory_layout();
    }

    // Helper function to compute the partition of worker L1 memory between
    // kernel config buffer and unreserved space
    // The kernel config buffer occupies [kernel_config_base_, unreserved_base_).
    // User-allocatable (unreserved) space occupies [unreserved_base_, L1_top).
    // The worker_l1_size parameter sets the size of the unreserved region, which indirectly
    // determines where unreserved_base_ is positioned and thus the kernel config buffer size.
    // Larger worker_l1_size -> higher unreserved_base_ -> smaller kernel config buffer and vice versa
    void compute_memory_layout() {
        TT_FATAL(!devices_.empty() && !devices_[0]->get_devices().empty(), "No devices available for testing");

        auto* single_device = devices_[0]->get_devices()[0];
        unreserved_base_ = single_device->allocator()->get_base_allocator_addr(HalMemType::L1);
        auto l1_base = hal_.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
        auto l1_size = hal_.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
        // User unreserved space extends from unreserved_base_ to the top of L1
        unreserved_size_ = (l1_base + l1_size) - unreserved_base_;
        // kernel config buffer size
        actual_kernel_config_size_ = unreserved_base_ - kernel_config_base_;
    }

    void log_l1_memory_layout() {
        log_info(LogTest, "Kernel config base                          : 0x{:x}", kernel_config_base_);
        log_info(LogTest, "Max available for kernel config + unreserved: {} B", max_worker_l1_size_);
        log_info(LogTest, "Actual space used by kernel config buffer   : {} B", actual_kernel_config_size_);
        log_info(LogTest, "Actual unreserved space                     : {} B", unreserved_size_);
    }

    void skip_if_unsupported_arch() {
        if (arch_ == tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Kernel size config tests are only supported on Wormhole and Blackhole";
        }
    }
};

// Test a big kernel config buffer size by setting the worker_l1_size to 64 KB
// This creates a kernel config buffer of size max_worker_l1_size_ - 64 KB (~1.4MB)
// and enables testing of larger kernels
class KernelSizeTestBigBuffer : public KernelSizeTestDefaultBuffer {
    static constexpr uint32_t MIN_WORKER_L1_SIZE = 64 * 1024;

protected:
    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        skip_if_unsupported_arch();

        create_devices(DEFAULT_TRACE_REGION_SIZE, MIN_WORKER_L1_SIZE);
        compute_memory_layout();
        log_l1_memory_layout();
    }
};

// Helper for RAII kernel nullification
class ScopedNullifyKernels {
public:
    ScopedNullifyKernels() { tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true); }
    ~ScopedNullifyKernels() { tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false); }

    ScopedNullifyKernels(const ScopedNullifyKernels&) = delete;
    ScopedNullifyKernels& operator=(const ScopedNullifyKernels&) = delete;
    ScopedNullifyKernels(ScopedNullifyKernels&&) = delete;
    ScopedNullifyKernels& operator=(ScopedNullifyKernels&&) = delete;
};

using namespace tt::tt_metal;

// Test 1: Verify that a large kernel (1MB) succeeds with a big kernel config buffer
// Uses KernelSizeTestBigBuffer which provides ~1.4MB of kernel config space
TEST_F(KernelSizeTestBigBuffer, LargeKernelWorks) {
    ScopedNullifyKernels nullify_kernels;

    log_info(LogTest, "Testing big kernel config buffer size");
    const uint32_t kernel_size = 1024 * 1024;
    log_info(LogTest, "Kernel config buffer size: {} B", actual_kernel_config_size_);

    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});
    std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

    auto& device = devices_[0];
    distributed::MeshWorkload workload;
    Program program;

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

    workload.add_program(device_range_, std::move(program));
    EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
    distributed::Finish(device->mesh_command_queue());
}

// Test 2: Verify that kernels exceeding kernel config buffer size are rejected
// This test intentionally creates an oversized kernel (100 KB) and expects
// the dispatch system to throw TT_FATAL in ProgramImpl::finalize_program_offsets()
TEST_F(KernelSizeTestDefaultBuffer, LargeKernelFails) {
    ScopedNullifyKernels nullify_kernels;

    const uint32_t kernel_size = 1024 * 100;
    log_info(
        LogTest,
        "Testing kernel size of {} bytes exceeding kernel config buffer size of {} bytes",
        kernel_size,
        actual_kernel_config_size_);

    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});
    std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

    auto& device = devices_[0];
    distributed::MeshWorkload workload;
    Program program;

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

    workload.add_program(device_range_, std::move(program));
    EXPECT_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false), std::runtime_error);

    log_info(LogTest, "Correctly rejected oversized kernel configuration");
}

// Test 3: Test kernels with sizes around the 64KB boundary (65532, 65536, 65540 bytes)
// Verify system handles this boundary condition correctly
TEST_F(KernelSizeTestDefaultBuffer, KernelSizesBoundaryConditions) {
    ScopedNullifyKernels nullify_kernels;

    constexpr uint32_t PACKED_WRITE_LARGE_MAX_CHUNK_SIZE = 65536 - 256;  // 256 bytes accounts for elf metadata, etc.
    log_info(LogTest, "Testing different kernel sizes and edge case of {} bytes", PACKED_WRITE_LARGE_MAX_CHUNK_SIZE);

    // Test sizes around critical boundaries
    const std::vector<uint32_t> kernel_sizes = {
        PACKED_WRITE_LARGE_MAX_CHUNK_SIZE - 4,  // 64KB - 4
        PACKED_WRITE_LARGE_MAX_CHUNK_SIZE,      // Exactly 64KB
        PACKED_WRITE_LARGE_MAX_CHUNK_SIZE + 4,  // 64KB + 4
    };

    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    // Test each kernel size if it fits within the kernel config buffer
    for (const auto& kernel_size : kernel_sizes) {
        if (kernel_size > actual_kernel_config_size_) {
            log_info(
                LogTest,
                "Skipping kernel size: {} B (exceeds available space: {} B)",
                kernel_size,
                actual_kernel_config_size_);
            continue;
        }

        log_info(LogTest, "Testing kernel size: {} B", kernel_size);
        std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

        auto& device = devices_[0];
        distributed::MeshWorkload workload;
        Program program;

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

        workload.add_program(device_range_, std::move(program));
        EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
        distributed::Finish(device->mesh_command_queue());
    }
}

// Test 4: Verify that a large kernel (200KB) loads correctly and executes with correct results
// Creates a kernel with 50000 (x 4 bytes / instruction = 200KB) add instructions and verifies the output equals 50000
// This tests both kernel loading correctness and execution integrity for very large kernels
TEST_F(KernelSizeTestBigBuffer, BigKernelExecutionCorrectness) {
    log_info(LogTest, "Testing big kernel execution correctness");

    constexpr uint32_t NUM_ADDS = 50000;
    std::map<std::string, std::string> defines = {{"NUM_ADDS", std::to_string(NUM_ADDS)}};

    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    auto& device = devices_[0];
    distributed::MeshWorkload workload;
    Program program;

    // Get unreserved L1 base address - this is where we can safely write
    uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/large_kernel_add_test.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

    // Pass the L1 address directly as runtime arg
    SetRuntimeArgs(program, kernel, cr_set, {l1_unreserved_base});

    // Execute the kernel
    workload.add_program(device_range_, std::move(program));
    EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
    distributed::Finish(device->mesh_command_queue());

    // Read the result back directly from L1 using device API
    auto* single_device = device->get_devices()[0];
    std::vector<uint32_t> result;
    tt::tt_metal::detail::ReadFromDeviceL1(
        single_device,
        CoreCoord(0, 0),  // Physical or logical core (0,0)
        l1_unreserved_base,
        sizeof(uint32_t),
        result);

    log_info(LogTest, "Result: {} (expected: {})", result[0], NUM_ADDS);
    EXPECT_EQ(result[0], NUM_ADDS) << "Kernel should have performed " << NUM_ADDS << " additions";
}

}  // namespace tt::tt_metal::kernel_size_tests
