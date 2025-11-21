// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>

#include "tests/tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/dispatch/util/size_literals.hpp"
#include "tt_metal/common/env_lib.hpp"

namespace tt::tt_metal {
namespace kernel_size_tests {

class KernelSizeTest : public UnitMeshCQFixture {
protected:
    // Runtime-determined values
    uint32_t actual_kernel_config_size_{};
    uint32_t kernel_config_base_{};
    uint32_t unreserved_base_{};
    uint32_t unreserved_size_{};
    const Hal& hal_{MetalContext::instance().hal()};

    // 65536 bytes is an edge case for testing the CQDispatchWritePackedLargeSubCmd.length
    // as its equal to scratch_db_size/2 (64KB) which overflows to 0 when cast to uint16_t in dispatch system
    static constexpr uint32_t packed_write_large_max_chunk_size = 65536;
    static constexpr uint32_t NUM_RISCV_PROCESSORS = 5;  // BRISC, NCRISC, TRISC0/1/2
    static constexpr uint32_t NUM_KERNELS_AGGREGATE_TEST = 4;
    static constexpr uint32_t OVERSIZED_KERNEL_PADDING = 32;  // Bytes beyond limit for rejection tests

    // Near limit of kernel config buffer size
    // The kernel size is greater than whats set by KERNEL_BYTES due elf headers, alignment, etc.
    // So we need to subtract the overhead to get the near limit of the kernel config buffer size
    static constexpr uint32_t NEAR_LIMIT_BUFFER = 324;

    void SetUp() override {
        UnitMeshCQFixture::SetUp();

        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Kernel size config tests are only supported on Blackhole";
        }

        kernel_config_base_ = hal_.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
        unreserved_base_ = hal_.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        unreserved_size_ = hal_.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

        // kernel config buffer size
        actual_kernel_config_size_ = unreserved_base_ - kernel_config_base_;

        log_l1_memory_layout();
    }

    void log_l1_memory_layout() {
        log_info(
            LogTest,
            "Kernel Config Buffer Size: {} KB from 0x{:x} to 0x{:x}",
            actual_kernel_config_size_ / 1024,
            kernel_config_base_,
            unreserved_base_);
        log_info(
            LogTest,
            "User unreserved space: {} KB from 0x{:x} to 0x{:x}",
            unreserved_size_ / 1024,
            unreserved_base_,
            unreserved_base_ + unreserved_size_);
    }
};

using namespace tt::tt_metal;

// Test 1: Verify default aggregate buffer works with small programs
TEST_F(KernelSizeTest, DefaultAggregateBuffer) {
    log_info(LogTest, "Testing default aggregate buffer");

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        // Create a simple kernel
        CoreRange cr({0, 0}, {0, 0});
        CoreRangeSet cr_set({cr});

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/arbiter_hang.cpp",
            cr_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        workload.add_program(device_range_, std::move(program));

        EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
        distributed::Finish(device->mesh_command_queue());
    }
}

// Test 2: Check if env var TT_METAL_KERNEL_CONFIG_BUFFER_SIZE is respected
TEST_F(KernelSizeTest, EnvVarConfigBufferSize) {
    log_info(LogTest, "Testing env var TT_METAL_KERNEL_CONFIG_BUFFER_SIZE parameter control");

    uint32_t user_set_kernel_config_size = tt::parse_env<uint32_t>("TT_METAL_KERNEL_CONFIG_BUFFER_SIZE", 0);

    if (user_set_kernel_config_size == 0) {
        GTEST_SKIP() << "Run with TT_METAL_KERNEL_CONFIG_BUFFER_SIZE set to test huge kernel config support";
    }

    const uint32_t l1_alignment = hal_.get_alignment(HalMemType::L1);
    const uint32_t dram_alignment = hal_.get_alignment(HalMemType::DRAM);
    uint32_t max_alignment = std::max(dram_alignment, l1_alignment);

    // Calculate the expected end address by aligning (kernel_config_base_ + user_set_kernel_config_size) up
    // to the max_alignment boundary, and then subtract the base address to get the aligned size.
    // Same as bh_hal_tensix.cpp
    uint32_t aligned_end_addr = (((kernel_config_base_ + user_set_kernel_config_size - 1) | (max_alignment - 1)) + 1);
    uint32_t expected_kernel_config_size = aligned_end_addr - kernel_config_base_;

    log_info(LogTest, "User set kernel config size                 : {} B", user_set_kernel_config_size);
    log_info(LogTest, "Expected kernel config size  after alignment: {} B", expected_kernel_config_size);
    log_info(LogTest, "Actual kernel config size                   : {} B", actual_kernel_config_size_);

    // Verify that the actual kernel config size is equal to the expected kernel config size after alignment
    EXPECT_EQ(actual_kernel_config_size_, expected_kernel_config_size);

    // Verify that the actual kernel config size is at least the user set kernel config size
    // Due to alignment, the actual kernel config size may be larger than the user set kernel config size
    EXPECT_LE(user_set_kernel_config_size, actual_kernel_config_size_);
}

// Test 3: Verify kernel config buffer respects worker_l1_size parameter
TEST_F(KernelSizeTest, WorkerL1SizeParameterControl) {
    log_info(LogTest, "Testing worker_l1_size parameter control");

    auto l1_base = hal_.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    auto l1_size = hal_.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);

    uint32_t l1_end = l1_base + l1_size;

    // Max available space for kernel config buffer
    uint32_t max_available = l1_end - kernel_config_base_;

    log_info(LogTest, "Kernel config base                          : 0x{:x}", kernel_config_base_);
    log_info(LogTest, "L1 end                                      : 0x{:x}", l1_end);
    log_info(LogTest, "Max available for kernel config + unreserved: {} B", max_available);
    log_info(LogTest, "Actual space used by kernel config buffer   : {} B", actual_kernel_config_size_);

    EXPECT_LE(actual_kernel_config_size_, max_available);
}

// Test 4: Test kernels with different sizes and also edge case of 65536 bytes (64KB)
// Tests the edge case where 65536 bytes overflows to 0 in CQDispatchWritePackedLargeSubCmd.length
// This validates the encoding/decoding of the maximum chunk size in the dispatch system
TEST_F(KernelSizeTest, KernelSizesBoundaryConditions) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    log_info(LogTest, "Testing different kernel sizes and edge case of {} bytes", packed_write_large_max_chunk_size);

    // Test sizes around critical boundaries
    std::vector<uint32_t> kernel_sizes = {
        1_KB,                                            // 1 KB
        16_KB,                                           // 16 KB
        32_KB,                                           // 32 KB
        packed_write_large_max_chunk_size - 1,           // 64KB - 1
        packed_write_large_max_chunk_size,               // Exactly 64KB
        packed_write_large_max_chunk_size + 1,           // 64KB + 1
        2 * packed_write_large_max_chunk_size,           // 128KB
        3 * packed_write_large_max_chunk_size,           // 192KB
        212992,                                          // 212992 bytes
        actual_kernel_config_size_ / 8,                  // 1/8 of kernel config buffer size
        actual_kernel_config_size_ / 4,                  // 1/4 of kernel config buffer size
        actual_kernel_config_size_ / 2,                  // Half of kernel config buffer size
        actual_kernel_config_size_ - NEAR_LIMIT_BUFFER,  // Near limit of kernel config buffer size
    };

    // Select a kernel size and if it's within kernel config buffer size,
    // test it on all devices
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

        for (const auto& device : devices_) {
            distributed::MeshWorkload workload;
            Program program;

            CoreRange cr({0, 0}, {0, 0});
            CoreRangeSet cr_set({cr});

            std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/large_kernel_test.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

            workload.add_program(device_range_, std::move(program));
            EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
            distributed::Finish(device->mesh_command_queue());
        }
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);
}

// Test 5: Multiple kernels filling up kernel config buffer on different cores
TEST_F(KernelSizeTest, MultipleKernelsAggregateSize) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    log_info(LogTest, "Testing multiple kernels filling up kernel config buffer");

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        // Use near limit of kernel config buffer size to test the
        // multiple kernels filling up the buffer
        const uint32_t kernel_size = actual_kernel_config_size_ - NEAR_LIMIT_BUFFER;

        for (uint32_t i = 0; i < NUM_KERNELS_AGGREGATE_TEST; i++) {
            CoreRange cr({i, 0}, {i, 0});
            CoreRangeSet cr_set({cr});

            std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/large_kernel_test.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

            log_info(LogTest, "Created kernel {} with size {} B", i, kernel_size);
        }

        workload.add_program(device_range_, std::move(program));
        EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
        distributed::Finish(device->mesh_command_queue());
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);
}

// Test 6: Test all five RISC-V processors with large kernels on same core
TEST_F(KernelSizeTest, AllRISCVProcessorsWithLargeKernels) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    log_info(LogTest, "Testing all five RISC-V processors with large kernels");

    // Distribute the kernel size equally among the five RISC-V processors
    const uint32_t kernel_size = ((actual_kernel_config_size_) / NUM_RISCV_PROCESSORS) - NEAR_LIMIT_BUFFER;

    log_info(LogTest, "Kernel size: {} B", kernel_size);

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        CoreRange cr({0, 0}, {0, 0});
        CoreRangeSet cr_set({cr});

        std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

        // BRISC
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/large_kernel_test.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

        // NCRISC
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/large_kernel_test.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = defines});

        // TRISC0, TRISC1, TRISC2
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/large_kernel_test.cpp",
            cr_set,
            ComputeConfig{.compile_args = {}, .defines = defines});

        workload.add_program(device_range_, std::move(program));
        EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
        distributed::Finish(device->mesh_command_queue());
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);
}

// Test 7: Stress test - rapidly vary kernel sizes on a single device
TEST_F(KernelSizeTest, StressTestRapidlyVaryKernelSizes) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    log_info(LogTest, "Stress test - rapidly vary kernel sizes");

    // Rapidly alternate between small and large kernels
    std::vector<uint32_t> alternating_sizes = {
        1024,                            // Small
        actual_kernel_config_size_ / 2,  // Large
        2048,                            // Small
        actual_kernel_config_size_ / 4,  // Large
        512,                             // Boundary
        actual_kernel_config_size_ / 2,  // Large
    };

    // Take a device and vary the kernel sizes rapidly
    for (const auto& device : devices_) {
        for (const auto& kernel_size : alternating_sizes) {
            if (kernel_size > actual_kernel_config_size_) {
                log_info(
                    LogTest,
                    "Skipping kernel size: {} B (exceeds available space: {} B)",
                    kernel_size,
                    actual_kernel_config_size_);
                continue;
            }

            distributed::MeshWorkload workload;
            Program program;

            CoreRange cr({0, 0}, {0, 0});
            CoreRangeSet cr_set({cr});

            std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/large_kernel_test.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

            log_info(LogTest, "Testing kernel size: {} B", kernel_size);

            workload.add_program(device_range_, std::move(program));
            EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
            distributed::Finish(device->mesh_command_queue());
        }
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);
}

// Test 8: Verify that kernels exceeding L1 limits are properly rejected
TEST_F(KernelSizeTest, KernelSizeExceedsL1Limit) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    log_info(LogTest, "Testing kernel size exceeding L1 limits");

    auto l1_size = hal_.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);

    // Try to create a kernel larger than the entire L1 memory
    // The kernel size is typically greater than whats specified by KERNEL_BYTES due elf headers, alignment, etc.
    // So even adding a small amount of padding can reject the kernel
    uint32_t oversized_kernel = l1_size + OVERSIZED_KERNEL_PADDING;

    log_info(LogTest, "L1 size: {} B", l1_size);
    log_info(LogTest, "Attempting to create kernel of size: {} B", oversized_kernel);
    log_info(LogTest, "Kernel config buffer size: {} B", actual_kernel_config_size_);

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        CoreRange cr({0, 0}, {0, 0});
        CoreRangeSet cr_set({cr});

        std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(oversized_kernel)}};

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/large_kernel_test.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

        workload.add_program(device_range_, std::move(program));

        // Expect this to throw an exception
        EXPECT_THROW(
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false), std::runtime_error);

        log_info(LogTest, "Correctly rejected oversized kernel");
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);
}

// Test 9: Verify that kernel size exceeding kernel config buffer is rejected
TEST_F(KernelSizeTest, ExceedsKernelConfigBufferSize) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    log_info(LogTest, "Testing kernel size exceeding kernel config buffer size");

    // Kernel config buffer size plus padding for rejection test
    // The kernel size is typically greater than whats specified by KERNEL_BYTES due elf headers, alignment, etc.
    // So even adding a small amount of padding can reject the kernel
    const uint32_t kernel_size = actual_kernel_config_size_ + OVERSIZED_KERNEL_PADDING;

    log_info(LogTest, "Kernel config buffer size: {} B", actual_kernel_config_size_);
    log_info(LogTest, "Kernel size: {} B (exceeds by {} B)", kernel_size, kernel_size - actual_kernel_config_size_);

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        CoreRange cr({0, 0}, {0, 0});
        CoreRangeSet cr_set({cr});

        std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/large_kernel_test.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

        workload.add_program(device_range_, std::move(program));
        // Expect this to throw because kernel size exceeds kernel config buffer
        EXPECT_THROW(
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false), std::runtime_error);

        log_info(LogTest, "Correctly rejected oversized kernel configuration");
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);
}

}  // namespace kernel_size_tests
}  // namespace tt::tt_metal
