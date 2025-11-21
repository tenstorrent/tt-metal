// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>

#include "tests/tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/common.h"

namespace tt::tt_metal {
namespace kernel_size_tests {

class KernelSizeTest : public UnitMeshCQFixture {
protected:
    uint32_t actual_kernel_config_size_{};
    uint32_t kernel_config_base_{};
    uint32_t unreserved_base_{};
    uint32_t unreserved_size_{};

    void SetUp() override {
        UnitMeshCQFixture::SetUp();

        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Kernel size config tests are only supported on Blackhole";
        }

        const auto& hal = MetalContext::instance().hal();
        kernel_config_base_ = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
        unreserved_base_ = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
        unreserved_size_ = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

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
    uint32_t user_set_kernel_config_size = tt::parse_env<uint32_t>("TT_METAL_KERNEL_CONFIG_BUFFER_SIZE", 0);

    if (user_set_kernel_config_size == 0) {
        GTEST_SKIP() << "Run with TT_METAL_KERNEL_CONFIG_BUFFER_SIZE set to test huge kernel config support";
    }

    const uint32_t l1_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t dram_alignment = tt::tt_metal::MetalContext::instance().hal().get_alignment(HalMemType::DRAM);
    uint32_t max_alignment = std::max(dram_alignment, l1_alignment);

    uint32_t expected_kernel_config_size =
        (((kernel_config_base_ + user_set_kernel_config_size - 1) | (max_alignment - 1)) + 1) - kernel_config_base_;

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
    const auto& hal = MetalContext::instance().hal();
    auto l1_base = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    auto l1_size = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);

    uint32_t l1_end = l1_base + l1_size;

    // Max available space for kernel config buffer
    uint32_t max_available = l1_end - kernel_config_base_;

    log_info(LogTest, "Kernel config base                          : 0x{:x}", kernel_config_base_);
    log_info(LogTest, "L1 end                                      : 0x{:x}", l1_end);
    log_info(LogTest, "Max available for kernel config + unreserved: {} B", max_available);
    log_info(LogTest, "Actual space used by kernel config buffer   : {} B", actual_kernel_config_size_);

    EXPECT_LE(actual_kernel_config_size_, max_available);
}

// Test 4: Test kernels with sizes around 65335 / 65536 bytes
TEST_F(KernelSizeTest, KernelSizesBoundaryConditions) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    // Test boundary conditions slightly below the default 69KB kernel config buffer size
    std::vector<uint32_t> kernel_sizes = {
        1024,                               // 1 KB
        16 * 1024,                          // 16 KB
        32 * 1024,                          // 32 KB
        65336 - 2,                          // 64KB - 2
        65336 - 1,                          // 64KB - 1 bytes
        65536,                              // Exactly 64KB
        65536 + 1,                          // 64KB + 1 bytes
        65536 + 2,                          // 64KB + 2 bytes
        2 * 65336,                          // ~128KB
        131072,                             // 131072 bytes
        3 * 65336,                          // ~192KB
        212992,                             // 212992 bytes
        actual_kernel_config_size_ / 8,     // 1/8 of available
        actual_kernel_config_size_ / 4,     // 1/4 of available
        actual_kernel_config_size_ / 2,     // Half of available
        actual_kernel_config_size_ - 2048,  // Near limit
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

// Test 5: Multiple kernels filling up kernel config buffer
TEST_F(KernelSizeTest, MultipleAggregatedKernelSize) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    log_info(LogTest, "Testing multiple kernels filling up kernel config buffer");

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        const uint32_t num_kernels = 4;
        const uint32_t kernel_size = actual_kernel_config_size_ / num_kernels;

        for (uint32_t i = 0; i < num_kernels; i++) {
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

// Test 6: Test all five RISC-V processors with large kernels
TEST_F(KernelSizeTest, AllRISCVProcessorsWithLargeKernels) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    log_info(LogTest, "Testing all five RISC-V processors with large kernels");

    const uint32_t kernel_size = actual_kernel_config_size_ / 5;

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
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/arbiter_hang.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

        // NCRISC
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/arbiter_hang.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = defines});

        // TRISC0, TRISC2, TRISC2
        CreateKernel(
            program,
            "tt_metal/kernels/compute/blank.cpp",
            cr_set,
            ComputeConfig{.compile_args = {}, .defines = defines});

        workload.add_program(device_range_, std::move(program));
        EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
        distributed::Finish(device->mesh_command_queue());
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);
}

// Test 7: Stress test - rapidly vary kernel sizes
TEST_F(KernelSizeTest, StressTestRapidlyVaryKernelSizes) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    log_info(LogTest, "Stress test - rapidly vary kernel sizes");

    // Rapidly alternate between small and large kernels
    std::vector<uint32_t> alternating_sizes = {
        1024,                            // Small
        actual_kernel_config_size_ / 2,  // Large
        2048,                            // Small
        actual_kernel_config_size_ / 4,  // Large
        65336,                           // Boundary
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

}  // namespace kernel_size_tests
}  // namespace tt::tt_metal
