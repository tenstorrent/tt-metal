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
    uint32_t actual_kernel_config_size_{};
    uint32_t unreserved_base_{};
    uint32_t unreserved_size_{};
    const Hal& hal_{MetalContext::instance().hal()};
    // Maximum combined size available for kernel config buffer + unreserved user space in worker L1 memory
    const uint32_t max_worker_l1_size_{hal::get_max_worker_l1_unreserved_size()};
    const uint32_t kernel_config_base_{
        hal_.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG)};

    static constexpr uint32_t PACKED_WRITE_LARGE_MAX_CHUNK_SIZE = 65536;
    static constexpr uint32_t NUM_RISCV_PROCESSORS = 5;  // BRISC, NCRISC, TRISC0/1/2
    static constexpr uint32_t NUM_KERNELS_AGGREGATE_TEST = 4;  // Test on 4 separate cores simultaneously
    static constexpr uint32_t OVERSIZED_KERNEL_PADDING = 32;  // Bytes beyond limit for rejection tests

    // Estimated overhead added to KERNEL_BYTES define when creating actual kernel binaries.
    // This accounts for ELF headers, alignment and other binary metadata
    // When targeting the maximum kernel config buffer size, subtract this value from the
    // available space to avoid exceeding limits during ELF loading
    // Conservative estimate based on observed overhead
    static constexpr uint32_t KERNEL_BINARY_OVERHEAD_ESTIMATE = 320;  // Bytes
    // NCRISC IRAM size on Wormhole as defined in dev_mem_map.h
    static constexpr uint32_t WORMHOLE_NCRISC_IRAM_SIZE = 16 * 1024;  // Bytes

    void SetUp() override {
        UnitMeshCQFixture::SetUp();

        // Kernel size config tests are only supported on Wormhole and Blackhole
        skip_if_unsupported_arch();

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
        TT_ASSERT(!devices_.empty() && !devices_[0]->get_devices().empty(), "No devices available for testing");

        auto single_device = devices_[0]->get_devices()[0];
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

// This test suite is used to create devices to force a non-default worker_l1_size
// We set the worker_l1_size to the minimum possible value for the architecture
class KernelSizeTestCustomWorkerL1SizeMin : public KernelSizeTest {
protected:
    // Clamp Wormhole at 1280 B. Dropping below pushes the synthetic pgm_dispatch_perf
    // kernels over MEM_MAX_KERNEL_SIZE (1432 KB per BRISC/TRISC) and ElfFile::LoadImage
    // aborts with "phdr overflow". Blackhole's IRAM budget is larger, so 1024 B still works there
    static constexpr uint32_t WORMHOLE_MIN_WORKER_L1_SIZE = 1280;
    static constexpr uint32_t BLACKHOLE_MIN_WORKER_L1_SIZE = 1024;
    uint32_t custom_worker_l1_size_min_{};
    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        // Kernel size config tests are only supported on Wormhole and Blackhole
        skip_if_unsupported_arch();
        custom_worker_l1_size_min_ =
            arch_ == tt::ARCH::WORMHOLE_B0 ? WORMHOLE_MIN_WORKER_L1_SIZE : BLACKHOLE_MIN_WORKER_L1_SIZE;
        create_devices(DEFAULT_TRACE_REGION_SIZE, custom_worker_l1_size_min_);
        compute_memory_layout();
        log_l1_memory_layout();
    }
};

// This test suite is used to create devices to force a non-default worker_l1_size
// We set the worker_l1_size to the maximum possible value for the architecture
class KernelSizeTestCustomWorkerL1SizeMax : public KernelSizeTest {
protected:
    // Hardcoded floor (~29.9 KiB) observed from ProgramImpl::finalize_program_offsets()
    // TT_FATAL when shrinking worker_l1_size. Update if binaries change
    static constexpr uint32_t MARGIN = 5 * 1024;
    static constexpr uint32_t WORMHOLE_KERNEL_CONFIG_BUFFER_SIZE_MIN = 29904 + MARGIN;
    // Blackhole's sizes are slightly larger (~31.6 KB); same rationale as Wormhole
    static constexpr uint32_t BLACKHOLE_KERNEL_CONFIG_BUFFER_SIZE_MIN = 31568 + MARGIN;
    uint32_t custom_worker_l1_size_max_{};

    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        // Kernel size config tests are only supported on Wormhole and Blackhole
        skip_if_unsupported_arch();
        custom_worker_l1_size_max_ =
            max_worker_l1_size_ - (arch_ == tt::ARCH::WORMHOLE_B0 ? WORMHOLE_KERNEL_CONFIG_BUFFER_SIZE_MIN
                                                                  : BLACKHOLE_KERNEL_CONFIG_BUFFER_SIZE_MIN);
        create_devices(DEFAULT_TRACE_REGION_SIZE, custom_worker_l1_size_max_);
        compute_memory_layout();
        log_l1_memory_layout();
    }
};

// Helper for RAII kernel nullification
class ScopedNullifyKernels {
public:
    ScopedNullifyKernels(bool enable = true) {
        tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(enable);
    }
    ~ScopedNullifyKernels() { tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false); }

    ScopedNullifyKernels(const ScopedNullifyKernels&) = delete;
    ScopedNullifyKernels& operator=(const ScopedNullifyKernels&) = delete;
};

using namespace tt::tt_metal;

// Test 1: Verify default kernel config buffer works
TEST_F(KernelSizeTest, DefaultKernelConfigBuffer) {
    log_info(LogTest, "Testing default kernel config buffer");

    // Verify basic kernel creation and execution with default kernel config buffer sizing
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

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

// Test 2: Verify kernel config buffer respects default worker_l1_size parameter
TEST_F(KernelSizeTest, WorkerL1SizeParameterControlDefault) {
    log_info(LogTest, "Testing default worker_l1_size parameter control");
    log_info(LogTest, "Worker L1 size: {} B", DEFAULT_L1_SMALL_SIZE);
    log_l1_memory_layout();

    EXPECT_LE(actual_kernel_config_size_ + unreserved_size_, max_worker_l1_size_);
}

// Test 3: Actually test custom worker_l1_size parameter
// Initialize a device with a custom worker_l1_size and
// Verify that the kernel config buffer size = to the max available L1 space - custom worker_l1_size
TEST_F(KernelSizeTestCustomWorkerL1SizeMin, WorkerL1SizeParameterControlCustomMin) {
    log_info(LogTest, "Testing custom minimum worker_l1_size parameter control");
    log_l1_memory_layout();

    EXPECT_EQ(actual_kernel_config_size_, max_worker_l1_size_ - custom_worker_l1_size_min_);
    EXPECT_EQ(unreserved_size_, custom_worker_l1_size_min_);
}

// Test 4: Actually test custom worker_l1_size parameter
// Initialize a device with a custom worker_l1_size and
// Verify that the kernel config buffer size = to the max available L1 space - custom worker_l1_size
TEST_F(KernelSizeTestCustomWorkerL1SizeMax, WorkerL1SizeParameterControlCustomMax) {
    log_info(LogTest, "Testing custom maximum worker_l1_size parameter control");
    log_l1_memory_layout();

    EXPECT_EQ(actual_kernel_config_size_, max_worker_l1_size_ - custom_worker_l1_size_max_);
    EXPECT_EQ(unreserved_size_, custom_worker_l1_size_max_);
}

// Test 5: Test kernels with different sizes and also edge case of 65536 B (64KB)
// Verify system handles this boundary condition correctly
TEST_F(KernelSizeTestCustomWorkerL1SizeMin, KernelSizesBoundaryConditions) {
    ScopedNullifyKernels nullify_kernels;

    log_info(LogTest, "Testing different kernel sizes and edge case of {} bytes", PACKED_WRITE_LARGE_MAX_CHUNK_SIZE);

    // Test sizes around critical boundaries
    const std::vector<uint32_t> kernel_sizes = {
        1_KB,                                                          // 1 KB
        32_KB,                                                         // 32 KB
        PACKED_WRITE_LARGE_MAX_CHUNK_SIZE - 1,                         // 64KB - 1
        PACKED_WRITE_LARGE_MAX_CHUNK_SIZE,                             // Exactly 64KB
        PACKED_WRITE_LARGE_MAX_CHUNK_SIZE + 1,                         // 64KB + 1
        actual_kernel_config_size_ / 2,                                // Half of kernel config buffer size
        actual_kernel_config_size_ - KERNEL_BINARY_OVERHEAD_ESTIMATE,  // Near limit of kernel config buffer size
    };

    // Select a kernel size and if it's within kernel config buffer size and
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
        std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

        for (const auto& device : devices_) {
            distributed::MeshWorkload workload;
            Program program;

            CoreRange cr({0, 0}, {0, 0});
            CoreRangeSet cr_set({cr});

            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/arbiter_hang.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

            workload.add_program(device_range_, std::move(program));
            EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
            distributed::Finish(device->mesh_command_queue());
        }
    }
}

// Test 6: Verify multiple kernels filling up kernel config buffer on multiple separate cores simultaneously
TEST_F(KernelSizeTestCustomWorkerL1SizeMin, MultipleKernelsAggregateSize) {
    ScopedNullifyKernels nullify_kernels;

    log_info(LogTest, "Testing multiple kernels filling up kernel config buffer");

    // Use near limit of kernel config buffer size to test max utilization per core
    const uint32_t kernel_size = actual_kernel_config_size_ - KERNEL_BINARY_OVERHEAD_ESTIMATE;
    std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        for (uint32_t i = 0; i < NUM_KERNELS_AGGREGATE_TEST; i++) {
            // Assign each large kernel to a different core to verify per-core limits
            CoreRange cr({i, 0}, {i, 0});
            CoreRangeSet cr_set({cr});

            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                cr_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

            log_info(LogTest, "Created kernel {} with size {} B", i, kernel_size);
        }

        workload.add_program(device_range_, std::move(program));
        EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
        distributed::Finish(device->mesh_command_queue());
    }
}

// Test 7: Validate that kernel config buffer is correctly partitioned among all 5 RISC-V
// processors (BRISC, NCRISC, TRISC0/1/2) running concurrently on the same core
// Architecture-specific constraints:
//   Wormhole: NCRISC limited to 16 KB IRAM; remaining space split among 4 other processors
//   Blackhole: All 5 processors share kernel config buffer equally (no NCRISC cap)
TEST_F(KernelSizeTestCustomWorkerL1SizeMin, AllRISCVProcessorsWithLargeKernels) {
    ScopedNullifyKernels nullify_kernels;

    log_info(LogTest, "Testing all five RISC-V processors with large kernels");

    uint32_t kernel_size;
    uint32_t ncrisc_kernel_size;

    // Wormhole: reserve the 16 KiB NCRISC IRAM first, then split the leftover evenly across
    // BRISC/TRISC. Subtract KERNEL_BINARY_OVERHEAD_ESTIMATE to keep ELF headers/alignment under the IRAM cap.
    // Blackhole: no NCRISC cap, so divide evenly across all processors
    if (this->arch_ == tt::ARCH::WORMHOLE_B0) {
        ncrisc_kernel_size = WORMHOLE_NCRISC_IRAM_SIZE - KERNEL_BINARY_OVERHEAD_ESTIMATE;
        // Remaining space distributed among the other 4 processors (BRISC + 3 TRISCs)
        constexpr uint32_t NUM_NON_NCRISC_PROCESSORS = NUM_RISCV_PROCESSORS - 1;

        TT_ASSERT(
            actual_kernel_config_size_ >
                ncrisc_kernel_size + (NUM_NON_NCRISC_PROCESSORS * KERNEL_BINARY_OVERHEAD_ESTIMATE),
            "Insufficient kernel config buffer space for test: {} B available, need at least {} B",
            actual_kernel_config_size_,
            ncrisc_kernel_size + (NUM_NON_NCRISC_PROCESSORS * KERNEL_BINARY_OVERHEAD_ESTIMATE));

        kernel_size = ((actual_kernel_config_size_ - ncrisc_kernel_size) / NUM_NON_NCRISC_PROCESSORS) -
                      KERNEL_BINARY_OVERHEAD_ESTIMATE;
    } else {
        TT_ASSERT(
            actual_kernel_config_size_ > NUM_RISCV_PROCESSORS * KERNEL_BINARY_OVERHEAD_ESTIMATE,
            "Insufficient kernel config buffer space for test");

        kernel_size = ((actual_kernel_config_size_) / NUM_RISCV_PROCESSORS) - KERNEL_BINARY_OVERHEAD_ESTIMATE;
        ncrisc_kernel_size = kernel_size;
    }

    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};
    std::map<std::string, std::string> ncrisc_defines = {{"KERNEL_BYTES", std::to_string(ncrisc_kernel_size)}};

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        // BRISC
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

        // NCRISC
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = ncrisc_defines});

        // TRISC0, TRISC1, TRISC2
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
            cr_set,
            ComputeConfig{.compile_args = {}, .defines = defines});

        workload.add_program(device_range_, std::move(program));
        EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false));
        distributed::Finish(device->mesh_command_queue());
    }
}

// Test 8: Verify that kernels exceeding kernel config buffer size are rejected
// This test intentionally creates an oversized kernel (kernel_config_size + 32B) and expects
// the dispatch system to throw TT_FATAL in ProgramImpl::finalize_program_offsets()
TEST_F(KernelSizeTest, ExceedsKernelConfigBufferSize) {
    ScopedNullifyKernels nullify_kernels;

    log_info(LogTest, "Testing kernel size exceeding kernel config buffer size");

    // Kernel config buffer size plus padding for rejection test
    // The kernel size is typically greater than what's specified by KERNEL_BYTES due to ELF headers, alignment, etc.
    // So even adding a small amount of padding can reject the kernel
    const uint32_t kernel_size = actual_kernel_config_size_ + OVERSIZED_KERNEL_PADDING;

    log_info(LogTest, "Kernel config buffer size: {} B", actual_kernel_config_size_);
    log_info(LogTest, "Kernel size: {} B (exceeds by {} B)", kernel_size, kernel_size - actual_kernel_config_size_);
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});
    std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(kernel_size)}};

    for (const auto& device : devices_) {
        distributed::MeshWorkload workload;
        Program program;

        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
            cr_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

        workload.add_program(device_range_, std::move(program));
        // Expect this to throw because kernel size exceeds kernel config buffer
        EXPECT_THROW(
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload, false), std::runtime_error);

        log_info(LogTest, "Correctly rejected oversized kernel configuration");
    }
}

}  // namespace kernel_size_tests
}  // namespace tt::tt_metal
