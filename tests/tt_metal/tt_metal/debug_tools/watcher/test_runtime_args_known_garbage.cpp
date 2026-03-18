// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <gtest/gtest.h>

#include <tt-metalium/allocator.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include "debug_tools_fixture.hpp"

namespace tt::tt_metal {

// Fixture for RTA/CRTA watcher bounds check tests
class RTATestFixture : public MeshWatcherFixture {
protected:
    const std::string rta_crta_kernel_path =
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/test_rta_crta_asserts.cpp";

    void SetUp() override {
        bool watcher_assert_disabled = MetalContext::instance().rtoptions().watcher_assert_disabled();
        if (watcher_assert_disabled) {
            GTEST_SKIP() << "This test requires watcher assert checks (RTA/CRTA) to be enabled";
        }
        MeshWatcherFixture::SetUp();
    }
    // Common test data
    const std::vector<uint32_t> default_rtas{0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD, 0xEEEEEEEE};
    const std::vector<uint32_t> default_crtas{0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, 0x77777777};

    distributed::MeshCoordinate zero_coord{0, 0};
    distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};

    // Helper: Create DM kernel (arch-aware)
    // On Quasar, launches kernel on all DMs with compile_args[0] = dm_select so only
    // the target DM executes; others exit early via get_my_thread_id() guard.
    // On BH/WH, launches on RISCV_0 (BRISC) only.
    // l1_scratch_addr: L1 address for writing RTA/CRTA results (passed as compile-time arg)
    KernelHandle CreateDMKernel(
        Program& program,
        const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
        uint32_t l1_scratch_addr,
        std::map<std::string, std::string> defines = {},
        uint32_t dm_select = 0) {
        const auto& hal = MetalContext::instance().hal();
        if (hal.get_arch() == tt::ARCH::QUASAR) {
            auto num_dms = hal.get_processor_types_count(
                HalProgrammableCoreType::TENSIX, static_cast<uint32_t>(HalProcessorClassType::DM));
            return experimental::quasar::CreateKernel(
                program,
                rta_crta_kernel_path,
                core_spec,
                experimental::quasar::QuasarDataMovementConfig{
                    .num_threads_per_cluster = num_dms,
                    .compile_args = {dm_select, l1_scratch_addr},
                    .defines = defines});
        }
        return CreateKernel(
            program,
            rta_crta_kernel_path,
            core_spec,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {l1_scratch_addr},
                .defines = defines});
    }

    // Helper: Create compute kernel (arch-aware)
    KernelHandle CreateComputeKernel(
        Program& program,
        const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
        uint32_t compute_scratch_addr,
        const std::map<std::string, std::string>& defines = {}) {
        const auto& hal = MetalContext::instance().hal();
        if (hal.get_arch() == tt::ARCH::QUASAR) {
            return experimental::quasar::CreateKernel(
                program,
                rta_crta_kernel_path,
                core_spec,
                experimental::quasar::QuasarComputeConfig{.compile_args = {compute_scratch_addr}, .defines = defines});
        }
        return CreateKernel(
            program,
            rta_crta_kernel_path,
            core_spec,
            ComputeConfig{.compile_args = {compute_scratch_addr}, .defines = defines});
    }

    // Helper: Read and validate RTA/CRTA results
    void ValidateArgResults(
        IDevice* device,
        const CoreCoord& core,
        uint32_t cb_addr,
        const std::vector<uint32_t>& expected_rtas,
        const std::vector<uint32_t>& expected_crtas) {
        const uint32_t total_size = (2 + expected_rtas.size() + expected_crtas.size()) * sizeof(uint32_t);
        std::vector<uint32_t> read_result;

        tt::tt_metal::detail::ReadFromDeviceL1(device, core, cb_addr, total_size, read_result);

        // Validate counts
        EXPECT_EQ(read_result[0], expected_rtas.size());
        EXPECT_EQ(read_result[1], expected_crtas.size());

        // Validate RTA payload
        for (uint32_t i = 0; i < expected_rtas.size(); i++) {
            EXPECT_EQ(read_result[i + 2], expected_rtas[i]);
        }

        // Validate CRTA payload
        for (uint32_t i = 0; i < expected_crtas.size(); i++) {
            EXPECT_EQ(read_result[i + expected_rtas.size() + 2], expected_crtas[i]);
        }
    }

    // Helper: Wait for watcher exception with specific message
    void ExpectWatcherException(
        const std::string& expected_message, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        std::string exception;
        auto start = std::chrono::steady_clock::now();

        while (std::chrono::steady_clock::now() - start < timeout) {
            exception = MetalContext::instance().watcher_server()->exception_message();
            if (!exception.empty()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ASSERT_FALSE(exception.empty()) << "Timeout (" << timeout.count() << "ms) waiting for watcher exception.\n"
                                        << "Expected: " << expected_message;

        EXPECT_TRUE(exception.find(expected_message) != std::string::npos)
            << "Watcher exception mismatch:\n"
            << "  Expected substring: " << expected_message << "\n"
            << "  Actual exception:   " << exception;
    }
};

// Test parameters structure
struct RTAAssertTestParams {
    std::string test_name;                  // For readable test names
    bool test_rta;                          // true = test RTA, false = test CRTA
    std::string expected_message;           // Expected watcher exception message
    HalProcessorClassType processor_class;  // DM or COMPUTE
};

// Parameterized test fixture for RTA/CRTA out-of-bounds assertions
class RTAAssertTest : public RTATestFixture, public ::testing::WithParamInterface<RTAAssertTestParams> {};

// Verifies watcher handles 0xFFFF sentinel pattern (used when no RTAs set):
// - Device interprets 0xBEEF#### as rta_count = 0 (validated on BRISC/TRISC0)
// - Kernels not accessing args run successfully with sentinel pattern
// - Kernels accessing args when count = 0 trigger "out of bounds" assert
// - Catches bug: kernel placed on cores but SetRuntimeArgs() only called for subset
TEST_F(RTATestFixture, SentinelPatternHandlingAndMissingRTADetection) {
    if (IsSlowDispatch()) {
        GTEST_SKIP() << "This test can only be run with fast dispatch mode";
    }

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];

    // Part A: Zero-arg kernels should see rta_count = 0, crta_count = 0
    {
        CoreRange core_range(CoreCoord(0, 0), CoreCoord(2, 2));
        CoreRangeSet core_range_set(std::vector{core_range});

        const uint32_t total_read_size = 2 * sizeof(uint32_t);
        uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);

        // Zero-init the L1 scratch space on all cores
        std::vector<uint32_t> zero_init(2, 0);
        for (const auto& core : core_range) {
            tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
        }

        distributed::MeshWorkload workload;
        Program program;

        CreateDMKernel(program, core_range_set, l1_unreserved_base);
        CreateComputeKernel(program, core_range_set, l1_unreserved_base);

        // No SetRuntimeArgs called - all cores have RTA/CRTA offset = 0xFFFF pattern
        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device, workload);

        std::vector<uint32_t> read_result;

        for (const auto& core : core_range) {
            // DM: verify rta_count = 0, crta_count = 0
            read_result.clear();
            tt::tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, total_read_size, read_result);
            EXPECT_EQ(read_result[0], 0);  // rta_count
            EXPECT_EQ(read_result[1], 0);  // crta_count

            // TRISC0: verify rta_count = 0, crta_count = 0
            // Note: Both DM and compute write to the same L1 address, so we're validating
            // the last writer (compute kernel) here
            read_result.clear();
            tt::tt_metal::detail::ReadFromDeviceL1(device, core, l1_unreserved_base, total_read_size, read_result);
            EXPECT_EQ(read_result[0], 0);  // rta_count
            EXPECT_EQ(read_result[1], 0);  // crta_count
        }
    }

    // Part B: Accessing unset RTAs on a subset of cores should trigger assert
    {
        CoreRange cores_with_rtas(CoreCoord(0, 0), CoreCoord(1, 1));
        CoreRange cores_without_rtas(CoreCoord(2, 2), CoreCoord(3, 3));
        CoreRangeSet core_range_set(std::vector{cores_with_rtas, cores_without_rtas});

        uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);

        distributed::MeshWorkload workload;
        Program program;

        auto kernel = CreateDMKernel(program, core_range_set, l1_unreserved_base, {{"MAX_RTA_IDX", "0"}});

        SetRuntimeArgs(program, kernel, cores_with_rtas, default_rtas);
        // cores_without_rtas has NO RTAs set - 0xBEEF#### pattern -> rta_count = 0 -> assert

        workload.add_program(device_range, std::move(program));
        RunProgram(mesh_device, workload);
        ExpectWatcherException("unique runtime arg index out of bounds");
    }
}

// Correct RTA/CRTA Dispatch and Payload Validation
// Validates happy path: RTAs/CRTAs are correctly dispatched and read back
// Tests multiple core ranges with different RTA sizes, common RTAs shared across
// cores, and re-dispatching on subsequent runs. Ensures arg counts and payload
// values match what was set via SetRuntimeArgs/SetCommonRuntimeArgs
TEST_F(RTATestFixture, CorrectArgDispatchAndPayloadValidation) {
    const auto& hal = MetalContext::instance().hal();
    const bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];

    // Quasar: single core; other archs: multi-core with two ranges
    CoreRange core_range1 =
        is_quasar ? CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)) : CoreRange(CoreCoord(0, 0), CoreCoord(1, 1));
    CoreRange core_range2(CoreCoord(2, 2), CoreCoord(3, 3));
    CoreRangeSet core_range_set =
        is_quasar ? CoreRangeSet(std::vector{core_range1}) : CoreRangeSet(std::vector{core_range1, core_range2});

    // Use l1_unreserved space instead of CB for scratch storage
    const uint32_t total_word_count = 2 + default_rtas.size() + default_crtas.size();
    uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);

    // Zero-init the L1 scratch space on all cores before running
    std::vector<uint32_t> zero_init(total_word_count, 0);
    for (const auto& core : core_range1) {
        tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
    }
    if (!is_quasar) {
        for (const auto& core : core_range2) {
            tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
        }
    }

    distributed::MeshWorkload workload;
    Program program;

    auto kernel = CreateDMKernel(program, core_range_set, l1_unreserved_base);

    // Range 1 RTAs
    SetRuntimeArgs(program, kernel, core_range1, default_rtas);

    // Range 2: different RTA size (non-Quasar only)
    std::vector<uint32_t> rtas_range2 = {0x1000, 0x1001, 0x1002};
    if (!is_quasar) {
        SetRuntimeArgs(program, kernel, core_range2, rtas_range2);
    }

    // Common args shared across all cores
    SetCommonRuntimeArgs(program, kernel, default_crtas);
    workload.add_program(device_range, std::move(program));
    RunProgram(mesh_device, workload);

    // Validate first run
    for (const auto& core : core_range1) {
        ValidateArgResults(device, core, l1_unreserved_base, default_rtas, default_crtas);
    }
    if (!is_quasar) {
        for (const auto& core : core_range2) {
            ValidateArgResults(device, core, l1_unreserved_base, rtas_range2, default_crtas);
        }
    }

    // Zero-init again before second run
    for (const auto& core : core_range1) {
        tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
    }
    if (!is_quasar) {
        for (const auto& core : core_range2) {
            tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_unreserved_base, zero_init);
        }
    }

    // Second run: call SetRuntimeArgs again to test re-dispatch with updated arg counts
    auto& program_from_workload = workload.get_programs().at(device_range);
    SetRuntimeArgs(program_from_workload, kernel, core_range1, default_rtas);
    if (!is_quasar) {
        SetRuntimeArgs(program_from_workload, kernel, core_range2, rtas_range2);
    }
    RunProgram(mesh_device, workload);

    // Validate second run
    for (const auto& core : core_range1) {
        ValidateArgResults(device, core, l1_unreserved_base, default_rtas, default_crtas);
    }
    if (!is_quasar) {
        for (const auto& core : core_range2) {
            ValidateArgResults(device, core, l1_unreserved_base, rtas_range2, default_crtas);
        }
    }
}

// Parameterized test: Out-of-Bounds Arg Access Detection
// Verifies watcher detects kernels accessing args beyond bounds (index >= count)
// Tests RTA/CRTA access on DM0 and TRISC0 (single processor per test)
TEST_P(RTAAssertTest, OutOfBoundsArgAccessDetection) {
    const auto& params = GetParam();
    const auto& hal = MetalContext::instance().hal();
    const bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;

    // Dispatch mode validation:
    // - Quasar: SD only (FD not yet available)
    // - Other archs: FD only
    if (IsSlowDispatch() && !is_quasar) {
        GTEST_SKIP() << "This test requires fast dispatch mode (except on Quasar)";
    }

    // Quasar: single core; other archs: 5x5 grid
    CoreRange core_range =
        is_quasar ? CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)) : CoreRange(CoreCoord(0, 0), CoreCoord(4, 4));
    CoreRangeSet core_range_set(std::vector{core_range});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];

    uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);

    distributed::MeshWorkload workload;
    Program program;

    std::map<std::string, std::string> defines;
    if (params.test_rta) {
        defines["MAX_RTA_IDX"] = std::to_string(default_rtas.size());  // Out of bounds
    } else {
        defines["MAX_CRTA_IDX"] = std::to_string(default_crtas.size());  // Out of bounds
    }

    // Create kernel based on processor class (arch-specific creation handled by helpers)
    KernelHandle kernel;
    switch (params.processor_class) {
        case HalProcessorClassType::DM:
            kernel = CreateDMKernel(program, core_range_set, l1_unreserved_base, defines);
            break;
        case HalProcessorClassType::COMPUTE:
            kernel = CreateComputeKernel(program, core_range_set, l1_unreserved_base, defines);
            break;
        default: TT_THROW("Unsupported processor class");
    }

    if (params.test_rta) {
        SetRuntimeArgs(program, kernel, core_range, default_rtas);
    } else {
        SetCommonRuntimeArgs(program, kernel, default_crtas);
    }
    workload.add_program(device_range, std::move(program));
    RunProgram(mesh_device, workload);

    ExpectWatcherException(params.expected_message);
}

// Multi-DM test: verifies all DMs running concurrently can trigger RTA bounds check on Quasar.
// Uses a sync barrier so all DMs hit the OOB access together, stress-testing
// watcher's first-writer-wins assert mechanism. Any DM can report the error.
TEST_F(RTATestFixture, QuasarMultiDMOutOfBoundsArgDetection) {
    const auto& hal = MetalContext::instance().hal();
    if (hal.get_arch() != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Test only applicable to Quasar";
    }

    CoreRange core_range(CoreCoord(0, 0), CoreCoord(0, 0));
    CoreRangeSet core_range_set(std::vector{core_range});

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];

    auto num_dms = hal.get_processor_types_count(
        HalProgrammableCoreType::TENSIX, static_cast<uint32_t>(HalProcessorClassType::DM));

    // L1 layout: [sync_counter (8 bytes)] [scratch space for RTA/CRTA results]
    uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    uint32_t l1_sync_addr = l1_unreserved_base;
    uint32_t l1_scratch_addr = l1_unreserved_base + 8;  // After 8-byte sync counter

    distributed::MeshWorkload workload;
    Program program;

    // All DMs sync then attempt OOB RTA access together
    std::map<std::string, std::string> defines = {
        {"MAX_RTA_IDX", std::to_string(default_rtas.size())}, {"TEST_MULTI_DM_RTA", "1"}};

    auto kernel = experimental::quasar::CreateKernel(
        program,
        rta_crta_kernel_path,
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = num_dms,
            .compile_args = {num_dms, l1_sync_addr, l1_scratch_addr},
            .defines = defines});

    SetRuntimeArgs(program, kernel, core_range, default_rtas);
    workload.add_program(device_range, std::move(program));

    // Zero out sync counter before launch
    std::vector<uint32_t> zero_sync = {0, 0};
    tt::tt_metal::detail::WriteToDeviceL1(device, core_range.start_coord, l1_sync_addr, zero_sync);

    RunProgram(mesh_device, workload);

    // Any DM can report the error; just verify the bounds-check message appears
    ExpectWatcherException("unique runtime arg index out of bounds");
}

INSTANTIATE_TEST_SUITE_P(
    WatcherArgAsserts,
    RTAAssertTest,
    ::testing::Values(
        // RTA out-of-bounds on DM0
        RTAAssertTestParams{"RTA_DM0", true, "unique runtime arg index out of bounds", HalProcessorClassType::DM},
        // RTA out-of-bounds on TRISC0
        RTAAssertTestParams{
            "RTA_TRISC0", true, "unique runtime arg index out of bounds", HalProcessorClassType::COMPUTE},
        // CRTA out-of-bounds on DM0
        RTAAssertTestParams{"CRTA_DM0", false, "common runtime arg index out of bounds", HalProcessorClassType::DM},
        // CRTA out-of-bounds on TRISC0
        RTAAssertTestParams{
            "CRTA_TRISC0", false, "common runtime arg index out of bounds", HalProcessorClassType::COMPUTE}),
    [](const ::testing::TestParamInfo<RTAAssertTestParams>& info) { return info.param.test_name; });
}  // namespace tt::tt_metal
