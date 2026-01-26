// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

constexpr uint32_t NUM_TEST_CASES = 1024;

// Test result values (for individual tests at buffer[2+test_id])
constexpr uint32_t TEST_PASS = 1;
constexpr uint32_t TEST_FAIL = 2;
constexpr uint32_t TEST_NOT_RUN = 0;

// Overall test status (at buffer[0])
enum class TestCompletionStatus : uint32_t {
    NOT_STARTED = 0,
    COMPLETED = 1,
    CRASHED = 2  // Will remain 0 if kernel crashes before completion
};

// Buffer layout:
// buffer[0] = TestCompletionStatus (written last by kernel)
// buffer[1] = num_tests_run (written last by kernel)
// buffer[2+test_id] = individual test results
constexpr uint32_t TEST_STATUS_OFFSET = 0;
constexpr uint32_t NUM_TESTS_RUN_OFFSET = 1;
constexpr uint32_t FIRST_TEST_RESULT_OFFSET = 2;
constexpr uint32_t BUFFER_SIZE_WORDS = FIRST_TEST_RESULT_OFFSET + NUM_TEST_CASES;

/**
 * Test harness for packed credits functionality
 *
 * This test launches a Metal kernel that runs various tests on the credit packing
 * implementation. Each test writes a result (pass/fail/not_run) to a specific index
 * in a buffer.
 */
void RunPackedCreditsTests(BaseFabricFixture* fixture) {
    const auto& devices = fixture->get_devices();
    if (devices.empty()) {
        GTEST_SKIP() << "No devices available";
    }

    auto device = devices[0];
    CoreCoord test_logical_core = {0, 0};

    // Allocate results buffer on host (status + count + test results)
    std::vector<uint32_t> results_host(BUFFER_SIZE_WORDS, TEST_NOT_RUN);

    // Allocate L1 buffer for test results
    uint32_t base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    uint32_t results_l1_address = base_addr;
    uint32_t results_size_bytes = BUFFER_SIZE_WORDS * sizeof(uint32_t);

    // Write initial buffer to L1 (all zeros = TEST_NOT_RUN)
    tt_metal::detail::WriteToDeviceL1(
        device->get_devices()[0],
        test_logical_core,
        results_l1_address,
        results_host,
        CoreType::WORKER);

    // Create program with test kernel
    auto program = tt_metal::CreateProgram();

    std::vector<uint32_t> compile_args = {
        results_l1_address,
        NUM_TEST_CASES
    };

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/fabric_router/kernels/test_packed_credits_kernel.cpp",
        {test_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_args});

    // Run the test kernel
    fixture->RunProgramNonblocking(device, program);
    fixture->WaitForSingleProgramDone(device, program);

    // Read results from L1
    std::vector<uint32_t> results_device;
    tt_metal::detail::ReadFromDeviceL1(
        device->get_devices()[0],
        test_logical_core,
        results_l1_address,
        results_size_bytes,
        results_device,
        CoreType::WORKER);

    // Check overall test completion status
    auto completion_status = static_cast<TestCompletionStatus>(results_device[TEST_STATUS_OFFSET]);
    uint32_t num_tests_run = results_device[NUM_TESTS_RUN_OFFSET];

    std::cout << "Packed Credits Test Results:\n";
    std::cout << "  Completion Status: " << (completion_status == TestCompletionStatus::COMPLETED ? "COMPLETED" : "NOT COMPLETED") << "\n";
    std::cout << "  Tests Run: " << num_tests_run << "\n";

    ASSERT_EQ(completion_status, TestCompletionStatus::COMPLETED)
        << "Kernel did not complete successfully (may have crashed)";

    // Validate individual test results
    bool all_tests_passed = true;
    std::vector<uint32_t> failed_tests;
    uint32_t num_passed = 0;

    for (uint32_t i = 0; i < num_tests_run; i++) {
        uint32_t result = results_device[FIRST_TEST_RESULT_OFFSET + i];
        if (result == TEST_PASS) {
            num_passed++;
        } else if (result == TEST_FAIL) {
            failed_tests.push_back(i);
            all_tests_passed = false;
        } else {
            // Result is TEST_NOT_RUN - this shouldn't happen for tests within num_tests_run
            std::cout << "  Warning: Test " << i << " reported as run but has status NOT_RUN\n";
        }
    }

    std::cout << "  Passed: " << num_passed << " / " << num_tests_run << "\n";
    std::cout << "  Failed: " << failed_tests.size() << " / " << num_tests_run << "\n";

    if (!failed_tests.empty()) {
        std::cout << "  Failed test IDs: ";
        for (size_t i = 0; i < failed_tests.size() && i < 10; i++) {
            std::cout << failed_tests[i] << " ";
        }
        if (failed_tests.size() > 10) {
            std::cout << "... (" << (failed_tests.size() - 10) << " more)";
        }
        std::cout << "\n";
    }

    EXPECT_TRUE(all_tests_passed) << "Some packed credit tests failed";
    EXPECT_GT(num_passed, 0) << "No tests ran successfully";

}

TEST_F(Fabric1DFixture, TestPackedCredits1D) {
    RunPackedCreditsTests(this);
}

TEST_F(Fabric2DFixture, TestPackedCredits2D) {
    RunPackedCreditsTests(this);
}

}  // namespace tt::tt_fabric::fabric_router_tests
