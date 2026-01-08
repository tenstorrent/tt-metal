// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/matmul_test.hpp"

class MatmulTestOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// ============================================================================
// Test Cases: Mixed Precision Matmul
// ============================================================================
// These tests validate the behavior of matmul with different data format
// combinations in circular buffers.
//
// Test methodology:
// 1. Create two 32x32 (single tile) test tensors
// 2. Run matmul with different CB format configurations
// 3. Check if the operation completes successfully
//
// Expected behavior:
// - BF16 @ BF16: Should work
// - FP32 @ FP32: Should work
// - BF16 @ FP32: Expected to fail (this is the bug we're testing)
// - FP32 @ BF16: Expected to fail (this is the bug we're testing)
// ============================================================================

namespace {

/**
 * Reference implementation of matmul using xtensor
 * C = A @ B for single tiles (32x32 matrices)
 */
xt::xarray<float> matmul_reference(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    // Strip the first two singleton dimensions and perform matrix multiplication
    // a, b have shape [1, 1, 32, 32]
    auto a_mat = xt::squeeze(a, {0, 1});  // [32, 32]
    auto b_mat = xt::squeeze(b, {0, 1});  // [32, 32]

    // Perform matrix multiplication: result [32, 32]
    auto result_mat = xt::linalg::dot(a_mat, b_mat);

    // Reshape back to [1, 1, 32, 32]
    return xt::reshape_view(result_mat, {1, 1, 32, 32});
}

// Helper to create a single tile tensor (32x32) filled with random values
ttnn::Tensor create_single_tile_tensor(float min_val, float max_val) {
    using namespace ttml;

    // Create shape [1, 1, 32, 32] - single tile
    std::vector<uint32_t> shape = {1, 1, 32, 32};

    // Generate random input data
    auto& rng = autograd::ctx().get_generator();
    xt::xarray<float> host_tensor = xt::empty<float>(shape);
    core::parallel_generate<float>(
        host_tensor,
        [min_val, max_val]() { return std::uniform_real_distribution<float>(min_val, max_val); },
        /* seed */ rng());

    // Convert to device tensor
    auto device_tensor = core::from_xtensor(host_tensor, &autograd::ctx().get_device());

    return device_tensor;
}

void run_matmul_test(
    ttml::metal::ops::matmul_test::device::TestCase test_case, const std::string& test_name, bool expect_success) {
    using namespace ttml;

    fmt::print("\n=== Running test: {} ===\n", test_name);

    // Generate random input data
    auto& rng = autograd::ctx().get_generator();
    std::vector<uint32_t> shape = {1, 1, 32, 32};

    xt::xarray<float> input_a_host = xt::empty<float>(shape);
    xt::xarray<float> input_b_host = xt::empty<float>(shape);

    core::parallel_generate<float>(
        input_a_host, []() { return std::uniform_real_distribution<float>(-2.0f, 2.0f); }, rng());
    core::parallel_generate<float>(
        input_b_host, []() { return std::uniform_real_distribution<float>(-2.0f, 2.0f); }, rng());

    // Create device tensors
    auto input_a = core::from_xtensor(input_a_host, &autograd::ctx().get_device());
    auto input_b = core::from_xtensor(input_b_host, &autograd::ctx().get_device());

    // Compute reference result
    auto reference_result = matmul_reference(input_a_host, input_b_host);

    try {
        // Run the matmul test operation
        auto output = metal::ops::matmul_test::MatmulTestOperation::invoke(input_a, input_b, test_case);

        // Read back output
        auto output_host = core::to_xtensor(output);

        // Print first row (32 elements) for debugging
        fmt::print("\nFirst row of output_host:\n");
        for (int i = 0; i < 32; ++i) {
            fmt::print("{:8.4f} ", output_host(0, 0, 0, i));
        }
        fmt::print("\n\nFirst row of reference_result:\n");
        for (int i = 0; i < 32; ++i) {
            fmt::print("{:8.4f} ", reference_result(0, 0, 0, i));
        }
        fmt::print("\n\n");

        // Set tolerance based on test case
        float rtol = 1e-2f;
        float atol = 1e-1f;

        // Check correctness
        bool is_correct = xt::allclose(output_host, reference_result, rtol, atol);
        if (expect_success) {
            if (is_correct) {
                fmt::print("✓ Test PASSED: {} produced correct results\n", test_name);
            } else {
                fmt::print("✗ Test FAILED: {} produced incorrect results (garbage output)\n", test_name);
                auto max_diff = xt::amax(xt::abs(output_host - reference_result))();
                fmt::print("  Max absolute difference: {}\n", max_diff);
                FAIL() << test_name << " produced incorrect results";
            }
        } else {
            if (is_correct) {
                fmt::print("✗ Test FAILED: {} produced correct results but was expected to fail\n", test_name);
                FAIL() << test_name << " should have produced garbage but gave correct results";
            } else {
                fmt::print("✓ Test PASSED: {} produced garbage as expected (BUG CONFIRMED)\n", test_name);
                auto max_diff = xt::amax(xt::abs(output_host - reference_result))();
                fmt::print("  Max absolute difference: {} (indicates garbage output)\n", max_diff);
            }
        }

        // Verify output is valid
        ASSERT_TRUE(output.is_allocated());

        // Read back and verify dimensions
        auto output_shape = output.logical_shape();
        ASSERT_EQ(output_shape[0], 1);
        ASSERT_EQ(output_shape[1], 1);
        ASSERT_EQ(output_shape[2], 32);
        ASSERT_EQ(output_shape[3], 32);

    } catch (const std::exception& e) {
        // Operation failed with exception
        fmt::print("✗ Test FAILED: {} threw exception\n", test_name);
        fmt::print("  Error message: {}\n", e.what());
        FAIL() << test_name << " threw exception: " << e.what();
    }
}

}  // namespace

// Test Case 1: BF16 @ BF16 - Should work
TEST_F(MatmulTestOpTest, Matmul_BF16_BF16) {
    run_matmul_test(
        ttml::metal::ops::matmul_test::device::TestCase::BF16_BF16,
        "BF16 @ BF16",
        /* expect_success = */ true);
}

// Test Case 2: FP32 @ FP32 - Should work
TEST_F(MatmulTestOpTest, Matmul_FP32_FP32) {
    run_matmul_test(
        ttml::metal::ops::matmul_test::device::TestCase::FP32_FP32,
        "FP32 @ FP32",
        /* expect_success = */ true);
}

// Test Case 3: BF16 @ FP32 - Expected to fail (demonstrating the bug)
TEST_F(MatmulTestOpTest, Matmul_BF16_FP32) {
    run_matmul_test(
        ttml::metal::ops::matmul_test::device::TestCase::BF16_FP32,
        "BF16 @ FP32",
        /* expect_success = */ false);
}

// Test Case 4: FP32 @ BF16 - Expected to fail (demonstrating the bug)
TEST_F(MatmulTestOpTest, Matmul_FP32_BF16) {
    run_matmul_test(
        ttml::metal::ops::matmul_test::device::TestCase::FP32_BF16,
        "FP32 @ BF16",
        /* expect_success = */ false);
}

// ============================================================================
// Summary Test: Run all cases and report results
// ============================================================================
TEST_F(MatmulTestOpTest, Matmul_AllCases_Summary) {
    fmt::print("\n");
    fmt::print("========================================\n");
    fmt::print("Matmul Mixed Precision Test Summary\n");
    fmt::print("========================================\n");
    fmt::print("\n");
    fmt::print("This test demonstrates the bug where matmul_tiles() fails\n");
    fmt::print("when circular buffers have different data formats.\n");
    fmt::print("\n");
    fmt::print("Expected results:\n");
    fmt::print("  [PASS] BF16 @ BF16 - Both operands same format\n");
    fmt::print("  [PASS] FP32 @ FP32 - Both operands same format\n");
    fmt::print("  [FAIL] BF16 @ FP32 - Mixed format (bug)\n");
    fmt::print("  [FAIL] FP32 @ BF16 - Mixed format (bug)\n");
    fmt::print("\n");

    struct TestResult {
        std::string name;
        bool passed;
        std::string message;
    };

    std::vector<TestResult> results;

    auto input_a = create_single_tile_tensor(-2.0f, 2.0f);
    auto input_b = create_single_tile_tensor(-2.0f, 2.0f);

    // Test 1: BF16 @ BF16
    try {
        auto output = ttml::metal::ops::matmul_test::MatmulTestOperation::invoke(
            input_a, input_b, ttml::metal::ops::matmul_test::device::TestCase::BF16_BF16);
        results.push_back({"BF16 @ BF16", true, "Success"});
    } catch (const std::exception& e) {
        results.push_back({"BF16 @ BF16", false, std::string("Failed: ") + e.what()});
    }

    // Test 2: FP32 @ FP32
    try {
        auto output = ttml::metal::ops::matmul_test::MatmulTestOperation::invoke(
            input_a, input_b, ttml::metal::ops::matmul_test::device::TestCase::FP32_FP32);
        results.push_back({"FP32 @ FP32", true, "Success"});
    } catch (const std::exception& e) {
        results.push_back({"FP32 @ FP32", false, std::string("Failed: ") + e.what()});
    }

    // Test 3: BF16 @ FP32 (expected to fail)
    try {
        auto output = ttml::metal::ops::matmul_test::MatmulTestOperation::invoke(
            input_a, input_b, ttml::metal::ops::matmul_test::device::TestCase::BF16_FP32);
        results.push_back({"BF16 @ FP32", true, "Unexpectedly succeeded (bug may be fixed?)"});
    } catch (const std::exception& e) {
        results.push_back({"BF16 @ FP32", false, "Failed as expected (confirms bug)"});
    }

    // Test 4: FP32 @ BF16 (expected to fail)
    try {
        auto output = ttml::metal::ops::matmul_test::MatmulTestOperation::invoke(
            input_a, input_b, ttml::metal::ops::matmul_test::device::TestCase::FP32_BF16);
        results.push_back({"FP32 @ BF16", true, "Unexpectedly succeeded (bug may be fixed?)"});
    } catch (const std::exception& e) {
        results.push_back({"FP32 @ BF16", false, "Failed as expected (confirms bug)"});
    }

    // Print results
    fmt::print("Results:\n");
    fmt::print("--------\n");
    for (const auto& result : results) {
        fmt::print("  {} {}: {}\n", result.passed ? "✓" : "✗", result.name, result.message);
    }
    fmt::print("\n");

    // Check if results match expectations
    bool bf16_bf16_ok = results[0].passed;
    bool fp32_fp32_ok = results[1].passed;
    bool bf16_fp32_failed = !results[2].passed;
    bool fp32_bf16_failed = !results[3].passed;

    if (bf16_bf16_ok && fp32_fp32_ok && bf16_fp32_failed && fp32_bf16_failed) {
        fmt::print("========================================\n");
        fmt::print("BUG CONFIRMED: Mixed precision matmul fails!\n");
        fmt::print("========================================\n");
    } else {
        fmt::print("========================================\n");
        fmt::print("Unexpected results - bug behavior changed?\n");
        fmt::print("========================================\n");
    }
}
