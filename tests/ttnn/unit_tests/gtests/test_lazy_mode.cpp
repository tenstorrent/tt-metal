// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/lazy_mode.hpp"
#include "ttnn/experimental/jit/context.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace test {

class LazyModeFixture : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        TTNNFixtureWithDevice::SetUp();
        // Enable lazy mode by setting environment variable
        setenv("TTNN_LAZY_MODE", "1", 1);

        // Clear any existing lazy graph
        auto& context = ttnn::experimental::jit::Context::instance();
        context.clear();
    }

    void TearDown() override {
        // Clear lazy graph before closing device
        auto& context = ttnn::experimental::jit::Context::instance();
        context.clear();

        // Disable lazy mode
        unsetenv("TTNN_LAZY_MODE");

        TTNNFixtureWithDevice::TearDown();
    }
};

// Test: Simple unary operations in lazy mode with verification
TEST_F(LazyModeFixture, SimpleUnaryOperationsLazy) {
    auto& device = *device_;
    auto& context = ttnn::experimental::jit::Context::instance();

    log_info(tt::LogTest, "==== Starting SimpleUnaryOperationsLazy test ====");

    // Verify lazy mode is enabled
    ASSERT_TRUE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be enabled";

    // Create input tensor
    ttnn::Shape shape({32, 64});
    const auto input_tensor = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    log_info(tt::LogTest, "Created input tensor with shape [{}, {}]", shape[0], shape[1]);

    // Apply unary operations - these should be captured lazily
    log_info(tt::LogTest, "Applying unary operations in lazy mode...");
    const auto relu_output = ttnn::relu(input_tensor);
    const auto exp_output = ttnn::exp(relu_output);
    const auto sqrt_output = ttnn::sqrt(exp_output);

    // Check that we have captured 3 operations
    ASSERT_EQ(context.size(), 3) << "Expected 3 nodes in lazy graph";
    log_info(tt::LogTest, "Lazy graph size: {} nodes", context.size());

    // Now execute the lazy graph
    log_info(tt::LogTest, "Executing lazy graph...");
    context.execute_node(sqrt_output.producer_node());

    // Get lazy result to host for comparison
    const auto lazy_result = ttnn::from_device(sqrt_output);

    // Clear the lazy graph and disable lazy mode
    context.clear();
    ttnn::lazy_mode::disable();
    ASSERT_FALSE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be disabled";

    // Run the same operations in eager mode
    log_info(tt::LogTest, "Running same operations in eager mode for verification...");
    const auto input_tensor_eager = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto relu_eager = ttnn::relu(input_tensor_eager);
    const auto exp_eager = ttnn::exp(relu_eager);
    const auto sqrt_eager = ttnn::sqrt(exp_eager);
    const auto eager_result = ttnn::from_device(sqrt_eager);

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished SimpleUnaryOperationsLazy test ====");
}

// Test: Binary operations in lazy mode with verification
TEST_F(LazyModeFixture, BinaryOperationsLazy) {
    auto& device = *device_;
    auto& context = ttnn::experimental::jit::Context::instance();

    log_info(tt::LogTest, "==== Starting BinaryOperationsLazy test ====");

    // Verify lazy mode is enabled
    ASSERT_TRUE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be enabled";

    // Create input tensors
    ttnn::Shape shape({32, 64});
    const auto input1 = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto input2 = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    log_info(tt::LogTest, "Created input tensors with shape [{}, {}]", shape[0], shape[1]);

    // Apply binary operations - these should be captured lazily
    log_info(tt::LogTest, "Applying binary operations in lazy mode...");
    const auto add_output = ttnn::add(input1, input2);
    const auto mul_output = ttnn::multiply(add_output, input1);

    // Check that we have captured 2 operations
    ASSERT_EQ(context.size(), 2) << "Expected 2 nodes in lazy graph";
    log_info(tt::LogTest, "Lazy graph size: {} nodes", context.size());

    // Now execute the lazy graph
    log_info(tt::LogTest, "Executing lazy graph...");
    context.execute_node(mul_output.producer_node());

    // Get lazy result to host for comparison
    const auto lazy_result = ttnn::from_device(mul_output);

    // Clear the lazy graph and disable lazy mode
    context.clear();
    ttnn::lazy_mode::disable();

    // Run the same operations in eager mode
    log_info(tt::LogTest, "Running same operations in eager mode for verification...");
    const auto input1_eager = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto input2_eager = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto add_eager = ttnn::add(input1_eager, input2_eager);
    const auto mul_eager = ttnn::multiply(add_eager, input1_eager);
    const auto eager_result = ttnn::from_device(mul_eager);

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished BinaryOperationsLazy test ====");
}

// Test: Mixed unary and binary operations in lazy mode with verification
TEST_F(LazyModeFixture, MixedOperationsLazy) {
    auto& device = *device_;
    auto& context = ttnn::experimental::jit::Context::instance();

    log_info(tt::LogTest, "==== Starting MixedOperationsLazy test ====");

    // Verify lazy mode is enabled
    ASSERT_TRUE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be enabled";

    // Create input tensors
    ttnn::Shape shape({32, 32});
    const auto input1 = ttnn::full(shape, 1.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto input2 = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    log_info(tt::LogTest, "Created input tensors with shape [{}, {}]", shape[0], shape[1]);

    // Apply mixed operations - these should be captured lazily
    log_info(tt::LogTest, "Applying mixed operations in lazy mode...");
    const auto add_result = ttnn::add(input1, input2);            // 1 + 2 = 3
    const auto relu_result = ttnn::relu(add_result);              // relu(3) = 3
    const auto mul_result = ttnn::multiply(relu_result, input2);  // 3 * 2 = 6
    const auto sqrt_result = ttnn::sqrt(mul_result);              // sqrt(6)
    const auto sub_result = ttnn::subtract(sqrt_result, input1);  // sqrt(6) - 1

    // Check that we have captured 5 operations
    ASSERT_EQ(context.size(), 5) << "Expected 5 nodes in lazy graph";
    log_info(tt::LogTest, "Lazy graph size: {} nodes", context.size());

    // List all operations in the graph
    log_info(tt::LogTest, "Operations in lazy graph:");
    const auto& nodes = context.get_all_nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        log_info(tt::LogTest, "  Node {}: {}", nodes[i].id(), nodes[i].operation_name());
    }

    // Now execute the lazy graph
    log_info(tt::LogTest, "Executing lazy graph with {} operations...", context.size());
    context.execute_node(sub_result.producer_node());

    // Get lazy result to host for comparison
    const auto lazy_result = ttnn::from_device(sub_result);

    // Clear the lazy graph and disable lazy mode
    context.clear();
    ttnn::lazy_mode::disable();

    // Run the same operations in eager mode
    log_info(tt::LogTest, "Running same operations in eager mode for verification...");
    const auto input1_eager = ttnn::full(shape, 1.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto input2_eager = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto add_eager = ttnn::add(input1_eager, input2_eager);
    const auto relu_eager = ttnn::relu(add_eager);
    const auto mul_eager = ttnn::multiply(relu_eager, input2_eager);
    const auto sqrt_eager = ttnn::sqrt(mul_eager);
    const auto sub_eager = ttnn::subtract(sqrt_eager, input1_eager);
    const auto eager_result = ttnn::from_device(sub_eager);

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished MixedOperationsLazy test ====");
}

// Test: Verify execution order is correct with diamond-shaped graph
TEST_F(LazyModeFixture, ExecutionOrderCorrect) {
    auto& device = *device_;
    auto& context = ttnn::experimental::jit::Context::instance();

    log_info(tt::LogTest, "==== Starting ExecutionOrderCorrect test ====");

    // Verify lazy mode is enabled
    ASSERT_TRUE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be enabled";

    // Create a diamond-shaped dependency graph
    ttnn::Shape shape({32, 32});
    const auto input = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    log_info(tt::LogTest, "Building diamond-shaped computation graph...");
    // Diamond structure:
    //       input
    //       /   \
    //    relu   exp
    //       \   /
    //       add
    const auto relu_branch = ttnn::relu(input);
    const auto exp_branch = ttnn::exp(input);
    const auto final_result = ttnn::add(relu_branch, exp_branch);

    // Check that we have captured 3 operations
    ASSERT_EQ(context.size(), 3) << "Expected 3 nodes in lazy graph";
    log_info(tt::LogTest, "Lazy graph size: {} nodes", context.size());

    // Now execute the lazy graph - should execute in topological order
    log_info(tt::LogTest, "Executing lazy graph (watch for execution order in logs)...");
    context.execute_node(final_result.producer_node());

    // Get lazy result to host for comparison
    const auto lazy_result = ttnn::from_device(final_result);

    // Clear the lazy graph and disable lazy mode
    context.clear();
    ttnn::lazy_mode::disable();

    // Run the same operations in eager mode
    log_info(tt::LogTest, "Running same operations in eager mode for verification...");
    const auto input_eager = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto relu_eager = ttnn::relu(input_eager);
    const auto exp_eager = ttnn::exp(input_eager);
    const auto final_eager = ttnn::add(relu_eager, exp_eager);
    const auto eager_result = ttnn::from_device(final_eager);

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished ExecutionOrderCorrect test ====");
}

}  // namespace test
}  // namespace ttnn
