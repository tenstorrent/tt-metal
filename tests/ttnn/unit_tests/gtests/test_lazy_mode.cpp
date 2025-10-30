// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/experimental/jit/lazy_mode.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/operations/rand/rand.hpp"

namespace ttnn {
namespace test {

class LazyModeFixture : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        TTNNFixtureWithDevice::SetUp();
        // Enable lazy mode by setting environment variable
        setenv("TTNN_LAZY_MODE", "1", 1);
    }

    void TearDown() override {
        // Disable lazy mode
        unsetenv("TTNN_LAZY_MODE");

        TTNNFixtureWithDevice::TearDown();
    }

public:
    // Constructor that sets L1_SMALL size to 16KB for conv2d operations
    LazyModeFixture() : TTNNFixtureWithDevice(DEFAULT_TRACE_REGION_SIZE, 16384) {}
};

TEST_F(LazyModeFixture, LazyTensorCreation) {
    log_info(tt::LogTest, "==== Starting LazyTensorCreation test ====");

    // Verify lazy mode is enabled
    ttnn::experimental::jit::enable();
    ASSERT_TRUE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be enabled";

    // 1. From materialized tensor
    auto random =
        ttnn::rand(ttnn::Shape({32, 32}), *device_, DataType::BFLOAT16, Layout::TILE, ttnn::types::DRAM_MEMORY_CONFIG);

    ASSERT_EQ(random.lazy().materialized_tensor().logical_shape(), ttnn::Shape({32, 32}))
        << "Lazy tensor should have the same shape as the materialized tensor";
    ASSERT_EQ(random.lazy().materialized_tensor().dtype(), DataType::BFLOAT16)
        << "Lazy tensor should have the same dtype as the materialized tensor";
    ASSERT_EQ(random.lazy().materialized_tensor().layout(), Layout::TILE)
        << "Lazy tensor should have the same layout as the materialized tensor";
    ASSERT_EQ(random.lazy().materialized_tensor().memory_config(), ttnn::types::DRAM_MEMORY_CONFIG)
        << "Lazy tensor should have the same memory config as the materialized tensor";
    ASSERT_EQ(random.lazy().materialized_tensor().device(), device_)
        << "Lazy tensor should have the same device as the materialized tensor";
    ASSERT_EQ(random.lazy().materialized_tensor().mesh_buffer(), random.mesh_buffer())
        << "Lazy tensor should have the same mesh buffer as the materialized tensor";
    ASSERT_EQ(random.lazy().materialized_tensor().mesh_buffer()->device(), device_)
        << "Lazy tensor should have the same mesh buffer device as the materialized tensor";

    ASSERT_EQ(random.lazy().state(), experimental::jit::LazyTensorState::MATERIALIZED)
        << "Lazy tensor should be in materialized state";
    ASSERT_EQ(random.lazy().tensor_spec(), random.tensor_spec())
        << "Lazy tensor should have the same tensor spec as the materialized tensor";
    ASSERT_EQ(random.lazy().op().get(), nullptr)
        << "Lazy tensor created from materialized tensor should have no operation";
    ASSERT_TRUE(random.lazy().is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_EQ(random.lazy().op_inputs().size(), 0) << "Lazy tensor should have no op inputs";
    ASSERT_EQ(random.lazy().siblings().size(), 0) << "Lazy tensor should have no siblings";
    ASSERT_EQ(random.lazy().materialized_tensors().size(), 1) << "Lazy tensor should have one materialized tensor";

    // Note: you can't create a lazy operation without op and it's inputs because it won't be possible to materialize it
    // later.
}

// Test: Simple unary operations in lazy mode with verification
TEST_F(LazyModeFixture, SimpleUnaryOperationsLazy) {
    log_info(tt::LogTest, "==== Starting SimpleUnaryOperationsLazy test ====");

    // Verify lazy mode is enabled
    ttnn::experimental::jit::enable();
    ASSERT_TRUE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be enabled";

    // Create input tensor
    ttnn::Shape shape({32, 32});

    // Create a tensor of 1, 0, -1 values in a pattern
    auto spec = TensorSpec(
        shape,
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(ttnn::TILE_LAYOUT),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)));
    std::vector<bfloat16> data(shape.volume(), 1.0f);
    for (int i = 0; i < shape.volume(); i++) {
        if (i % 3 == 0) {
            data[i] = -1.0f;
        }
        if (i % 3 == 1) {
            data[i] = 0.0f;
        }
    }
    auto input_tensor = Tensor::from_vector(data, spec, device_);

    ttnn::set_printoptions(TensorPrintProfile::Full);
    // log_info(tt::LogTest, "Input tensor: {}", input_tensor.write_to_string());

    log_info(tt::LogTest, "Created input tensor with shape [{}, {}]", shape[0], shape[1]);

    // Apply unary operations - these should be captured lazily
    log_info(tt::LogTest, "Applying unary operations in lazy mode...");
    const auto relu_output = ttnn::relu(input_tensor);
    const auto exp_output = ttnn::exp(relu_output);
    const auto sqrt_output = ttnn::sqrt(exp_output);

    // // Check that we have captured 3 operations
    // ASSERT_EQ(context.size(), 3) << "Expected 3 nodes in lazy graph";
    // log_info(tt::LogTest, "Lazy graph size: {} nodes", context.size());

    // // Now execute the lazy graph
    // log_info(tt::LogTest, "Executing lazy graph...");
    // context.execute_node(sqrt_output.producer_node());

    // // Get the materialized tensor and bring it to host for comparison
    // const auto materialized_output = context.get_materialized_tensor(sqrt_output);
    // const auto lazy_result = ttnn::from_device(materialized_output);

    // // Clear the lazy graph and disable lazy mode
    // context.clear();
    // ttnn::lazy_mode::disable();
    // ASSERT_FALSE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be disabled";

    // // Run the same operations in eager mode
    // log_info(tt::LogTest, "Running same operations in eager mode for verification...");
    // // Create the same input data for eager mode
    // const auto input_tensor_eager = Tensor::from_vector(data, spec, device_);
    // const auto relu_eager = ttnn::relu(input_tensor_eager);
    // const auto exp_eager = ttnn::exp(relu_eager);
    // const auto sqrt_eager = ttnn::sqrt(exp_eager);
    // const auto eager_result = ttnn::from_device(sqrt_eager);

    // // Compare results
    // log_info(tt::LogTest, "Comparing lazy and eager results...");
    // ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
    //     << "Lazy and eager execution results should match";

    // log_info(tt::LogTest, "✓ Lazy and eager results match!");
    // log_info(tt::LogTest, "==== Finished SimpleUnaryOperationsLazy test ====");
}

// // Test: Binary operations in lazy mode with verification
// TEST_F(LazyModeFixture, BinaryOperationsLazy) {
//     auto& device = *device_;
//     auto& context = ttnn::experimental::jit::Context::instance();

//     log_info(tt::LogTest, "==== Starting BinaryOperationsLazy test ====");

//     // Verify lazy mode is enabled
//     ttnn::lazy_mode::enable();
//     ASSERT_TRUE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be enabled";

//     // Create input tensors
//     ttnn::Shape shape({32, 64});
//     const auto input1 = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto input2 = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

//     log_info(tt::LogTest, "Created input tensors with shape [{}, {}]", shape[0], shape[1]);

//     // Apply binary operations - these should be captured lazily
//     log_info(tt::LogTest, "Applying binary operations in lazy mode...");
//     const auto add_output = ttnn::add(input1, input2);
//     const auto mul_output = ttnn::multiply(add_output, input1);

//     // Check that we have captured 2 operations
//     ASSERT_EQ(context.size(), 2) << "Expected 2 nodes in lazy graph";
//     log_info(tt::LogTest, "Lazy graph size: {} nodes", context.size());

//     // Now execute the lazy graph
//     log_info(tt::LogTest, "Executing lazy graph...");
//     context.execute_node(mul_output.producer_node());
//     auto mul_output_materialized = context.get_materialized_tensor(mul_output);

//     // Get lazy result to host for comparison
//     const auto lazy_result = ttnn::from_device(mul_output_materialized);

//     // Clear the lazy graph and disable lazy mode
//     context.clear();
//     ttnn::lazy_mode::disable();

//     // Run the same operations in eager mode
//     log_info(tt::LogTest, "Running same operations in eager mode for verification...");
//     const auto input1_eager = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto input2_eager = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto add_eager = ttnn::add(input1_eager, input2_eager);
//     const auto mul_eager = ttnn::multiply(add_eager, input1_eager);
//     const auto eager_result = ttnn::from_device(mul_eager);

//     // Compare results
//     log_info(tt::LogTest, "Comparing lazy and eager results...");
//     ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
//         << "Lazy and eager execution results should match";

//     log_info(tt::LogTest, "✓ Lazy and eager results match!");
//     log_info(tt::LogTest, "==== Finished BinaryOperationsLazy test ====");
// }

// // Test: Mixed unary and binary operations in lazy mode with verification
// TEST_F(LazyModeFixture, MixedOperationsLazy) {
//     auto& device = *device_;
//     auto& context = ttnn::experimental::jit::Context::instance();

//     log_info(tt::LogTest, "==== Starting MixedOperationsLazy test ====");

//     // Verify lazy mode is enabled
//     ttnn::lazy_mode::enable();
//     ASSERT_TRUE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be enabled";

//     // Create input tensors
//     ttnn::Shape shape({32, 32});
//     const auto input1 = ttnn::full(shape, 1.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto input2 = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

//     log_info(tt::LogTest, "Created input tensors with shape [{}, {}]", shape[0], shape[1]);

//     // Apply mixed operations - these should be captured lazily
//     log_info(tt::LogTest, "Applying mixed operations in lazy mode...");
//     const auto add_result = ttnn::add(input1, input2);            // 1 + 2 = 3
//     const auto relu_result = ttnn::relu(add_result);              // relu(3) = 3
//     const auto mul_result = ttnn::multiply(relu_result, input2);  // 3 * 2 = 6
//     const auto sqrt_result = ttnn::sqrt(mul_result);              // sqrt(6)
//     const auto sub_result = ttnn::subtract(sqrt_result, input1);  // sqrt(6) - 1

//     // Check that we have captured 5 operations
//     ASSERT_EQ(context.size(), 5) << "Expected 5 nodes in lazy graph";
//     log_info(tt::LogTest, "Lazy graph size: {} nodes", context.size());

//     // List all operations in the graph
//     log_info(tt::LogTest, "Operations in lazy graph:");
//     const auto& nodes = context.get_all_nodes();
//     for (size_t i = 0; i < nodes.size(); ++i) {
//         log_info(tt::LogTest, "  Node {}: {}", nodes[i].id(), nodes[i].operation_name());
//     }

//     // Now execute the lazy graph
//     log_info(tt::LogTest, "Executing lazy graph with {} operations...", context.size());
//     context.execute_node(sub_result.producer_node());
//     auto sub_result_materialized = context.get_materialized_tensor(sub_result);

//     // Get lazy result to host for comparison
//     const auto lazy_result = ttnn::from_device(sub_result_materialized);

//     // Clear the lazy graph and disable lazy mode
//     context.clear();
//     ttnn::lazy_mode::disable();

//     // Run the same operations in eager mode
//     log_info(tt::LogTest, "Running same operations in eager mode for verification...");
//     const auto input1_eager = ttnn::full(shape, 1.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto input2_eager = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto add_eager = ttnn::add(input1_eager, input2_eager);
//     const auto relu_eager = ttnn::relu(add_eager);
//     const auto mul_eager = ttnn::multiply(relu_eager, input2_eager);
//     const auto sqrt_eager = ttnn::sqrt(mul_eager);
//     const auto sub_eager = ttnn::subtract(sqrt_eager, input1_eager);
//     const auto eager_result = ttnn::from_device(sub_eager);

//     // Compare results
//     log_info(tt::LogTest, "Comparing lazy and eager results...");
//     ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
//         << "Lazy and eager execution results should match";

//     log_info(tt::LogTest, "✓ Lazy and eager results match!");
//     log_info(tt::LogTest, "==== Finished MixedOperationsLazy test ====");
// }

// // Test: Verify execution order is correct with diamond-shaped graph
// TEST_F(LazyModeFixture, ExecutionOrderCorrect) {
//     auto& device = *device_;
//     auto& context = ttnn::experimental::jit::Context::instance();

//     log_info(tt::LogTest, "==== Starting ExecutionOrderCorrect test ====");

//     // Verify lazy mode is enabled
//     ttnn::lazy_mode::enable();
//     ASSERT_TRUE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be enabled";

//     // Create a diamond-shaped dependency graph
//     ttnn::Shape shape({32, 32});
//     const auto input = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

//     log_info(tt::LogTest, "Building diamond-shaped computation graph...");
//     // Diamond structure:
//     //       input
//     //       /   \
//     //    relu   exp
//     //       \   /
//     //       add
//     const auto relu_branch = ttnn::relu(input);
//     const auto exp_branch = ttnn::exp(input);
//     const auto final_result = ttnn::add(relu_branch, exp_branch);

//     // Check that we have captured 3 operations
//     ASSERT_EQ(context.size(), 3) << "Expected 3 nodes in lazy graph";
//     log_info(tt::LogTest, "Lazy graph size: {} nodes", context.size());

//     // Now execute the lazy graph - should execute in topological order
//     log_info(tt::LogTest, "Executing lazy graph (watch for execution order in logs)...");
//     context.execute_node(final_result.producer_node());
//     auto final_result_materialized = context.get_materialized_tensor(final_result);

//     // Get lazy result to host for comparison
//     const auto lazy_result = ttnn::from_device(final_result_materialized);

//     // Clear the lazy graph and disable lazy mode
//     context.clear();
//     ttnn::lazy_mode::disable();

//     // Run the same operations in eager mode
//     log_info(tt::LogTest, "Running same operations in eager mode for verification...");
//     const auto input_eager = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto relu_eager = ttnn::relu(input_eager);
//     const auto exp_eager = ttnn::exp(input_eager);
//     const auto final_eager = ttnn::add(relu_eager, exp_eager);
//     const auto eager_result = ttnn::from_device(final_eager);

//     // Compare results
//     log_info(tt::LogTest, "Comparing lazy and eager results...");
//     ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
//         << "Lazy and eager execution results should match";

//     log_info(tt::LogTest, "✓ Lazy and eager results match!");
//     log_info(tt::LogTest, "==== Finished ExecutionOrderCorrect test ====");
// }

// // Test: Matmul with element-wise operations in lazy mode
// TEST_F(LazyModeFixture, MatmulWithElementwiseLazy) {
//     auto& device = *device_;
//     auto& context = ttnn::experimental::jit::Context::instance();

//     log_info(tt::LogTest, "==== Starting MatmulWithElementwiseLazy test ====");

//     // Verify lazy mode is enabled
//     ttnn::lazy_mode::enable();
//     ASSERT_TRUE(ttnn::lazy_mode::is_lazy_enabled()) << "Lazy mode should be enabled";

//     // Create input tensors for matmul
//     // Use smaller values to avoid numerical overflow in exp()
//     ttnn::Shape matmul_shape1({32, 64});
//     ttnn::Shape matmul_shape2({64, 32});
//     ttnn::Shape matmul_shape3({32, 32});

//     const auto matmul_input1 = ttnn::full(matmul_shape1, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto matmul_input2 = ttnn::full(matmul_shape2, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto add_input = ttnn::full(matmul_shape3, 0.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

//     log_info(
//         tt::LogTest,
//         "Created matmul input tensors with shapes [{}, {}] and [{}, {}]",
//         matmul_shape1[0],
//         matmul_shape1[1],
//         matmul_shape2[0],
//         matmul_shape2[1]);

//     // Apply matmul and element-wise operations - these should be captured lazily
//     log_info(tt::LogTest, "Applying matmul and element-wise operations in lazy mode...");
//     const auto matmul_result =
//         ttnn::matmul(matmul_input1, matmul_input2, /* transpose_a */ false, /* transpose_b */ false);
//     const auto add_result = ttnn::add(matmul_result, add_input);  // Add a constant
//     const auto relu_result = ttnn::relu(add_result);              // Apply relu
//     const auto matmul_result2 =
//         ttnn::matmul(relu_result, matmul_input1, /* transpose_a */ false, /* transpose_b */ false);
//     const auto exp_result = ttnn::exp(matmul_result2);                    // Apply exp
//     const auto final_result = ttnn::multiply(exp_result, matmul_input1);  // Multiply

//     // Check that we have captured the expected number of operations
//     // matmul: 2, add: 2, relu: 1, exp: 1 = 6 total
//     ASSERT_EQ(context.size(), 6) << "Expected 6 nodes in lazy graph";
//     log_info(tt::LogTest, "Lazy graph size: {} nodes", context.size());

//     // List all operations in the graph
//     log_info(tt::LogTest, "Operations in lazy graph:");
//     const auto& nodes = context.get_all_nodes();
//     for (size_t i = 0; i < nodes.size(); ++i) {
//         log_info(tt::LogTest, "  Node {}: {}", nodes[i].id(), nodes[i].operation_name());
//     }

//     // Now execute the lazy graph
//     log_info(tt::LogTest, "Executing lazy graph with {} operations...", context.size());
//     context.execute_node(final_result.producer_node());
//     auto final_result_materialized = context.get_materialized_tensor(final_result);

//     // Get lazy result to host for comparison
//     const auto lazy_result = ttnn::from_device(final_result_materialized);

//     // Clear the lazy graph and disable lazy mode
//     context.clear();
//     ttnn::lazy_mode::disable();

//     // Run the same operations in eager mode
//     log_info(tt::LogTest, "Running same operations in eager mode for verification...");
//     const auto matmul_input1_eager = ttnn::full(matmul_shape1, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto matmul_input2_eager = ttnn::full(matmul_shape2, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
//     const auto add_input_eager = ttnn::full(matmul_shape3, 0.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

//     const auto matmul_result_eager =
//         ttnn::matmul(matmul_input1_eager, matmul_input2_eager, /* transpose_a */ false, /* transpose_b */ false);
//     const auto add_result_eager = ttnn::add(matmul_result_eager, add_input_eager);
//     const auto relu_result_eager = ttnn::relu(add_result_eager);
//     const auto matmul_result2_eager =
//         ttnn::matmul(relu_result_eager, matmul_input1_eager, /* transpose_a */ false, /* transpose_b */ false);
//     const auto exp_result_eager = ttnn::exp(matmul_result2_eager);
//     const auto final_eager = ttnn::multiply(exp_result_eager, matmul_input1_eager);
//     const auto eager_result = ttnn::from_device(final_eager);

//     // Compare results
//     log_info(tt::LogTest, "Comparing lazy and eager results...");
//     ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
//         << "Lazy and eager execution results should match";

//     log_info(tt::LogTest, "✓ Lazy and eager results match!");
//     log_info(tt::LogTest, "==== Finished MatmulWithElementwiseLazy test ====");
// }

}  // namespace test
}  // namespace ttnn
