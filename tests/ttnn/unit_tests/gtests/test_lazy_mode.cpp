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

    ASSERT_EQ(random.lazy()->materialized_tensor().logical_shape(), ttnn::Shape({32, 32}))
        << "Lazy tensor should have the same shape as the materialized tensor";
    ASSERT_EQ(random.lazy()->materialized_tensor().dtype(), DataType::BFLOAT16)
        << "Lazy tensor should have the same dtype as the materialized tensor";
    ASSERT_EQ(random.lazy()->materialized_tensor().layout(), Layout::TILE)
        << "Lazy tensor should have the same layout as the materialized tensor";
    ASSERT_EQ(random.lazy()->materialized_tensor().memory_config(), ttnn::types::DRAM_MEMORY_CONFIG)
        << "Lazy tensor should have the same memory config as the materialized tensor";
    ASSERT_EQ(random.lazy()->materialized_tensor().device(), device_)
        << "Lazy tensor should have the same device as the materialized tensor";
    ASSERT_EQ(random.lazy()->materialized_tensor().mesh_buffer(), random.mesh_buffer())
        << "Lazy tensor should have the same mesh buffer as the materialized tensor";
    ASSERT_EQ(random.lazy()->materialized_tensor().mesh_buffer()->device(), device_)
        << "Lazy tensor should have the same mesh buffer device as the materialized tensor";

    ASSERT_EQ(random.lazy()->state(), experimental::jit::LazyTensorState::MATERIALIZED)
        << "Lazy tensor should be in materialized state";
    ASSERT_EQ(random.lazy()->tensor_spec(), random.tensor_spec())
        << "Lazy tensor should have the same tensor spec as the materialized tensor";
    ASSERT_EQ(random.lazy()->op().get(), nullptr)
        << "Lazy tensor created from materialized tensor should have no operation";
    ASSERT_TRUE(random.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_EQ(random.lazy()->op_inputs().size(), 0) << "Lazy tensor should have no op inputs";
    ASSERT_EQ(random.lazy()->siblings().size(), 0) << "Lazy tensor should have no siblings";
    ASSERT_EQ(random.lazy()->materialized_tensors().size(), 1) << "Lazy tensor should have one materialized tensor";

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
    // TODO: creation ops should add op to lazy tensor
    auto input_tensor = Tensor::from_vector(data, spec, device_);

    log_info(tt::LogTest, "Created input tensor with shape [{}, {}]", shape[0], shape[1]);

    // Apply unary operations - these should be captured lazily
    log_info(tt::LogTest, "Applying unary operations in lazy mode...");
    auto relu_output = ttnn::relu(input_tensor);
    auto exp_output = ttnn::exp(relu_output);
    auto sqrt_output = ttnn::sqrt(exp_output);

    ASSERT_TRUE(input_tensor.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_FALSE(relu_output.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(exp_output.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(sqrt_output.lazy()->is_materialized()) << "Lazy tensor should not be materialized";

    log_info(
        tt::LogTest,
        "relu_out id: {}, exp_out id: {}, sqrt_out id: {}",
        relu_output.lazy()->id(),
        exp_output.lazy()->id(),
        sqrt_output.lazy()->id());

    sqrt_output.materialize();
    // TODO: materialize should be implicit in cpu() / to_string() / etc.
    auto lazy_result = sqrt_output.cpu();

    ASSERT_TRUE(sqrt_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(exp_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(relu_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input_tensor.lazy()->is_materialized()) << "Lazy tensor should be materialized";

    ttnn::experimental::jit::disable();
    ASSERT_FALSE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be disabled";

    // Run the same operations in eager mode
    log_info(tt::LogTest, "Running same operations in eager mode for verification...");
    auto input_tensor_eager = Tensor::from_vector(data, spec, device_);
    auto relu_eager = ttnn::relu(input_tensor_eager);
    auto exp_eager = ttnn::exp(relu_eager);
    auto sqrt_eager = ttnn::sqrt(exp_eager);
    auto eager_result = sqrt_eager.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished SimpleUnaryOperationsLazy test ====");
}

// Test: Binary operations in lazy mode with verification
TEST_F(LazyModeFixture, BinaryOperationsLazy) {
    log_info(tt::LogTest, "==== Starting BinaryOperationsLazy test ====");

    // Verify lazy mode is enabled
    ttnn::experimental::jit::enable();
    ASSERT_TRUE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be enabled";

    // Create input tensors
    ttnn::Shape shape({32, 64});
    const auto input1 = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    const auto input2 = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    log_info(tt::LogTest, "Created input tensors with shape [{}, {}]", shape[0], shape[1]);

    // Apply binary operations - these should be captured lazily
    log_info(tt::LogTest, "Applying binary operations in lazy mode...");
    auto add_output = ttnn::add(input1, input2);
    auto mul_output = ttnn::multiply(add_output, input1);

    ASSERT_TRUE(input1.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input2.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_FALSE(add_output.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(mul_output.lazy()->is_materialized()) << "Lazy tensor should not be materialized";

    log_info(tt::LogTest, "add_output id: {}, mul_output id: {}", add_output.lazy()->id(), mul_output.lazy()->id());

    mul_output.materialize();
    auto lazy_result = mul_output.cpu();

    ASSERT_TRUE(mul_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(add_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input1.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input2.lazy()->is_materialized()) << "Lazy tensor should be materialized";

    ttnn::experimental::jit::disable();
    ASSERT_FALSE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be disabled";

    // Run the same operations in eager mode
    log_info(tt::LogTest, "Running same operations in eager mode for verification...");
    const auto input1_eager = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    const auto input2_eager = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    const auto add_eager = ttnn::add(input1_eager, input2_eager);
    const auto mul_eager = ttnn::multiply(add_eager, input1_eager);
    const auto eager_result = mul_eager.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished BinaryOperationsLazy test ====");
}

// Test: Mixed unary and binary operations in lazy mode with verification
TEST_F(LazyModeFixture, MixedOperationsLazy) {
    log_info(tt::LogTest, "==== Starting MixedOperationsLazy test ====");

    // Verify lazy mode is enabled
    ttnn::experimental::jit::enable();
    ASSERT_TRUE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be enabled";

    // Create input tensors
    ttnn::Shape shape({32, 32});
    const auto input1 = ttnn::full(shape, 1.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    const auto input2 = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    log_info(tt::LogTest, "Created input tensors with shape [{}, {}]", shape[0], shape[1]);

    // Apply mixed operations - these should be captured lazily
    log_info(tt::LogTest, "Applying mixed operations in lazy mode...");
    auto add_result = ttnn::add(input1, input2);            // 1 + 2 = 3
    auto relu_result = ttnn::relu(add_result);              // relu(3) = 3
    auto mul_result = ttnn::multiply(relu_result, input2);  // 3 * 2 = 6
    auto sqrt_result = ttnn::sqrt(mul_result);              // sqrt(6)
    auto sub_result = ttnn::subtract(sqrt_result, input1);  // sqrt(6) - 1

    ASSERT_TRUE(input1.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input2.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_FALSE(add_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(relu_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(mul_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(sqrt_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(sub_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";

    log_info(
        tt::LogTest,
        "add_result id: {}, relu_result id: {}, mul_result id: {}, sqrt_result id: {}, sub_result id: {}",
        add_result.lazy()->id(),
        relu_result.lazy()->id(),
        mul_result.lazy()->id(),
        sqrt_result.lazy()->id(),
        sub_result.lazy()->id());

    sub_result.materialize();
    auto lazy_result = sub_result.cpu();

    ASSERT_TRUE(sub_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(sqrt_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(mul_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(relu_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(add_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input1.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input2.lazy()->is_materialized()) << "Lazy tensor should be materialized";

    ttnn::experimental::jit::disable();
    ASSERT_FALSE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be disabled";

    // Run the same operations in eager mode
    log_info(tt::LogTest, "Running same operations in eager mode for verification...");
    const auto input1_eager = ttnn::full(shape, 1.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    const auto input2_eager = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    const auto add_eager = ttnn::add(input1_eager, input2_eager);
    const auto relu_eager = ttnn::relu(add_eager);
    const auto mul_eager = ttnn::multiply(relu_eager, input2_eager);
    const auto sqrt_eager = ttnn::sqrt(mul_eager);
    const auto sub_eager = ttnn::subtract(sqrt_eager, input1_eager);
    const auto eager_result = sub_eager.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished MixedOperationsLazy test ====");
}

// Test: Verify execution order is correct with diamond-shaped graph
TEST_F(LazyModeFixture, ExecutionOrderCorrect) {
    log_info(tt::LogTest, "==== Starting ExecutionOrderCorrect test ====");

    // Verify lazy mode is enabled
    ttnn::experimental::jit::enable();
    ASSERT_TRUE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be enabled";

    // Create a diamond-shaped dependency graph
    ttnn::Shape shape({32, 32});
    const auto input = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    log_info(tt::LogTest, "Building diamond-shaped computation graph...");
    // Diamond structure:
    //       input
    //       /   \
    //    relu   exp
    //       \   /
    //       add
    auto relu_branch = ttnn::relu(input);
    auto exp_branch = ttnn::exp(input);
    auto final_result = ttnn::add(relu_branch, exp_branch);

    ASSERT_TRUE(input.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_FALSE(relu_branch.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(exp_branch.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(final_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";

    log_info(
        tt::LogTest,
        "relu_branch id: {}, exp_branch id: {}, final_result id: {}",
        relu_branch.lazy()->id(),
        exp_branch.lazy()->id(),
        final_result.lazy()->id());

    // Now execute the lazy graph - should execute in topological order
    log_info(tt::LogTest, "Executing lazy graph (watch for execution order in logs)...");
    final_result.materialize();
    auto lazy_result = final_result.cpu();

    ASSERT_TRUE(final_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(relu_branch.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(exp_branch.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input.lazy()->is_materialized()) << "Lazy tensor should be materialized";

    ttnn::experimental::jit::disable();
    ASSERT_FALSE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be disabled";

    // Run the same operations in eager mode
    log_info(tt::LogTest, "Running same operations in eager mode for verification...");
    const auto input_eager = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    const auto relu_eager = ttnn::relu(input_eager);
    const auto exp_eager = ttnn::exp(input_eager);
    const auto final_eager = ttnn::add(relu_eager, exp_eager);
    const auto eager_result = final_eager.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished ExecutionOrderCorrect test ====");
}

// Test: Matmul with element-wise operations in lazy mode
TEST_F(LazyModeFixture, MatmulWithElementwiseLazy) {
    log_info(tt::LogTest, "==== Starting MatmulWithElementwiseLazy test ====");

    // Create input shapes
    // Use smaller values to avoid numerical overflow in exp()
    ttnn::Shape matmul_shape1({32, 64});
    ttnn::Shape matmul_shape2({64, 32});
    ttnn::Shape matmul_shape3({32, 32});

    // ========== Run EAGER mode first to establish baseline ==========
    log_info(tt::LogTest, "Running operations in EAGER mode for baseline...");
    ttnn::experimental::jit::disable();
    ASSERT_FALSE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be disabled";

    auto matmul_input1_eager = ttnn::full(matmul_shape1, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto matmul_input2_eager = ttnn::full(matmul_shape2, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto add_input_eager = ttnn::full(matmul_shape3, 0.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    log_info(
        tt::LogTest,
        "Created matmul input tensors with shapes [{}, {}] and [{}, {}]",
        matmul_shape1[0],
        matmul_shape1[1],
        matmul_shape2[0],
        matmul_shape2[1]);

    auto matmul_result_eager =
        ttnn::matmul(matmul_input1_eager, matmul_input2_eager, /* transpose_a */ false, /* transpose_b */ false);
    auto add_result_eager = ttnn::add(matmul_result_eager, add_input_eager);
    auto relu_result_eager = ttnn::relu(add_result_eager);
    auto matmul_result2_eager =
        ttnn::matmul(relu_result_eager, matmul_input1_eager, /* transpose_a */ false, /* transpose_b */ false);
    auto exp_result_eager = ttnn::exp(matmul_result2_eager);
    auto final_eager = ttnn::multiply(exp_result_eager, matmul_input1_eager);
    auto eager_result = final_eager.cpu();

    // Clean up eager tensors to free device memory before running lazy mode
    log_info(tt::LogTest, "Cleaning up eager tensors...");
    // TODO:: Test fails without this, why???
    matmul_input1_eager = Tensor();
    matmul_input2_eager = Tensor();
    add_input_eager = Tensor();
    matmul_result_eager = Tensor();
    add_result_eager = Tensor();
    relu_result_eager = Tensor();
    matmul_result2_eager = Tensor();
    exp_result_eager = Tensor();
    final_eager = Tensor();

    // ========== Now run LAZY mode and compare ==========
    log_info(tt::LogTest, "Running same operations in LAZY mode...");
    ttnn::experimental::jit::enable();
    ASSERT_TRUE(ttnn::experimental::jit::is_lazy_enabled()) << "Lazy mode should be enabled";

    auto matmul_input1 = ttnn::full(matmul_shape1, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto matmul_input2 = ttnn::full(matmul_shape2, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto add_input = ttnn::full(matmul_shape3, 0.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    // Apply matmul and element-wise operations - these should be captured lazily
    log_info(tt::LogTest, "Applying operations in lazy mode...");
    auto matmul_result = ttnn::matmul(matmul_input1, matmul_input2, /* transpose_a */ false, /* transpose_b */ false);
    auto add_result = ttnn::add(matmul_result, add_input);
    auto relu_result = ttnn::relu(add_result);
    auto matmul_result2 = ttnn::matmul(relu_result, matmul_input1, /* transpose_a */ false, /* transpose_b */ false);
    auto exp_result = ttnn::exp(matmul_result2);
    auto final_result = ttnn::multiply(exp_result, matmul_input1);

    ASSERT_TRUE(matmul_input1.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(matmul_input2.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(add_input.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_FALSE(matmul_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(add_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(relu_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(matmul_result2.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(exp_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    ASSERT_FALSE(final_result.lazy()->is_materialized()) << "Lazy tensor should not be materialized";

    log_info(
        tt::LogTest,
        "matmul_result id: {}, add_result id: {}, relu_result id: {}, matmul_result2 id: {}, exp_result id: {}, "
        "final_result id: {}",
        matmul_result.lazy()->id(),
        add_result.lazy()->id(),
        relu_result.lazy()->id(),
        matmul_result2.lazy()->id(),
        exp_result.lazy()->id(),
        final_result.lazy()->id());

    // Now execute the lazy graph
    log_info(tt::LogTest, "Executing lazy graph...");
    final_result.materialize();

    ASSERT_TRUE(final_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(exp_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(matmul_result2.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(relu_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(add_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(matmul_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(matmul_input1.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(matmul_input2.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(add_input.lazy()->is_materialized()) << "Lazy tensor should be materialized";

    auto lazy_result = final_result.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished MatmulWithElementwiseLazy test ====");
}

}  // namespace test
}  // namespace ttnn
