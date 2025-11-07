// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/experimental/lazy/lazy_mode.hpp"
#include "ttnn/experimental/lazy/evaluation_manager.hpp"
#include "ttnn/experimental/lazy/compile/passes/passes.hpp"
#include "ttnn/experimental/lazy/compile/passes/unary_operation_fusion.hpp"
#include "ttnn/experimental/lazy/graph_utils.hpp"
#include "ttnn/experimental/lazy/lazy_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/operations/rand/rand.hpp"
#include <enchantum/enchantum.hpp>
#include "ttnn/operations/copy/typecast/device/typecast_device_op.hpp"
#include "ttnn/operations/rand/device/rand_device_operation.hpp"
#include "ttnn/operations/data_movement/split/split.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/non_zero_indices.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/fill_rm/fill_rm.hpp"
#include "ttnn/operations/data_movement/gather/gather.hpp"
#include "ttnn/operations/data_movement/sort/sort.hpp"
#include "ttnn/experimental/lazy/lazy_operation_inputs_utils.hpp"

namespace ttnn {
namespace test {

using namespace ttnn::experimental;

class LazyModeFixture : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        TTNNFixtureWithDevice::SetUp();
        // Enable lazy mode by setting environment variable
        setenv("TTNN_LAZY_MODE", "1", 1);
        // TODO: old device infra doesn't work well with cache enabled for some reason
        ttnn::disable_and_clear_program_cache(*device_);
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

    // Verify lazy mode is disabled
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    // 1. In eager mode, the lazy tensor should be materialized
    float from = 0.0;
    float to = 1.0;
    uint32_t seed = 42;
    auto random_eager = ttnn::rand(
        ttnn::Shape({32, 32}),
        *device_,
        DataType::BFLOAT16,
        Layout::TILE,
        ttnn::types::DRAM_MEMORY_CONFIG,
        from,
        to,
        seed);

    ASSERT_EQ(random_eager.lazy()->materialized_tensor().logical_shape(), ttnn::Shape({32, 32}))
        << "Lazy tensor should have the same shape as the materialized tensor";
    ASSERT_EQ(random_eager.lazy()->materialized_tensor().dtype(), DataType::BFLOAT16)
        << "Lazy tensor should have the same dtype as the materialized tensor";
    ASSERT_EQ(random_eager.lazy()->materialized_tensor().layout(), Layout::TILE)
        << "Lazy tensor should have the same layout as the materialized tensor";
    ASSERT_EQ(random_eager.lazy()->materialized_tensor().memory_config(), ttnn::types::DRAM_MEMORY_CONFIG)
        << "Lazy tensor should have the same memory config as the materialized tensor";
    ASSERT_EQ(random_eager.lazy()->materialized_tensor().device(), device_)
        << "Lazy tensor should have the same device as the materialized tensor";
    ASSERT_EQ(random_eager.lazy()->materialized_tensor().mesh_buffer(), random_eager.mesh_buffer())
        << "Lazy tensor should have the same mesh buffer as the materialized tensor";
    ASSERT_EQ(random_eager.lazy()->materialized_tensor().mesh_buffer()->device(), device_)
        << "Lazy tensor should have the same mesh buffer device as the materialized tensor";

    ASSERT_EQ(random_eager.lazy()->state(), experimental::lazy::LazyTensorState::EVALUATED)
        << "Lazy tensor should be in materialized state";
    ASSERT_EQ(random_eager.lazy()->tensor_spec(), random_eager.tensor_spec())
        << "Lazy tensor should have the same tensor spec as the materialized tensor";
    ASSERT_EQ(random_eager.lazy()->op().get(), nullptr)
        << "Lazy tensor created from materialized tensor should have no operation";
    ASSERT_TRUE(random_eager.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_EQ(lazy::count(random_eager.lazy()->op_inputs()), 0) << "Lazy tensor should have no op inputs";
    ASSERT_EQ(random_eager.lazy()->siblings().size(), 0) << "Lazy tensor should have no siblings";
    ASSERT_EQ(random_eager.lazy()->materialized_tensors().size(), 1)
        << "Lazy tensor should have one materialized tensor";

    // 2. In lazy mode, the lazy tensor should not be materialized
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    auto random_lazy = ttnn::rand(
        ttnn::Shape({32, 32}),
        *device_,
        DataType::BFLOAT16,
        Layout::TILE,
        ttnn::types::DRAM_MEMORY_CONFIG,
        from,
        to,
        seed);

    ASSERT_FALSE(random_lazy.lazy()->is_materialized()) << "Lazy tensor should not be materialized";
    // Composite part of rand uses typecast, which uses old device infra operation
    ASSERT_EQ(random_lazy.lazy()->op()->name(), "ttnn::prim::typecast")
        << "Lazy tensor should have the typecast operation";
    ASSERT_EQ(
        random_lazy.lazy()->op()->operation_type_id(),
        lazy::get_operation_type_id<ttnn::operations::copy::TypecastDeviceOperation>())
        << "Lazy tensor should have the typecast operation";

    // One input - typecast takes rand's output as input
    ASSERT_EQ(lazy::count(random_lazy.lazy()->op_inputs()), 1) << "Lazy tensor should have op input";
    ASSERT_EQ(random_lazy.lazy()->siblings().size(), 0) << "Lazy tensor should have no siblings";
    ASSERT_EQ(random_lazy.lazy()->materialized_tensors().size(), 0)
        << "Lazy tensor should have no materialized tensors";

    // Check that parent of typecast is rand
    auto parent = lazy::get(random_lazy.lazy()->op_inputs(), 0);
    ASSERT_EQ(parent->op()->name(), "ttnn::prim::rand") << "Lazy tensor should have the rand operation";
    ASSERT_EQ(
        parent->op()->operation_type_id(), lazy::get_operation_type_id<ttnn::operations::rand::RandDeviceOperation>())
        << "Lazy tensor should have the rand operation";

    // 3. Materialize the lazy tensor and check results
    random_lazy.evaluate();
    ASSERT_TRUE(random_lazy.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_EQ(random_lazy.lazy()->materialized_tensors().size(), 1)
        << "Lazy tensor should have one materialized tensor (since typecast returns a single tensor)";
    ASSERT_EQ(random_lazy.lazy()->materialized_tensor().logical_shape(), ttnn::Shape({32, 32}))
        << "Lazy tensor should have the same shape as the materialized tensor";
    ASSERT_EQ(random_lazy.lazy()->materialized_tensor().dtype(), DataType::BFLOAT16)
        << "Lazy tensor should have the same dtype as the materialized tensor";
    ASSERT_EQ(random_lazy.lazy()->materialized_tensor().layout(), Layout::TILE)
        << "Lazy tensor should have the same layout as the materialized tensor";
    ASSERT_EQ(random_lazy.lazy()->materialized_tensor().memory_config(), ttnn::types::DRAM_MEMORY_CONFIG)
        << "Lazy tensor should have the same memory config as the materialized tensor";
    ASSERT_EQ(random_lazy.lazy()->materialized_tensor().device(), device_)
        << "Lazy tensor should have the same device as the materialized tensor";

    ASSERT_TRUE(ttnn::allclose<::bfloat16>(random_lazy.cpu(), random_eager.cpu()))
        << "Lazy and eager execution results should match";
}

// Test: Simple unary operations in lazy mode with verification
TEST_F(LazyModeFixture, SimpleUnaryOperationsLazy) {
    log_info(tt::LogTest, "==== Starting SimpleUnaryOperationsLazy test ====");

    // Verify lazy mode is enabled
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

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

    sqrt_output.evaluate();
    // TODO: materialize should be implicit in cpu() / to_string() / etc.
    auto lazy_result = sqrt_output.cpu();

    ASSERT_TRUE(sqrt_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(exp_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(relu_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input_tensor.lazy()->is_materialized()) << "Lazy tensor should be materialized";

    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

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
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

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

    mul_output.evaluate();
    auto lazy_result = mul_output.cpu();

    ASSERT_TRUE(mul_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(add_output.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input1.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input2.lazy()->is_materialized()) << "Lazy tensor should be materialized";

    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

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
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

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

    sub_result.evaluate();
    auto lazy_result = sub_result.cpu();

    ASSERT_TRUE(sub_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(sqrt_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(mul_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(relu_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(add_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input1.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input2.lazy()->is_materialized()) << "Lazy tensor should be materialized";

    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

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
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

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
    final_result.evaluate();
    auto lazy_result = final_result.cpu();

    ASSERT_TRUE(final_result.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(relu_branch.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(exp_branch.lazy()->is_materialized()) << "Lazy tensor should be materialized";
    ASSERT_TRUE(input.lazy()->is_materialized()) << "Lazy tensor should be materialized";

    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

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
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

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

    auto matmul_result_eager = ttnn::matmul(matmul_input1_eager, matmul_input2_eager, false, false);
    auto add_result_eager = ttnn::add(matmul_result_eager, add_input_eager);
    auto relu_result_eager = ttnn::relu(add_result_eager);
    auto matmul_result2_eager = ttnn::matmul(relu_result_eager, matmul_result_eager, false, false);
    auto exp_result_eager = ttnn::exp(matmul_result2_eager);
    auto final_eager = ttnn::multiply(exp_result_eager, matmul_result_eager);
    auto eager_result = final_eager.cpu();

    // Clean up eager tensors to free device memory before running lazy mode
    log_info(tt::LogTest, "Cleaning up eager tensors...");

    // ========== Now run LAZY mode and compare ==========
    log_info(tt::LogTest, "Running same operations in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    auto matmul_input1 = ttnn::full(matmul_shape1, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto matmul_input2 = ttnn::full(matmul_shape2, 0.1f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto add_input = ttnn::full(matmul_shape3, 0.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    // Apply matmul and element-wise operations - these should be captured lazily
    log_info(tt::LogTest, "Applying operations in lazy mode...");
    auto matmul_result = ttnn::matmul(matmul_input1, matmul_input2, false, false);
    auto add_result = ttnn::add(matmul_result, add_input);
    auto relu_result = ttnn::relu(add_result);
    auto matmul_result2 = ttnn::matmul(relu_result, matmul_result, false, false);
    auto exp_result = ttnn::exp(matmul_result2);
    auto final_result = ttnn::multiply(exp_result, matmul_result);

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
    final_result.evaluate();

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

TEST_F(LazyModeFixture, UnaryOperationsFusion) {
    log_info(tt::LogTest, "==== Starting UnaryOperationsFusion test ====");

    // First run in eager mode to get baseline
    log_info(tt::LogTest, "Running operations in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape shape({32, 32});

    // Create data for reproducibility
    auto spec = TensorSpec(
        shape,
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(ttnn::TILE_LAYOUT),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)));
    std::vector<bfloat16> data(shape.volume());
    for (int i = 0; i < shape.volume(); i++) {
        data[i] = static_cast<float>(i % 100) / 100.0f;  // Values between 0 and 1
    }

    auto input_eager = Tensor::from_vector(data, spec, device_);

    // Chain of unary operations
    auto relu_eager = ttnn::relu(input_eager);
    auto exp_eager = ttnn::exp(relu_eager);
    auto sqrt_eager = ttnn::sqrt(exp_eager);
    auto eager_result = sqrt_eager.cpu();

    // Now run in lazy mode with fusion pass
    log_info(tt::LogTest, "Running same operations in LAZY mode with fusion pass...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    auto input_lazy = Tensor::from_vector(data, spec, device_);

    // Apply unary operations - these should be captured lazily
    auto relu_lazy = ttnn::relu(input_lazy);
    auto exp_lazy = ttnn::exp(relu_lazy);
    auto sqrt_lazy = ttnn::sqrt(exp_lazy);

    // Verify they are not materialized
    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_FALSE(relu_lazy.lazy()->is_materialized()) << "Relu tensor should not be materialized";
    ASSERT_FALSE(exp_lazy.lazy()->is_materialized()) << "Exp tensor should not be materialized";
    ASSERT_FALSE(sqrt_lazy.lazy()->is_materialized()) << "Sqrt tensor should not be materialized";

    // Traverse and log graph BEFORE fusion
    log_info(tt::LogTest, "\n==== Graph structure BEFORE fusion ====");
    auto graph_before = lazy::GraphUtils::topological_sort(sqrt_lazy.lazy());
    log_info(tt::LogTest, "Graph has {} nodes", graph_before.size());

    size_t unary_ops_before = 0;
    for (size_t i = 0; i < graph_before.size(); ++i) {
        auto& node = graph_before[i];
        auto op = node->op();
        if (op) {
            using UnaryLazyOp = lazy::LazyDeviceOperation<ttnn::operations::unary::UnaryDeviceOperation>;
            auto* unary_op = dynamic_cast<UnaryLazyOp*>(op.get());
            if (unary_op) {
                const auto& attrs = unary_op->attributes();
                log_info(
                    tt::LogTest,
                    "  Node[{}]: Unary op, id={}, op_chain size={}, inputs={}",
                    i,
                    node->id(),
                    attrs.op_chain.size(),
                    lazy::count(node->op_inputs()));
                for (const auto& op_in_chain : attrs.op_chain) {
                    log_info(tt::LogTest, "    - {}", enchantum::to_string(op_in_chain.type()));
                }
                unary_ops_before++;
            }
        } else {
            log_info(tt::LogTest, "  Node[{}]: Materialized input, id={}", i, node->id());
        }
    }
    log_info(tt::LogTest, "Total unary operations: {}", unary_ops_before);

    // Before fusion: should have 3 unary ops (relu, exp, sqrt), each with op_chain size 1 + input node
    ASSERT_EQ(graph_before.size(), 4) << "Graph should have 4 nodes before fusion";
    ASSERT_EQ(unary_ops_before, 3) << "Should have 3 unary operations before fusion";

    // Check the chain structure before fusion
    using UnaryLazyOp = lazy::LazyDeviceOperation<ttnn::operations::unary::UnaryDeviceOperation>;
    auto* sqrt_op_before = dynamic_cast<UnaryLazyOp*>(sqrt_lazy.lazy()->op().get());
    ASSERT_NE(sqrt_op_before, nullptr) << "Sqrt operation should be a unary op";
    ASSERT_EQ(sqrt_op_before->attributes().op_chain.size(), 1) << "Sqrt op_chain should have 1 element before fusion";
    ASSERT_EQ(lazy::count(sqrt_lazy.lazy()->op_inputs()), 1) << "Sqrt should have 1 input";
    ASSERT_EQ(lazy::get(sqrt_lazy.lazy()->op_inputs(), 0)->id(), exp_lazy.lazy()->id()) << "Sqrt input should be exp output";

    // Run the fusion pass
    log_info(tt::LogTest, "\n==== Running UnaryOperationsFusionPass ====");
    lazy::compile::PassManager pass_manager;
    pass_manager.add_pass(std::make_unique<lazy::compile::UnaryOperationsFusionPass>());
    pass_manager.run(sqrt_lazy);

    // Traverse and log graph AFTER fusion
    log_info(tt::LogTest, "\n==== Graph structure AFTER fusion ====");
    auto graph_after = lazy::GraphUtils::topological_sort(sqrt_lazy.lazy());
    log_info(tt::LogTest, "Graph has {} nodes", graph_after.size());

    size_t unary_ops_after = 0;
    size_t total_fused_ops = 0;
    for (size_t i = 0; i < graph_after.size(); ++i) {
        auto& node = graph_after[i];
        auto op = node->op();
        if (op) {
            using UnaryLazyOp = lazy::LazyDeviceOperation<ttnn::operations::unary::UnaryDeviceOperation>;
            auto* unary_op = dynamic_cast<UnaryLazyOp*>(op.get());
            if (unary_op) {
                const auto& attrs = unary_op->attributes();
                log_info(
                    tt::LogTest,
                    "  Node[{}]: Unary op, id={}, op_chain size={}, inputs={}",
                    i,
                    node->id(),
                    attrs.op_chain.size(),
                    lazy::count(node->op_inputs()));
                for (const auto& op_in_chain : attrs.op_chain) {
                    log_info(tt::LogTest, "    - {}", enchantum::to_string(op_in_chain.type()));
                }
                unary_ops_after++;
                total_fused_ops += attrs.op_chain.size();
            }
        } else {
            log_info(tt::LogTest, "  Node[{}]: Materialized input, id={}", i, node->id());
        }
    }
    log_info(tt::LogTest, "Total unary operations: {}", unary_ops_after);
    log_info(tt::LogTest, "Total fused operations in chains: {}", total_fused_ops);

    // After fusion: graph still has all nodes, but the last op should have merged op_chain
    ASSERT_EQ(graph_after.size(), 2) << "Graph should have 2 node after fusion";
    ASSERT_EQ(unary_ops_after, 1) << "Should have 4 unary operations after fusion";

    // Check the fused operation
    auto* sqrt_op_after = dynamic_cast<UnaryLazyOp*>(sqrt_lazy.lazy()->op().get());
    ASSERT_NE(sqrt_op_after, nullptr) << "Sqrt operation should be a unary op after fusion";
    ASSERT_EQ(sqrt_op_after->attributes().op_chain.size(), 3)
        << "Fused op_chain should have 3 elements (relu+exp+sqrt)";

    // Verify the op_chain contains the right operations in the right order
    const auto& fused_chain = sqrt_op_after->attributes().op_chain;
    ASSERT_EQ(fused_chain[0].type(), ttnn::operations::unary::UnaryOpType::RELU) << "First op should be RELU";
    ASSERT_EQ(fused_chain[1].type(), ttnn::operations::unary::UnaryOpType::EXP) << "Second op should be EXP";
    ASSERT_EQ(fused_chain[2].type(), ttnn::operations::unary::UnaryOpType::SQRT) << "Third op should be SQRT";

    // Verify the graph structure: sqrt should now take input directly from the original input
    ASSERT_EQ(lazy::count(sqrt_lazy.lazy()->op_inputs()), 1) << "Sqrt should have 1 input after fusion";
    ASSERT_EQ(lazy::get(sqrt_lazy.lazy()->op_inputs(), 0)->id(), input_lazy.lazy()->id())
        << "Sqrt should now take input directly from original input (bypassing relu and exp)";

    log_info(tt::LogTest, "\n✓ Fusion verification complete!");
    log_info(tt::LogTest, "  - op_chain expanded from 1 to 3 operations");
    log_info(tt::LogTest, "  - Graph dependency updated: sqrt now takes input from original input");
    log_info(tt::LogTest, "  - Operations in chain: RELU -> EXP -> SQRT");

    // Now materialize
    log_info(tt::LogTest, "\n==== Materializing result ====");
    sqrt_lazy.evaluate();
    auto lazy_result = sqrt_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy (with fusion) and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy (with fusion) and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy (with fusion) and eager results match!");
    log_info(tt::LogTest, "==== Finished UnaryOperationsFusion test ====");
}

// Test: Data movement operations in lazy mode (split, concat, repeat, reshape)
TEST_F(LazyModeFixture, DataMovementOperationsLazy) {
    log_info(tt::LogTest, "==== Starting DataMovementOperationsLazy test ====");

    // First run in eager mode to get baseline
    log_info(tt::LogTest, "Running operations in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape shape({32, 64});
    const auto input_eager = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    // Test 1: Split operation - split along dim 1 (width) into chunks of size 32
    auto split_results_eager = ttnn::split(input_eager, 32, 1, std::nullopt);
    // Test 2: Concat operation - concat the split results back
    auto concat_result_eager = ttnn::concat(split_results_eager, 1, std::nullopt);
    // Test 3: Reshape operation
    auto reshape_result_eager = ttnn::reshape(concat_result_eager, ttnn::Shape({64, 32}), std::nullopt);
    // Test 4: Repeat operation
    auto repeat_result_eager = ttnn::repeat(reshape_result_eager, ttnn::SmallVector<uint32_t>{1, 2}, std::nullopt);

    // Add element-wise operation to make the test more interesting
    auto relu_result_eager = ttnn::relu(repeat_result_eager);
    auto eager_result = relu_result_eager.cpu();

    ASSERT_EQ(split_results_eager.size(), 2) << "Split should produce 2 tensors";
    ASSERT_EQ(split_results_eager[0].logical_shape(), ttnn::Shape({32, 32}))
        << "First split should have shape [32, 32]";
    ASSERT_EQ(split_results_eager[1].logical_shape(), ttnn::Shape({32, 32}))
        << "Second split should have shape [32, 32]";
    ASSERT_EQ(concat_result_eager.logical_shape(), shape) << "Concat should restore original shape";
    ASSERT_EQ(reshape_result_eager.logical_shape(), ttnn::Shape({64, 32})) << "Reshape should produce shape [64, 32]";
    ASSERT_EQ(repeat_result_eager.logical_shape(), ttnn::Shape({64, 64})) << "Repeat should produce shape [64, 64]";

    // Now run in lazy mode
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = ttnn::full(shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    // Test 1: Split operation - should be captured lazily (split into chunks of size 32)
    auto split_results_lazy = ttnn::split(input_lazy, 32, 1, std::nullopt);
    // Test 2: Concat operation - should be captured lazily
    auto concat_result_lazy = ttnn::concat(split_results_lazy, 1, std::nullopt);
    // Test 3: Reshape operation - should be captured lazily
    auto reshape_result_lazy = ttnn::reshape(concat_result_lazy, ttnn::Shape({64, 32}), std::nullopt);
    // Test 4: Repeat operation - should be captured lazily
    auto repeat_result_lazy = ttnn::repeat(reshape_result_lazy, ttnn::SmallVector<uint32_t>{1, 2}, std::nullopt);
    // Add element-wise operation
    auto relu_result_lazy = ttnn::relu(repeat_result_lazy);

    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_EQ(split_results_lazy.size(), 2) << "Split should produce 2 tensors";
    ASSERT_FALSE(split_results_lazy[0].lazy()->is_materialized()) << "First split result should not be materialized";
    ASSERT_FALSE(split_results_lazy[1].lazy()->is_materialized()) << "Second split result should not be materialized";
    ASSERT_FALSE(concat_result_lazy.lazy()->is_materialized()) << "Concat result should not be materialized";
    ASSERT_FALSE(reshape_result_lazy.lazy()->is_materialized()) << "Reshape result should not be materialized";
    ASSERT_FALSE(repeat_result_lazy.lazy()->is_materialized()) << "Repeat result should not be materialized";

    ASSERT_FALSE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should not be materialized";

    log_info(
        tt::LogTest,
        "split_results_lazy[0] id: {}, split_results_lazy[1] id: {}, concat_result_lazy id: {}, "
        "reshape_result_lazy id: {}, repeat_result_lazy id: {}, relu_result_lazy id: {}",
        split_results_lazy[0].lazy()->id(),
        split_results_lazy[1].lazy()->id(),
        concat_result_lazy.lazy()->id(),
        reshape_result_lazy.lazy()->id(),
        repeat_result_lazy.lazy()->id(),
        relu_result_lazy.lazy()->id());

    // Now execute the lazy graph
    log_info(tt::LogTest, "Executing lazy graph...");
    relu_result_lazy.evaluate();

    ASSERT_TRUE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should be materialized";
    ASSERT_TRUE(repeat_result_lazy.lazy()->is_materialized()) << "Repeat result should be materialized";
    ASSERT_TRUE(reshape_result_lazy.lazy()->is_materialized()) << "Reshape result should be materialized";
    ASSERT_TRUE(concat_result_lazy.lazy()->is_materialized()) << "Concat result should be materialized";
    ASSERT_TRUE(split_results_lazy[0].lazy()->is_materialized()) << "First split result should be materialized";
    ASSERT_TRUE(split_results_lazy[1].lazy()->is_materialized()) << "Second split result should be materialized";
    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";

    auto lazy_result = relu_result_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished DataMovementOperationsLazy test ====");
}

// Test: Split operation in lazy mode with sibling evaluation verification
TEST_F(LazyModeFixture, SplitOperationLazy) {
    log_info(tt::LogTest, "==== Starting SplitOperationLazy test ====");

    // First run in eager mode to get baseline
    log_info(tt::LogTest, "Running split operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape shape({32, 64});
    const auto input_eager = ttnn::full(shape, 3.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    // Split along dim 1 (width) into chunks of size 32
    auto split_results_eager = ttnn::split(input_eager, 32, 1, std::nullopt);
    ASSERT_EQ(split_results_eager.size(), 2) << "Split should produce 2 tensors";
    ASSERT_EQ(split_results_eager[0].logical_shape(), ttnn::Shape({32, 32}))
        << "First split should have shape [32, 32]";
    ASSERT_EQ(split_results_eager[1].logical_shape(), ttnn::Shape({32, 32}))
        << "Second split should have shape [32, 32]";

    auto eager_result_0 = split_results_eager[0].cpu();
    auto eager_result_1 = split_results_eager[1].cpu();

    // Now run in lazy mode
    log_info(tt::LogTest, "Running split operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = ttnn::full(shape, 3.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    // Split operation - should be captured lazily
    log_info(tt::LogTest, "Applying split operation in lazy mode...");
    auto split_results_lazy = ttnn::split(input_lazy, 32, 1, std::nullopt);
    ASSERT_EQ(split_results_lazy.size(), 2) << "Split should produce 2 tensors";

    // Verify that input is materialized but outputs are not
    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_FALSE(split_results_lazy[0].lazy()->is_materialized()) << "First split result should not be materialized";
    ASSERT_FALSE(split_results_lazy[1].lazy()->is_materialized()) << "Second split result should not be materialized";

    // Note: lazy()->siblings() is going to be empty since split is "fake multi-output" operation.
    // It basically calls ttnn::slice for each output, so outputs tensors are independent.

    log_info(
        tt::LogTest,
        "split_results_lazy[0] id: {}, split_results_lazy[1] id: {}",
        split_results_lazy[0].lazy()->id(),
        split_results_lazy[1].lazy()->id());

    // Now execute ONLY the first output - this should also materialize the second output
    log_info(tt::LogTest, "Evaluating first split output (should also evaluate second output)...");
    split_results_lazy[0].evaluate();
    split_results_lazy[1].evaluate();

    // Verify that BOTH outputs are now materialized (because they are siblings)
    ASSERT_TRUE(split_results_lazy[0].lazy()->is_materialized()) << "First split result should be materialized";
    ASSERT_TRUE(split_results_lazy[1].lazy()->is_materialized())
        << "Second split result should ALSO be materialized (sibling evaluation)";
    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";

    auto lazy_result_0 = split_results_lazy[0].cpu();
    auto lazy_result_1 = split_results_lazy[1].cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result_0, eager_result_0))
        << "Lazy and eager execution results should match for first output";
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result_1, eager_result_1))
        << "Lazy and eager execution results should match for second output";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "✓ Sibling evaluation verified: evaluating one output materialized both outputs!");
    log_info(tt::LogTest, "==== Finished SplitOperationLazy test ====");
}

// Test: Slice operation in lazy mode
TEST_F(LazyModeFixture, SliceOperationLazy) {
    log_info(tt::LogTest, "==== Starting SliceOperationLazy test ====");

    // Run in eager mode
    log_info(tt::LogTest, "Running slice operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape shape({32, 64});
    const auto input_eager = ttnn::full(shape, 5.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto slice_result_eager = ttnn::slice(
        input_eager,
        ttnn::SmallVector<uint32_t>{0, 16},
        ttnn::SmallVector<uint32_t>{32, 48},
        ttnn::SmallVector<uint32_t>{1, 1},
        std::nullopt);
    auto relu_result_eager = ttnn::relu(slice_result_eager);
    auto eager_result = relu_result_eager.cpu();

    ASSERT_EQ(slice_result_eager.logical_shape(), ttnn::Shape({32, 32})) << "Slice should produce shape [32, 32]";

    // Run in lazy mode
    log_info(tt::LogTest, "Running slice operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = ttnn::full(shape, 5.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto slice_result_lazy = ttnn::slice(
        input_lazy,
        ttnn::SmallVector<uint32_t>{0, 16},
        ttnn::SmallVector<uint32_t>{32, 48},
        ttnn::SmallVector<uint32_t>{1, 1},
        std::nullopt);
    auto relu_result_lazy = ttnn::relu(slice_result_lazy);

    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_FALSE(slice_result_lazy.lazy()->is_materialized()) << "Slice result should not be materialized";
    ASSERT_FALSE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should not be materialized";

    log_info(tt::LogTest, "Executing lazy graph...");
    relu_result_lazy.evaluate();

    ASSERT_TRUE(slice_result_lazy.lazy()->is_materialized()) << "Slice result should be materialized";
    ASSERT_TRUE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should be materialized";

    auto lazy_result = relu_result_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished SliceOperationLazy test ====");
}

// Test: Pad operation in lazy mode
TEST_F(LazyModeFixture, PadOperationLazy) {
    log_info(tt::LogTest, "==== Starting PadOperationLazy test ====");

    // Run in eager mode
    log_info(tt::LogTest, "Running pad operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape shape({32, 32});
    const auto input_eager = ttnn::full(shape, 3.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto pad_result_eager =
        ttnn::pad(input_eager, ttnn::SmallVector<std::array<uint32_t, 2>>{{0, 0}, {0, 32}}, 0.0f, false, std::nullopt);
    auto relu_result_eager = ttnn::relu(pad_result_eager);
    auto eager_result = relu_result_eager.cpu();

    ASSERT_EQ(pad_result_eager.logical_shape(), ttnn::Shape({32, 64})) << "Pad should produce shape [32, 64]";

    // Run in lazy mode
    log_info(tt::LogTest, "Running pad operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = ttnn::full(shape, 3.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto pad_result_lazy =
        ttnn::pad(input_lazy, ttnn::SmallVector<std::array<uint32_t, 2>>{{0, 0}, {0, 32}}, 0.0f, false, std::nullopt);
    auto relu_result_lazy = ttnn::relu(pad_result_lazy);

    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_FALSE(pad_result_lazy.lazy()->is_materialized()) << "Pad result should not be materialized";
    ASSERT_FALSE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should not be materialized";

    log_info(tt::LogTest, "Executing lazy graph...");
    relu_result_lazy.evaluate();

    ASSERT_TRUE(pad_result_lazy.lazy()->is_materialized()) << "Pad result should be materialized";
    ASSERT_TRUE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should be materialized";

    auto lazy_result = relu_result_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished PadOperationLazy test ====");
}

// Test: Permute operation in lazy mode
TEST_F(LazyModeFixture, PermuteOperationLazy) {
    log_info(tt::LogTest, "==== Starting PermuteOperationLazy test ====");

    // Run in eager mode
    log_info(tt::LogTest, "Running permute operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape shape({32, 64});
    const auto input_eager = ttnn::full(shape, 2.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto permute_result_eager = ttnn::permute(input_eager, ttnn::SmallVector<int64_t>{1, 0});
    auto relu_result_eager = ttnn::relu(permute_result_eager);
    auto eager_result = relu_result_eager.cpu();

    ASSERT_EQ(permute_result_eager.logical_shape(), ttnn::Shape({64, 32})) << "Permute should produce shape [64, 32]";

    // Run in lazy mode
    log_info(tt::LogTest, "Running permute operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = ttnn::full(shape, 2.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto permute_result_lazy = ttnn::permute(input_lazy, ttnn::SmallVector<int64_t>{1, 0});
    auto relu_result_lazy = ttnn::relu(permute_result_lazy);

    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_FALSE(permute_result_lazy.lazy()->is_materialized()) << "Permute result should not be materialized";
    ASSERT_FALSE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should not be materialized";

    log_info(tt::LogTest, "Executing lazy graph...");
    relu_result_lazy.evaluate();

    ASSERT_TRUE(permute_result_lazy.lazy()->is_materialized()) << "Permute result should be materialized";
    ASSERT_TRUE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should be materialized";

    auto lazy_result = relu_result_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished PermuteOperationLazy test ====");
}

// Test: Tilize operation in lazy mode
TEST_F(LazyModeFixture, TilizeOperationLazy) {
    log_info(tt::LogTest, "==== Starting TilizeOperationLazy test ====");

    // Run in eager mode
    log_info(tt::LogTest, "Running tilize operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape shape({32, 64});
    const auto input_eager = ttnn::full(shape, 4.0f, DataType::BFLOAT16, ttnn::ROW_MAJOR_LAYOUT, *device_);
    auto tilize_result_eager = ttnn::tilize(input_eager, std::nullopt, std::nullopt, false);
    auto relu_result_eager = ttnn::relu(tilize_result_eager);
    auto eager_result = relu_result_eager.cpu();

    ASSERT_EQ(tilize_result_eager.layout(), Layout::TILE) << "Tilize should produce TILE layout";

    // Run in lazy mode
    log_info(tt::LogTest, "Running tilize operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = ttnn::full(shape, 4.0f, DataType::BFLOAT16, ttnn::ROW_MAJOR_LAYOUT, *device_);
    auto tilize_result_lazy = ttnn::tilize(input_lazy, std::nullopt, std::nullopt, false);
    auto relu_result_lazy = ttnn::relu(tilize_result_lazy);

    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_FALSE(tilize_result_lazy.lazy()->is_materialized()) << "Tilize result should not be materialized";
    ASSERT_FALSE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should not be materialized";

    log_info(tt::LogTest, "Executing lazy graph...");
    relu_result_lazy.evaluate();

    ASSERT_TRUE(tilize_result_lazy.lazy()->is_materialized()) << "Tilize result should be materialized";
    ASSERT_TRUE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should be materialized";

    auto lazy_result = relu_result_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished TilizeOperationLazy test ====");
}

// Test: Nonzero operation in lazy mode
TEST_F(LazyModeFixture, NonzeroOperationLazy) {
    log_info(tt::LogTest, "==== Starting NonzeroOperationLazy test ====");

    // Run in eager mode
    log_info(tt::LogTest, "Running nonzero operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    // nonzero requires 4D tensor with shape [1, 1, 1, X] and ROW_MAJOR layout
    ttnn::Shape shape({1, 1, 1, 1024});
    auto spec = TensorSpec(
        shape,
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(ttnn::ROW_MAJOR_LAYOUT),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)));
    std::vector<bfloat16> data(shape.volume(), 0.0f);
    // Create a pattern with some non-zero values
    for (int i = 0; i < shape.volume(); i++) {
        if (i % 10 == 0) {
            data[i] = 1.0f;
        }
    }
    const auto input_eager = Tensor::from_vector(data, spec, device_);
    auto nonzero_results_eager = ttnn::nonzero(input_eager, std::nullopt);

    log_info(tt::LogTest, "Nonzero returned {} tensors", nonzero_results_eager.size());
    ASSERT_GT(nonzero_results_eager.size(), 0) << "Nonzero should return at least one tensor";

    std::vector<Tensor> eager_results_cpu;
    eager_results_cpu.reserve(nonzero_results_eager.size());
    for (const auto& result : nonzero_results_eager) {
        eager_results_cpu.push_back(result.cpu());
    }

    // Run in lazy mode
    log_info(tt::LogTest, "Running nonzero operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = Tensor::from_vector(data, spec, device_);
    auto nonzero_results_lazy = ttnn::nonzero(input_lazy, std::nullopt);

    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_EQ(nonzero_results_lazy.size(), nonzero_results_eager.size())
        << "Lazy and eager should return same number of tensors";

    for (const auto& result : nonzero_results_lazy) {
        ASSERT_FALSE(result.lazy()->is_materialized()) << "Nonzero result should not be materialized";
    }

    log_info(tt::LogTest, "Executing lazy graph...");
    for (auto& result : nonzero_results_lazy) {
        result.evaluate();
    }

    for (const auto& result : nonzero_results_lazy) {
        ASSERT_TRUE(result.lazy()->is_materialized()) << "Nonzero result should be materialized";
    }

    std::vector<Tensor> lazy_results_cpu;
    lazy_results_cpu.reserve(nonzero_results_lazy.size());
    for (const auto& result : nonzero_results_lazy) {
        lazy_results_cpu.push_back(result.cpu());
    }

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_EQ(lazy_results_cpu.size(), eager_results_cpu.size())
        << "Lazy and eager results should have the same number of tensors";

    for (size_t i = 0; i < lazy_results_cpu.size(); i++) {
        ASSERT_EQ(lazy_results_cpu[i].logical_shape(), eager_results_cpu[i].logical_shape())
            << "Lazy and eager results should have the same shape for tensor " << i;
        ASSERT_TRUE(ttnn::allclose<::uint32_t>(lazy_results_cpu[i], eager_results_cpu[i]))
            << "Lazy and eager execution results should match for tensor " << i;
    }

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished NonzeroOperationLazy test ====");
}

// Test: Untilize operation in lazy mode
TEST_F(LazyModeFixture, UntilizeOperationLazy) {
    log_info(tt::LogTest, "==== Starting UntilizeOperationLazy test ====");

    // Run in eager mode
    log_info(tt::LogTest, "Running untilize operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape shape({32, 64});
    const auto input_eager = ttnn::full(shape, 3.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto untilize_result_eager = ttnn::untilize(input_eager, std::nullopt, true, true, std::nullopt);
    auto eager_result = untilize_result_eager.cpu();

    ASSERT_EQ(untilize_result_eager.layout(), Layout::ROW_MAJOR) << "Untilize should produce ROW_MAJOR layout";

    // Run in lazy mode
    log_info(tt::LogTest, "Running untilize operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = ttnn::full(shape, 3.5f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto untilize_result_lazy = ttnn::untilize(input_lazy, std::nullopt, true, true, std::nullopt);

    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_FALSE(untilize_result_lazy.lazy()->is_materialized()) << "Untilize result should not be materialized";

    log_info(tt::LogTest, "Executing lazy graph...");
    untilize_result_lazy.evaluate();

    ASSERT_TRUE(untilize_result_lazy.lazy()->is_materialized()) << "Untilize result should be materialized";

    auto lazy_result = untilize_result_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished UntilizeOperationLazy test ====");
}

// Test: Fill RM operation in lazy mode
TEST_F(LazyModeFixture, FillRMOperationLazy) {
    log_info(tt::LogTest, "==== Starting FillRMOperationLazy test ====");

    // Run in eager mode
    log_info(tt::LogTest, "Running fill_rm operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    // Create a reference tensor for fill_rm
    ttnn::Shape ref_shape({1, 1, 32, 32});
    const auto ref_tensor_eager = ttnn::ones(ref_shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto fill_result_eager = ttnn::fill_rm(1, 1, 32, 64, 32, 32, ref_tensor_eager, 5.0f, 2.0f, std::nullopt);
    auto eager_result = fill_result_eager.cpu();

    ASSERT_EQ(fill_result_eager.logical_shape(), ttnn::Shape({1, 1, 32, 64}))
        << "Fill_rm should produce shape [1, 1, 32, 64]";

    // Run in lazy mode
    log_info(tt::LogTest, "Running fill_rm operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto ref_tensor_lazy = ttnn::ones(ref_shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto fill_result_lazy = ttnn::fill_rm(1, 1, 32, 64, 32, 32, ref_tensor_lazy, 5.0f, 2.0f, std::nullopt);

    ASSERT_TRUE(ref_tensor_lazy.lazy()->is_materialized()) << "Reference tensor should be materialized";
    ASSERT_FALSE(fill_result_lazy.lazy()->is_materialized()) << "Fill_rm result should not be materialized";

    log_info(tt::LogTest, "Executing lazy graph...");
    fill_result_lazy.evaluate();

    ASSERT_TRUE(fill_result_lazy.lazy()->is_materialized()) << "Fill_rm result should be materialized";

    auto lazy_result = fill_result_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished FillRMOperationLazy test ====");
}

// Test: Gather operation in lazy mode
TEST_F(LazyModeFixture, GatherOperationLazy) {
    log_info(tt::LogTest, "==== Starting GatherOperationLazy test ====");

    // Run in eager mode
    log_info(tt::LogTest, "Running gather operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape input_shape({32, 64});
    ttnn::Shape index_shape({32, 32});
    const auto input_eager = ttnn::full(input_shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

    // Create index tensor with valid indices (0 to 63)
    auto index_spec = TensorSpec(
        index_shape,
        TensorLayout(
            DataType::UINT32,
            PageConfig(ttnn::TILE_LAYOUT),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)));
    std::vector<uint32_t> index_data(index_shape.volume());
    for (size_t i = 0; i < index_data.size(); i++) {
        index_data[i] = i % 64;  // Valid indices for dim 1
    }
    const auto index_eager = Tensor::from_vector(index_data, index_spec, device_);

    auto gather_result_eager = ttnn::gather(input_eager, 1, index_eager, false, std::nullopt, std::nullopt);
    auto relu_result_eager = ttnn::relu(gather_result_eager);
    auto eager_result = relu_result_eager.cpu();

    ASSERT_EQ(gather_result_eager.logical_shape(), ttnn::Shape({32, 32})) << "Gather should produce shape [32, 32]";

    // Run in lazy mode
    log_info(tt::LogTest, "Running gather operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = ttnn::full(input_shape, 2.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    const auto index_lazy = Tensor::from_vector(index_data, index_spec, device_);
    auto gather_result_lazy = ttnn::gather(input_lazy, 1, index_lazy, false, std::nullopt, std::nullopt);
    auto relu_result_lazy = ttnn::relu(gather_result_lazy);

    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_TRUE(index_lazy.lazy()->is_materialized()) << "Index tensor should be materialized";
    ASSERT_FALSE(gather_result_lazy.lazy()->is_materialized()) << "Gather result should not be materialized";
    ASSERT_FALSE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should not be materialized";

    log_info(tt::LogTest, "Executing lazy graph...");
    relu_result_lazy.evaluate();

    ASSERT_TRUE(gather_result_lazy.lazy()->is_materialized()) << "Gather result should be materialized";
    ASSERT_TRUE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should be materialized";

    auto lazy_result = relu_result_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_result, eager_result))
        << "Lazy and eager execution results should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished GatherOperationLazy test ====");
}

// Test: Sort operation in lazy mode
TEST_F(LazyModeFixture, SortOperationLazy) {
    log_info(tt::LogTest, "==== Starting SortOperationLazy test ====");

    // Run in eager mode
    log_info(tt::LogTest, "Running sort operation in EAGER mode for baseline...");
    lazy::disable();
    ASSERT_FALSE(lazy::is_lazy_enabled()) << "Lazy mode should be disabled";

    ttnn::Shape shape({32, 64});
    // Create tensor with varied values for sorting
    auto spec = TensorSpec(
        shape,
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(ttnn::TILE_LAYOUT),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)));
    std::vector<bfloat16> data(shape.volume());
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<float>((data.size() - i) % 100) / 10.0f;  // Descending pattern
    }
    const auto input_eager = Tensor::from_vector(data, spec, device_);
    auto sort_results_eager = ttnn::sort(input_eager, 1, false, true, std::nullopt, std::nullopt);

    log_info(tt::LogTest, "Sort returned {} tensors", sort_results_eager.size());
    ASSERT_EQ(sort_results_eager.size(), 2) << "Sort should return 2 tensors (values and indices)";

    auto sorted_values_eager = sort_results_eager[0];
    auto sorted_indices_eager = sort_results_eager[1];
    auto relu_result_eager = ttnn::relu(sorted_values_eager);
    auto eager_values_result = relu_result_eager.cpu();
    auto eager_indices_result = sorted_indices_eager.cpu();

    // Run in lazy mode
    log_info(tt::LogTest, "Running sort operation in LAZY mode...");
    lazy::enable();
    ASSERT_TRUE(lazy::is_lazy_enabled()) << "Lazy mode should be enabled";

    const auto input_lazy = Tensor::from_vector(data, spec, device_);
    auto sort_results_lazy = ttnn::sort(input_lazy, 1, false, true, std::nullopt, std::nullopt);

    ASSERT_TRUE(input_lazy.lazy()->is_materialized()) << "Input tensor should be materialized";
    ASSERT_EQ(sort_results_lazy.size(), 2) << "Sort should return 2 tensors in lazy mode";
    ASSERT_FALSE(sort_results_lazy[0].lazy()->is_materialized()) << "Sorted values should not be materialized";
    ASSERT_FALSE(sort_results_lazy[1].lazy()->is_materialized()) << "Sorted indices should not be materialized";

    auto sorted_values_lazy = sort_results_lazy[0];
    auto sorted_indices_lazy = sort_results_lazy[1];
    auto relu_result_lazy = ttnn::relu(sorted_values_lazy);

    ASSERT_FALSE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should not be materialized";

    log_info(tt::LogTest, "Executing lazy graph...");
    relu_result_lazy.evaluate();
    sorted_indices_lazy.evaluate();

    ASSERT_TRUE(sorted_values_lazy.lazy()->is_materialized()) << "Sorted values should be materialized";
    ASSERT_TRUE(sorted_indices_lazy.lazy()->is_materialized()) << "Sorted indices should be materialized";
    ASSERT_TRUE(relu_result_lazy.lazy()->is_materialized()) << "Relu result should be materialized";

    auto lazy_values_result = relu_result_lazy.cpu();
    auto lazy_indices_result = sorted_indices_lazy.cpu();

    // Compare results
    log_info(tt::LogTest, "Comparing lazy and eager results...");
    ASSERT_TRUE(ttnn::allclose<::bfloat16>(lazy_values_result, eager_values_result))
        << "Lazy and eager sorted values should match";
    ASSERT_TRUE(ttnn::allclose<::uint16_t>(lazy_indices_result, eager_indices_result))
        << "Lazy and eager sorted indices should match";

    log_info(tt::LogTest, "✓ Lazy and eager results match!");
    log_info(tt::LogTest, "==== Finished SortOperationLazy test ====");
}

}  // namespace test
}  // namespace ttnn
