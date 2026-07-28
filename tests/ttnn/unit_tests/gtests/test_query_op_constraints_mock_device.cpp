// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests that query_op_constraints works on a mock device created via MetalEnv.

#include <gtest/gtest.h>

#include <any>
#include <optional>

#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device/mock_allocator.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/system_mesh.hpp>

#include <umd/device/types/arch.hpp>

#include "device/mock_device_util.hpp"

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/device.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal {

class QueryOpConstraintsMockDevice : public ::testing::Test {
protected:
    std::unique_ptr<MetalEnv> mock_env_;
    std::shared_ptr<distributed::MeshDevice> mock_device_;

    void SetUp() override {
        mock_env_ = std::make_unique<MetalEnv>(
            MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)));
        auto mesh_shape = mock_env_->get_system_mesh().shape();
        mock_device_ = mock_env_->create_mesh_device(distributed::MeshDeviceConfig(mesh_shape));
        ASSERT_GT(mock_device_->num_devices(), 0u);

        // Constraint queries run ops in NORMAL mode, which enqueues a MeshWorkload. With the program
        // cache enabled, a cached workload outlives the sub-devices it references and crashes at
        // teardown (tenstorrent/tt-metal#45646).
        mock_device_->disable_and_clear_program_cache();
    }

    void TearDown() override {
        mock_device_.reset();
        mock_env_.reset();
    }
};

// ============================================================================
// NO_DISPATCH / NORMAL graph-capture tests
// ============================================================================

TEST_F(QueryOpConstraintsMockDevice, DeviceTensorCreationInGraphCapture) {
    const auto input_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // ttnn::create_device_tensor inside graph capture should work on mock device
    auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
    auto tensor = ttnn::create_device_tensor(input_spec, mock_device_.get());
    EXPECT_TRUE(tensor.is_allocated());
}

// ============================================================================
// NORMAL mode graph capture tests — verify real allocator interaction
// ============================================================================

TEST_F(QueryOpConstraintsMockDevice, NormalModeGraphCaptureAllocatesRealMemory) {
    const auto input_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Record allocator state before
    auto stats_before = mock_device_->allocator()->get_statistics(BufferType::L1);

    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto tensor = ttnn::create_device_tensor(input_spec, mock_device_.get());

        // Tensor should have a real address (not 0 like in NO_DISPATCH)
        EXPECT_GT(tensor.buffer()->address(), 0u);

        // Allocator state should reflect the allocation
        auto stats_during = mock_device_->allocator()->get_statistics(BufferType::L1);
        EXPECT_GT(stats_during.total_allocated_bytes, stats_before.total_allocated_bytes);
    }

    // After scope ends and tensor is destroyed, allocator should be back to original state
    auto stats_after = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_after.total_allocated_bytes, stats_before.total_allocated_bytes);
}

TEST_F(QueryOpConstraintsMockDevice, NormalModeVsNoDispatchAllocatorBehavior) {
    const auto input_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // NO_DISPATCH: allocation is dry (address=0, allocator untouched)
    size_t allocated_no_dispatch = 0;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
        auto tensor = ttnn::create_device_tensor(input_spec, mock_device_.get());
        EXPECT_EQ(tensor.buffer()->address(), 0u);
        allocated_no_dispatch = mock_device_->allocator()->get_statistics(BufferType::L1).total_allocated_bytes;
    }

    // NORMAL: allocation is real (address > 0, allocator updated)
    size_t allocated_normal = 0;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto tensor = ttnn::create_device_tensor(input_spec, mock_device_.get());
        EXPECT_GT(tensor.buffer()->address(), 0u);
        allocated_normal = mock_device_->allocator()->get_statistics(BufferType::L1).total_allocated_bytes;
    }

    EXPECT_GT(allocated_normal, allocated_no_dispatch);
}

TEST_F(QueryOpConstraintsMockDevice, NormalModeMultipleTensorsAccumulateState) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 32, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    auto stats_before = mock_device_->allocator()->get_statistics(BufferType::L1);

    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);

        auto tensor1 = ttnn::create_device_tensor(spec, mock_device_.get());
        auto stats_after_1 = mock_device_->allocator()->get_statistics(BufferType::L1);

        auto tensor2 = ttnn::create_device_tensor(spec, mock_device_.get());
        auto stats_after_2 = mock_device_->allocator()->get_statistics(BufferType::L1);

        // Each tensor adds to the allocator state
        EXPECT_GT(stats_after_1.total_allocated_bytes, stats_before.total_allocated_bytes);
        EXPECT_GT(stats_after_2.total_allocated_bytes, stats_after_1.total_allocated_bytes);

        // Tensors should have distinct addresses
        EXPECT_NE(tensor1.buffer()->address(), tensor2.buffer()->address());
    }

    // All cleaned up after scope
    auto stats_after = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_after.total_allocated_bytes, stats_before.total_allocated_bytes);
}

// ============================================================================
// MockAllocator tests — verify experimental allocator state management
// ============================================================================

TEST_F(QueryOpConstraintsMockDevice, ExtractStateReturnsValidState) {
    auto* mock_alloc = experimental::get_mock_allocator(*mock_device_);
    ASSERT_NE(mock_alloc, nullptr);

    auto state = experimental::extract_mock_allocator_state(*mock_device_);

    // Fresh device should have no L1 allocations
    EXPECT_TRUE(state.is_empty(BufferType::L1));
}

TEST_F(QueryOpConstraintsMockDevice, OverrideStateRestoresCheckpoint) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 32, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Take empty checkpoint
    auto empty_state = experimental::extract_mock_allocator_state(*mock_device_);

    // Allocate a tensor (under NORMAL capture so it's a real allocation)
    experimental::MockAllocatorState state_with_tensor;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto tensor = ttnn::create_device_tensor(spec, mock_device_.get());
        state_with_tensor = experimental::extract_mock_allocator_state(*mock_device_);
        EXPECT_GT(state_with_tensor.total_allocated_size(), 0u);
    }

    // After tensor destroyed, allocator is clean
    auto stats_clean = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_clean.total_allocated_bytes, 0u);

    // Restore the state with tensor — allocator should show occupied memory
    experimental::override_mock_allocator_state(*mock_device_, state_with_tensor);
    auto stats_restored = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_GT(stats_restored.total_allocated_bytes, 0u);

    // Restore empty — allocator should be clean again
    experimental::override_mock_allocator_state(*mock_device_, empty_state);
    auto stats_empty = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_empty.total_allocated_bytes, 0u);
}

TEST_F(QueryOpConstraintsMockDevice, CheckpointRestoreWorkflow) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 32, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Simulate op1: allocate output, take checkpoint
    experimental::MockAllocatorState checkpoint_after_op1;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto op1_output = ttnn::create_device_tensor(spec, mock_device_.get());
        checkpoint_after_op1 = experimental::extract_mock_allocator_state(*mock_device_);
    }

    // Simulate op2 attempt A (L1 output): restore checkpoint, allocate more
    experimental::MockAllocatorState state_after_op2a;
    {
        experimental::override_mock_allocator_state(*mock_device_, checkpoint_after_op1);
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto op2_output = ttnn::create_device_tensor(spec, mock_device_.get());
        state_after_op2a = experimental::extract_mock_allocator_state(*mock_device_);
    }

    // Simulate op2 attempt B (backtrack): restore same checkpoint, try different config
    experimental::MockAllocatorState state_after_op2b;
    {
        experimental::override_mock_allocator_state(*mock_device_, checkpoint_after_op1);

        // Use DRAM config this time
        const auto dram_spec = tt::tt_metal::TensorSpec(
            ttnn::Shape(ttnn::Array4D{1, 1, 32, 64}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));

        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto op2_output = ttnn::create_device_tensor(dram_spec, mock_device_.get());
        state_after_op2b = experimental::extract_mock_allocator_state(*mock_device_);
    }

    // DRAM path should have less L1 usage than L1 path
    EXPECT_LT(
        state_after_op2b.total_allocated_size(BufferType::L1), state_after_op2a.total_allocated_size(BufferType::L1));
}

// ============================================================================
// query_op_constraints op tests
// ============================================================================

TEST_F(QueryOpConstraintsMockDevice, UnaryRelu) {
    const auto input_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    auto query = ttnn::graph::query_op_constraints(
        [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); },
        mock_device_.get(),
        input_spec,
        input_spec.tensor_layout().get_memory_config());

    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "Error: " << query.error_message.value_or("none");
    EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 0u);
    EXPECT_GT(query.resource_usage.peak_memory_usage_per_core, 0u);
    ASSERT_TRUE(query.output_tensor_specs.has_value());
    EXPECT_EQ(query.output_tensor_specs->size(), 1u);
}

TEST_F(QueryOpConstraintsMockDevice, UnaryReluHeightSharded) {
    const auto input_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{3, 1, 32 * 32, 32 * 32}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::TILE),
            MemoryConfig{
                TensorMemoryLayout::HEIGHT_SHARDED,
                BufferType::L1,
                ShardSpec{
                    CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                    {6 * 32, 32 * 32},
                    ShardOrientation::ROW_MAJOR}}));

    auto query = ttnn::graph::query_op_constraints(
        [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); },
        mock_device_.get(),
        input_spec,
        input_spec.tensor_layout().get_memory_config());

    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "Error: " << query.error_message.value_or("none");
    EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 0u);
    EXPECT_GT(query.resource_usage.peak_memory_usage_per_core, 0u);
    ASSERT_TRUE(query.output_tensor_specs.has_value());
    EXPECT_EQ(query.output_tensor_specs->size(), 1u);
}

TEST_F(QueryOpConstraintsMockDevice, BinaryAdd) {
    const auto spec_a = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{4, 2, 5 * 32, 7 * 32}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    const auto& spec_b = spec_a;

    constexpr ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> none{};

    auto query = ttnn::graph::query_op_constraints(
        [](auto&&... args) { return ttnn::add(args...); },
        mock_device_.get(),
        spec_a,
        spec_b,
        spec_a.data_type(),
        spec_a.tensor_layout().get_memory_config(),
        std::nullopt,
        none,
        none,
        none,
        std::nullopt);

    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "Error: " << query.error_message.value_or("none");
    EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 0u);
    EXPECT_GT(query.resource_usage.peak_memory_usage_per_core, 0u);
}

TEST_F(QueryOpConstraintsMockDevice, Matmul) {
    const auto spec_a = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    const auto spec_b = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 128, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    auto query = ttnn::graph::query_op_constraints(
        ttnn::matmul,
        mock_device_.get(),
        spec_a,
        spec_b,
        false,  // transpose_a
        false,  // transpose_b
        ttnn::L1_MEMORY_CONFIG,
        DataType::BFLOAT16,
        std::nullopt,   // program_config
        std::nullopt,   // activation
        std::nullopt,   // compute_kernel_config
        std::nullopt,   // core_grid
        std::nullopt,   // output_tile
        std::nullopt,   // optional_output_tensor
        std::nullopt,   // global_cb
        std::nullopt);  // sub_device_id

    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "Error: " << query.error_message.value_or("none");
    EXPECT_GT(query.resource_usage.cb_peak_size_per_core, 0u);
    EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 0u);
    EXPECT_GT(query.resource_usage.peak_memory_usage_per_core, 0u);
    ASSERT_TRUE(query.output_tensor_specs.has_value());
    EXPECT_EQ(query.output_tensor_specs->size(), 1u);
}

// ============================================================================
// Capture the ttnn-auto-selected program config through the uniform query API
// ============================================================================

TEST_F(QueryOpConstraintsMockDevice, MatmulProgramConfigCaptured) {
    const auto spec_a = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    const auto spec_b = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 128, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    auto initial_state = experimental::extract_mock_allocator_state(*mock_device_);
    ASSERT_TRUE(initial_state.is_empty(BufferType::L1));

    // Same op-agnostic entry point every op uses. No explicit program config: ttnn auto-selects one
    // internally, and the query's built-in capture records it into captured_config.
    auto out = ttnn::graph::query_op_constraints_with_initial_state(
        ttnn::matmul,
        mock_device_.get(),
        initial_state,
        spec_a,
        spec_b,
        false,  // transpose_a
        false,  // transpose_b
        ttnn::L1_MEMORY_CONFIG,
        DataType::BFLOAT16,
        std::nullopt,   // program_config
        std::nullopt,   // activation
        std::nullopt,   // compute_kernel_config
        std::nullopt,   // core_grid
        std::nullopt,   // output_tile
        std::nullopt,   // optional_output_tensor
        std::nullopt,   // global_cb
        std::nullopt);  // sub_device_id

    EXPECT_EQ(out.response.status, ttnn::graph::ExecutionStatus::Success)
        << "Error: " << out.response.error_message.value_or("none");

    // Captured, and unpacks (above the API) to the concrete matmul config type.
    ASSERT_TRUE(out.captured_config.has_value());
    const auto& captured = std::any_cast<const ttnn::operations::matmul::MatmulProgramConfig&>(*out.captured_config);

    // Cross-check: feeding the captured config back in as an explicit program config reproduces the
    // op's own kernel footprint — i.e. we captured the config the op actually ran, not an
    // approximation. This round-trip is the contract every op extractor must satisfy.
    auto verify = ttnn::graph::query_op_constraints_with_initial_state(
        ttnn::matmul,
        mock_device_.get(),
        initial_state,
        spec_a,
        spec_b,
        false,
        false,
        ttnn::L1_MEMORY_CONFIG,
        DataType::BFLOAT16,
        std::make_optional(captured),  // explicit config = the captured one
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt);

    EXPECT_EQ(verify.response.status, ttnn::graph::ExecutionStatus::Success)
        << "Error: " << verify.response.error_message.value_or("none");
    EXPECT_EQ(verify.response.resource_usage.cb_peak_size_per_core, out.response.resource_usage.cb_peak_size_per_core);
    EXPECT_EQ(
        verify.response.resource_usage.l1_buffers_peak_per_core, out.response.resource_usage.l1_buffers_peak_per_core);
}

// Requesting a width-sharded output drives ttnn to auto-select a 1D (multi-cast) matmul program
// config — a different variant than the interleaved case — and the query still captures it and
// round-trips it to the same footprint.
TEST_F(QueryOpConstraintsMockDevice, MatmulWidthShardedProgramConfigCaptured) {
    // M=512 (16 tiles), K=256 (8 tiles), N=1024 (32 tiles).
    const auto spec_a = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 512, 256}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    const auto spec_b = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 256, 1024}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Width-shard the [512, 1024] output across an 8-core row: each core holds the full height
    // (512) and 1024/8 = 128 columns.
    const auto width_sharded = MemoryConfig{
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec{
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 0}}}},
            {512, 128},
            ShardOrientation::ROW_MAJOR}};

    auto initial_state = experimental::extract_mock_allocator_state(*mock_device_);
    ASSERT_TRUE(initial_state.is_empty(BufferType::L1));

    auto out = ttnn::graph::query_op_constraints_with_initial_state(
        ttnn::matmul,
        mock_device_.get(),
        initial_state,
        spec_a,
        spec_b,
        false,          // transpose_a
        false,          // transpose_b
        width_sharded,  // width-sharded output memory config
        DataType::BFLOAT16,
        std::nullopt,   // program_config — let ttnn auto-select
        std::nullopt,   // activation
        std::nullopt,   // compute_kernel_config
        std::nullopt,   // core_grid
        std::nullopt,   // output_tile
        std::nullopt,   // optional_output_tensor
        std::nullopt,   // global_cb
        std::nullopt);  // sub_device_id

    EXPECT_EQ(out.response.status, ttnn::graph::ExecutionStatus::Success)
        << "Error: " << out.response.error_message.value_or("none");

    ASSERT_TRUE(out.captured_config.has_value());
    const auto& captured = std::any_cast<const ttnn::operations::matmul::MatmulProgramConfig&>(*out.captured_config);
    // A width-sharded output yields the 1D multi-cast config, not the interleaved 2D one.
    EXPECT_TRUE(
        std::holds_alternative<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(captured));

    auto verify = ttnn::graph::query_op_constraints_with_initial_state(
        ttnn::matmul,
        mock_device_.get(),
        initial_state,
        spec_a,
        spec_b,
        false,
        false,
        width_sharded,
        DataType::BFLOAT16,
        std::make_optional(captured),  // explicit config = the captured one
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt);

    EXPECT_EQ(verify.response.status, ttnn::graph::ExecutionStatus::Success)
        << "Error: " << verify.response.error_message.value_or("none");
    EXPECT_EQ(verify.response.resource_usage.cb_peak_size_per_core, out.response.resource_usage.cb_peak_size_per_core);
    EXPECT_EQ(
        verify.response.resource_usage.l1_buffers_peak_per_core, out.response.resource_usage.l1_buffers_peak_per_core);
}

// ============================================================================
// query_op_constraints_with_initial_state — pure state-in / state-out variant
// ============================================================================

TEST_F(QueryOpConstraintsMockDevice, WithInitialStateReturnsResponseAndNewState) {
    const auto input_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Start from an empty allocator state.
    auto initial_state = experimental::extract_mock_allocator_state(*mock_device_);
    ASSERT_TRUE(initial_state.is_empty(BufferType::L1));

    auto out = ttnn::graph::query_op_constraints_with_initial_state(
        [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); },
        mock_device_.get(),
        initial_state,
        input_spec,
        input_spec.tensor_layout().get_memory_config());

    // The embedded response behaves exactly like the existing API.
    EXPECT_EQ(out.response.status, ttnn::graph::ExecutionStatus::Success)
        << "Error: " << out.response.error_message.value_or("none");
    EXPECT_GT(out.response.resource_usage.l1_buffers_peak_per_core, 0u);
    ASSERT_TRUE(out.response.output_tensor_specs.has_value());
    EXPECT_EQ(out.response.output_tensor_specs->size(), 1u);

    // new_state reflects the op output allocated on top of the (empty) initial state.
    EXPECT_GT(out.new_state.total_allocated_size(BufferType::L1), 0u);

    // An op with no registered program-config extractor captures nothing; the always-on capture is
    // transparent to such ops.
    EXPECT_FALSE(out.captured_config.has_value());
}

TEST_F(QueryOpConstraintsMockDevice, WithInitialStateThreadsStateAcrossOps) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    auto relu = [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); };
    const auto mem_cfg = spec.tensor_layout().get_memory_config();

    auto empty = experimental::extract_mock_allocator_state(*mock_device_);

    // Op 1 from empty state.
    auto out1 = ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), empty, spec, mem_cfg);
    ASSERT_EQ(out1.response.status, ttnn::graph::ExecutionStatus::Success)
        << "Error: " << out1.response.error_message.value_or("none");
    const auto after_op1 = out1.new_state.total_allocated_size(BufferType::L1);
    EXPECT_GT(after_op1, 0u);

    // Op 2 threaded on op1's output state: op1's output is still live, so op2's output
    // is allocated on top of it — total occupancy strictly grows.
    auto out2 =
        ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), out1.new_state, spec, mem_cfg);
    ASSERT_EQ(out2.response.status, ttnn::graph::ExecutionStatus::Success)
        << "Error: " << out2.response.error_message.value_or("none");
    EXPECT_GT(out2.new_state.total_allocated_size(BufferType::L1), after_op1);
}

TEST_F(QueryOpConstraintsMockDevice, WithInitialStateIsPureAcrossCalls) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    auto relu = [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); };
    const auto mem_cfg = spec.tensor_layout().get_memory_config();

    auto empty = experimental::extract_mock_allocator_state(*mock_device_);

    // Same initial_state + same op => same resulting occupancy, regardless of prior calls.
    auto a = ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), empty, spec, mem_cfg);
    auto b = ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), empty, spec, mem_cfg);
    ASSERT_EQ(a.response.status, ttnn::graph::ExecutionStatus::Success);
    ASSERT_EQ(b.response.status, ttnn::graph::ExecutionStatus::Success);
    EXPECT_EQ(a.new_state.total_allocated_size(BufferType::L1), b.new_state.total_allocated_size(BufferType::L1));

    // The caller's initial_state is not mutated.
    EXPECT_TRUE(empty.is_empty(BufferType::L1));
}

TEST_F(QueryOpConstraintsMockDevice, OptionalStateNulloptRunsStatelessQuery) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    auto relu = [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); };
    const auto mem_cfg = spec.tensor_layout().get_memory_config();

    // std::nullopt => dispatch to the stateless query: same response as query_op_constraints,
    // and an empty new_state (nothing was allocated against a supplied state).
    auto out =
        ttnn::graph::query_op_constraints_with_optional_state(relu, mock_device_.get(), std::nullopt, spec, mem_cfg);

    ASSERT_EQ(out.response.status, ttnn::graph::ExecutionStatus::Success)
        << "Error: " << out.response.error_message.value_or("none");
    EXPECT_GT(out.response.resource_usage.l1_buffers_peak_per_core, 0u);
    ASSERT_TRUE(out.response.output_tensor_specs.has_value());
    EXPECT_EQ(out.response.output_tensor_specs->size(), 1u);
    EXPECT_TRUE(out.new_state.is_empty(BufferType::L1));

    // The embedded response matches the default stateless API for the same op.
    auto baseline = ttnn::graph::query_op_constraints(relu, mock_device_.get(), spec, mem_cfg);
    ASSERT_EQ(baseline.status, ttnn::graph::ExecutionStatus::Success);
    EXPECT_EQ(out.response.resource_usage.l1_buffers_peak_per_core, baseline.resource_usage.l1_buffers_peak_per_core);
    EXPECT_EQ(
        out.response.resource_usage.peak_memory_usage_per_core, baseline.resource_usage.peak_memory_usage_per_core);
}

TEST_F(QueryOpConstraintsMockDevice, OptionalStateWithValueMatchesStatefulCore) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    auto relu = [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); };
    const auto mem_cfg = spec.tensor_layout().get_memory_config();

    auto empty = experimental::extract_mock_allocator_state(*mock_device_);

    // With a state present, the dispatcher delegates to the stateful core: same resulting occupancy.
    auto via_core =
        ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), empty, spec, mem_cfg);
    auto via_dispatch = ttnn::graph::query_op_constraints_with_optional_state(
        relu, mock_device_.get(), std::optional<experimental::MockAllocatorState>(empty), spec, mem_cfg);

    ASSERT_EQ(via_core.response.status, ttnn::graph::ExecutionStatus::Success);
    ASSERT_EQ(via_dispatch.response.status, ttnn::graph::ExecutionStatus::Success);
    EXPECT_GT(via_dispatch.new_state.total_allocated_size(BufferType::L1), 0u);
    EXPECT_EQ(
        via_dispatch.new_state.total_allocated_size(BufferType::L1),
        via_core.new_state.total_allocated_size(BufferType::L1));
}

TEST_F(QueryOpConstraintsMockDevice, WithAllocationsReproducesQueryPlacement) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    auto relu = [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); };
    const auto mem_cfg = spec.tensor_layout().get_memory_config();

    auto empty_template = experimental::extract_mock_allocator_state(*mock_device_);
    auto out =
        ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), empty_template, spec, mem_cfg);
    ASSERT_EQ(out.response.status, ttnn::graph::ExecutionStatus::Success)
        << "Error: " << out.response.error_message.value_or("none");
    ASSERT_EQ(out.output_allocations.size(), 1u);
    EXPECT_EQ(out.output_allocations[0].buffer_type, BufferType::L1);
    EXPECT_GT(out.output_allocations[0].size_per_bank, 0u);

    // Rebuilding the state from the output's record must reproduce the allocator's real placement.
    auto built = empty_template.with_allocations(out.output_allocations);
    EXPECT_EQ(built.total_allocated_size(BufferType::L1), out.new_state.total_allocated_size(BufferType::L1));
    EXPECT_EQ(built.lowest_occupied(BufferType::L1), out.new_state.lowest_occupied(BufferType::L1));
}

TEST_F(QueryOpConstraintsMockDevice, WithAllocationsDropRecordFreesSpace) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    auto relu = [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); };
    const auto mem_cfg = spec.tensor_layout().get_memory_config();

    auto empty_template = experimental::extract_mock_allocator_state(*mock_device_);

    // Collect two live records by threading op 2 on op 1's state.
    auto out1 =
        ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), empty_template, spec, mem_cfg);
    ASSERT_EQ(out1.response.status, ttnn::graph::ExecutionStatus::Success);
    auto out2 =
        ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), out1.new_state, spec, mem_cfg);
    ASSERT_EQ(out2.response.status, ttnn::graph::ExecutionStatus::Success);

    const auto rec1 = out1.output_allocations[0];
    const auto rec2 = out2.output_allocations[0];

    auto both = empty_template.with_allocations({rec1, rec2});
    auto dropped = empty_template.with_allocations({rec2});  // "evict" op 1 = drop its record

    EXPECT_EQ(both.total_allocated_size(BufferType::L1), rec1.size_per_bank + rec2.size_per_bank);
    EXPECT_EQ(dropped.total_allocated_size(BufferType::L1), rec2.size_per_bank);
    EXPECT_FALSE(dropped.is_empty(BufferType::L1));
}

TEST_F(QueryOpConstraintsMockDevice, WithAllocationsFreedSlotIsReused) {
    // Proves build-from-records ≡ regular dealloc placement: after dropping a record, a fresh
    // allocation reuses the freed slot exactly as it would on a real device.
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    auto relu = [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); };
    const auto mem_cfg = spec.tensor_layout().get_memory_config();

    auto empty_template = experimental::extract_mock_allocator_state(*mock_device_);

    auto out1 =
        ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), empty_template, spec, mem_cfg);
    ASSERT_EQ(out1.response.status, ttnn::graph::ExecutionStatus::Success);
    auto out2 =
        ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), out1.new_state, spec, mem_cfg);
    ASSERT_EQ(out2.response.status, ttnn::graph::ExecutionStatus::Success);
    const auto rec1 = out1.output_allocations[0];
    const auto rec2 = out2.output_allocations[0];

    // Drop rec1, then run a new op against the rebuilt state: its output reuses rec1's freed slot.
    auto after_drop = empty_template.with_allocations({rec2});
    auto out3 =
        ttnn::graph::query_op_constraints_with_initial_state(relu, mock_device_.get(), after_drop, spec, mem_cfg);
    ASSERT_EQ(out3.response.status, ttnn::graph::ExecutionStatus::Success);
    const auto rec3 = out3.output_allocations[0];

    EXPECT_EQ(rec3.address, rec1.address) << "freed slot should be reused";
    EXPECT_EQ(rec3.size_per_bank, rec1.size_per_bank);
}

TEST_F(QueryOpConstraintsMockDevice, WithInitialStateReportsOomAsError) {
    // An L1 output far larger than total L1 cannot be allocated in Phase 2 -> Error.
    const auto huge = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 16384, 16384}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    auto relu = [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); };

    auto empty = experimental::extract_mock_allocator_state(*mock_device_);
    auto out = ttnn::graph::query_op_constraints_with_initial_state(
        relu, mock_device_.get(), empty, huge, huge.tensor_layout().get_memory_config());

    EXPECT_EQ(out.response.status, ttnn::graph::ExecutionStatus::Error);
    EXPECT_TRUE(out.response.error_message.has_value());
}

// ============================================================================
// Coexistence: mock allocator probed while a real silicon device is open in
// the same process. Validates the production scenario from issue #38445 — a
// graph-planning component (forge / tt-mlir) probing mock allocator state
// while another component drives real-device inference.
// ============================================================================

class MockAllocatorCoexistence : public ::testing::Test {
protected:
    std::unique_ptr<MetalEnv> silicon_env_;
    std::shared_ptr<distributed::MeshDevice> real_device_;
    std::unique_ptr<MetalEnv> mock_env_;
    std::shared_ptr<distributed::MeshDevice> mock_device_;

    void SetUp() override {
        silicon_env_ = std::make_unique<MetalEnv>();
        real_device_ =
            silicon_env_->create_mesh_device(distributed::MeshDeviceConfig(silicon_env_->get_system_mesh().shape()));
        ASSERT_GT(real_device_->num_devices(), 0u);

        mock_env_ = std::make_unique<MetalEnv>(
            MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1)));
        mock_device_ =
            mock_env_->create_mesh_device(distributed::MeshDeviceConfig(mock_env_->get_system_mesh().shape()));
        ASSERT_EQ(mock_device_->num_devices(), 1u);
    }

    void TearDown() override {
        mock_device_.reset();
        mock_env_.reset();
        real_device_.reset();
        silicon_env_.reset();
    }
};

TEST_F(MockAllocatorCoexistence, CheckpointRestoreWhileRealDeviceOpen) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 32, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Empty checkpoint while real device is open
    auto empty_state = experimental::extract_mock_allocator_state(*mock_device_);

    // Allocate on mock under graph capture
    experimental::MockAllocatorState state_with_tensor;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto tensor = ttnn::create_device_tensor(spec, mock_device_.get());
        state_with_tensor = experimental::extract_mock_allocator_state(*mock_device_);
        EXPECT_GT(state_with_tensor.total_allocated_size(), 0u);
    }

    // Tensor destructed; mock allocator clean
    auto stats_clean = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_clean.total_allocated_bytes, 0u);

    // Restore allocated state — mock allocator shows occupied memory
    experimental::override_mock_allocator_state(*mock_device_, state_with_tensor);
    auto stats_restored = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_GT(stats_restored.total_allocated_bytes, 0u);

    // Restore empty
    experimental::override_mock_allocator_state(*mock_device_, empty_state);
    auto stats_empty = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_empty.total_allocated_bytes, 0u);

    // Real device should still be open and reachable
    EXPECT_GT(real_device_->num_devices(), 0u);
}

TEST_F(MockAllocatorCoexistence, GraphCaptureOnMockDoesNotAffectRealDevice) {
    const auto spec = tt::tt_metal::TensorSpec(
        ttnn::Shape(ttnn::Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    auto real_stats_before = real_device_->allocator()->get_statistics(BufferType::L1);

    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto tensor = ttnn::create_device_tensor(spec, mock_device_.get());
        EXPECT_TRUE(tensor.is_allocated());
        // Mock allocator advanced
        auto mock_state = experimental::extract_mock_allocator_state(*mock_device_);
        EXPECT_GT(mock_state.total_allocated_size(), 0u);
    }

    // Real device's allocator state untouched by mock activity
    auto real_stats_after = real_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(real_stats_after.total_allocated_bytes, real_stats_before.total_allocated_bytes);
}

}  // namespace tt::tt_metal
