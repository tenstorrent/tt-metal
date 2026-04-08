// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests that query_op_constraints works on a mock device created via MetalEnv.
//
// STATUS: query_op_constraints CRASHES on MetalEnv mock devices.
// Root cause: CreateKernel -> Kernel::register_kernel_with_watcher() dereferences
// a null WatcherServer because MetalEnv mock contexts don't initialize the global
// watcher. Tracked in https://github.com/tenstorrent/tt-metal/issues/39849
//
// The op tests below are DISABLED_ until #39849 is resolved. The baseline tests
// verify that MetalEnv mock device creation and tensor allocation work correctly,
// confirming the issue is specifically in the kernel creation path.

#include <gtest/gtest.h>

#include <tt-metalium/allocator_state.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_allocator.hpp>
#include <tt-metalium/experimental/mock_device.hpp>
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
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
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
    }

    void TearDown() override {
        if (mock_device_) {
            mock_device_->close();
            mock_device_.reset();
        }
        mock_env_.reset();
    }
};

// ============================================================================
// Baseline tests — verify MetalEnv mock device plumbing works
// ============================================================================

TEST_F(QueryOpConstraintsMockDevice, MockDeviceCreationAndQueries) {
    EXPECT_EQ(mock_env_->get_arch(), tt::ARCH::WORMHOLE_B0);
    EXPECT_GT(mock_env_->get_l1_size(), 0u);
    EXPECT_GT(mock_env_->get_dram_alignment(), 0u);
    EXPECT_GT(mock_env_->get_l1_alignment(), 0u);
    EXPECT_EQ(mock_device_->num_devices(), 1u);
}

TEST_F(QueryOpConstraintsMockDevice, BufferAllocationOnMockDevice) {
    constexpr size_t page_size = 4096;
    constexpr size_t buffer_size = page_size * 12;
    distributed::DeviceLocalBufferConfig local_config{.page_size = buffer_size, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto buffer = distributed::MeshBuffer::create(buffer_config, local_config, mock_device_.get());
    EXPECT_GT(buffer->address(), 0u);
    EXPECT_TRUE(buffer->is_allocated());
    buffer->deallocate();
    EXPECT_FALSE(buffer->is_allocated());
}

TEST_F(QueryOpConstraintsMockDevice, DeviceTensorCreationInGraphCapture) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // create_device_tensor inside graph capture should work on mock device
    auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
    auto tensor = create_device_tensor(input_spec, mock_device_.get());
    EXPECT_TRUE(tensor.is_allocated());
}

// ============================================================================
// NORMAL mode graph capture tests — verify real allocator interaction
// ============================================================================

TEST_F(QueryOpConstraintsMockDevice, NormalModeGraphCaptureAllocatesRealMemory) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Record allocator state before
    auto stats_before = mock_device_->allocator()->get_statistics(BufferType::L1);

    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto tensor = create_device_tensor(input_spec, mock_device_.get());

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
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // NO_DISPATCH: allocation is dry (address=0, allocator untouched)
    size_t allocated_no_dispatch = 0;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
        auto tensor = create_device_tensor(input_spec, mock_device_.get());
        EXPECT_EQ(tensor.buffer()->address(), 0u);
        allocated_no_dispatch = mock_device_->allocator()->get_statistics(BufferType::L1).total_allocated_bytes;
    }

    // NORMAL: allocation is real (address > 0, allocator updated)
    size_t allocated_normal = 0;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto tensor = create_device_tensor(input_spec, mock_device_.get());
        EXPECT_GT(tensor.buffer()->address(), 0u);
        allocated_normal = mock_device_->allocator()->get_statistics(BufferType::L1).total_allocated_bytes;
    }

    EXPECT_GT(allocated_normal, allocated_no_dispatch);
}

TEST_F(QueryOpConstraintsMockDevice, NormalModeMultipleTensorsAccumulateState) {
    const auto spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 32, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    auto stats_before = mock_device_->allocator()->get_statistics(BufferType::L1);

    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);

        auto tensor1 = create_device_tensor(spec, mock_device_.get());
        auto stats_after_1 = mock_device_->allocator()->get_statistics(BufferType::L1);

        auto tensor2 = create_device_tensor(spec, mock_device_.get());
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

TEST_F(QueryOpConstraintsMockDevice, GetMockAllocatorReturnsNonNull) {
    auto* mock_alloc = experimental::get_mock_allocator(mock_device_.get());
    ASSERT_NE(mock_alloc, nullptr);
}

TEST_F(QueryOpConstraintsMockDevice, ExtractStateReturnsValidState) {
    auto* mock_alloc = experimental::get_mock_allocator(mock_device_.get());
    ASSERT_NE(mock_alloc, nullptr);

    auto state = experimental::extract_mock_allocator_state(mock_device_.get());

    // Fresh device should have no L1 allocations
    auto l1_regions = state.get_allocated_regions(BufferType::L1);
    EXPECT_EQ(l1_regions.size(), 0u);
}

TEST_F(QueryOpConstraintsMockDevice, OverrideStateRestoresCheckpoint) {
    const auto spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 32, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Take empty checkpoint
    auto empty_state = experimental::extract_mock_allocator_state(mock_device_.get());

    // Allocate a tensor (under NORMAL capture so it's a real allocation)
    AllocatorState state_with_tensor;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto tensor = create_device_tensor(spec, mock_device_.get());
        state_with_tensor = experimental::extract_mock_allocator_state(mock_device_.get());
        EXPECT_GT(state_with_tensor.total_allocated_size(), 0u);
    }

    // After tensor destroyed, allocator is clean
    auto stats_clean = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_clean.total_allocated_bytes, 0u);

    // Restore the state with tensor — allocator should show occupied memory
    experimental::override_mock_allocator_state(mock_device_.get(), state_with_tensor);
    auto stats_restored = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_GT(stats_restored.total_allocated_bytes, 0u);

    // Restore empty — allocator should be clean again
    experimental::override_mock_allocator_state(mock_device_.get(), empty_state);
    auto stats_empty = mock_device_->allocator()->get_statistics(BufferType::L1);
    EXPECT_EQ(stats_empty.total_allocated_bytes, 0u);
}

TEST_F(QueryOpConstraintsMockDevice, CheckpointRestoreWorkflow) {
    const auto spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 32, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Simulate op1: allocate output, take checkpoint
    AllocatorState checkpoint_after_op1;
    {
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto op1_output = create_device_tensor(spec, mock_device_.get());
        checkpoint_after_op1 = experimental::extract_mock_allocator_state(mock_device_.get());
    }

    // Simulate op2 attempt A (L1 output): restore checkpoint, allocate more
    AllocatorState state_after_op2a;
    {
        experimental::override_mock_allocator_state(mock_device_.get(), checkpoint_after_op1);
        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto op2_output = create_device_tensor(spec, mock_device_.get());
        state_after_op2a = experimental::extract_mock_allocator_state(mock_device_.get());
    }

    // Simulate op2 attempt B (backtrack): restore same checkpoint, try different config
    AllocatorState state_after_op2b;
    {
        experimental::override_mock_allocator_state(mock_device_.get(), checkpoint_after_op1);

        // Use DRAM config this time
        const auto dram_spec = ttnn::TensorSpec(
            ttnn::Shape(Array4D{1, 1, 32, 64}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));

        auto capture = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
        auto op2_output = create_device_tensor(dram_spec, mock_device_.get());
        state_after_op2b = experimental::extract_mock_allocator_state(mock_device_.get());
    }

    // DRAM path should have less L1 usage than L1 path
    size_t l1_usage_a = 0;
    for (const auto& [start, end] : state_after_op2a.get_allocated_regions(BufferType::L1)) {
        l1_usage_a += (end - start);
    }
    size_t l1_usage_b = 0;
    for (const auto& [start, end] : state_after_op2b.get_allocated_regions(BufferType::L1)) {
        l1_usage_b += (end - start);
    }
    EXPECT_LT(l1_usage_b, l1_usage_a);
}

// ============================================================================
// query_op_constraints tests — DISABLED until #39849 is resolved
//
// These segfault in WatcherServer::register_kernel because CreateKernel
// (called internally by ops like relu) dereferences the global WatcherServer,
// which is null for MetalEnv mock contexts.
// https://github.com/tenstorrent/tt-metal/issues/39849
// ============================================================================

TEST_F(QueryOpConstraintsMockDevice, DISABLED_UnaryRelu) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 64, 128}),
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

TEST_F(QueryOpConstraintsMockDevice, DISABLED_UnaryReluHeightSharded) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{3, 1, 32 * 32, 32 * 32}),
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

TEST_F(QueryOpConstraintsMockDevice, DISABLED_BinaryAdd) {
    const auto spec_a = ttnn::TensorSpec(
        ttnn::Shape(Array4D{4, 2, 5 * 32, 7 * 32}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    const auto spec_b = spec_a;

    constexpr tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> none{};

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
        false);

    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "Error: " << query.error_message.value_or("none");
    EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 0u);
    EXPECT_GT(query.resource_usage.peak_memory_usage_per_core, 0u);
}

TEST_F(QueryOpConstraintsMockDevice, DISABLED_Matmul) {
    const auto spec_a = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    const auto spec_b = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 128, 64}),
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
// configure_mock_mode tests — ops work via configure_mock_mode + open_mesh_device
//
// Unlike MetalEnv, the configure_mock_mode path initializes the full MetalContext
// including WatcherServer, so ops can run without the #39849 crash.
// ============================================================================

class QueryOpConstraintsMockMode : public ::testing::Test {
protected:
    std::shared_ptr<distributed::MeshDevice> device_;

    void SetUp() override {
        experimental::configure_mock_mode(tt::ARCH::WORMHOLE_B0, 1);
        device_ = ttnn::open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
        ASSERT_GT(device_->num_devices(), 0u);
    }

    void TearDown() override {
        if (device_) {
            device_->close();
            device_.reset();
        }
        experimental::disable_mock_mode();
    }
};

TEST_F(QueryOpConstraintsMockMode, UnaryRelu) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    auto query = ttnn::graph::query_op_constraints(
        [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); },
        device_.get(),
        input_spec,
        input_spec.tensor_layout().get_memory_config());

    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "Error: " << query.error_message.value_or("none");
    EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 0u);
    EXPECT_GT(query.resource_usage.peak_memory_usage_per_core, 0u);
    ASSERT_TRUE(query.output_tensor_specs.has_value());
    EXPECT_EQ(query.output_tensor_specs->size(), 1u);
}

TEST_F(QueryOpConstraintsMockMode, UnaryReluHeightSharded) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{3, 1, 32 * 32, 32 * 32}),
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
        device_.get(),
        input_spec,
        input_spec.tensor_layout().get_memory_config());

    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "Error: " << query.error_message.value_or("none");
    EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 0u);
    EXPECT_GT(query.resource_usage.peak_memory_usage_per_core, 0u);
    ASSERT_TRUE(query.output_tensor_specs.has_value());
    EXPECT_EQ(query.output_tensor_specs->size(), 1u);
}

TEST_F(QueryOpConstraintsMockMode, BinaryAdd) {
    const auto spec_a = ttnn::TensorSpec(
        ttnn::Shape(Array4D{4, 2, 5 * 32, 7 * 32}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));
    const auto spec_b = spec_a;

    constexpr tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> none{};

    auto query = ttnn::graph::query_op_constraints(
        [](auto&&... args) { return ttnn::add(args...); },
        device_.get(),
        spec_a,
        spec_b,
        spec_a.data_type(),
        spec_a.tensor_layout().get_memory_config(),
        std::nullopt,
        none,
        none,
        none,
        false);

    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "Error: " << query.error_message.value_or("none");
    EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 0u);
    EXPECT_GT(query.resource_usage.peak_memory_usage_per_core, 0u);
}

TEST_F(QueryOpConstraintsMockMode, Matmul) {
    const auto spec_a = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    const auto spec_b = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 128, 64}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    auto query = ttnn::graph::query_op_constraints(
        ttnn::matmul,
        device_.get(),
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
// MockAllocator state extraction with query_op_constraints
// ============================================================================

TEST_F(QueryOpConstraintsMockMode, GetMockAllocatorReturnsNonNull) {
    auto* mock_alloc = experimental::get_mock_allocator(device_.get());
    ASSERT_NE(mock_alloc, nullptr);
}

TEST_F(QueryOpConstraintsMockMode, ReluWithAllocatorStateExtraction) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape(Array4D{1, 1, 64, 128}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::L1_MEMORY_CONFIG));

    // Checkpoint before op
    auto state_before = experimental::extract_mock_allocator_state(device_.get());

    auto query = ttnn::graph::query_op_constraints(
        [](auto&&... args) { return ttnn::relu(std::forward<decltype(args)>(args)...); },
        device_.get(),
        input_spec,
        input_spec.tensor_layout().get_memory_config());

    EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success) << "Error: " << query.error_message.value_or("none");

    // Checkpoint after op — allocator state should be unchanged since
    // query_op_constraints cleans up allocations internally
    auto state_after = experimental::extract_mock_allocator_state(device_.get());

    // Restore original state to verify override works
    experimental::override_mock_allocator_state(device_.get(), state_before);
}

}  // namespace tt::tt_metal
