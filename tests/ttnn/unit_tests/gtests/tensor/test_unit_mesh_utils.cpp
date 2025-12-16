// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include <tracy/Tracy.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/distributed.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/tensor/unit_mesh/unit_mesh_utils.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::experimental::unit_mesh {
namespace {

using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::ThrowsMessage;
using ::tt::tt_metal::distributed::MeshShape;

using UnitMeshUtils2x4Test = ::tt::tt_metal::MeshDevice2x4Fixture;

class UnitMeshUtils2x4FabricTest : public ::tt::tt_metal::MeshDeviceFixtureBase {
protected:
    UnitMeshUtils2x4FabricTest() :
        ::tt::tt_metal::MeshDeviceFixtureBase(::tt::tt_metal::MeshDeviceFixtureBase::Config{
            .mesh_shape = ::tt::tt_metal::distributed::MeshShape{2, 4},
            .trace_region_size = 64 * 1024,  // 64KB for trace capture
            .fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D}) {}
};

TEST_F(UnitMeshUtils2x4Test, AggregateAndDisaggregate) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));
    ASSERT_THAT(unit_meshes, SizeIs(mesh_device_->shape().mesh_size()));

    // Allocates and deallocates a buffer, returning the allocation address.
    // Used to probe where new buffers are being allocated on the parent mesh, as a proxy for the parent mesh allocator
    // state.
    auto get_parent_allocation_address = [&]() {
        auto buffer = tt::tt_metal::distributed::MeshBuffer::create(
            tt::tt_metal::distributed::ReplicatedBufferConfig{.size = 16 << 10},
            tt::tt_metal::distributed::DeviceLocalBufferConfig{
                .page_size = 1024, .buffer_type = tt::tt_metal::BufferType::DRAM},
            mesh_device_.get());
        EXPECT_TRUE(buffer->is_allocated());
        return buffer->address();
    };
    const auto initial_parent_address = get_parent_allocation_address();
    EXPECT_EQ(get_parent_allocation_address(), initial_parent_address);

    // Create a tensor spec for testing
    auto shape = ttnn::Shape(std::array<uint32_t, 2>{32, 32});
    auto dtype = tt::tt_metal::DataType::BFLOAT16;
    auto layout = tt::tt_metal::Layout::TILE;

    // Create tensors on each unit mesh at the same address, assuming deterministic lock-step allocation.
    std::vector<Tensor> unit_tensors;
    unit_tensors.reserve(unit_meshes.size());

    for (const auto& unit_mesh : unit_meshes) {
        auto tensor = create_device_tensor(
            tt::tt_metal::TensorSpec(
                shape,
                tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), tt::tt_metal::MemoryConfig())),
            unit_mesh.get());
        unit_tensors.push_back(tensor);
    }

    // Verify all tensors are at the same address
    auto reference_address = unit_tensors[0].mesh_buffer()->address();
    for (size_t i = 1; i < unit_tensors.size(); i++) {
        EXPECT_EQ(unit_tensors[i].mesh_buffer()->address(), reference_address);
    }

    // Test aggregate
    auto aggregated_tensor = aggregate(unit_tensors);
    EXPECT_NE(get_parent_allocation_address(), initial_parent_address);

    EXPECT_EQ(aggregated_tensor.device(), mesh_device_.get());
    EXPECT_EQ(aggregated_tensor.logical_shape(), shape);
    EXPECT_EQ(aggregated_tensor.dtype(), dtype);
    EXPECT_EQ(aggregated_tensor.layout(), layout);
    EXPECT_EQ(aggregated_tensor.mesh_buffer()->address(), reference_address);

    // Test disaggregate
    auto disaggregated_tensors = disaggregate(aggregated_tensor);

    ASSERT_THAT(disaggregated_tensors, SizeIs(unit_meshes.size()));

    for (size_t i = 0; i < disaggregated_tensors.size(); i++) {
        const auto& tensor = disaggregated_tensors[i];

        EXPECT_NE(tensor.device(), nullptr);
        EXPECT_EQ(tensor.device()->shape().mesh_size(), 1);
        EXPECT_EQ(tensor.logical_shape(), shape);
        EXPECT_EQ(tensor.dtype(), dtype);
        EXPECT_EQ(tensor.layout(), layout);
        EXPECT_EQ(tensor.mesh_buffer()->address(), reference_address);
    }
}

TEST_F(UnitMeshUtils2x4Test, AggregateEmptyVector) {
    std::vector<Tensor> empty_tensors;
    EXPECT_THAT(
        ([&]() { aggregate(empty_tensors); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Cannot aggregate empty tensor vector")));
}

TEST_F(UnitMeshUtils2x4Test, AggregateNonUnitMeshes) {
    auto non_unit_meshes = mesh_device_->create_submeshes(MeshShape(2, 2));
    auto shape = ttnn::Shape(std::array<uint32_t, 2>{32, 32});
    auto dtype = tt::tt_metal::DataType::BFLOAT16;
    auto layout = tt::tt_metal::Layout::TILE;

    std::vector<Tensor> tensors;
    tensors.reserve(non_unit_meshes.size());
    for (const auto& non_unit_mesh : non_unit_meshes) {
        tensors.push_back(create_device_tensor(
            tt::tt_metal::TensorSpec(
                shape,
                tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), tt::tt_metal::MemoryConfig())),
            non_unit_mesh.get()));
    }

    EXPECT_THAT(
        ([&]() { aggregate(tensors); }), ThrowsMessage<std::runtime_error>(HasSubstr("Expected unit mesh (1x1)")));
}

TEST_F(UnitMeshUtils2x4Test, AggregateMismatchedTensorSpecs) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));

    auto shape1 = ttnn::Shape(std::array<uint32_t, 2>{32, 32});
    auto shape2 = ttnn::Shape(std::array<uint32_t, 2>{64, 64});

    std::vector<Tensor> tensors;
    tensors.reserve(unit_meshes.size());
    for (int i = 0; i < unit_meshes.size(); i++) {
        if (i % 2 == 0) {
            tensors.push_back(create_device_tensor(
                tt::tt_metal::TensorSpec(
                    shape1,
                    tt::tt_metal::TensorLayout(
                        tt::tt_metal::DataType::BFLOAT16,
                        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                        tt::tt_metal::MemoryConfig())),
                unit_meshes[i].get()));
        } else {
            tensors.push_back(create_device_tensor(
                tt::tt_metal::TensorSpec(
                    shape2,
                    tt::tt_metal::TensorLayout(
                        tt::tt_metal::DataType::BFLOAT16,
                        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                        tt::tt_metal::MemoryConfig())),
                unit_meshes[i].get()));
        }
    }

    EXPECT_THAT(
        ([&]() { aggregate(tensors); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("All tensors must have the same TensorSpec")));
}

TEST_F(UnitMeshUtils2x4Test, AggregateMismatchedAddresses) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));

    auto shape = ttnn::Shape(std::array<uint32_t, 2>{64, 64});

    // Make an additional allocation on the first unit mesh to make the addresses mismatch.
    auto tensor = create_device_tensor(
        tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig())),
        unit_meshes[0].get());

    std::vector<Tensor> tensors;
    tensors.reserve(unit_meshes.size());
    for (const auto& unit_mesh : unit_meshes) {
        tensors.push_back(create_device_tensor(
            tt::tt_metal::TensorSpec(
                shape,
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    tt::tt_metal::MemoryConfig())),
            unit_mesh.get()));
    }

    EXPECT_THAT(
        ([&]() { aggregate(tensors); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("All mesh buffers must be at the same address")));
}

TEST_F(UnitMeshUtils2x4Test, AggregateWrongNumberOfTensors) {
    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));
    ASSERT_GE(unit_meshes.size(), 2);

    auto shape = ttnn::Shape(std::array<uint32_t, 2>{32, 32});

    auto tensor = create_device_tensor(
        tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig())),
        unit_meshes[0].get());

    std::vector<Tensor> tensors = {tensor};
    EXPECT_THAT(
        ([&]() { aggregate(tensors); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Input tensors must span the entire parent mesh")));
}

TEST_F(UnitMeshUtils2x4Test, DisaggregateWithoutSubmeshes) {
    // Create a tensor on the parent mesh directly (no submeshes created yet)
    auto shape = ttnn::Shape(std::array<uint32_t, 2>{32, 32});

    auto tensor = create_device_tensor(
        tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                tt::tt_metal::MemoryConfig())),
        mesh_device_.get());

    // Should throw because no submeshes exist
    EXPECT_THAT(
        ([&]() { disaggregate(tensor); }),
        ThrowsMessage<std::runtime_error>(HasSubstr("Number of submeshes (0) must match mesh size")));
}

TEST_F(UnitMeshUtils2x4FabricTest, MorehProfiling) {
    FrameMark;  // Mark the start of a frame for Tracy

    auto unit_meshes = mesh_device_->create_submeshes(MeshShape(1, 1));
    ASSERT_THAT(unit_meshes, SizeIs(mesh_device_->shape().mesh_size()));

    std::vector<Tensor> unit_tensors;
    unit_tensors.reserve(unit_meshes.size());
    {
        ZoneScopedN("CreateUnitTensors");
        for (const auto& unit_mesh : unit_meshes) {
            unit_tensors.push_back(create_device_tensor(
                tt::tt_metal::TensorSpec(
                    ttnn::Shape(std::array<uint32_t, 2>{32, 32}),
                    tt::tt_metal::TensorLayout(
                        tt::tt_metal::DataType::BFLOAT16,
                        tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                        tt::tt_metal::MemoryConfig())),
                unit_mesh.get()));
        }
    }

    // Aggregate unit tensors first (this is a host-side view operation)
    Tensor aggregated_tensor;
    {
        ZoneScopedN("Aggregate");
        aggregated_tensor = aggregate(unit_tensors);
    }

    // Tensors to be reused across warmup and trace
    Tensor abs_output;
    Tensor all_gathered_output;

    // Run full flow during warmup to:
    // 1. Populate program cache
    // 2. Create semaphores (which involves writes)
    // 3. Get the correctly-shaped output tensors for reuse in trace
    {
        ZoneScopedN("CacheWarmup");

        // Run abs on the aggregated tensor (operates on parent mesh)
        {
            ZoneScopedN("RunAbsOp");
            abs_output = ttnn::abs(aggregated_tensor);
        }

        // Quiesce the parent mesh before all gather
        {
            ZoneScopedN("QuiesceDevices");
            mesh_device_->quiesce_devices();
        }

        {
            ZoneScopedN("AllGather");
            // Use cluster_axis=1 (columns) for 2x4 mesh -> ring_size=4
            all_gathered_output = ttnn::all_gather(abs_output, /*dim=*/0, /*cluster_axis=*/1);
        }

        // Quiesce parent mesh after all gather to ensure command queues are finished
        {
            ZoneScopedN("QuiesceDevicesAfterAllGather");
            mesh_device_->quiesce_devices();
        }

        {
            ZoneScopedN("Disaggregate");
            auto disaggregated_tensors = disaggregate(all_gathered_output);
        }
    }

    // Trace capture - reuse same tensors for cache hit
    distributed::MeshTraceId trace_id;
    {
        ZoneScopedN("TraceCapture");

        // Begin trace on parent mesh (trace_region_size was set in fixture config)
        uint8_t cq_id = mesh_device_->mesh_command_queue().id();
        trace_id = distributed::BeginTraceCapture(mesh_device_.get(), cq_id);

        // Run abs on aggregated tensor with pre-allocated output
        {
            ZoneScopedN("RunAbsOp");
            ttnn::abs(aggregated_tensor, std::nullopt, abs_output);
        }

        {
            ZoneScopedN("AllGather");
            // Use same cluster_axis and pass pre-allocated output
            ttnn::all_gather(
                abs_output,
                /*dim=*/0,
                /*cluster_axis=*/1,
                /*subdevice_id=*/std::nullopt,
                /*memory_config=*/std::nullopt,
                /*output_tensor=*/all_gathered_output);
        }

        // End trace on parent mesh
        mesh_device_->end_mesh_trace(cq_id, trace_id);
    }

    // Measure TraceExecution
    for (int j = 0; j < 10; j++) {
        ZoneScopedN("TraceExecution");

        uint8_t cq_id = mesh_device_->mesh_command_queue().id();
        mesh_device_->replay_mesh_trace(cq_id, trace_id, /*blocking=*/false);
    }

    // Release trace and quiesce
    mesh_device_->release_mesh_trace(trace_id);
    mesh_device_->quiesce_devices();
    FrameMark;  // Mark the end of a frame
}

}  // namespace
}  // namespace tt::tt_metal::experimental::unit_mesh
