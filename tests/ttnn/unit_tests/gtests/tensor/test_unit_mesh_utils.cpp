// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <numeric>
#include <vector>

#include <tracy/Tracy.hpp>
#include <tt-metalium/bfloat16.hpp>
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
#include "ttnn/operations/eltwise/binary/binary.hpp"
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

    const uint32_t tensor_height = 32;
    const uint32_t tensor_width = 32;
    const size_t tensor_size = tensor_height * tensor_width;
    const float add_scalar = 1.0f;

    // Create unit tensors with known values: each tensor filled with its device index
    std::vector<Tensor> unit_tensors;
    unit_tensors.reserve(unit_meshes.size());
    {
        ZoneScopedN("CreateUnitTensors");
        for (size_t i = 0; i < unit_meshes.size(); ++i) {
            // Create host tensor filled with device index value
            std::vector<bfloat16> host_data(tensor_size, bfloat16(static_cast<float>(i)));
            auto tensor_spec = tt::tt_metal::TensorSpec(
                ttnn::Shape(std::array<uint32_t, 2>{tensor_height, tensor_width}),
                tt::tt_metal::TensorLayout(
                    tt::tt_metal::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                    tt::tt_metal::MemoryConfig()));

            // Create tensor from host data and move to device
            auto host_tensor = Tensor::from_vector(host_data, tensor_spec);
            unit_tensors.push_back(host_tensor.to_device(unit_meshes[i].get()));
        }
    }

    // Pre-allocate add outputs for each unit tensor (to reuse in trace)
    std::vector<Tensor> add_outputs;
    add_outputs.reserve(unit_tensors.size());
    Tensor all_gathered_output;

    // Run full flow during warmup to:
    // 1. Populate program cache
    // 2. Create semaphores (which involves writes)
    // 3. Get the correctly-shaped output tensors for reuse in trace
    {
        ZoneScopedN("CacheWarmup");

        // Run add with scalar on each unit tensor
        {
            ZoneScopedN("RunAddOps");
            for (const auto& unit_tensor : unit_tensors) {
                add_outputs.push_back(ttnn::add(unit_tensor, add_scalar));
            }
        }

        // Aggregate the add outputs
        Tensor aggregated_add;
        {
            ZoneScopedN("Aggregate");
            aggregated_add = aggregate(add_outputs);
        }

        // Quiesce the parent mesh before all gather
        {
            ZoneScopedN("QuiesceDevices");
            mesh_device_->quiesce_devices();
        }

        {
            ZoneScopedN("AllGather");
            // Use cluster_axis=1 (columns) for 2x4 mesh -> ring_size=4
            all_gathered_output = ttnn::all_gather(aggregated_add, /*dim=*/0, /*cluster_axis=*/1);
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

    // Verify correctness of all_gather results
    // For a 2x4 mesh with cluster_axis=1 (columns), each row gathers from 4 column devices
    // Row 0 devices: 0, 1, 2, 3 (indices in unit_meshes)
    // Row 1 devices: 4, 5, 6, 7 (indices in unit_meshes)
    // After all_gather on dim=0, output shape is [4*32, 32] = [128, 32]
    {
        ZoneScopedN("VerifyResults");

        auto disaggregated_tensors = disaggregate(all_gathered_output);
        ASSERT_THAT(disaggregated_tensors, SizeIs(unit_meshes.size()));

        const uint32_t ring_size = 4;  // 4 columns
        const uint32_t expected_height = ring_size * tensor_height;

        for (size_t device_idx = 0; device_idx < disaggregated_tensors.size(); ++device_idx) {
            auto& output_tensor = disaggregated_tensors[device_idx];
            auto output_shape = output_tensor.logical_shape();

            // Verify output shape
            EXPECT_EQ(output_shape[0], expected_height) << "Device " << device_idx << " has wrong output height";
            EXPECT_EQ(output_shape[1], tensor_width) << "Device " << device_idx << " has wrong output width";

            // Read back data and verify values
            auto output_data = output_tensor.to_vector<bfloat16>();
            ASSERT_EQ(output_data.size(), expected_height * tensor_width)
                << "Device " << device_idx << " has wrong output size";

            // Determine which row this device is in (0 or 1)
            size_t row = device_idx / ring_size;

            // Each chunk of 32x32 should contain data from the corresponding column device in this row
            // The order depends on all_gather implementation, but typically it's the gather order
            for (uint32_t chunk = 0; chunk < ring_size; ++chunk) {
                // The source device for this chunk is in the same row, column = chunk
                size_t source_device = row * ring_size + chunk;
                float expected_value = static_cast<float>(source_device) + add_scalar;

                // Check a sample of values in this chunk
                size_t chunk_start = chunk * tensor_height * tensor_width;
                for (size_t j = 0; j < std::min<size_t>(10, tensor_size); ++j) {
                    float actual_value = static_cast<float>(output_data[chunk_start + j]);
                    EXPECT_NEAR(actual_value, expected_value, 0.1f)
                        << "Device " << device_idx << ", chunk " << chunk << ", element " << j << ": expected "
                        << expected_value << " (from device " << source_device << ")";
                }
            }
        }
    }

    // Trace capture on each unit mesh individually
    // Captures full flow: add -> aggregate -> all_gather
    //
    // Operations on the parent mesh (like all_gather) internally route through the
    // submesh's command queues via the workaround in FDMeshCommandQueue::enqueue_mesh_workload.
    // When submeshes have active traces, parent mesh workloads are dispatched through
    // the submesh CQs, allowing the submesh traces to capture everything.
    std::vector<distributed::MeshTraceId> unit_trace_ids;
    unit_trace_ids.reserve(unit_meshes.size());

    {
        ZoneScopedN("TraceCapture");

        // Begin trace capture on ALL unit meshes simultaneously
        {
            ZoneScopedN("BeginTraces");
            for (size_t i = 0; i < unit_meshes.size(); ++i) {
                uint8_t cq_id = unit_meshes[i]->mesh_command_queue().id();
                auto trace_id = distributed::BeginTraceCapture(unit_meshes[i].get(), cq_id);
                unit_trace_ids.push_back(trace_id);
            }
        }

        // Run add on each unit tensor with pre-allocated outputs
        {
            ZoneScopedN("RunAddOps");
            for (size_t i = 0; i < unit_tensors.size(); ++i) {
                ttnn::add(unit_tensors[i], add_scalar, std::nullopt, std::nullopt, add_outputs[i]);
            }
        }

        // Aggregate the add outputs (host-side view operation)
        Tensor aggregated_add;
        {
            ZoneScopedN("Aggregate");
            aggregated_add = aggregate(add_outputs);
        }

        // NOTE: Cannot quiesce during trace capture - it tries to reset worker state
        // which is not allowed. Quiescing would need to be handled differently
        // (perhaps as a traced synchronization primitive).

        // Run all_gather on aggregated tensor
        // CURRENT LIMITATION: all_gather goes through parent mesh's CQ, not submesh CQs.
        // For this to be captured in submesh traces, we need the architectural change
        // where parent mesh ops internally route through submesh command queues.
        {
            ZoneScopedN("AllGather");
            ttnn::all_gather(
                aggregated_add,
                /*dim=*/0,
                /*cluster_axis=*/1,
                /*subdevice_id=*/std::nullopt,
                /*memory_config=*/std::nullopt,
                /*output_tensor=*/all_gathered_output);
        }

        // End traces on each unit mesh
        {
            ZoneScopedN("EndTraces");
            for (size_t i = 0; i < unit_meshes.size(); ++i) {
                uint8_t cq_id = unit_meshes[i]->mesh_command_queue().id();
                unit_meshes[i]->end_mesh_trace(cq_id, unit_trace_ids[i]);
            }
        }
    }

    // Measure TraceExecution - replay all unit traces
    for (int j = 0; j < 10; j++) {
        ZoneScopedN("TraceExecution");

        // Replay traces on each unit mesh (contains full flow: add + all_gather)
        {
            ZoneScopedN("ReplayUnitTraces");
            for (size_t i = 0; i < unit_meshes.size(); ++i) {
                uint8_t cq_id = unit_meshes[i]->mesh_command_queue().id();
                unit_meshes[i]->replay_mesh_trace(cq_id, unit_trace_ids[i], /*blocking=*/false);
            }
        }
    }

    // Quiesce to ensure all trace replays complete before verification
    mesh_device_->quiesce_devices();

    // Verify correctness after trace replay
    auto disaggregated_tensors = disaggregate(all_gathered_output);
    ASSERT_THAT(disaggregated_tensors, SizeIs(unit_meshes.size()));

    const uint32_t ring_size = 4;  // 4 columns
    const uint32_t expected_height = ring_size * tensor_height;

    for (size_t device_idx = 0; device_idx < disaggregated_tensors.size(); ++device_idx) {
        auto& output_tensor = disaggregated_tensors[device_idx];
        auto output_shape = output_tensor.logical_shape();

        // Verify output shape
        EXPECT_EQ(output_shape[0], expected_height)
            << "Trace replay: Device " << device_idx << " has wrong output height";
        EXPECT_EQ(output_shape[1], tensor_width) << "Trace replay: Device " << device_idx << " has wrong output width";

        // Read back data and verify values
        auto output_data = output_tensor.to_vector<bfloat16>();
        ASSERT_EQ(output_data.size(), expected_height * tensor_width)
            << "Trace replay: Device " << device_idx << " has wrong output size";

        // Determine which row this device is in (0 or 1)
        size_t row = device_idx / ring_size;

        // Each chunk of 32x32 should contain data from the corresponding column device in this row
        for (uint32_t chunk = 0; chunk < ring_size; ++chunk) {
            // The source device for this chunk is in the same row, column = chunk
            size_t source_device = row * ring_size + chunk;
            float expected_value = static_cast<float>(source_device) + add_scalar;

            // Check a sample of values in this chunk
            size_t chunk_start = chunk * tensor_height * tensor_width;
            for (size_t j = 0; j < std::min<size_t>(10, tensor_size); ++j) {
                float actual_value = static_cast<float>(output_data[chunk_start + j]);
                EXPECT_NEAR(actual_value, expected_value, 0.1f)
                    << "Trace replay: Device " << device_idx << ", chunk " << chunk << ", element " << j
                    << ": expected " << expected_value << " (from device " << source_device << ")";
            }
        }
    }

    // Release all traces
    for (size_t i = 0; i < unit_meshes.size(); ++i) {
        unit_meshes[i]->release_mesh_trace(unit_trace_ids[i]);
    }
    FrameMark;  // Mark the end of a frame
}

}  // namespace
}  // namespace tt::tt_metal::experimental::unit_mesh
