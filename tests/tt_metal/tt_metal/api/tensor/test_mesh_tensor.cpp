// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sanity tests for MeshTensor class.
//
// These are minimal compile-time and runtime checks to verify MeshTensor's
// basic type properties and construction semantics. MeshTensor enforces unique
// ownership of device memory: movable but non-copyable.
//
// Device-based tests use GenericMeshDeviceFixture and require hardware access.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/float8.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/math.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Type trait tests verifying MeshTensor's semantic constraints

TEST(MeshTensorTypeTraitsTest, IsDefaultConstructible) { EXPECT_FALSE(std::is_default_constructible_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsDestructible) { EXPECT_TRUE(std::is_destructible_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsNotCopyConstructible) { EXPECT_FALSE(std::is_copy_constructible_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsNotCopyAssignable) { EXPECT_FALSE(std::is_copy_assignable_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsMoveConstructible) { EXPECT_TRUE(std::is_move_constructible_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsMoveAssignable) { EXPECT_TRUE(std::is_move_assignable_v<MeshTensor>); }

TEST(MeshTensorTypeTraitsTest, IsNothrowMoveConstructible) {
    EXPECT_TRUE(std::is_nothrow_move_constructible_v<MeshTensor>);
}

TEST(MeshTensorTypeTraitsTest, IsNothrowMoveAssignable) { EXPECT_TRUE(std::is_nothrow_move_assignable_v<MeshTensor>); }

TEST(MeshTensorTest, ConstructionWithNullMeshBufferFails) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec = TensorSpec(Shape{1, 32}, tensor_layout);
    auto topology = TensorTopology();

    EXPECT_ANY_THROW(MeshTensor(nullptr, std::move(spec), std::move(topology)));
}

// Device-based tests using GenericMeshDeviceFixture
using MeshTensorDeviceTest = GenericMeshDeviceFixture;

auto create_mesh_buffer(
    distributed::MeshDevice& mesh_device, const TensorSpec& spec, float buffer_size_multiplier = 1.0f) {
    auto page_size = spec.compute_page_size_bytes();

    distributed::DeviceLocalBufferConfig local_config{
        .page_size = static_cast<uint32_t>(page_size),
        .buffer_type = BufferType::DRAM,
    };

    auto buffer_size_needed = spec.compute_packed_buffer_size_bytes();
    auto scaled_size = static_cast<DeviceAddr>(buffer_size_needed * buffer_size_multiplier);
    // Round down to nearest multiple of page_size (buffer_size % page_size must == 0).
    // Ensure minimum of one page.
    auto buffer_size = std::max(round_down(scaled_size, page_size), page_size);

    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    return distributed::MeshBuffer::create(buffer_config, local_config, &mesh_device);
}

TEST_F(MeshTensorDeviceTest, ConstructionWithMeshBuffer) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec = TensorSpec(Shape{1, 32}, tensor_layout);

    auto mesh_buffer = create_mesh_buffer(*mesh_device_, spec);
    ASSERT_NE(mesh_buffer, nullptr);
    ASSERT_TRUE(mesh_buffer->is_allocated());

    auto topology = TensorTopology();

    MeshTensor tensor(mesh_buffer, std::move(spec), std::move(topology));

    EXPECT_EQ(&tensor.mesh_buffer(), mesh_buffer.get());
    EXPECT_EQ(&tensor.device(), mesh_device_.get());
    EXPECT_EQ(tensor.dtype(), DataType::BFLOAT16);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), Shape({1, 32}));
}

TEST_F(MeshTensorDeviceTest, MoveConstructionTransfersOwnership) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec = TensorSpec(Shape{2, 64}, tensor_layout);

    auto mesh_buffer = create_mesh_buffer(*mesh_device_, spec);
    auto topology = TensorTopology();

    MeshTensor original(mesh_buffer, std::move(spec), std::move(topology));
    const auto* buffer_ptr = &original.mesh_buffer();

    MeshTensor moved(std::move(original));

    EXPECT_EQ(&moved.mesh_buffer(), buffer_ptr);
    EXPECT_EQ(moved.logical_shape(), Shape({2, 64}));
}

TEST_F(MeshTensorDeviceTest, MoveAssignmentTransfersOwnership) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);

    auto spec1 = TensorSpec(Shape{1, 32}, tensor_layout);
    auto spec2 = TensorSpec(Shape{2, 64}, tensor_layout);

    auto mesh_buffer1 = create_mesh_buffer(*mesh_device_, spec1);
    auto mesh_buffer2 = create_mesh_buffer(*mesh_device_, spec2);

    MeshTensor tensor1(mesh_buffer1, std::move(spec1), TensorTopology());
    MeshTensor tensor2(mesh_buffer2, std::move(spec2), TensorTopology());

    const auto* buffer1_ptr = &tensor1.mesh_buffer();

    tensor2 = std::move(tensor1);

    EXPECT_EQ(&tensor2.mesh_buffer(), buffer1_ptr);
    EXPECT_EQ(tensor2.logical_shape(), Shape({1, 32}));
}

TEST_F(MeshTensorDeviceTest, TensorProperties) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::FLOAT32, page_config, memory_config);
    auto spec = TensorSpec(Shape{4, 8, 16}, tensor_layout);

    auto mesh_buffer = create_mesh_buffer(*mesh_device_, spec);
    auto topology = TensorTopology();

    MeshTensor tensor(mesh_buffer, std::move(spec), std::move(topology));

    EXPECT_EQ(tensor.dtype(), DataType::FLOAT32);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), Shape({4, 8, 16}));
    EXPECT_EQ(tensor.logical_volume(), 4 * 8 * 16);
    EXPECT_EQ(tensor.element_size(), sizeof(float));
    EXPECT_FALSE(tensor.is_sharded());
    EXPECT_EQ(tensor.memory_config().buffer_type(), BufferType::DRAM);
}

TEST_F(MeshTensorDeviceTest, IsValuelessAfterMoveReturnsFalse) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec = TensorSpec(Shape{1, 32}, tensor_layout);

    // A freshly constructed tensor is not valueless.
    auto tensor = MeshTensor::allocate_on_device(*mesh_device_, spec, TensorTopology());
    EXPECT_FALSE(tensor.is_valueless_after_move());
}

TEST_F(MeshTensorDeviceTest, IsValuelessAfterMoveReturnsTrueAfterMoveConstruction) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec = TensorSpec(Shape{1, 32}, tensor_layout);

    auto original = MeshTensor::allocate_on_device(*mesh_device_, spec, TensorTopology());
    MeshTensor moved(std::move(original));

    // Intentional use-after-move: verifying valueless-after-move semantics.
    EXPECT_TRUE(original.is_valueless_after_move());  // NOLINT(bugprone-use-after-move)
    EXPECT_FALSE(moved.is_valueless_after_move());    // NOLINT(bugprone-use-after-move)
}

TEST_F(MeshTensorDeviceTest, IsValuelessAfterMoveReturnsTrueAfterMoveAssignment) {
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    auto spec1 = TensorSpec(Shape{1, 32}, tensor_layout);
    auto spec2 = TensorSpec(Shape{2, 64}, tensor_layout);

    auto source = MeshTensor::allocate_on_device(*mesh_device_, spec1, TensorTopology());
    auto target = MeshTensor::allocate_on_device(*mesh_device_, spec2, TensorTopology());

    target = std::move(source);

    // Intentional use-after-move: verifying valueless-after-move semantics.
    EXPECT_TRUE(source.is_valueless_after_move());   // NOLINT(bugprone-use-after-move)
    EXPECT_FALSE(target.is_valueless_after_move());  // NOLINT(bugprone-use-after-move)
}

TEST_F(MeshTensorDeviceTest, ConstructionWithTooSmallBufferFails) {
    // Create a TensorSpec that requires multiple pages, then create a buffer
    // that's too small to hold the tensor but still valid (multiple of page_size).
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    // Use a larger shape to ensure multiple pages are required.
    auto spec = TensorSpec(Shape{4, 512}, tensor_layout);

    const size_t required_size = spec.compute_packed_buffer_size_bytes();
    const size_t page_size = spec.compute_page_size_bytes();
    ASSERT_GT(required_size, page_size);  // Ensure we need multiple pages

    // Create a buffer at half the required size (rounded to page boundary).
    auto mesh_buffer = create_mesh_buffer(*mesh_device_, spec, 0.5f);
    ASSERT_NE(mesh_buffer, nullptr);
    ASSERT_TRUE(mesh_buffer->is_allocated());
    ASSERT_LT(mesh_buffer->size(), required_size);

    auto topology = TensorTopology();

    EXPECT_ANY_THROW(MeshTensor(mesh_buffer, std::move(spec), std::move(topology)));
}

// FP8_E4M3 device-side support is intentionally narrow. The TensorSpec validator enforces
// ROW_MAJOR layout for FP8_E4M3; MeshTensor::allocate_on_device adds the Blackhole-arch
// requirement (FP8_E4M3's only producer today is the DeepSeek V3 Prefill combine op).
// These tests pin both constraints and confirm a plain H2D->D2H round-trip works.

TensorSpec make_fp8_row_major_spec(const Shape& shape) {
    return TensorSpec(
        shape,
        TensorLayout(
            DataType::FP8_E4M3,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
}

TEST_F(MeshTensorDeviceTest, AllocateOnDeviceFp8RowMajorOnBlackholeSucceeds) {
    if (mesh_device_->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "FP8_E4M3 device allocation requires Blackhole.";
    }
    const Shape shape{2, 32};
    auto spec = make_fp8_row_major_spec(shape);

    testing::internal::CaptureStdout();
    auto tensor = MeshTensor::allocate_on_device(*mesh_device_, spec, TensorTopology());
    std::fflush(stdout);
    const std::string captured = testing::internal::GetCapturedStdout();

    EXPECT_EQ(tensor.dtype(), DataType::FP8_E4M3);
    EXPECT_EQ(tensor.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(tensor.logical_shape(), shape);
    EXPECT_EQ(tensor.element_size(), sizeof(float8_e4m3));
    // The unconditional FP8 warning fires even on Blackhole as a signpost of limited support.
    EXPECT_THAT(captured, ::testing::HasSubstr("FP8_E4M3 has limited tensor infra support"));
}

TEST_F(MeshTensorDeviceTest, AllocateOnDeviceFp8OnNonBlackholeThrows) {
    if (mesh_device_->arch() == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Negative arch guard only fires off-Blackhole.";
    }
    auto spec = make_fp8_row_major_spec(Shape{2, 32});

    testing::internal::CaptureStdout();
    bool threw_with_message = false;
    try {
        (void)MeshTensor::allocate_on_device(*mesh_device_, spec, TensorTopology());
    } catch (const std::runtime_error& e) {
        threw_with_message =
            std::string(e.what()).find("FP8_E4M3 is only supported on Blackhole hardware") != std::string::npos;
    }
    std::fflush(stdout);
    const std::string captured = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(threw_with_message) << "Expected std::runtime_error with the Blackhole-only message; captured stdout: "
                                    << captured;
    EXPECT_THAT(captured, ::testing::HasSubstr("FP8_E4M3 has limited tensor infra support"));
}

TEST_F(MeshTensorDeviceTest, Fp8H2DToD2HRoundTripPreservesBytes) {
    if (mesh_device_->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "H2D/D2H for FP8_E4M3 requires Blackhole.";
    }

    const Shape shape{2, 32};
    auto spec = make_fp8_row_major_spec(shape);

    const auto& mesh_shape = mesh_device_->shape();
    auto dhb = DistributedHostBuffer::create(mesh_shape);
    distributed::MeshCoordinateRange range(mesh_shape);
    std::vector<distributed::MeshCoordinate> coords(range.begin(), range.end());
    // Per-shard distinct fills make cross-shard corruption easy to spot; exact FP8 quantization
    // doesn't matter since we compare raw bytes after the round-trip.
    dhb.emplace_shards(coords, [&, idx = size_t{0}](const distributed::MeshCoordinate&) mutable {
        const float v = 0.5f * static_cast<float>(idx++ + 1);
        return HostBuffer(std::vector<float8_e4m3>(shape.volume(), float8_e4m3(v)));
    });
    auto topology = TensorTopology::create_sharded_tensor_topology(mesh_shape);
    HostTensor host_tensor(std::move(dhb), spec, topology);

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = enqueue_write_tensor(cq, host_tensor, *mesh_device_);
    HostTensor result = enqueue_read_tensor(cq, device_tensor);

    const auto& exp_coords = host_tensor.buffer().shard_coords();
    const auto& act_coords = result.buffer().shard_coords();
    ASSERT_EQ(exp_coords, act_coords);
    for (const auto& coord : exp_coords) {
        auto exp_shard = host_tensor.buffer().get_shard(coord);
        auto act_shard = result.buffer().get_shard(coord);
        ASSERT_TRUE(exp_shard.has_value());
        ASSERT_TRUE(act_shard.has_value());
        const auto exp_bytes = exp_shard->view_bytes();
        const auto act_bytes = act_shard->view_bytes();
        ASSERT_EQ(exp_bytes.size(), act_bytes.size());
        EXPECT_TRUE(std::equal(exp_bytes.begin(), exp_bytes.end(), act_bytes.begin()));
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

}  // namespace tt::tt_metal
