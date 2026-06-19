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
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/math.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/impl/tensor_impl.hpp>
#include <tt-metalium/experimental/core_subset_write/tensor.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt_stl/aligned_allocator.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "tt_metal/distributed/pinned_memory_cache.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using ::testing::Eq;
using ::testing::Pointwise;

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

// ======================================================================================
//                    Data Movement Tests (2x2 mesh, Runtime Tensor)
// ======================================================================================

using MeshTensorDataMovementTest = MeshDevice2x2Fixture;
using MeshTensorPinnedMemoryBudgetTest = MeshDevice1x1Fixture;

constexpr int kPinnedMemoryTestAlignment = 64;
constexpr size_t kPinnedWriteThresholdBytesForTest = 32 * 1024 * 1024;

using AlignedUInt32Vector = std::vector<uint32_t, tt::stl::aligned_allocator<uint32_t, kPinnedMemoryTestAlignment>>;

class ScopedPinnedMemoryCacheLimit {
public:
    explicit ScopedPinnedMemoryCacheLimit(size_t limit_bytes) :
        previous_limit_bytes_(
            tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes()) {
        tt::tt_metal::MetalContext::instance().rtoptions().set_pinned_memory_cache_limit_bytes(limit_bytes);
    }

    ~ScopedPinnedMemoryCacheLimit() {
        tt::tt_metal::MetalContext::instance().rtoptions().set_pinned_memory_cache_limit_bytes(previous_limit_bytes_);
    }

private:
    size_t previous_limit_bytes_;
};

HostBuffer make_aligned_host_buffer(size_t num_words, uint32_t fill) {
    auto data = std::make_shared<AlignedUInt32Vector>(num_words, fill);
    return HostBuffer(tt::stl::Span<uint32_t>(data->data(), data->size()), tt::tt_metal::MemoryPin(data));
}

HostTensor make_full_coverage_aligned_host_tensor(
    const Shape& shape, const distributed::MeshShape& mesh_shape, const std::vector<uint32_t>& shard_fills) {
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto dhb = DistributedHostBuffer::create(mesh_shape);
    distributed::MeshCoordinateRange range(mesh_shape);
    std::vector<distributed::MeshCoordinate> coords(range.begin(), range.end());
    dhb.emplace_shards(coords, [&, idx = size_t{0}](const distributed::MeshCoordinate&) mutable {
        return make_aligned_host_buffer(static_cast<size_t>(shape.volume()), shard_fills.at(idx++));
    });
    auto topology = TensorTopology::create_sharded_tensor_topology(mesh_shape);
    return HostTensor(std::move(dhb), spec, topology);
}

// Helper: create a HostTensor with a single shard at [0,0].
HostTensor make_single_shard_host_tensor(const Shape& shape, uint32_t fill) {
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    return HostTensor(HostBuffer(std::vector<uint32_t>(shape.volume(), fill)), spec, TensorTopology{});
}

// Helper: create a HostTensor with one shard per coordinate in a mesh.
// Each shard is filled with `volume` copies of `shard_fills[i]`.
HostTensor make_full_coverage_host_tensor(
    const Shape& shape, const distributed::MeshShape& mesh_shape, const std::vector<uint32_t>& shard_fills) {
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto dhb = DistributedHostBuffer::create(mesh_shape);
    distributed::MeshCoordinateRange range(mesh_shape);
    std::vector<distributed::MeshCoordinate> coords(range.begin(), range.end());
    dhb.emplace_shards(coords, [&, idx = size_t{0}](const distributed::MeshCoordinate&) mutable {
        return HostBuffer(std::vector<uint32_t>(shape.volume(), shard_fills.at(idx++)));
    });
    auto topology = TensorTopology::create_sharded_tensor_topology(mesh_shape);
    return HostTensor(std::move(dhb), spec, topology);
}

// Helper: create a HostTensor with shards at only a subset of mesh coordinates.
// Each shard is filled with `volume` copies of `shard_fills[i]`.
HostTensor make_partial_coverage_host_tensor(
    const Shape& shape,
    const distributed::MeshShape& mesh_shape,
    const std::vector<distributed::MeshCoordinate>& coords,
    const std::vector<uint32_t>& shard_fills) {
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto dhb = DistributedHostBuffer::create(mesh_shape);
    dhb.emplace_shards(coords, [&, idx = size_t{0}](const distributed::MeshCoordinate&) mutable {
        return HostBuffer(std::vector<uint32_t>(shape.volume(), shard_fills.at(idx++)));
    });
    auto topology = TensorTopology::create_sharded_tensor_topology(distributed::MeshShape(coords.size()));
    return HostTensor(std::move(dhb), spec, topology);
}

// Helper: assert that two HostTensors have the same populated coords and identical shard contents.
void expect_host_tensors_eq(const HostTensor& expected, const HostTensor& actual) {
    const auto& exp_coords = expected.buffer().shard_coords();
    const auto& act_coords = actual.buffer().shard_coords();
    ASSERT_EQ(exp_coords, act_coords);
    for (const auto& coord : exp_coords) {
        auto exp_shard = expected.buffer().get_shard(coord);
        auto act_shard = actual.buffer().get_shard(coord);
        ASSERT_TRUE(exp_shard.has_value());
        ASSERT_TRUE(act_shard.has_value());
        auto exp_span = exp_shard->view_as<uint32_t>();
        auto act_span = act_shard->view_as<uint32_t>();
        EXPECT_THAT(
            std::vector<uint32_t>(act_span.begin(), act_span.end()),
            Pointwise(Eq(), std::vector<uint32_t>(exp_span.begin(), exp_span.end())));
    }
}

// Verify the contract for a HEIGHT_SHARDED tensor with shard_shape {rows_per_shard, cols} on cores
// {(0,0), (0,1)}: rows [0, rows_per_shard) live on (0,0); rows [rows_per_shard, 2*rows_per_shard)
// live on (0,1).  Asserts that, in the *logical* row-major view of the shard, rows backed by a
// filtered core hold `new_value` while the rest hold `sentinel_value`.
void expect_height_sharded_filter_applied(
    const std::vector<uint32_t>& logical_data,
    uint32_t cols,
    uint32_t rows_per_shard,
    bool first_shard_was_filtered,
    bool second_shard_was_filtered,
    uint32_t sentinel_value,
    uint32_t new_value) {
    ASSERT_EQ(logical_data.size(), 2u * rows_per_shard * cols);
    const uint32_t first_expected = first_shard_was_filtered ? new_value : sentinel_value;
    const uint32_t second_expected = second_shard_was_filtered ? new_value : sentinel_value;
    for (uint32_t row = 0; row < 2u * rows_per_shard; ++row) {
        const uint32_t expected = (row < rows_per_shard) ? first_expected : second_expected;
        for (uint32_t col = 0; col < cols; ++col) {
            const size_t idx = static_cast<size_t>(row) * cols + col;
            ASSERT_EQ(logical_data[idx], expected) << "mismatch at row=" << row << " col=" << col;
        }
    }
}

// ---------------------------------------------------------------------------
//  is_uniform_write
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_FullCoverage) {
    const Shape shape{1, 1, 32, 32};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {0, 0, 0, 0});
    EXPECT_TRUE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_NO_THROW(enqueue_write_tensor(cq, host_tensor, *mesh_device_));
}

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_SingleShard) {
    const Shape shape{1, 1, 32, 32};
    auto host_tensor = make_single_shard_host_tensor(shape, 1);
    EXPECT_FALSE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_ANY_THROW(enqueue_write_tensor(cq, host_tensor, *mesh_device_));
}

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_PartialCoverage) {
    const Shape shape{1, 1, 32, 32};
    std::vector<distributed::MeshCoordinate> coords = {{0, 0}, {0, 1}};
    auto host_tensor = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), coords, {0, 0});
    EXPECT_FALSE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_ANY_THROW(enqueue_write_tensor(cq, host_tensor, *mesh_device_));
}

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_SmallerMeshShape) {
    const Shape shape{1, 1, 32, 32};
    distributed::MeshShape smaller_mesh{1, 2};
    auto host_tensor = make_full_coverage_host_tensor(shape, smaller_mesh, {0, 0});
    EXPECT_FALSE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_ANY_THROW(enqueue_write_tensor(cq, host_tensor, *mesh_device_));
}

TEST_F(MeshTensorDataMovementTest, IsUniformWrite_EmptyDistributedHostBuffer) {
    const Shape shape{1, 1, 32, 32};
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto dhb = DistributedHostBuffer::create(mesh_device_->shape());
    HostTensor host_tensor(std::move(dhb), spec, TensorTopology{});
    EXPECT_FALSE(is_uniform_write(host_tensor, *mesh_device_));
    auto& cq = mesh_device_->mesh_command_queue();
    EXPECT_ANY_THROW(enqueue_write_tensor(cq, host_tensor, *mesh_device_));
}

// ---------------------------------------------------------------------------
//  Uniform to_device / to_host roundtrip
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, UniformToDevice_ToHost_Roundtrip) {
    const Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills = {10, 20, 30, 40};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = enqueue_write_tensor(cq, host_tensor, *mesh_device_);
    HostTensor result = enqueue_read_tensor(cq, device_tensor);

    expect_host_tensors_eq(host_tensor, result);
}

// ---------------------------------------------------------------------------
//  Uniform copy_to_device / copy_to_host roundtrip
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, UniformCopyToDevice_CopyToHost_Roundtrip) {
    const Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills = {100, 200, 300, 400};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    MeshTensor device_tensor = MeshTensor::allocate_on_device(*mesh_device_, spec, host_tensor.tensor_topology());
    enqueue_write_tensor(cq, host_tensor, device_tensor);

    auto result_dhb = DistributedHostBuffer::create(mesh_device_->shape());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device_->shape())) {
        result_dhb.emplace_shard(coord, [&]() { return tensor_impl::allocate_host_buffer(spec); });
    }
    HostTensor result(std::move(result_dhb), spec, TensorTopology{});
    enqueue_read_tensor(cq, device_tensor, result);

    expect_host_tensors_eq(host_tensor, result);
}

TEST_F(MeshTensorPinnedMemoryBudgetTest, LargeHostWriteOverHardwarePinBudgetFallsBackAndKeepsDeviceUsable) {
    const auto pinning_params = tt::tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_);
    if (pinning_params.max_pins == 0 || !pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }
    if (pinning_params.max_total_pin_size == std::numeric_limits<uint64_t>::max()) {
        GTEST_SKIP() << "Hardware does not advertise a finite pinned-memory budget";
        return;
    }

    // The runtime cache limit can be larger than the hardware NOC-mappable pin budget. This write is intentionally
    // larger than that hardware budget, so it must fall back to the regular host-to-device path instead of trying to
    // create an oversized NOC mapping.
    const Shape large_shape{1, 1, 160, 5210112};
    const Shape large_shard_shape{1, 1, 32, 131072};
    const size_t oversized_buffer_size = static_cast<size_t>(large_shape.volume()) * sizeof(uint32_t);
    if (oversized_buffer_size <= pinning_params.max_total_pin_size) {
        GTEST_SKIP() << "Test tensor is not larger than the hardware pinned-memory budget";
        return;
    }
    ScopedPinnedMemoryCacheLimit cache_limit(oversized_buffer_size);

    void* mapping = mmap(
        nullptr, oversized_buffer_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    ASSERT_NE(mapping, MAP_FAILED) << "mmap failed: " << std::strerror(errno);

    auto mapping_owner = std::shared_ptr<std::byte>(
        static_cast<std::byte*>(mapping),
        [oversized_buffer_size](std::byte* ptr) { munmap(static_cast<void*>(ptr), oversized_buffer_size); });

    const size_t num_words = oversized_buffer_size / sizeof(uint32_t);
    ASSERT_EQ(oversized_buffer_size % sizeof(uint32_t), 0u);
    ASSERT_EQ(num_words, large_shape.volume());

    auto dram_grid_size = mesh_device_->dram_grid_size();
    if (dram_grid_size.x == 0 || dram_grid_size.y == 0) {
        GTEST_SKIP() << "Device does not expose a DRAM grid";
        return;
    }
    // Only use the first DRAM row; higher y values can be replicas of the same DRAM cores.
    CoreRangeSet dram_cores(CoreRange(CoreCoord{0, 0}, CoreCoord{dram_grid_size.x - 1, 0}));
    MemoryConfig large_mem_cfg(
        BufferType::DRAM, NdShardSpec{large_shard_shape, dram_cores, ShardOrientation::ROW_MAJOR});
    auto large_spec = TensorSpec(large_shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, large_mem_cfg));

    auto large_dhb = DistributedHostBuffer::create(mesh_device_->shape());
    const auto first_coord = *distributed::MeshCoordinateRange(mesh_device_->shape()).begin();
    large_dhb.emplace_shard(first_coord, [&]() {
        return HostBuffer(
            tt::stl::Span<uint32_t>(reinterpret_cast<uint32_t*>(mapping_owner.get()), num_words),
            tt::tt_metal::MemoryPin(std::static_pointer_cast<void>(mapping_owner)));
    });
    HostTensor large_host_tensor(
        std::move(large_dhb), large_spec, TensorTopology::create_sharded_tensor_topology(mesh_device_->shape()));

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor large_device_tensor = enqueue_write_tensor(cq, large_host_tensor, *mesh_device_);
    cq.finish();
    EXPECT_EQ(large_device_tensor.tensor_spec().logical_shape(), large_shape);

    const Shape small_shape{1, 1, 32, 32};
    auto small_host_tensor = make_full_coverage_host_tensor(small_shape, mesh_device_->shape(), {0x5Au});
    MeshTensor small_device_tensor = enqueue_write_tensor(cq, small_host_tensor, *mesh_device_);
    HostTensor small_result = enqueue_read_tensor(cq, small_device_tensor);

    expect_host_tensors_eq(small_host_tensor, small_result);
}

TEST_F(MeshTensorDeviceTest, UniformCopyToDevice_ReusesPinnedMemoryCacheEntries) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0) {
        GTEST_SKIP() << "Pinned memory cache is disabled";
        return;
    }
    auto& cache = tt::tt_metal::experimental::PinnedMemoryCache::instance();
    const auto pinning_params = tt::tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_);
    if (!pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }

    const size_t shard_count = mesh_device_->shape().mesh_size();
    if (static_cast<size_t>(pinning_params.max_pins) < shard_count) {
        GTEST_SKIP() << "Requires enough pin slots to keep one cache entry per tensor shard";
        return;
    }

    // 1024 * 8193 * 4 bytes per shard exceeds the 32 MiB pinned-write threshold even on a 1x1 mesh.
    const Shape shape{1, 1, 1024, 8193};
    const size_t total_size_bytes = static_cast<size_t>(shape.volume()) * sizeof(uint32_t) * shard_count;
    ASSERT_GT(total_size_bytes, kPinnedWriteThresholdBytesForTest);
    if (pinning_params.max_total_pin_size != 0 && pinning_params.max_total_pin_size < total_size_bytes) {
        GTEST_SKIP() << "Pinned memory budget is too small for the test tensor";
        return;
    }

    std::vector<uint32_t> shard_fills(shard_count);
    std::iota(shard_fills.begin(), shard_fills.end(), 1u);
    auto host_tensor = make_full_coverage_aligned_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    MeshTensor device_tensor = MeshTensor::allocate_on_device(*mesh_device_, spec, host_tensor.tensor_topology());

    const size_t entries_before = cache.num_entries();
    enqueue_write_tensor(cq, host_tensor, device_tensor);
    cq.finish();
    EXPECT_EQ(cache.num_entries(), entries_before + shard_count);

    const auto first_coord = *distributed::MeshCoordinateRange(mesh_device_->shape()).begin();
    auto first_shard = host_tensor.buffer().get_shard(first_coord);
    ASSERT_TRUE(first_shard.has_value());
    auto first_coord_range =
        distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(first_coord, first_coord));
    auto first_pin = cache.try_pin(*mesh_device_, first_coord_range, *first_shard, /*map_to_noc=*/true);
    ASSERT_TRUE(first_pin);
    std::weak_ptr<tt::tt_metal::experimental::PinnedMemory> first_weak = first_pin;
    first_pin.reset();

    enqueue_write_tensor(cq, host_tensor, device_tensor);
    cq.finish();

    EXPECT_EQ(cache.num_entries(), entries_before + shard_count);
    EXPECT_FALSE(first_weak.expired());
}

TEST_F(MeshTensorDeviceTest, HostTensorCopyKeepsPinnedMemoryCacheEntriesUntilLastPinRelease) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0) {
        GTEST_SKIP() << "Pinned memory cache is disabled";
        return;
    }
    auto& cache = tt::tt_metal::experimental::PinnedMemoryCache::instance();
    const auto pinning_params = tt::tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_);
    if (!pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }

    const size_t shard_count = mesh_device_->shape().mesh_size();
    if (static_cast<size_t>(pinning_params.max_pins) < shard_count) {
        GTEST_SKIP() << "Requires enough pin slots to keep one cache entry per tensor shard";
        return;
    }

    const Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills(shard_count);
    std::iota(shard_fills.begin(), shard_fills.end(), 1u);

    const size_t entries_before = cache.num_entries();
    std::unique_ptr<HostTensor> copied_host_tensor;
    {
        auto host_tensor = make_full_coverage_aligned_host_tensor(shape, mesh_device_->shape(), shard_fills);
        for (const auto& coord : distributed::MeshCoordinateRange(mesh_device_->shape())) {
            auto shard = host_tensor.buffer().get_shard(coord);
            ASSERT_TRUE(shard.has_value());
            auto coord_range = distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(coord, coord));
            auto pinned = cache.try_pin(*mesh_device_, coord_range, *shard, /*map_to_noc=*/true);
            ASSERT_TRUE(pinned);
            pinned.reset();
        }
        ASSERT_EQ(cache.num_entries(), entries_before + shard_count);
        copied_host_tensor = std::make_unique<HostTensor>(host_tensor);
    }

    EXPECT_EQ(cache.num_entries(), entries_before + shard_count);
    copied_host_tensor.reset();
    EXPECT_EQ(cache.num_entries(), entries_before);
}

TEST_F(MeshTensorDeviceTest, UniformCopyToHost_ReusesPinnedMemoryCacheEntries) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0) {
        GTEST_SKIP() << "Pinned memory cache is disabled";
        return;
    }
    auto& cache = tt::tt_metal::experimental::PinnedMemoryCache::instance();
    const auto pinning_params = tt::tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_);
    if (!pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
        return;
    }

    const size_t shard_count = mesh_device_->shape().mesh_size();
    if (static_cast<size_t>(pinning_params.max_pins) < shard_count) {
        GTEST_SKIP() << "Requires enough pin slots to keep one cache entry per tensor shard";
        return;
    }

    const Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills(shard_count);
    std::iota(shard_fills.begin(), shard_fills.end(), 10u);
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = enqueue_write_tensor(cq, host_tensor, *mesh_device_);

    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto result_dhb = DistributedHostBuffer::create(mesh_device_->shape());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device_->shape())) {
        result_dhb.emplace_shard(coord, [&]() { return tensor_impl::allocate_host_buffer(spec); });
    }
    HostTensor result(std::move(result_dhb), spec, TensorTopology{});

    const size_t entries_before = cache.num_entries();
    enqueue_read_tensor(cq, device_tensor, result);
    cq.finish();
    EXPECT_EQ(cache.num_entries(), entries_before + shard_count);

    const auto first_coord = *distributed::MeshCoordinateRange(mesh_device_->shape()).begin();
    auto first_shard = result.buffer().get_shard(first_coord);
    ASSERT_TRUE(first_shard.has_value());
    auto first_coord_range =
        distributed::MeshCoordinateRangeSet(distributed::MeshCoordinateRange(first_coord, first_coord));
    auto first_pin = cache.try_pin(*mesh_device_, first_coord_range, *first_shard, /*map_to_noc=*/true);
    ASSERT_TRUE(first_pin);
    std::weak_ptr<tt::tt_metal::experimental::PinnedMemory> first_weak = first_pin;
    first_pin.reset();

    enqueue_read_tensor(cq, device_tensor, result);
    cq.finish();

    EXPECT_EQ(cache.num_entries(), entries_before + shard_count);
    EXPECT_FALSE(first_weak.expired());
}

// Verifies the user-facing contract: in a HEIGHT_SHARDED tensor with one shard per core, applying
// a core filter restricts a write to the logical rows backed by those cores. The test does not
// touch BufferPageMapping or raw device byte layout; it asserts on the logical row-major view
// returned by enqueue_read_tensor.
TEST_F(MeshTensorDataMovementTest, UniformCopyToDevice_WithCoreFilter_WritesOnlyFilteredCores) {
    constexpr uint32_t kRowsPerShard = 32;
    constexpr uint32_t kCols = 32;
    constexpr uint32_t kSentinel = 0x11u;
    constexpr uint32_t kNew = 0x22u;
    const Shape shape{1, 1, 2 * kRowsPerShard, kCols};
    CoreRangeSet shard_grid(CoreRange(CoreCoord(0, 0), CoreCoord(0, 1)));
    ShardSpec shard_spec(shard_grid, {kRowsPerShard, kCols});
    // Use L1 so that BufferPageMapping::all_cores equals the user's shard grid (Tensix cores).
    MemoryConfig mem_cfg(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec);
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, mem_cfg));
    auto host_sentinel =
        make_full_coverage_host_tensor(shape, mesh_device_->shape(), {kSentinel, kSentinel, kSentinel, kSentinel});
    auto host_new = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {kNew, kNew, kNew, kNew});

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = MeshTensor::allocate_on_device(*mesh_device_, spec, host_sentinel.tensor_topology());
    enqueue_write_tensor(cq, host_sentinel, device_tensor);
    // Filter to logical core (0,0) -- the first shard slot of the ShardSpec grid.
    CoreRangeSet filter(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    tt::tt_metal::experimental::core_subset_write::enqueue_write_tensor(cq, host_new, device_tensor, filter);

    HostTensor result = enqueue_read_tensor(cq, device_tensor);
    for (const auto& coord : result.buffer().shard_coords()) {
        auto shard = result.buffer().get_shard(coord);
        ASSERT_TRUE(shard.has_value());
        auto span = shard->view_as<uint32_t>();
        std::vector<uint32_t> data(span.begin(), span.end());
        expect_height_sharded_filter_applied(
            data,
            kCols,
            kRowsPerShard,
            /*first_shard_was_filtered=*/true,
            /*second_shard_was_filtered=*/false,
            kSentinel,
            kNew);
    }
}

TEST_F(MeshTensorDataMovementTest, EnqueueWriteTensor_FilterEmpty_Noop) {
    const Shape shape{1, 1, 64, 32};
    CoreRangeSet shard_grid(CoreRange(CoreCoord(0, 0), CoreCoord(0, 1)));
    ShardSpec shard_spec(shard_grid, {32, 32});
    MemoryConfig mem_cfg(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::DRAM, shard_spec);
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, mem_cfg));
    auto host_sentinel = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {5u, 5u, 5u, 5u});
    auto host_new = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {9u, 9u, 9u, 9u});

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = MeshTensor::allocate_on_device(*mesh_device_, spec, host_sentinel.tensor_topology());
    enqueue_write_tensor(cq, host_sentinel, device_tensor);
    CoreRangeSet empty_filter;
    tt::tt_metal::experimental::core_subset_write::enqueue_write_tensor(cq, host_new, device_tensor, empty_filter);
    cq.finish();

    auto result_dhb = DistributedHostBuffer::create(mesh_device_->shape());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device_->shape())) {
        result_dhb.emplace_shard(coord, [&]() { return tensor_impl::allocate_host_buffer(spec); });
    }
    HostTensor result(std::move(result_dhb), spec, TensorTopology{});
    enqueue_read_tensor(cq, device_tensor, result);
    expect_host_tensors_eq(host_sentinel, result);
}

// ---------------------------------------------------------------------------
//  Uniform copy_to_device rejects non-uniform host tensor
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, UniformCopyToDevice_RejectsPartialCoverage) {
    const Shape shape{1, 1, 32, 32};
    std::vector<distributed::MeshCoordinate> coords = {{0, 0}, {1, 1}};
    auto host_tensor = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), coords, {0, 0});

    auto& cq = mesh_device_->mesh_command_queue();
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    MeshTensor device_tensor = MeshTensor::allocate_on_device(*mesh_device_, spec, TensorTopology{});
    EXPECT_ANY_THROW(enqueue_write_tensor(cq, host_tensor, device_tensor));
}

// ---------------------------------------------------------------------------
//  Non-uniform to_device / to_host roundtrip
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, NonUniformToDevice_ToHost_Roundtrip) {
    const Shape shape{1, 1, 32, 32};
    std::vector<distributed::MeshCoordinate> coords = {{0, 0}, {1, 0}};
    std::vector<uint32_t> shard_fills = {7, 13};
    auto host_tensor = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), coords, shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    auto [device_tensor, written_coords] =
        non_uniform_data_movement::enqueue_write_tensor(cq, host_tensor, *mesh_device_);
    ASSERT_EQ(written_coords.size(), coords.size());

    HostTensor result = non_uniform_data_movement::enqueue_read_tensor(cq, device_tensor, written_coords);
    expect_host_tensors_eq(host_tensor, result);
}

// ---------------------------------------------------------------------------
//  Non-uniform copy_to_device / copy_to_host roundtrip
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, NonUniformCopyToDevice_CopyToHost_Roundtrip) {
    const Shape shape{1, 1, 32, 32};
    std::vector<distributed::MeshCoordinate> coords = {{0, 1}, {1, 1}};
    std::vector<uint32_t> shard_fills = {42, 99};
    auto host_tensor = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), coords, shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    MeshTensor device_tensor = MeshTensor::allocate_on_device(*mesh_device_, spec, TensorTopology{});

    auto written_coords = non_uniform_data_movement::enqueue_write_tensor(cq, host_tensor, device_tensor);
    ASSERT_EQ(written_coords.size(), coords.size());

    auto result_dhb = DistributedHostBuffer::create(mesh_device_->shape());
    for (const auto& coord : written_coords) {
        result_dhb.emplace_shard(coord, [&]() { return tensor_impl::allocate_host_buffer(spec); });
    }
    HostTensor result(std::move(result_dhb), spec, TensorTopology{});
    non_uniform_data_movement::enqueue_read_tensor(cq, device_tensor, result, written_coords);
    expect_host_tensors_eq(host_tensor, result);
}

// ---------------------------------------------------------------------------
//  Non-uniform to_device with single shard (replicate)
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, NonUniformToDevice_SingleShard_Roundtrip) {
    const Shape shape{1, 1, 32, 32};
    auto host_tensor = make_single_shard_host_tensor(shape, 55);

    auto& cq = mesh_device_->mesh_command_queue();
    auto [device_tensor, written_coords] =
        non_uniform_data_movement::enqueue_write_tensor(cq, host_tensor, *mesh_device_);
    HostTensor result = non_uniform_data_movement::enqueue_read_tensor(cq, device_tensor, written_coords);
    auto expected = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {55, 55, 55, 55});
    expect_host_tensors_eq(expected, result);
}

// ---------------------------------------------------------------------------
//  Non-uniform D2H sheds extra shards from the host tensor
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDataMovementTest, NonUniformToHost_ShedsExtraShards) {
    const Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills = {10, 20, 30, 40};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = enqueue_write_tensor(cq, host_tensor, *mesh_device_);

    std::vector<distributed::MeshCoordinate> subset = {{0, 1}, {1, 0}};
    HostTensor result = non_uniform_data_movement::enqueue_read_tensor(cq, device_tensor, subset);

    ASSERT_EQ(result.buffer().shard_coords().size(), subset.size());
    auto expected = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), subset, {20, 30});
    expect_host_tensors_eq(expected, result);
}

TEST_F(MeshTensorDataMovementTest, NonUniformCopyToHost_ShedsExtraShards) {
    const Shape shape{1, 1, 32, 32};
    std::vector<uint32_t> shard_fills = {5, 15, 25, 35};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), shard_fills);

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = enqueue_write_tensor(cq, host_tensor, *mesh_device_);

    // Pass a full-coverage host tensor as the destination, but only read back a subset of coords.
    std::vector<distributed::MeshCoordinate> subset = {{1, 1}};
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    auto dest = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {0, 0, 0, 0});
    non_uniform_data_movement::enqueue_read_tensor(cq, device_tensor, dest, subset);

    ASSERT_EQ(dest.buffer().shard_coords().size(), subset.size());
    auto expected = make_partial_coverage_host_tensor(shape, mesh_device_->shape(), subset, {35});
    expect_host_tensors_eq(expected, dest);
}

TEST_F(MeshTensorDataMovementTest, UniformEnqueueWriteTensor_FilteredWriteRejectsInterleavedLayout) {
    const Shape shape{1, 1, 32, 32};
    auto host_tensor = make_full_coverage_host_tensor(shape, mesh_device_->shape(), {1u, 2u, 3u, 4u});
    auto& cq = mesh_device_->mesh_command_queue();
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));
    MeshTensor device_tensor = MeshTensor::allocate_on_device(*mesh_device_, spec, host_tensor.tensor_topology());
    enqueue_write_tensor(cq, host_tensor, device_tensor);
    CoreRangeSet filter(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    EXPECT_ANY_THROW(
        tt::tt_metal::experimental::core_subset_write::enqueue_write_tensor(cq, host_tensor, device_tensor, filter));
}

// ---------------------------------------------------------------------------
//  Large write roundtrip (exceeds pinned-memory threshold)
// ---------------------------------------------------------------------------

TEST_F(MeshTensorDeviceTest, LargeWriteRoundtrip_PinnedMemoryPath) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0) {
        GTEST_SKIP() << "Pinned memory cache is disabled";
        return;
    }
    // 1024 * 9216 * 4 bytes = 36 MB, above the 32 MB pinned-write threshold.
    const Shape shape{1, 1, 1024, 9216};
    auto spec = TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{}));

    auto dhb = DistributedHostBuffer::create(mesh_device_->shape());
    distributed::MeshCoordinateRange range(mesh_device_->shape());
    std::vector<distributed::MeshCoordinate> coords(range.begin(), range.end());
    dhb.emplace_shards(coords, [&](const distributed::MeshCoordinate&) {
        std::vector<uint32_t> data(shape.volume());
        std::iota(data.begin(), data.end(), 0);
        return HostBuffer(std::move(data));
    });
    auto topology = TensorTopology::create_sharded_tensor_topology(mesh_device_->shape());
    HostTensor host_tensor(std::move(dhb), spec, topology);

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = enqueue_write_tensor(cq, host_tensor, *mesh_device_);
    HostTensor result = enqueue_read_tensor(cq, device_tensor);

    expect_host_tensors_eq(host_tensor, result);
}

TEST_F(MeshTensorDeviceTest, LargeWriteRoundtrip_HeightShardedDeviceRequiringShardPadding) {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0) {
        GTEST_SKIP() << "Pinned memory cache is disabled";
        return;
    }
    // 1024 * 9216 * 4 bytes = 36 MB, above the 32 MB pinned-write threshold.
    // Device uses HEIGHT_SHARDED with shard_height=257 on 4 DRAM cores;
    // ceil(1024/257)=4, and 4*257=1028 > 1024, so the last shard carries
    // 4 padding rows. This padding may prevent the pinned-memory fast path
    // at the dispatch level.
    auto& cache = tt::tt_metal::experimental::PinnedMemoryCache::instance();
    const auto pinning_params = tt::tt_metal::experimental::GetMemoryPinningParameters(*mesh_device_);

    const Shape shape{1, 1, 1024, 9216};
    constexpr uint32_t kShardHeight = 257;
    constexpr uint32_t kWidth = 9216;

    CoreRangeSet shard_grid(CoreRange(CoreCoord(0, 0), CoreCoord(3, 0)));
    ShardSpec shard_spec(shard_grid, {kShardHeight, kWidth});
    MemoryConfig sharded_mem_cfg(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::DRAM, shard_spec);

    auto dhb = DistributedHostBuffer::create(mesh_device_->shape());
    distributed::MeshCoordinateRange range(mesh_device_->shape());
    std::vector<distributed::MeshCoordinate> coords(range.begin(), range.end());
    dhb.emplace_shards(coords, [&](const distributed::MeshCoordinate&) {
        std::vector<uint32_t> data(shape.volume());
        std::iota(data.begin(), data.end(), 0);
        return HostBuffer(std::move(data));
    });
    auto topology = TensorTopology::create_sharded_tensor_topology(mesh_device_->shape());
    HostTensor host_tensor(
        std::move(dhb), TensorSpec(shape, TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, MemoryConfig{})), topology);

    const size_t entries_before = cache.num_entries();
    const size_t shard_count = mesh_device_->shape().mesh_size();

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = enqueue_write_tensor(cq, host_tensor, *mesh_device_, sharded_mem_cfg);

    // The write path attempts pinning because total data exceeds the 32 MB
    // threshold.  When the system supports pinning, the cache should grow by
    // one entry per mesh shard.
    const bool pinning_supported = pinning_params.can_map_to_noc && pinning_params.max_pins > 0;
    if (pinning_supported) {
        EXPECT_EQ(cache.num_entries(), entries_before + shard_count)
            << "Expected one new cache entry per mesh shard after the pinned write";
    }

    HostTensor result = enqueue_read_tensor(cq, device_tensor);

    const size_t logical_volume = shape.volume();
    const auto& exp_coords = host_tensor.buffer().shard_coords();
    const auto& act_coords = result.buffer().shard_coords();
    ASSERT_EQ(exp_coords, act_coords);
    for (const auto& coord : exp_coords) {
        auto exp_shard = host_tensor.buffer().get_shard(coord);
        auto act_shard = result.buffer().get_shard(coord);
        ASSERT_TRUE(exp_shard.has_value());
        ASSERT_TRUE(act_shard.has_value());
        auto exp_span = exp_shard->view_as<uint32_t>();
        auto act_span = act_shard->view_as<uint32_t>();
        ASSERT_GE(act_span.size(), logical_volume);
        EXPECT_TRUE(std::equal(exp_span.begin(), exp_span.end(), act_span.begin()))
            << "Data mismatch at mesh coordinate " << coord;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

}  // namespace tt::tt_metal
