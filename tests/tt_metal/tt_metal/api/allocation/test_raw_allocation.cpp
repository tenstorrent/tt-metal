// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <optional>
#include <utility>

#include <gtest/gtest.h>
#include <tt-metalium/experimental/allocation/raw_allocation.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>

#include "impl/tensor/mesh_tensor_impl.hpp"
#include "tests/tt_metal/tt_metal/api/allocator/hybrid_allocator_fixture.hpp"

namespace tt::tt_metal {
namespace {

enum class AllocationMode { Lockstep, PerCore, RangeLockstep };

TEST(RawAllocationTest, CannotReleaseEmptyOrMovedFromAllocation) {
    EXPECT_ANY_THROW(experimental::RawAllocation::release_as_meshtensor(experimental::RawAllocation{}));

    experimental::RawAllocation allocation;
    experimental::RawAllocation moved(std::move(allocation));
    EXPECT_ANY_THROW(experimental::RawAllocation::release_as_meshtensor(std::move(allocation)));
    EXPECT_ANY_THROW(experimental::RawAllocation::release_as_meshtensor(std::move(moved)));
}

void check_tensor_adoption(distributed::MeshDevice& device, AllocationMode mode) {
    const CoreCoord first(0, 0);
    const CoreCoord second(1, 0);
    MemoryConfig config(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(CoreRangeSet(CoreRange(first, second)), {1, 32}, ShardOrientation::ROW_MAJOR));
    if (mode == AllocationMode::PerCore) {
        experimental::per_core_allocation::set_per_core_allocation(config, true);
    } else if (mode == AllocationMode::RangeLockstep) {
        experimental::range_lockstep_allocation::set_range_lockstep_allocation(config, true);
    }
    auto tensor = MeshTensor::allocate_on_device(
        device, TensorSpec(Shape{1, 64}, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), config)));
    std::weak_ptr<distributed::MeshBuffer> owner = tensor.impl().raw_mesh_buffer();
    const auto* buffer = tensor.mesh_buffer().get_reference_buffer();
    ASSERT_NE(buffer, nullptr);
    const auto expected_size = buffer->aligned_size_per_bank();
    const auto first_address = experimental::per_core_allocation::get_shard_base_address(*buffer, first);
    const auto second_address = experimental::per_core_allocation::get_shard_base_address(*buffer, second);

    std::optional<MeshTensor> released;
    {
        auto allocation = experimental::RawAllocation::adopt_from_meshtensor(std::move(tensor));
        EXPECT_TRUE(tensor.is_valueless_after_move());
        ASSERT_FALSE(owner.expired());
        EXPECT_TRUE(owner.lock()->is_allocated());

        experimental::RawAllocation moved(std::move(allocation));
        EXPECT_FALSE(owner.expired());
        allocation = std::move(moved);
        EXPECT_FALSE(owner.expired());

        released.emplace(experimental::RawAllocation::release_as_meshtensor(std::move(allocation)));
        EXPECT_EQ(&released->mesh_buffer(), owner.lock().get());
        EXPECT_EQ(released->logical_shape(), Shape({1, 64}));
        EXPECT_EQ(released->dtype(), DataType::BFLOAT16);
        EXPECT_EQ(released->memory_config(), config);
        const auto* released_buffer = released->mesh_buffer().get_reference_buffer();
        EXPECT_EQ(released_buffer->aligned_size_per_bank(), expected_size);
        EXPECT_EQ(experimental::per_core_allocation::get_shard_base_address(*released_buffer, first), first_address);
        EXPECT_EQ(experimental::per_core_allocation::get_shard_base_address(*released_buffer, second), second_address);
        EXPECT_ANY_THROW(experimental::RawAllocation::release_as_meshtensor(std::move(allocation)));
    }
    ASSERT_FALSE(owner.expired());
    EXPECT_TRUE(released->mesh_buffer().is_allocated());
    {
        auto readopted = experimental::RawAllocation::adopt_from_meshtensor(std::move(*released));
        released.reset();
        EXPECT_FALSE(owner.expired());
    }
    EXPECT_TRUE(owner.expired());
}

TEST_F(HybridAllocatorTest, AdoptLockstepMeshTensor) { check_tensor_adoption(*devices_[0], AllocationMode::Lockstep); }

TEST_F(HybridAllocatorTest, AdoptPerCoreMeshTensor) { check_tensor_adoption(*devices_[0], AllocationMode::PerCore); }

TEST_F(HybridAllocatorTest, AdoptRangeLockstepMeshTensor) {
    check_tensor_adoption(*devices_[0], AllocationMode::RangeLockstep);
}

TensorSpec interleaved_tensor_spec(BufferType buffer_type = BufferType::L1) {
    return TensorSpec(
        Shape{1, 32},
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, buffer_type)));
}

TEST_F(HybridAllocatorTest, AdoptDramMeshTensorFails) {
    auto tensor = MeshTensor::allocate_on_device(*devices_[0], interleaved_tensor_spec(BufferType::DRAM));

    EXPECT_ANY_THROW(experimental::RawAllocation::adopt_from_meshtensor(std::move(tensor)));
}

TEST_F(HybridAllocatorTest, RawAllocationMoveAssignmentReleasesPreviousOwner) {
    auto first = MeshTensor::allocate_on_device(*devices_[0], interleaved_tensor_spec());
    auto second = MeshTensor::allocate_on_device(*devices_[0], interleaved_tensor_spec());
    std::weak_ptr<distributed::MeshBuffer> first_owner = first.impl().raw_mesh_buffer();
    std::weak_ptr<distributed::MeshBuffer> second_owner = second.impl().raw_mesh_buffer();

    auto destination = experimental::RawAllocation::adopt_from_meshtensor(std::move(first));
    auto source = experimental::RawAllocation::adopt_from_meshtensor(std::move(second));
    destination = std::move(source);

    EXPECT_TRUE(first_owner.expired());
    ASSERT_FALSE(second_owner.expired());
    EXPECT_TRUE(second_owner.lock()->is_allocated());
    EXPECT_ANY_THROW(experimental::RawAllocation::release_as_meshtensor(std::move(source)));

    destination = experimental::RawAllocation{};
    EXPECT_TRUE(second_owner.expired());
    EXPECT_ANY_THROW(experimental::RawAllocation::release_as_meshtensor(std::move(destination)));
}

TEST_F(HybridAllocatorTest, AdoptMovedFromMeshTensorFails) {
    auto tensor = MeshTensor::allocate_on_device(*devices_[0], interleaved_tensor_spec());
    const auto* original_buffer = &tensor.mesh_buffer();
    auto allocation = experimental::RawAllocation::adopt_from_meshtensor(std::move(tensor));

    EXPECT_ANY_THROW(experimental::RawAllocation::adopt_from_meshtensor(std::move(tensor)));

    auto released = experimental::RawAllocation::release_as_meshtensor(std::move(allocation));
    EXPECT_EQ(&released.mesh_buffer(), original_buffer);
    EXPECT_TRUE(released.mesh_buffer().is_allocated());
}

TEST_F(HybridAllocatorTest, AdoptNonOwningMeshTensorFails) {
    auto spec = interleaved_tensor_spec();
    auto owner = MeshTensor::allocate_on_device(*devices_[0], spec);
    const auto& buffer = owner.mesh_buffer();
    auto view = distributed::MeshBuffer::create(
        buffer.global_config(), buffer.device_local_config(), devices_[0].get(), buffer.address());
    ASSERT_EQ(view->get_backing_buffer(), nullptr);
    auto tensor = MeshTensor::from_buffer(std::move(*view), spec);

    EXPECT_ANY_THROW(experimental::RawAllocation::adopt_from_meshtensor(std::move(tensor)));

    EXPECT_TRUE(owner.mesh_buffer().is_allocated());
    auto allocation = experimental::RawAllocation::adopt_from_meshtensor(std::move(owner));
    auto released = experimental::RawAllocation::release_as_meshtensor(std::move(allocation));
    EXPECT_EQ(&released.mesh_buffer(), &buffer);
    EXPECT_TRUE(released.mesh_buffer().is_allocated());
}

}  // namespace
}  // namespace tt::tt_metal
