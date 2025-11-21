// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer_distribution_spec.hpp>
#include "impl/buffers/buffer_distribution_spec.hpp"

namespace tt::tt_metal {

// Static factory method
BufferDistributionSpec BufferDistributionSpec::from_shard_spec(
    Shape tensor_shape,
    Shape shard_shape,
    Shape2D page_shape,
    const CoreRangeSet& core_range_set,
    ShardOrientation shard_orientation,
    ShardDistributionStrategy shard_distribution_strategy) {
    return BufferDistributionSpecImpl::from_shard_spec(
        std::move(tensor_shape),
        std::move(shard_shape),
        page_shape,
        core_range_set,
        shard_orientation,
        shard_distribution_strategy);
}

// Constructors
BufferDistributionSpec::BufferDistributionSpec(
    const Shape& tensor_shape_in_pages,
    const Shape& shard_shape_in_pages,
    const CoreRangeSet& core_range_set,
    ShardOrientation shard_orientation,
    ShardDistributionStrategy shard_distribution_strategy) :
    impl_(std::make_unique<BufferDistributionSpecImpl>(
        tensor_shape_in_pages, shard_shape_in_pages, core_range_set, shard_orientation, shard_distribution_strategy)) {}

BufferDistributionSpec::BufferDistributionSpec(
    const Shape& tensor_shape_in_pages, const Shape& shard_shape_in_pages, std::vector<CoreCoord> cores) :
    impl_(std::make_unique<BufferDistributionSpecImpl>(tensor_shape_in_pages, shard_shape_in_pages, std::move(cores))) {
}

// Destructor
BufferDistributionSpec::~BufferDistributionSpec() = default;

// Copy constructor
BufferDistributionSpec::BufferDistributionSpec(const BufferDistributionSpec& other) :
    impl_(other.impl_ ? std::make_unique<BufferDistributionSpecImpl>(*other.impl_) : nullptr) {}

// Move constructor
BufferDistributionSpec::BufferDistributionSpec(BufferDistributionSpec&& other) noexcept = default;

// Copy assignment operator
BufferDistributionSpec& BufferDistributionSpec::operator=(const BufferDistributionSpec& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<BufferDistributionSpecImpl>(*other.impl_) : nullptr;
    }
    return *this;
}

// Move assignment operator
BufferDistributionSpec& BufferDistributionSpec::operator=(BufferDistributionSpec&& other) noexcept = default;

// Member functions
const Shape& BufferDistributionSpec::tensor_shape_in_pages() const { return impl_->tensor_shape_in_pages(); }

const Shape& BufferDistributionSpec::shard_shape_in_pages() const { return impl_->shard_shape_in_pages(); }

size_t BufferDistributionSpec::num_shards() const { return impl_->num_shards(); }

size_t BufferDistributionSpec::max_num_dev_pages_per_core() const { return impl_->max_num_dev_pages_per_core(); }

size_t BufferDistributionSpec::num_cores_with_data() const { return impl_->num_cores_with_data(); }

const std::vector<CoreCoord>& BufferDistributionSpec::cores_with_data() const { return impl_->cores_with_data(); }

std::tuple<uint32_t, CoreRangeSet, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t>
BufferDistributionSpec::core_groups_tuple() const {
    return impl_->core_groups_tuple();
}

BufferDistributionSpecImpl* BufferDistributionSpec::impl() { return impl_.get(); }

const BufferDistributionSpecImpl* BufferDistributionSpec::impl() const { return impl_.get(); }

}  // namespace tt::tt_metal
