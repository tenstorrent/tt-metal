// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "buffer_distribution_spec.hpp"
#include "assert.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

struct PageMappingIntermData {
    BufferPageMapping* page_mapping = nullptr;

    const size_t num_cores = 0;
    const size_t num_shards_per_core = 0;
    const size_t rank = 0;
    const uint32_t* tensor_shape = nullptr;
    const uint32_t* shard_shape = nullptr;
    const uint64_t shard_volume = 0;
    const uint32_t* shard_grid = nullptr;
    const uint32_t* tensor_strides = nullptr;
    const uint32_t* shard_strides = nullptr;

    uint32_t* actual_shard_size = nullptr;
    size_t shard_id = 0;
};

void iterate_within_shard(
    PageMappingIntermData& params,
    size_t dim,
    size_t src_offset,
    size_t core_id,
    size_t core_offset,
    size_t dst_offset) {
    if (dim == params.rank) {
        params.page_mapping->core_host_page_indices[core_id][dst_offset] = src_offset;

        size_t dev_page = core_offset + dst_offset;
        params.page_mapping->dev_page_to_core_mapping[dev_page] = core_id;
        params.page_mapping->dev_page_to_host_page_mapping[dev_page] = src_offset;
        params.page_mapping->host_page_to_dev_page_mapping[src_offset] = dev_page;
        params.page_mapping->host_page_to_local_shard_page_mapping[src_offset] = dst_offset;
        return;
    }

    for (size_t i = 0; i < params.actual_shard_size[dim]; i++) {
        iterate_within_shard(params, dim + 1, src_offset, core_id, core_offset, dst_offset);
        src_offset += params.tensor_strides[dim];
        dst_offset += params.shard_strides[dim];
    }
}

void iterate_over_shards(
    PageMappingIntermData& params, size_t dim, size_t src_offset, std::array<uint32_t, 2> actual_shard_shape_2d) {
    if (dim == params.rank) {
        size_t core_id = params.shard_id % params.num_cores;
        size_t dst_offset = (params.shard_id / params.num_cores) * params.shard_volume;
        size_t core_offset = core_id * params.num_shards_per_core * params.shard_volume;

        iterate_within_shard(params, 0, src_offset, core_id, core_offset, dst_offset);

        params.shard_id++;
        return;
    }

    size_t shard_size = params.shard_shape[dim];
    params.actual_shard_size[dim] = shard_size;
    actual_shard_shape_2d[0] *= actual_shard_shape_2d[1];
    actual_shard_shape_2d[1] = shard_size;

    for (size_t i = 0; i < params.shard_grid[dim] - 1; i++) {
        iterate_over_shards(
            params,
            dim + 1,
            src_offset + i * params.shard_shape[dim] * params.tensor_strides[dim],
            actual_shard_shape_2d);
    }

    // Last shard may be partial, so we need to handle it separately
    size_t partial_shard_size = params.tensor_shape[dim] % shard_size;
    actual_shard_shape_2d[1] = partial_shard_size == 0 ? shard_size : partial_shard_size;
    params.actual_shard_size[dim] = actual_shard_shape_2d[1];
    iterate_over_shards(
        params,
        dim + 1,
        src_offset + (params.shard_grid[dim] - 1) * params.shard_shape[dim] * params.tensor_strides[dim],
        actual_shard_shape_2d);
}

tt::tt_metal::Shape convert_shape_to_pages(tt::tt_metal::Shape shape, const tt::tt_metal::Shape2D& page_shape) {
    if (shape.rank() >= 1) {
        shape[-1] = (shape[-1] + page_shape.width() - 1) / page_shape.width();
    }
    if (shape.rank() >= 2) {
        shape[-2] = (shape[-2] + page_shape.height() - 1) / page_shape.height();
    }
    return shape;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

BufferDistributionSpec BufferDistributionSpec::from_shard_spec(
    tt::tt_metal::Shape tensor_shape,
    tt::tt_metal::Shape shard_shape,
    tt::tt_metal::Shape2D page_shape,
    CoreRangeSet core_range_set,
    ShardOrientation shard_orientation) {
    auto tensor_shape_in_pages = CMAKE_UNIQUE_NAMESPACE::convert_shape_to_pages(tensor_shape, page_shape);
    auto shard_shape_in_pages = CMAKE_UNIQUE_NAMESPACE::convert_shape_to_pages(shard_shape, page_shape);
    return BufferDistributionSpec(tensor_shape_in_pages, shard_shape_in_pages, core_range_set, shard_orientation);
}

BufferDistributionSpec::BufferDistributionSpec(
    tt::tt_metal::Shape tensor_shape_in_pages,
    tt::tt_metal::Shape shard_shape_in_pages,
    CoreRangeSet core_range_set,
    ShardOrientation shard_orientation) :
    tensor_shape_in_pages_(std::move(tensor_shape_in_pages)),
    shard_shape_in_pages_(std::move(shard_shape_in_pages)),
    shard_orientation_(shard_orientation) {
    cores_ = corerange_to_cores(
        core_range_set, core_range_set.num_cores(), shard_orientation_ == ShardOrientation::ROW_MAJOR);
    TT_FATAL(
        tensor_shape_in_pages_.rank() == shard_shape_in_pages_.rank(),
        "Tensor shape rank ({}) must be same as shard shape rank ({})!",
        tensor_shape_in_pages_.rank(),
        shard_shape_in_pages_.rank());
    TT_FATAL(tensor_shape_in_pages_.rank() >= 1, "Tensor rank must be at least 1!");
    TT_FATAL(shard_shape_in_pages_.volume() != 0, "Shard shape must have non zero volume!");
    if (tensor_shape_in_pages_.volume() != 0) {
        TT_FATAL(cores_.size() != 0, "Can't distribute non zero volume tensor over an empty set of cores");
    }
}

size_t BufferDistributionSpec::num_shards() const {
    if (tensor_shape_in_pages_.volume() == 0) {
        return 0;
    }
    size_t num_shards = 1;
    for (size_t i = 0; i < tensor_shape_in_pages_.size(); i++) {
        num_shards *= (tensor_shape_in_pages_[i] + shard_shape_in_pages_[i] - 1) / shard_shape_in_pages_[i];
    }
    return num_shards;
}

size_t BufferDistributionSpec::num_shards_per_core() const {
    if (cores_.size() == 0) {
        return 0;
    }
    return (num_shards() + cores_.size() - 1) / cores_.size();
}

size_t BufferDistributionSpec::num_dev_pages_per_core() const {
    return num_shards_per_core() * shard_shape_in_pages_.volume();
}

BufferPageMapping BufferDistributionSpec::compute_page_mapping() const {
    BufferPageMapping page_mapping;
    page_mapping.all_cores = cores_;
    for (size_t i = 0; i < cores_.size(); i++) {
        page_mapping.core_to_core_id[cores_[i]] = i;
    }

    if (tensor_shape_in_pages_.volume() == 0) {
        return page_mapping;
    }

    size_t num_shards_per_core = this->num_shards_per_core();
    size_t shard_pages = shard_shape_in_pages_.volume();
    size_t host_pages = tensor_shape_in_pages_.volume();
    size_t dev_pages = cores_.size() * num_shards_per_core * shard_pages;

    page_mapping.core_host_page_indices.resize(cores_.size());
    for (size_t i = 0; i < cores_.size(); i++) {
        page_mapping.core_host_page_indices[i].resize(num_shards_per_core * shard_pages);
    }
    page_mapping.dev_page_to_core_mapping.resize(dev_pages);
    page_mapping.dev_page_to_host_page_mapping.resize(dev_pages);
    page_mapping.host_page_to_dev_page_mapping.resize(host_pages);
    page_mapping.host_page_to_local_shard_page_mapping.resize(host_pages);

    tt::stl::SmallVector<uint32_t> shard_grid(tensor_shape_in_pages_.rank());
    for (size_t i = 0; i < tensor_shape_in_pages_.rank(); i++) {
        shard_grid[i] = (tensor_shape_in_pages_[i] + shard_shape_in_pages_[i] - 1) / shard_shape_in_pages_[i];
    }

    tt::stl::SmallVector<uint32_t> tensor_strides = tt::tt_metal::compute_strides(tensor_shape_in_pages_);
    tt::stl::SmallVector<uint32_t> shard_strides = tt::tt_metal::compute_strides(shard_shape_in_pages_);
    tt::stl::SmallVector<uint32_t> actual_shard_size(tensor_shape_in_pages_.rank());

    CMAKE_UNIQUE_NAMESPACE::PageMappingIntermData params{
        .page_mapping = &page_mapping,
        .num_cores = cores_.size(),
        .num_shards_per_core = num_shards_per_core,
        .rank = tensor_shape_in_pages_.rank(),
        .tensor_shape = tensor_shape_in_pages_.view().data(),
        .shard_shape = shard_shape_in_pages_.view().data(),
        .shard_volume = shard_shape_in_pages_.volume(),
        .shard_grid = shard_grid.data(),
        .tensor_strides = tensor_strides.data(),
        .shard_strides = shard_strides.data(),
        .actual_shard_size = actual_shard_size.data(),
        .shard_id = 0,
    };
    CMAKE_UNIQUE_NAMESPACE::iterate_over_shards(params, 0, 0, {1, 1});

    return page_mapping;
}

}  // namespace tt::tt_metal
