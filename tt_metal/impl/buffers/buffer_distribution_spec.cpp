// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "buffer_distribution_spec.hpp"
#include "assert.hpp"

#include <algorithm>

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

struct PageMappingIntermData {
    UncompressedBufferPageMapping* page_mapping = nullptr;

    const size_t num_cores = 0;
    const size_t rank = 0;
    const uint32_t* tensor_shape = nullptr;
    const uint32_t* shard_shape = nullptr;
    const uint64_t shard_volume = 0;
    const uint32_t* shard_grid = nullptr;
    const uint32_t* tensor_strides = nullptr;
    const uint32_t* shard_strides = nullptr;

    uint32_t* actual_shard_size = nullptr;
    size_t core_id = 0;
    size_t shard_id = 0;
};

void iterate_within_shard(PageMappingIntermData& params, size_t dim, size_t src_offset, size_t dst_offset) {
    if (dim == params.rank) {
        params.page_mapping->core_host_page_indices[params.core_id][dst_offset] = src_offset;
        return;
    }

    for (size_t i = 0; i < params.actual_shard_size[dim]; i++) {
        iterate_within_shard(params, dim + 1, src_offset, dst_offset);
        src_offset += params.tensor_strides[dim];
        dst_offset += params.shard_strides[dim];
    }
}

void iterate_over_shards(PageMappingIntermData& params, size_t dim, size_t src_offset) {
    if (dim == params.rank) {
        params.core_id = params.shard_id % params.num_cores;
        size_t dst_offset = (params.shard_id / params.num_cores) * params.shard_volume;

        iterate_within_shard(params, 0, src_offset, dst_offset);

        params.shard_id++;
        return;
    }

    size_t shard_size = params.shard_shape[dim];
    params.actual_shard_size[dim] = shard_size;

    for (size_t i = 0; i < params.shard_grid[dim] - 1; i++) {
        iterate_over_shards(params, dim + 1, src_offset + i * params.shard_shape[dim] * params.tensor_strides[dim]);
    }

    // Last shard may be partial, so we need to handle it separately
    size_t partial_shard_size = params.tensor_shape[dim] % shard_size;
    params.actual_shard_size[dim] = partial_shard_size == 0 ? shard_size : partial_shard_size;
    iterate_over_shards(
        params,
        dim + 1,
        src_offset + (params.shard_grid[dim] - 1) * params.shard_shape[dim] * params.tensor_strides[dim]);
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

std::pair<Shape, Shape> squeeze_shape_ranks(const Shape& tensor_shape, const Shape& shard_shape) {
    TT_FATAL(
        tensor_shape.rank() >= shard_shape.rank(),
        "Tensor shape rank ({}) can't be less than shard shape rank ({})!",
        tensor_shape.rank(),
        shard_shape.rank());

    uint64_t tensor_volume = tensor_shape.volume();
    uint64_t shard_volume = shard_shape.volume();
    tt::stl::SmallVector<uint32_t> new_tensor_shape;
    tt::stl::SmallVector<uint32_t> new_shard_shape;

    bool last_dim_identical = false;
    bool last_dim_divisible = false;
    uint64_t cur_tensor_volume = 1;
    uint64_t cur_shard_volume = 1;
    for (int dim = -1; dim >= -static_cast<int>(shard_shape.rank()); dim--) {
        auto tensor_size = tensor_shape[dim];
        auto shard_size = shard_shape[dim];

        bool should_merge_dims = false;
        if (dim < -2) {
            should_merge_dims = last_dim_identical || (shard_size == 1 && last_dim_divisible);
        }

        if (should_merge_dims) {
            new_tensor_shape.back() *= tensor_size;
            new_shard_shape.back() *= shard_size;
        } else {
            new_tensor_shape.push_back(tensor_size);
            new_shard_shape.push_back(shard_size);
        }

        cur_tensor_volume *= tensor_size;
        cur_shard_volume *= shard_size;
        if (cur_tensor_volume == tensor_volume && cur_shard_volume == shard_volume) {
            break;
        }

        last_dim_identical = tensor_size == shard_size;
        last_dim_divisible = tensor_size % shard_size == 0;
    }

    for (int dim = -static_cast<int>(shard_shape.rank()) - 1; dim >= -static_cast<int>(tensor_shape.rank()); dim--) {
        new_tensor_shape.back() *= tensor_shape[dim];
    }

    std::reverse(new_tensor_shape.begin(), new_tensor_shape.end());
    std::reverse(new_shard_shape.begin(), new_shard_shape.end());
    return {Shape(std::move(new_tensor_shape)), Shape(std::move(new_shard_shape))};
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
    shard_orientation_(shard_orientation) {
    cores_ = corerange_to_cores(
        core_range_set, core_range_set.num_cores(), shard_orientation_ == ShardOrientation::ROW_MAJOR);
    TT_FATAL(tensor_shape_in_pages.rank() >= 1, "Tensor rank must be at least 1!");
    TT_FATAL(shard_shape_in_pages.rank() >= 1, "Shard rank must be at least 1!");
    TT_FATAL(shard_shape_in_pages.volume() != 0, "Shard shape must have non zero volume!");
    if (tensor_shape_in_pages.volume() != 0) {
        TT_FATAL(cores_.size() != 0, "Can't distribute non zero volume tensor over an empty set of cores");
    }
    std::tie(tensor_shape_in_pages_, shard_shape_in_pages_) =
        CMAKE_UNIQUE_NAMESPACE::squeeze_shape_ranks(tensor_shape_in_pages, shard_shape_in_pages);
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

size_t BufferDistributionSpec::num_cores_with_data() const { return std::min(num_cores(), num_shards()); }

std::vector<CoreCoord> BufferDistributionSpec::get_cores_with_data() const {
    return std::vector<CoreCoord>(cores_.begin(), cores_.begin() + num_cores_with_data());
}

size_t BufferDistributionSpec::max_num_shards_per_core() const {
    if (cores_.size() == 0) {
        return 0;
    }
    return (num_shards() + cores_.size() - 1) / cores_.size();
}

size_t BufferDistributionSpec::max_num_dev_pages_per_core() const {
    return max_num_shards_per_core() * shard_shape_in_pages_.volume();
}

size_t BufferDistributionSpec::num_shards_per_core(size_t core_idx) const {
    auto num_shards = this->num_shards();
    return num_shards / num_cores() + (core_idx < num_shards % num_cores() ? 1 : 0);
}

size_t BufferDistributionSpec::num_dev_pages_per_core(size_t core_idx) const {
    return num_shards_per_core(core_idx) * shard_shape_in_pages_.volume();
}

std::pair<BufferDistributionSpec::CoreGroup, BufferDistributionSpec::CoreGroup>
BufferDistributionSpec::get_core_groups_by_num_shards() const {
    auto num_shards = this->num_shards();
    if (num_shards == 0) {
        return {CoreGroup{}, CoreGroup{}};
    }

    auto num_cores_with_more_shards = num_shards % num_cores();
    if (num_cores_with_more_shards == 0) {
        return {CoreGroup{num_shards / num_cores(), cores_}, CoreGroup{}};
    }

    std::vector<CoreCoord> cores_with_more_shards(cores_.begin(), cores_.begin() + num_cores_with_more_shards);
    std::vector<CoreCoord> cores_with_less_shards;
    if (num_shards / num_cores() != 0) {
        cores_with_less_shards = std::vector<CoreCoord>(cores_.begin() + num_cores_with_more_shards, cores_.end());
    }
    return {
        CoreGroup{num_shards / num_cores() + 1, std::move(cores_with_more_shards)},
        CoreGroup{num_shards / num_cores(), std::move(cores_with_less_shards)},
    };
}

namespace detail {
UncompressedBufferPageMapping compute_page_mapping(
    const Shape& tensor_shape, const Shape& shard_shape, const std::vector<CoreCoord>& cores) {
    UncompressedBufferPageMapping page_mapping;
    page_mapping.all_cores = cores;

    if (tensor_shape.volume() == 0) {
        return page_mapping;
    }

    size_t num_shards = 1;
    for (size_t i = 0; i < tensor_shape.rank(); i++) {
        num_shards *= (tensor_shape[i] + shard_shape[i] - 1) / shard_shape[i];
    }

    size_t num_shards_per_core = (num_shards + cores.size() - 1) / cores.size();
    size_t shard_pages = shard_shape.volume();
    size_t dev_pages = cores.size() * num_shards_per_core * shard_pages;

    page_mapping.core_host_page_indices.resize(cores.size());
    for (size_t i = 0; i < cores.size(); i++) {
        page_mapping.core_host_page_indices[i].resize(
            num_shards_per_core * shard_pages, UncompressedBufferPageMapping::PADDING);
    }

    tt::stl::SmallVector<uint32_t> shard_grid(tensor_shape.rank());
    for (size_t i = 0; i < tensor_shape.rank(); i++) {
        shard_grid[i] = (tensor_shape[i] + shard_shape[i] - 1) / shard_shape[i];
    }

    tt::stl::SmallVector<uint32_t> tensor_strides = tt::tt_metal::compute_strides(tensor_shape);
    tt::stl::SmallVector<uint32_t> shard_strides = tt::tt_metal::compute_strides(shard_shape);
    tt::stl::SmallVector<uint32_t> actual_shard_size(tensor_shape.rank());

    CMAKE_UNIQUE_NAMESPACE::PageMappingIntermData params{
        .page_mapping = &page_mapping,
        .num_cores = cores.size(),
        .rank = tensor_shape.rank(),
        .tensor_shape = tensor_shape.view().data(),
        .shard_shape = shard_shape.view().data(),
        .shard_volume = shard_pages,
        .shard_grid = shard_grid.data(),
        .tensor_strides = tensor_strides.data(),
        .shard_strides = shard_strides.data(),
        .actual_shard_size = actual_shard_size.data(),
        .core_id = 0,
        .shard_id = 0,
    };
    CMAKE_UNIQUE_NAMESPACE::iterate_over_shards(params, 0, 0);

    return page_mapping;
}
}  // namespace detail

}  // namespace tt::tt_metal
