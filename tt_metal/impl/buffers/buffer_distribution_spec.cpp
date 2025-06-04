// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "buffer_distribution_spec.hpp"
#include "assert.hpp"

namespace tt::tt_metal {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

tt::tt_metal::Shape convert_shape_to_shape_in_pages(const tt::tt_metal::Shape& shape, const Shape2D& page_shape) {
    auto shape_in_pages = shape;
    TT_FATAL(
        shape[-2] % page_shape.height() == 0,
        "Shape height ({}) must be divisible by page height ({})!",
        shape[-2],
        page_shape.height());
    TT_FATAL(
        shape[-1] % page_shape.width() == 0,
        "Shape width ({}) must be divisible by page width ({})!",
        shape[-1],
        page_shape.width());
    shape_in_pages[-2] /= page_shape.height();
    shape_in_pages[-1] /= page_shape.width();
    return shape_in_pages;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

BufferDistributionSpec::BufferDistributionSpec(
    tt::tt_metal::Shape tensor_shape_in_pages,
    tt::tt_metal::Shape shard_shape_in_pages,
    CoreRangeSet core_range_set,
    ShardOrientation shard_orientation) :
    tensor_shape_in_pages_(std::move(tensor_shape_in_pages)),
    shard_shape_in_pages_(std::move(shard_shape_in_pages)),
    shard_orientation_(shard_orientation) {
    TT_FATAL(
        tensor_shape_in_pages.rank() == shard_shape_in_pages.rank(),
        "Tensor shape rank ({}) must be same as shard shape rank ({})!",
        tensor_shape_in_pages.rank(),
        shard_shape_in_pages.rank());
    TT_FATAL(shard_shape_in_pages.volume() == 0, "Shard shape must have non zero volume!");
    TT_FATAL(
        tensor_shape_in_pages.volume() != 0 || cores_.size() == 0,
        "Can't distribute non zero volume tensor over an empty set of cores");

    compute_page_mapping();
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

void BufferDistributionSpec::compute_page_mapping() {
    size_t num_shards_per_core = this->num_shards_per_core();
    size_t shard_pages = shard_shape_in_pages_.volume();
    page_mapping_.resize(cores_.size());
    for (size_t i = 0; i < cores_.size(); i++) {
        page_mapping_[i].resize(num_shards_per_core * shard_pages);
    }
}

/*
BufferDistributionSpec::BufferDistributionSpec(
    const DistributionSpec& page_distribution_spec, const std::vector<CoreCoord>& cores) :
    page_distribution_spec_(page_distribution_spec), cores_(cores) {};

BufferDistributionSpec BufferDistributionSpec::from_shard_spec(
    const tt::tt_metal::Shape& tensor_shape,
    const tt::tt_metal::Shape& physical_shard_shape,
    const Shape2D& page_shape,
    const CoreRangeSet& corerangeset,
    const ShardOrientation shard_orientation) {
    TT_FATAL(
        physical_shard_shape.rank() == tensor_shape.rank(),
        "Physical shard shape rank ({}) must be same as tensor shape rank ({})!",
        physical_shard_shape.rank(),
        tensor_shape.rank());
    auto tensor_shape_in_pages = CMAKE_UNIQUE_NAMESPACE::convert_shape_to_shape_in_pages(tensor_shape, page_shape);
    auto shard_shape_in_pages =
        CMAKE_UNIQUE_NAMESPACE::convert_shape_to_shape_in_pages(physical_shard_shape, page_shape);
    auto page_distribution_spec =
        DistributionSpec::from_shard_shape(tensor_shape_in_pages, shard_shape_in_pages, corerangeset.num_cores());

    const bool row_major = shard_orientation == ShardOrientation::ROW_MAJOR;
    auto cores = corerange_to_cores(corerangeset, page_distribution_spec.get_num_targets(), row_major);

    return BufferDistributionSpec(page_distribution_spec, cores);
}

const std::vector<DistributionSpec::TargetData>& BufferDistributionSpec::get_page_mapping(
    DistributionSpec::MappingMode mapping_mode) {
    const auto& page_mapping = page_distribution_spec_.get_metadata_for_targets(mapping_mode);
    TT_FATAL(
        page_mapping.size() == cores_.size(),
        "Number of targets for page mapping {} must match number of cores {}!",
        page_mapping.size(),
        cores_.size());
    return page_mapping;
};
*/

}  // namespace tt::tt_metal
