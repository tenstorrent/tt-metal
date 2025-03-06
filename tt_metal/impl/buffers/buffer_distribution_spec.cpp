// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "buffer_distribution_spec.hpp"
#include "work_split.hpp"
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
    const DistributionSpec& page_distribution_spec, const std::vector<CoreCoord>& cores) :
    page_distribution_spec_(page_distribution_spec), cores_(cores) {};

BufferDistributionSpec BufferDistributionSpec::from_primitives(
    const tt::tt_metal::Shape& tensor_shape,
    const tt::tt_metal::Shape& physical_shard_shape,
    const CoreCoord& grid_size,
    const ShardOrientation shard_orientation,
    const Shape2D& page_shape) {
    TT_FATAL(
        physical_shard_shape.rank() == tensor_shape.rank(),
        "Physical shard shape rank ({}) must be same as tensor shape rank ({})!",
        physical_shard_shape.rank(),
        tensor_shape.rank());
    auto tensor_shape_in_pages = CMAKE_UNIQUE_NAMESPACE::convert_shape_to_shape_in_pages(tensor_shape, page_shape);
    auto shard_shape_in_pages =
        CMAKE_UNIQUE_NAMESPACE::convert_shape_to_shape_in_pages(physical_shard_shape, page_shape);
    auto page_distribution_spec =
        DistributionSpec::from_shard_shape(tensor_shape_in_pages, shard_shape_in_pages, grid_size.x * grid_size.y);

    const bool row_major = shard_orientation == ShardOrientation::ROW_MAJOR;
    auto cores = grid_to_cores(page_distribution_spec.get_num_targets(), grid_size.x, grid_size.y, row_major);

    return BufferDistributionSpec(page_distribution_spec, cores);
}

}  // namespace tt::tt_metal
