#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/shape2d.hpp>

namespace tt::tt_metal {
// Helper to create a sharded buffer config similar to Python's mesh_mapper
// cluster_axis: 0 = shard along height (rows), 1 = shard along width (cols)
distributed::ShardedBufferConfig create_sharded_buffer_config(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const Shape2D& global_shape,
    uint32_t element_size_bytes,
    int cluster_axis = 0) {
    // Calculate shard shape based on cluster_axis
    Shape2D shard_shape(0, 0);
    if (cluster_axis == 0) {
        // Shard along height (rows) - divide height by num_rows
        shard_shape = Shape2D(
            global_shape.height() / mesh_device->num_rows(),
            global_shape.width()  // width is preserved
        );
    } else {
        // Shard along width (cols) - divide width by num_cols
        shard_shape = Shape2D(
            global_shape.height(),  // height is preserved
            global_shape.width() / mesh_device->num_cols());
    }

    // Calculate global buffer size
    uint32_t global_size = global_shape.height() * global_shape.width() * element_size_bytes;

    return distributed::ShardedBufferConfig{
        .global_size = global_size,
        .global_buffer_shape = global_shape,
        .shard_shape = shard_shape,
        .shard_orientation = (cluster_axis == 0) ? ShardOrientation::ROW_MAJOR : ShardOrientation::COL_MAJOR};
}
}  // namespace tt::tt_metal
