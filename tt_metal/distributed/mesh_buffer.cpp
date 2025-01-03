
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_buffer.hpp"

namespace tt::tt_metal::distributed {

MeshBuffer MeshBuffer::create(
    const MeshBufferConfig& mesh_buffer_config,
    const DeviceLocalLayoutConfig& device_local_layout,
    BufferType buffer_type,
    MeshDevice* mesh_device) {
    DeviceAddr device_local_size = std::visit(
        tt::stl::overloaded{
            [](const ReplicatedBufferConfig& c) { return c.buffer_size; },
            [mesh_device](const ShardedBufferConfig& config) {
                const auto [mesh_height, mesh_width] = mesh_device->shape();
                const auto [global_buffer_height, global_buffer_width] = config.global_buffer_shape;
                TT_FATAL(
                    (global_buffer_height % mesh_height == 0) and (global_buffer_width % mesh_width == 0),
                    "Global buffer shape must be aligned with the mesh shape: requested buffer shape: {}, {}, mesh "
                    "shape: {}, {}",
                    global_buffer_height,
                    global_buffer_width,
                    mesh_height,
                    mesh_width);
                return config.global_buffer_size / mesh_device->num_devices();
            }},
        mesh_buffer_config);

    return MeshBuffer(mesh_buffer_config, device_local_layout, device_local_size, mesh_device);
}
}  // namespace tt::tt_metal::distributed
