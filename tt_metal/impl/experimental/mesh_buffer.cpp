// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/mesh_buffer_allocation.hpp>

#include <memory>
#include <utility>

#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal::experimental {

distributed::MeshBuffer allocate_mesh_buffer(
    const distributed::MeshBufferConfig& mesh_buffer_config,
    const distributed::DeviceLocalBufferConfig& device_local_config,
    distributed::MeshDevice* mesh_device,
    std::optional<DeviceAddr> address) {
    // Allocate through the existing factory and move the buffer out of the returned shared_ptr, handing the caller
    // an owning value. The moved-from MeshBuffer left behind is deallocated (a no-op destructor) when the temporary
    // shared_ptr goes out of scope.
    auto mesh_buffer = distributed::MeshBuffer::create(mesh_buffer_config, device_local_config, mesh_device, address);
    return std::move(*mesh_buffer);
}

}  // namespace tt::tt_metal::experimental
