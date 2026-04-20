// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iterator>

#include <tt-metalium/experimental/core_subset_write/mesh_command_queue.hpp>
#include <tt-metalium/experimental/core_subset_write/tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

namespace tt::tt_metal::experimental::core_subset_write {

void enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    MeshTensor& device_tensor,
    const CoreRangeSet& logical_core_filter) {
    TT_FATAL(
        is_uniform_write(host_tensor, device_tensor.device()),
        "Incompatible shape between source host tensor and target MeshDevice. For non-uniform transfers, use the "
        "non-uniform data movement APIs.");
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();

    enqueue_write(cq, mesh_buffer, host_tensor.buffer(), /*blocking=*/false, logical_core_filter);
    device_tensor = MeshTensor(
        mesh_buffer,
        host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()),
        host_tensor.tensor_topology());
}

std::vector<distributed::MeshCoordinate> enqueue_write_tensor_non_uniform(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    MeshTensor& device_tensor,
    const CoreRangeSet& logical_core_filter) {
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& host_storage_shape = host_tensor.buffer().shape();
    const auto& dst_device_shape = device_tensor.device().shape();

    if (host_storage_shape.mesh_size() < dst_device_shape.mesh_size() &&
        host_storage_shape == distributed::MeshShape(1, 1)) {
        TT_FATAL(
            false,
            "enqueue_write_tensor with logical_core_filter does not support 1x1 host tensor replicated to a larger "
            "mesh; use the unfiltered enqueue_write_tensor path.");
    }

    auto mesh_buffer = device_tensor.mesh_buffer_invariant_breaking();
    enqueue_write(cq, mesh_buffer, host_tensor.buffer(), /*blocking=*/false, logical_core_filter);

    std::vector<distributed::MeshCoordinate> coords;
    const auto& shard_coords = host_tensor.buffer().shard_coords();
    coords.reserve(shard_coords.size());
    std::copy(shard_coords.begin(), shard_coords.end(), std::back_inserter(coords));

    device_tensor = MeshTensor(
        mesh_buffer,
        host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()),
        host_tensor.tensor_topology());

    return coords;
}

}  // namespace tt::tt_metal::experimental::core_subset_write
