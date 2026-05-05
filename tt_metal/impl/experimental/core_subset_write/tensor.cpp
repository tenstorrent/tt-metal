// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/core_subset_write/mesh_command_queue.hpp>
#include <tt-metalium/experimental/core_subset_write/tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

namespace tt::tt_metal::experimental::core_subset_write {

void enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    const MeshTensor& device_tensor,
    const CoreRangeSet& logical_core_filter) {
    TT_FATAL(
        is_uniform_write(host_tensor, device_tensor.device()),
        "core_subset_write::enqueue_write_tensor requires a uniform host->device transfer.");
    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    // Pure data write into the device tensor's existing buffer; spec/topology of the device tensor
    // are intentionally not modified by a partial host->device copy.
    enqueue_write(cq, device_tensor.mesh_buffer(), host_tensor.buffer(), /*blocking=*/false, logical_core_filter);
}

}  // namespace tt::tt_metal::experimental::core_subset_write
