// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/tensor/tensor_utils.hpp>

#include <vector>

#include <tt_stl/reflection.hpp>

namespace tt::tt_metal {

bool logical_matches_physical(const TensorSpec& tensor_spec) {
    return tensor_spec.layout() == Layout::ROW_MAJOR && tensor_spec.logical_2d_shape() == tensor_spec.physical_shape();
}

namespace host_buffer {

HostBuffer get_host_buffer(const HostTensor& tensor) {
    const auto& storage = tensor.get_legacy_host_storage();
    std::vector<HostBuffer> buffers;
    storage.buffer().apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
    TT_FATAL(
        buffers.size() == 1,
        "Can't get a single buffer from host storage distributed over mesh shape {}",
        storage.buffer().shape());
    return buffers.front();
}

}  // namespace host_buffer

}  // namespace tt::tt_metal
