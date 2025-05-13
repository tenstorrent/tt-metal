// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/span.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/assert.hpp>

#include <functional>
#include <vector>

namespace tt::tt_metal {

DistributedHostBuffer DistributedHostBuffer::create(size_t global_size) {
    return DistributedHostBuffer(
        [](size_t global_index) { return global_index; }, std::vector<HostBuffer>(global_size));
}

DistributedHostBuffer DistributedHostBuffer::create(
    const distributed::MeshShape& global_shape,
    const distributed::MeshShape& local_shape,
    const distributed::MeshCoordinate& local_offset) {
    TT_FATAL(
        global_shape.dims() == local_shape.dims(),
        "Global shape {} and local shape {} must have the same dimensions",
        global_shape,
        local_shape);
    TT_FATAL(
        global_shape.dims() == local_offset.dims(),
        "Global shape {} and local offset {} must have the same dimensions",
        global_shape,
        local_offset);

    for (size_t dim = 0; dim < global_shape.dims(); ++dim) {
        TT_FATAL(
            local_offset[dim] + local_shape[dim] <= global_shape[dim],
            "Local shape exceeds global shape at dimension {}: {} + {} > {}",
            dim,
            local_offset[dim],
            local_shape[dim],
            global_shape[dim]);
    }

    auto global_to_local_index = [global_shape, local_shape, local_offset](size_t global_idx) -> std::optional<size_t> {
        TT_FATAL(
            global_idx < global_shape.mesh_size(),
            "Global index {} is out of bounds for global shape {}",
            global_idx,
            global_shape);

        size_t local_idx = 0;
        size_t remaining = global_idx;
        for (size_t dim = 0; dim < global_shape.dims(); ++dim) {
            const uint32_t coord = remaining / global_shape.get_stride(dim);
            remaining %= global_shape.get_stride(dim);

            if (coord < local_offset[dim] || coord >= local_offset[dim] + local_shape[dim]) {
                return std::nullopt;
            }

            local_idx += (coord - local_offset[dim]) * local_shape.get_stride(dim);
        }

        return local_idx;
    };
    return DistributedHostBuffer(std::move(global_to_local_index), std::vector<HostBuffer>(local_shape.mesh_size()));
}

std::optional<HostBuffer> DistributedHostBuffer::get_shard(size_t linear_index) const {
    const auto local_index = global_to_local_index_(linear_index);
    return local_index.has_value() ? std::make_optional(local_buffers_.at(local_index.value())) : std::nullopt;
}

void DistributedHostBuffer::emplace_shard(size_t linear_index, HostBuffer buffer) {
    const auto local_index = global_to_local_index_(linear_index);
    if (local_index.has_value()) {
        local_buffers_.at(local_index.value()) = std::move(buffer);
    }
}

void DistributedHostBuffer::transform(const TransformFn& fn) {
    for (size_t i = 0; i < local_buffers_.size(); ++i) {
        local_buffers_.at(i) = fn(local_buffers_.at(i), i);
    }
}

void DistributedHostBuffer::apply(const ApplyFn& fn) {
    for (size_t i = 0; i < local_buffers_.size(); ++i) {
        fn(local_buffers_.at(i), i);
    }
}

bool DistributedHostBuffer::is_allocated() const {
    return std::all_of(
        local_buffers_.begin(), local_buffers_.end(), [](const HostBuffer& b) { return b.is_allocated(); });
}

void DistributedHostBuffer::deallocate() {
    for (auto& buffer : local_buffers_) {
        buffer.deallocate();
    }
}

}  // namespace tt::tt_metal
